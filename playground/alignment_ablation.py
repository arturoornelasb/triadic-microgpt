"""
E2: Alignment Loss Ablation — Isolate which triadic loss component drives semantic quality.

Trains 3 XL variants of the Run 15 config, each removing one loss component:
  1. FULL:       alpha=0.05, align=5.0, entropy=1.0  (Run 15 exact — control)
  2. NO_ALIGN:   alpha=0.05, align=0.0, entropy=1.0  (removes embedding alignment)
  3. NO_ENTROPY: alpha=0.05, align=5.0, entropy=0.0  (removes entropy regularization / diversity)

All share the same tokenizer (Run 15), seed (42), XL config (12L/512D/8H/64bits),
50K steps, batch 64. Each saves to playground/results/alignment_ablation/{variant}/.

After training, each variant is evaluated on:
  - PPL on held-out TinyStories (last 200 stories)
  - Semantic gap (13 related + 9 unrelated pairs)
  - Analogy verification (13 quadruples)
  - Dead bits and mean entropy
  - Semantic ordering: king-queen > king-dog

Each variant takes ~76 min on RTX 5060 Ti (total ~4h for all 3).

Usage:
  python playground/alignment_ablation.py --all                    # all 3 sequential
  python playground/alignment_ablation.py --variant full           # single variant
  python playground/alignment_ablation.py --variant no_align       # single variant
  python playground/alignment_ablation.py --variant no_entropy     # single variant
  python playground/alignment_ablation.py --aggregate-only         # compare existing results
"""

import os
import sys
import csv
import json
import math
import time
import random
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.torch_transformer import TriadicGPT, TriadicGPTConfig
try:
    from src.fast_tokenizer import FastBPETokenizer as BPETokenizer
except ImportError:
    from src.tokenizer import BPETokenizer
from src.triadic import PrimeMapper, TriadicValidator

# ============================================================
# Constants
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'TinyStories-train.txt')
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, 'checkpoints', 'torch_run15_strongalign', 'tokenizer.json')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'playground', 'results', 'alignment_ablation')
STORY_SEPARATOR = '<' + '|endoftext|' + '>'

SEED = 42

# Run 15 production config (shared base)
BASE_CONFIG = dict(
    n_layer=12, n_embd=512, n_head=8, n_triadic_bits=64,
    block_size=256, dropout=0.1,
    lr=3e-4, alpha=0.05, align_mode='mse', triadic_warmup_pct=0.8,
    batch_size=64, steps=50000, stories=50000, vocab=4096,
)

# Three ablation variants
VARIANTS = {
    'full': {
        'description': 'Run 15 exact (control)',
        'align_weight': 5.0,
        'entropy_weight': 1.0,
    },
    'no_align': {
        'description': 'No embedding alignment (align=0)',
        'align_weight': 0.0,
        'entropy_weight': 1.0,
    },
    'no_entropy': {
        'description': 'No entropy regularization (entropy=0)',
        'align_weight': 5.0,
        'entropy_weight': 0.0,
    },
}

# Evaluation concept pairs and analogies (same as Run 15 / multi_seed)
CONCEPT_PAIRS = {
    'related': [
        ("king", "queen"), ("dog", "cat"), ("happy", "sad"), ("mother", "father"),
        ("sun", "moon"), ("hot", "cold"), ("love", "hate"), ("big", "small"),
        ("bird", "fish"), ("doctor", "hospital"), ("teacher", "school"),
        ("princess", "prince"), ("old", "young"),
    ],
    'unrelated': [
        ("king", "fish"), ("dog", "moon"), ("happy", "river"), ("mother", "blue"),
        ("sun", "cat"), ("hot", "queen"), ("bird", "school"), ("love", "tree"),
        ("big", "night"),
    ],
}

ANALOGY_TRIPLES = [
    ("king", "queen", "man", "woman"),
    ("father", "mother", "brother", "sister"),
    ("father", "mother", "son", "daughter"),
    ("dog", "puppy", "cat", "kitten"),
    ("big", "small", "tall", "short"),
    ("hot", "cold", "day", "night"),
    ("happy", "sad", "love", "hate"),
    ("princess", "prince", "queen", "king"),
    ("bird", "fly", "fish", "swim"),
    ("old", "young", "big", "small"),
    ("doctor", "hospital", "teacher", "school"),
    ("sun", "day", "moon", "night"),
    ("red", "blue", "green", "yellow"),
]


# ============================================================
# Dataset
# ============================================================

class TextDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.tokens) - self.block_size - 1)

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.block_size + 1]
        return torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)


# ============================================================
# Reproducibility
# ============================================================

def set_all_seeds(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Data Loading
# ============================================================

def load_and_tokenize(tokenizer):
    """Load stories, shuffle with fixed seed (42), tokenize."""
    print("  Loading corpus...")
    with open(DATA_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()

    stories = [s.strip() for s in raw.split(STORY_SEPARATOR) if s.strip() and len(s.strip()) > 30]

    random.seed(SEED)
    random.shuffle(stories)
    stories = stories[:BASE_CONFIG['stories']]
    print(f"  Documents: {len(stories)}")

    print("  Tokenizing...")
    t0 = time.time()
    all_tokens = []
    for story in stories:
        ids = tokenizer.encode(story, add_special=True)
        all_tokens.extend(ids)
    tok_time = time.time() - t0
    print(f"  Tokens: {len(all_tokens):,} ({tok_time:.1f}s)")

    return all_tokens


# ============================================================
# Evaluation
# ============================================================

def compute_ppl(model, tokenizer, device, block_size, max_samples=200):
    """Compute perplexity on held-out validation stories (last N)."""
    with open(DATA_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()
    stories = [s.strip() for s in raw.split(STORY_SEPARATOR) if s.strip() and len(s.strip()) > 50]
    val_stories = stories[-max_samples:]

    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for story in val_stories:
            ids = tokenizer.encode(story, add_special=True)
            if len(ids) < 3:
                continue
            ids = ids[:block_size + 1]
            x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
            _, _, loss = model(x, targets=y)
            n = len(ids) - 1
            total_loss += loss.item() * n
            total_tokens += n

    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss), avg_loss


def compute_triadic_metrics(model, tokenizer, device, n_bits):
    """Compute semantic gap, analogy verification, dead bits, entropy, ordering."""
    mapper = PrimeMapper(n_bits)

    # Collect all unique words needed for evaluation
    all_words = set()
    for group in CONCEPT_PAIRS.values():
        for w1, w2 in group:
            all_words.update([w1, w2])
    for a, b, c, d in ANALOGY_TRIPLES:
        all_words.update([a, b, c, d])

    # Get triadic signatures for all concepts
    sigs = {}
    model.eval()
    with torch.no_grad():
        for word in all_words:
            ids = tokenizer.encode(word, add_special=False)
            if ids:
                x = torch.tensor([ids], dtype=torch.long, device=device)
                _, proj, _ = model(x)
                sigs[word] = proj[0].mean(dim=0).cpu().numpy()

    def cosine(a, b):
        dot = float(np.dot(a, b))
        norm = float(np.linalg.norm(a) * np.linalg.norm(b))
        return dot / (norm + 1e-10)

    # --- Semantic gap ---
    related_sims = [cosine(sigs[w1], sigs[w2])
                    for w1, w2 in CONCEPT_PAIRS['related'] if w1 in sigs and w2 in sigs]

    # Random baseline: 200 random pairs from all concept signatures
    rand_sims = []
    words = list(sigs.keys())
    rng = np.random.RandomState(0)  # fixed for eval consistency
    for _ in range(200):
        i, j = rng.choice(len(words), 2, replace=False)
        rand_sims.append(cosine(sigs[words[i]], sigs[words[j]]))
    gap = float(np.mean(related_sims) - np.mean(rand_sims))

    # --- Analogy verification ---
    correct, total = 0, 0
    for a, b, c, d in ANALOGY_TRIPLES:
        if not all(w in sigs for w in [a, b, c, d]):
            continue
        predicted = TriadicValidator.analogy(
            mapper.map(sigs[a]), mapper.map(sigs[b]), mapper.map(sigs[c]))
        if TriadicValidator.similarity(predicted, mapper.map(sigs[d])) > 0.3:
            correct += 1
        total += 1
    analogy_verif = correct / max(total, 1)

    # --- Bit entropy and dead bits ---
    all_projs = np.stack(list(sigs.values()))
    bit_means = (all_projs > 0).mean(axis=0)  # fraction of concepts where bit is active
    eps = 1e-7
    bit_entropy = -(bit_means * np.log2(bit_means + eps) +
                    (1 - bit_means) * np.log2(1 - bit_means + eps))
    dead_bits = int((bit_entropy < 0.3).sum())
    mean_entropy = float(bit_entropy.mean())

    # --- Semantic ordering: king-queen vs king-dog ---
    kq = cosine(sigs['king'], sigs['queen']) if 'king' in sigs and 'queen' in sigs else 0
    kd = cosine(sigs['king'], sigs['dog']) if 'king' in sigs and 'dog' in sigs else 0
    ordering_correct = kq > kd

    return {
        'semantic_gap': gap,
        'mean_related_sim': float(np.mean(related_sims)) if related_sims else 0.0,
        'mean_random_sim': float(np.mean(rand_sims)) if rand_sims else 0.0,
        'analogy_verification': analogy_verif,
        'analogy_correct': correct,
        'analogy_total': total,
        'dead_bits': dead_bits,
        'mean_entropy': mean_entropy,
        'ordering_correct': ordering_correct,
        'king_queen_sim': float(kq),
        'king_dog_sim': float(kd),
    }


# ============================================================
# Training loop for one variant
# ============================================================

def train_variant(variant_name, device):
    """Train a single ablation variant. Returns metrics dict."""
    variant = VARIANTS[variant_name]
    cfg = BASE_CONFIG
    align_weight = variant['align_weight']
    entropy_weight = variant['entropy_weight']

    ckpt_dir = os.path.join(RESULTS_DIR, variant_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    print()
    print("=" * 70)
    print(f"  E2: ALIGNMENT ABLATION — {variant_name.upper()}")
    print(f"  {variant['description']}")
    print(f"  align_weight={align_weight}, entropy_weight={entropy_weight}")
    print("=" * 70)

    # Set all seeds for reproducibility
    set_all_seeds(SEED)

    # Load shared tokenizer (Run 15)
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    actual_vocab = tokenizer.vocab_size
    print(f"  Tokenizer: {TOKENIZER_PATH} (vocab: {actual_vocab})")

    # Tokenize corpus
    all_tokens = load_and_tokenize(tokenizer)

    # Initialize model
    set_all_seeds(SEED)  # reset before model init for identical initialization
    config = TriadicGPTConfig(
        vocab_size=actual_vocab, block_size=cfg['block_size'],
        n_layer=cfg['n_layer'], n_embd=cfg['n_embd'], n_head=cfg['n_head'],
        n_triadic_bits=cfg['n_triadic_bits'], dropout=cfg['dropout'],
    )
    model = TriadicGPT(config).to(device)
    total_params = model.num_params()
    print(f"  Model: {total_params:,} params ({cfg['n_layer']}L/{cfg['n_embd']}D/{cfg['n_head']}H/{cfg['n_triadic_bits']}bits)")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=0.01, betas=(0.9, 0.95))

    # DataLoader
    dataset = TextDataset(all_tokens, cfg['block_size'])
    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, drop_last=True,
                            num_workers=0, generator=torch.Generator().manual_seed(SEED))

    # Mixed precision scaler
    amp_dtype = torch.bfloat16
    use_scaler = False  # bfloat16 doesn't need loss scaling
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    # Triadic warmup
    triadic_warmup = int(cfg['steps'] * cfg['triadic_warmup_pct'])

    # CSV log
    csv_path = os.path.join(ckpt_dir, 'training_log.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['step', 'loss', 'tri_loss', 'lr', 'elapsed_s'])

    # Training
    print(f"\n  Training {cfg['steps']} steps...")
    print(f"  Batch size: {cfg['batch_size']}")
    print(f"  Triadic activation: step {triadic_warmup}")
    print(f"  Alpha: {cfg['alpha']}, Align: {align_weight} ({cfg['align_mode']}), Entropy: {entropy_weight}")
    print("-" * 70)

    model.train()
    start_time = time.time()
    step = 0
    best_loss = float('inf')
    data_iter = iter(dataloader)

    while step < cfg['steps']:
        # Get batch (cycle through data)
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        # Cosine LR with warmup
        warmup_steps = min(500, cfg['steps'] // 10)
        if step < warmup_steps:
            lr_t = cfg['lr'] * (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / max(cfg['steps'] - warmup_steps, 1)
            lr_t = cfg['lr'] * max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        for pg in optimizer.param_groups:
            pg['lr'] = lr_t

        # Forward pass with mixed precision
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=(device.type == 'cuda')):
            logits, triadic_proj, lang_loss = model(x, targets=y)

            total_loss = lang_loss
            tri_loss_val = 0.0

            if step >= triadic_warmup:
                # Dynamic alpha warmup: linear from triadic_warmup to +20% of steps
                alpha_warmup_steps = int(cfg['steps'] * 0.2)
                alpha_factor = min(1.0, (step - triadic_warmup + 1) / alpha_warmup_steps)
                current_alpha = cfg['alpha'] * alpha_factor

                tri_loss = model.triadic_loss(
                    triadic_proj, entropy_weight=entropy_weight,
                    input_ids=x, align_weight=align_weight,
                    align_mode=cfg['align_mode']
                )
                total_loss = lang_loss + current_alpha * tri_loss
                tri_loss_val = tri_loss.item()

        # Backward + step
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # CSV log every step
        elapsed = time.time() - start_time
        csv_writer.writerow([step + 1, f'{lang_loss.item():.6f}', f'{tri_loss_val:.6f}',
                             f'{lr_t:.8f}', f'{elapsed:.1f}'])

        # Console log
        if step % 500 == 0 or step == cfg['steps'] - 1:
            sps = (step + 1) / elapsed if elapsed > 0 else 0
            remaining = (cfg['steps'] - step - 1) / sps if sps > 0 else 0
            pct = (step + 1) / cfg['steps'] * 100
            eta = f"{remaining/60:.1f}m" if remaining >= 60 else f"{remaining:.0f}s"

            bar_len = 30
            filled = int(bar_len * (step + 1) / cfg['steps'])
            bar = '#' * filled + '-' * (bar_len - filled)

            msg = f"  [{bar}] {pct:5.1f}%"
            msg += f" | step {step+1}/{cfg['steps']}"
            msg += f" | loss {lang_loss.item():.4f}"
            if step >= triadic_warmup:
                msg += f" | tri {tri_loss_val:.4f}"
            msg += f" | {sps:.1f} stp/s"
            msg += f" | ETA {eta}"
            print(msg)

        # Track best loss
        if lang_loss.item() < best_loss:
            best_loss = lang_loss.item()

        # Checkpoint every 10K steps and at the end
        if (step + 1) % 10000 == 0 or step == cfg['steps'] - 1:
            model_tag = f"L{cfg['n_layer']}_D{cfg['n_embd']}_B{cfg['n_triadic_bits']}"
            ckpt_path = os.path.join(ckpt_dir, f'model_{model_tag}_step{step+1}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': vars(config),
                'step': step + 1,
                'loss': lang_loss.item(),
                'variant': variant_name,
            }, ckpt_path)

            # Save best
            if lang_loss.item() <= best_loss:
                best_path = os.path.join(ckpt_dir, f'model_{model_tag}_best.pt')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': vars(config),
                    'step': step + 1,
                    'loss': best_loss,
                    'variant': variant_name,
                }, best_path)

            print(f"  >>> Checkpoint saved: {ckpt_path}")

        step += 1

    csv_file.close()
    total_time = time.time() - start_time

    print()
    print("-" * 70)
    print(f"  Training complete! ({total_time/60:.1f} min)")
    print(f"  Final loss: {lang_loss.item():.4f}")
    print(f"  Speed: {cfg['steps']/total_time:.1f} steps/s")

    # --- Evaluation ---
    print(f"\n  --- Evaluation ({variant_name}) ---")

    ppl, avg_loss = compute_ppl(model, tokenizer, device, cfg['block_size'])
    print(f"  PPL:               {ppl:.2f} (avg loss: {avg_loss:.4f})")

    triadic = compute_triadic_metrics(model, tokenizer, device, cfg['n_triadic_bits'])
    print(f"  Semantic gap:      {triadic['semantic_gap']:+.4f}")
    print(f"    Related sim:     {triadic['mean_related_sim']:.4f}")
    print(f"    Random sim:      {triadic['mean_random_sim']:.4f}")
    print(f"  Analogy verif:     {triadic['analogy_verification']:.1%} ({triadic['analogy_correct']}/{triadic['analogy_total']})")
    print(f"  Dead bits:         {triadic['dead_bits']}")
    print(f"  Mean entropy:      {triadic['mean_entropy']:.4f}")
    print(f"  Ordering (K-Q>K-D): {triadic['ordering_correct']} ({triadic['king_queen_sim']:.3f} vs {triadic['king_dog_sim']:.3f})")
    print(f"  Training time:     {total_time/60:.1f} min")

    # Generate samples
    model.eval()
    bos_id = tokenizer.special_tokens['<BOS>']
    samples = []
    print(f"\n  Sample generations:")
    for i in range(3):
        input_ids = torch.tensor([[bos_id]], dtype=torch.long, device=device)
        output = model.generate(input_ids, max_new_tokens=50, temperature=0.7, top_k=50)
        text = tokenizer.decode(output[0].tolist(), skip_special=True)
        samples.append(text[:120])
        print(f"    {i+1}. {text[:100]}")

    # Save results
    results = {
        'variant': variant_name,
        'description': variant['description'],
        'align_weight': align_weight,
        'entropy_weight': entropy_weight,
        'alpha': cfg['alpha'],
        'seed': SEED,
        'ppl': ppl,
        'avg_loss': avg_loss,
        'final_train_loss': best_loss,
        'training_time_min': total_time / 60,
        'total_params': total_params,
        'config': f"{cfg['n_layer']}L/{cfg['n_embd']}D/{cfg['n_head']}H/{cfg['n_triadic_bits']}bits",
        'samples': samples,
        **triadic,
    }

    results_path = os.path.join(ckpt_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {results_path}")

    # Free GPU memory before next variant
    del model, optimizer, scaler
    torch.cuda.empty_cache()

    return results


# ============================================================
# Aggregate comparison
# ============================================================

def aggregate_results():
    """Load all variant results and print comparison table."""
    all_results = {}
    for variant_name in VARIANTS:
        rpath = os.path.join(RESULTS_DIR, variant_name, 'results.json')
        if os.path.exists(rpath):
            with open(rpath) as f:
                all_results[variant_name] = json.load(f)

    if not all_results:
        print("  No results found. Run variants first.")
        return None

    print()
    print("=" * 86)
    print("  E2: ALIGNMENT LOSS ABLATION — Comparison")
    print("=" * 86)
    print()

    # Print description of each variant present
    for name, r in all_results.items():
        print(f"  {name.upper():>12s}: align={r.get('align_weight', '?')}, entropy={r.get('entropy_weight', '?')}  ({r.get('description', '')})")
    print()

    # Comparison table
    variants_present = [v for v in ['full', 'no_align', 'no_entropy'] if v in all_results]
    labels = [v.upper() for v in variants_present]

    header = f"  {'Metric':<25s}" + "".join(f" {l:>14s}" for l in labels)
    print(header)
    print("  " + "-" * (25 + 14 * len(labels)))

    rows = [
        ("PPL",                  'ppl',                   '.2f'),
        ("Avg Loss",             'avg_loss',              '.4f'),
        ("Semantic Gap",         'semantic_gap',          '+.4f'),
        ("  Related Sim",        'mean_related_sim',      '.4f'),
        ("  Random Sim",         'mean_random_sim',       '.4f'),
        ("Analogy Verif",        'analogy_verification',  '.1%'),
        ("Dead Bits",            'dead_bits',             'd'),
        ("Mean Entropy",         'mean_entropy',          '.4f'),
        ("Ordering (K-Q>K-D)",   'ordering_correct',      ''),
        ("  King-Queen Sim",     'king_queen_sim',        '.4f'),
        ("  King-Dog Sim",       'king_dog_sim',          '.4f'),
        ("Training Time (min)",  'training_time_min',     '.1f'),
    ]

    for label, key, fmt in rows:
        vals = []
        for v in variants_present:
            val = all_results[v].get(key, '-')
            if val == '-' or val is None:
                vals.append(f"{'—':>14s}")
            elif fmt == '':
                vals.append(f"{str(val):>14s}")
            elif fmt == 'd':
                vals.append(f"{int(val):>14d}")
            else:
                formatted = format(val, fmt)
                vals.append(f"{formatted:>14s}")
        print(f"  {label:<25s}" + "".join(vals))

    # Key findings
    print()
    print("  Key findings:")

    if 'full' in all_results and 'no_align' in all_results:
        gap_full = all_results['full'].get('semantic_gap', 0)
        gap_no_align = all_results['no_align'].get('semantic_gap', 0)
        delta = gap_full - gap_no_align
        direction = "HIGHER" if delta > 0 else "LOWER"
        print(f"    Alignment loss effect: FULL gap {gap_full:+.4f} vs NO_ALIGN {gap_no_align:+.4f} (delta {delta:+.4f}, FULL is {direction})")

        ppl_full = all_results['full'].get('ppl', 0)
        ppl_no_align = all_results['no_align'].get('ppl', 0)
        ppl_cost = ppl_full - ppl_no_align
        print(f"    Alignment PPL cost: FULL {ppl_full:.2f} vs NO_ALIGN {ppl_no_align:.2f} (delta {ppl_cost:+.2f})")

    if 'full' in all_results and 'no_entropy' in all_results:
        dead_full = all_results['full'].get('dead_bits', 0)
        dead_no_ent = all_results['no_entropy'].get('dead_bits', 0)
        print(f"    Entropy reg effect on dead bits: FULL {dead_full} vs NO_ENTROPY {dead_no_ent}")

        ent_full = all_results['full'].get('mean_entropy', 0)
        ent_no_ent = all_results['no_entropy'].get('mean_entropy', 0)
        print(f"    Entropy reg effect on bit entropy: FULL {ent_full:.4f} vs NO_ENTROPY {ent_no_ent:.4f}")

    if len(all_results) >= 2:
        best_gap_name = max(all_results, key=lambda v: all_results[v].get('semantic_gap', -999))
        best_gap = all_results[best_gap_name].get('semantic_gap', 0)
        print(f"    Best semantic gap: {best_gap_name.upper()} ({best_gap:+.4f})")

        best_ppl_name = min(all_results, key=lambda v: all_results[v].get('ppl', 9999))
        best_ppl = all_results[best_ppl_name].get('ppl', 0)
        print(f"    Best PPL: {best_ppl_name.upper()} ({best_ppl:.2f})")

    # Reference: Run 15
    print()
    print("  --- vs Run 15 (reference) ---")
    print("  Run 15: PPL=7.69, gap=+0.020, analogy=69.2%, dead=15, entropy=0.749")
    if 'full' in all_results:
        r = all_results['full']
        print(f"  FULL:   PPL={r.get('ppl', '?'):.2f}, gap={r.get('semantic_gap', 0):+.4f}, "
              f"analogy={r.get('analogy_verification', 0):.1%}, "
              f"dead={r.get('dead_bits', '?')}, entropy={r.get('mean_entropy', 0):.4f}")

    print("=" * 86)

    # Save aggregate JSON
    comparison = {
        'experiment': 'E2: Alignment Loss Ablation',
        'n_variants': len(all_results),
        'variants': list(all_results.keys()),
        'seed': SEED,
        'base_config': BASE_CONFIG,
        'variant_configs': VARIANTS,
        'results': all_results,
        'reference': {
            'name': 'Run 15 (v1.4-strongalign)',
            'ppl': 7.69,
            'semantic_gap': 0.020,
            'analogy_verification': 0.692,
            'dead_bits': 15,
            'mean_entropy': 0.749,
        },
    }

    agg_path = os.path.join(RESULTS_DIR, 'comparison.json')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(agg_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  Saved: {agg_path}")

    return comparison


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='E2: Alignment Loss Ablation')
    parser.add_argument('--variant', type=str, choices=['full', 'no_align', 'no_entropy'],
                        default=None, help='Run a single variant')
    parser.add_argument('--all', action='store_true', help='Run all 3 variants sequentially')
    parser.add_argument('--aggregate-only', action='store_true', help='Only compare existing results')
    args = parser.parse_args()

    if args.aggregate_only:
        aggregate_results()
        return

    if not args.variant and not args.all:
        parser.error("Specify --variant <name> or --all to train, or --aggregate-only to compare.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Determine which variants to run
    if args.all:
        variant_names = ['full', 'no_align', 'no_entropy']
    else:
        variant_names = [args.variant]

    print()
    print("=" * 70)
    print("  E2: ALIGNMENT LOSS ABLATION")
    print(f"  Variants: {[v.upper() for v in variant_names]}")
    print(f"  Seed: {SEED}")
    print(f"  Config: XL (12L/512D/8H/64bits), 50K steps, batch 64")
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    est_hours = len(variant_names) * 76 / 60
    print(f"  Estimated time: ~{est_hours:.1f}h ({len(variant_names)} x ~76 min)")
    print("=" * 70)

    all_results = []
    for i, variant_name in enumerate(variant_names):
        print(f"\n  >>> VARIANT {i+1}/{len(variant_names)}: {variant_name.upper()}")
        results = train_variant(variant_name, device)
        all_results.append(results)

    # Aggregate all available results (including previously completed variants)
    aggregate_results()


if __name__ == '__main__':
    main()
