"""
E5: Scale Interpolation — Where does the semantic gap flip from negative to positive?

Known data points:
  small  0.8M   gap -0.076
  base   5.8M   gap -0.040
  large  15.9M  gap -0.034
  xl     40M    gap +0.020   (PPL 7.69, analogy 69.2%, dead 15)

The transition from negative to positive semantic gap occurs somewhere between
15.9M and 40M. Is it a smooth crossover or a sharp phase transition?

This experiment adds two interpolation points:
  ~25M  (10L/448D/8H/64bits)
  ~30M  (10L/480D/8H/64bits)

Both use Run 15 training config: alpha=0.05, align=5.0 MSE, entropy=1.0,
warmup 80%, 50K steps, batch 64, lr=3e-4, seed 42.
Each takes ~50-60 min on RTX 5060 Ti (total ~2h for both).

Usage:
  python playground/scale_interpolation.py --config 25m       # single point
  python playground/scale_interpolation.py --config 30m       # single point
  python playground/scale_interpolation.py --all              # both sequentially
  python playground/scale_interpolation.py --aggregate-only   # just print table
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'TinyStories-train.txt')
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, 'checkpoints', 'torch_run15_strongalign', 'tokenizer.json')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'playground', 'results', 'scale_interpolation')
STORY_SEPARATOR = '<' + '|endoftext|' + '>'

# ============================================================
# Interpolation configs (~25M and ~30M between large=15.9M and xl=40M)
# ============================================================

SCALE_CONFIGS = {
    '25m': dict(n_layer=10, n_embd=448, n_head=8, n_triadic_bits=64, label='25M'),
    '30m': dict(n_layer=10, n_embd=480, n_head=8, n_triadic_bits=64, label='30M'),
}

# Training config matching Run 15 (v1.4-strongalign)
TRAIN_CONFIG = dict(
    block_size=256, dropout=0.1,
    lr=3e-4, alpha=0.05, entropy_weight=1.0, align_weight=5.0,
    align_mode='mse', triadic_warmup_pct=0.8,
    batch_size=64, steps=50000, stories=50000, vocab=4096,
    seed=42,
)

# Known scale data points for comparison table
KNOWN_POINTS = [
    {'scale': 'small',  'params': '0.8M',  'ppl': None,  'gap': -0.076, 'analogy': None,    'dead_bits': None},
    {'scale': 'base',   'params': '5.8M',  'ppl': None,  'gap': -0.040, 'analogy': None,    'dead_bits': None},
    {'scale': 'large',  'params': '15.9M', 'ppl': None,  'gap': -0.034, 'analogy': None,    'dead_bits': None},
    # 25M and 30M will be inserted here
    {'scale': 'xl',     'params': '40M',   'ppl': 7.69,  'gap': +0.020, 'analogy': 0.692,   'dead_bits': 15},
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
# Seed management
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
# Data loading
# ============================================================

def load_and_tokenize(tokenizer, seed):
    """Load stories, shuffle with seed, tokenize."""
    with open(DATA_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()
    stories = [s.strip() for s in raw.split(STORY_SEPARATOR) if s.strip() and len(s.strip()) > 30]

    random.seed(seed)
    random.shuffle(stories)
    stories = stories[:TRAIN_CONFIG['stories']]

    all_tokens = []
    for story in stories:
        ids = tokenizer.encode(story, add_special=True)
        all_tokens.extend(ids)

    print(f"  Loaded {len(stories)} stories, {len(all_tokens):,} tokens")
    return all_tokens


# ============================================================
# Evaluation: PPL
# ============================================================

def compute_ppl(model, tokenizer, device, block_size, max_samples=200):
    """Compute perplexity on held-out validation stories."""
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

    return math.exp(total_loss / max(total_tokens, 1))


# ============================================================
# Evaluation: Triadic metrics (semantic gap, analogy, dead bits, entropy)
# ============================================================

def compute_triadic_metrics(model, tokenizer, device, n_bits):
    """Compute semantic gap, analogy verification, dead bits, entropy, ordering."""
    mapper = PrimeMapper(n_bits)

    concept_pairs = {
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
    analogy_triples = [
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

    # Gather all words
    all_words = set()
    for group in concept_pairs.values():
        for w1, w2 in group:
            all_words.update([w1, w2])
    for a, b, c, d in analogy_triples:
        all_words.update([a, b, c, d])

    # Get triadic projections for each word
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
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    # Semantic gap: mean(related_sim) - mean(random_sim)
    related_sims = [cosine(sigs[w1], sigs[w2])
                    for w1, w2 in concept_pairs['related']
                    if w1 in sigs and w2 in sigs]
    rand_sims = []
    words = list(sigs.keys())
    rng = np.random.RandomState(0)
    for _ in range(200):
        i, j = rng.choice(len(words), 2, replace=False)
        rand_sims.append(cosine(sigs[words[i]], sigs[words[j]]))
    gap = float(np.mean(related_sims) - np.mean(rand_sims))

    # Analogy verification
    correct, total = 0, 0
    for a, b, c, d in analogy_triples:
        if not all(w in sigs for w in [a, b, c, d]):
            continue
        predicted = TriadicValidator.analogy(
            mapper.map(sigs[a]), mapper.map(sigs[b]), mapper.map(sigs[c])
        )
        if TriadicValidator.similarity(predicted, mapper.map(sigs[d])) > 0.3:
            correct += 1
        total += 1
    analogy_verif = correct / max(total, 1)

    # Bit entropy and dead bits
    all_projs = np.stack(list(sigs.values()))
    bit_means = (all_projs > 0).mean(axis=0)
    eps = 1e-7
    bit_entropy = -(bit_means * np.log2(bit_means + eps) +
                    (1 - bit_means) * np.log2(1 - bit_means + eps))
    dead_bits = int((bit_entropy < 0.3).sum())
    mean_entropy = float(bit_entropy.mean())

    # Semantic ordering: king-queen should be more similar than king-dog
    kq = cosine(sigs['king'], sigs['queen']) if 'king' in sigs and 'queen' in sigs else 0
    kd = cosine(sigs['king'], sigs['dog']) if 'king' in sigs and 'dog' in sigs else 0
    ordering_correct = kq > kd

    return {
        'semantic_gap': gap,
        'analogy_verification': analogy_verif,
        'analogy_correct': correct,
        'analogy_total': total,
        'dead_bits': dead_bits,
        'mean_entropy': mean_entropy,
        'bit_entropy_per_bit': bit_entropy.tolist(),
        'ordering_correct': ordering_correct,
        'king_queen_sim': kq,
        'king_dog_sim': kd,
        'related_mean_sim': float(np.mean(related_sims)),
        'random_mean_sim': float(np.mean(rand_sims)),
    }


# ============================================================
# Training
# ============================================================

def train_config(config_name, device):
    """Train one scale config and return metrics dict."""
    if config_name not in SCALE_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Choose from: {list(SCALE_CONFIGS.keys())}")

    scfg = SCALE_CONFIGS[config_name]
    tcfg = TRAIN_CONFIG
    seed = tcfg['seed']

    ckpt_dir = os.path.join(RESULTS_DIR, config_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  E5: SCALE INTERPOLATION -- {scfg['label']} ({config_name})")
    print(f"  Config: {scfg['n_layer']}L/{scfg['n_embd']}D/{scfg['n_head']}H/{scfg['n_triadic_bits']}bits")
    print(f"{'='*70}")

    # Set all seeds
    set_all_seeds(seed)

    # Load tokenizer (reuse Run 15's)
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    actual_vocab = tokenizer.vocab_size
    print(f"  Tokenizer: {TOKENIZER_PATH} (vocab: {actual_vocab})")

    # Tokenize
    print(f"  Tokenizing with seed {seed}...")
    all_tokens = load_and_tokenize(tokenizer, seed)

    # Init model
    set_all_seeds(seed)
    config = TriadicGPTConfig(
        vocab_size=actual_vocab,
        block_size=tcfg['block_size'],
        n_layer=scfg['n_layer'],
        n_embd=scfg['n_embd'],
        n_head=scfg['n_head'],
        n_triadic_bits=scfg['n_triadic_bits'],
        dropout=tcfg['dropout'],
    )
    model = TriadicGPT(config).to(device)
    total_params = model.num_params()
    print(f"  Parameters: {total_params:,}")
    print(f"  Target: ~{scfg['label']} params (actual: {total_params/1e6:.1f}M)")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg['lr'], weight_decay=0.01, betas=(0.9, 0.95))

    # DataLoader
    dataset = TextDataset(all_tokens, tcfg['block_size'])
    dataloader = DataLoader(dataset, batch_size=tcfg['batch_size'], shuffle=True,
                            drop_last=True, num_workers=0,
                            generator=torch.Generator().manual_seed(seed))

    # Mixed precision
    amp_dtype = torch.bfloat16
    use_scaler = False  # bfloat16 doesn't need loss scaling
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    # Triadic warmup
    triadic_warmup = int(tcfg['steps'] * tcfg['triadic_warmup_pct'])

    # CSV log
    csv_path = os.path.join(ckpt_dir, 'training_log.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['step', 'loss', 'tri_loss', 'lr', 'elapsed_s'])

    # Training loop
    print(f"\n  Training {tcfg['steps']} steps...")
    print(f"  Batch size: {tcfg['batch_size']}")
    print(f"  Triadic activation: step {triadic_warmup} ({tcfg['triadic_warmup_pct']:.0%} warmup)")
    print(f"  Alpha: {tcfg['alpha']}, Align: {tcfg['align_weight']} ({tcfg['align_mode']}), Entropy: {tcfg['entropy_weight']}")
    print("-" * 70)

    model.train()
    start_time = time.time()
    step = 0
    best_loss = float('inf')
    data_iter = iter(dataloader)

    while step < tcfg['steps']:
        # Get batch (cycle through data)
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        # Cosine LR with warmup
        warmup_steps = min(500, tcfg['steps'] // 10)
        if step < warmup_steps:
            lr_t = tcfg['lr'] * (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / max(tcfg['steps'] - warmup_steps, 1)
            lr_t = tcfg['lr'] * max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        for pg in optimizer.param_groups:
            pg['lr'] = lr_t

        # Forward pass
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=(device.type == 'cuda')):
            logits, triadic_proj, lang_loss = model(x, targets=y)
            total_loss = lang_loss
            tri_loss_val = 0.0

            if step >= triadic_warmup:
                # Alpha ramp: linear from triadic_warmup to triadic_warmup + 20% of steps
                alpha_warmup_steps = int(tcfg['steps'] * 0.2)
                alpha_factor = min(1.0, (step - triadic_warmup + 1) / alpha_warmup_steps)
                current_alpha = tcfg['alpha'] * alpha_factor

                tri_loss = model.triadic_loss(
                    triadic_proj,
                    entropy_weight=tcfg['entropy_weight'],
                    input_ids=x,
                    align_weight=tcfg['align_weight'],
                    align_mode=tcfg['align_mode'],
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

        # CSV logging
        elapsed = time.time() - start_time
        csv_writer.writerow([step + 1, f'{lang_loss.item():.6f}', f'{tri_loss_val:.6f}',
                             f'{lr_t:.8f}', f'{elapsed:.1f}'])

        # Console logging
        if step % 500 == 0 or step == tcfg['steps'] - 1:
            sps = (step + 1) / elapsed if elapsed > 0 else 0
            remaining = (tcfg['steps'] - step - 1) / sps if sps > 0 else 0
            pct = (step + 1) / tcfg['steps'] * 100
            eta = f"{remaining/60:.1f}m" if remaining >= 60 else f"{remaining:.0f}s"
            tri_str = f" | tri {tri_loss_val:.4f}" if step >= triadic_warmup else ""
            print(f"  {scfg['label']} | {pct:5.1f}% | step {step+1}/{tcfg['steps']} "
                  f"| loss {lang_loss.item():.4f}{tri_str} | {sps:.1f} stp/s | ETA {eta}")

        # Track best
        if lang_loss.item() < best_loss:
            best_loss = lang_loss.item()

        # Save checkpoint periodically
        if (step + 1) % 10000 == 0 or step == tcfg['steps'] - 1:
            model_tag = f"L{scfg['n_layer']}_D{scfg['n_embd']}_B{scfg['n_triadic_bits']}"
            ckpt_path = os.path.join(ckpt_dir, f'model_{model_tag}_step{step+1}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': vars(config),
                'step': step + 1,
                'loss': lang_loss.item(),
                'seed': seed,
            }, ckpt_path)
            print(f"  >>> Checkpoint saved: {ckpt_path}")

        step += 1

    csv_file.close()
    total_time = time.time() - start_time

    # Save best model
    model_tag = f"L{scfg['n_layer']}_D{scfg['n_embd']}_B{scfg['n_triadic_bits']}"
    best_path = os.path.join(ckpt_dir, f'model_{model_tag}_best.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': vars(config),
        'step': tcfg['steps'],
        'loss': best_loss,
        'seed': seed,
    }, best_path)

    # Copy tokenizer for eval convenience
    tokenizer.save(os.path.join(ckpt_dir, 'tokenizer.json'))

    # ── Evaluation ──
    print(f"\n  --- Evaluation ({scfg['label']}) ---")

    ppl = compute_ppl(model, tokenizer, device, tcfg['block_size'])
    print(f"  PPL:               {ppl:.2f}")

    triadic = compute_triadic_metrics(model, tokenizer, device, scfg['n_triadic_bits'])
    print(f"  Semantic gap:      {triadic['semantic_gap']:+.4f}")
    print(f"    Related mean:    {triadic['related_mean_sim']:.4f}")
    print(f"    Random mean:     {triadic['random_mean_sim']:.4f}")
    print(f"  Analogy verif:     {triadic['analogy_correct']}/{triadic['analogy_total']} "
          f"({triadic['analogy_verification']:.1%})")
    print(f"  Dead bits:         {triadic['dead_bits']}/{scfg['n_triadic_bits']}")
    print(f"  Mean entropy:      {triadic['mean_entropy']:.4f}")
    print(f"  Ordering (K-Q>K-D): {triadic['ordering_correct']} "
          f"({triadic['king_queen_sim']:.3f} vs {triadic['king_dog_sim']:.3f})")
    print(f"  Training time:     {total_time/60:.1f} min")

    # Generate samples
    print(f"\n  Sample generations:")
    bos_id = tokenizer.special_tokens['<BOS>']
    model.eval()
    for i in range(3):
        input_ids = torch.tensor([[bos_id]], dtype=torch.long, device=device)
        output = model.generate(input_ids, max_new_tokens=50, temperature=0.7, top_k=50)
        text = tokenizer.decode(output[0].tolist(), skip_special=True)
        print(f"    {i+1}. {text[:100]}")

    # Assemble results
    results = {
        'config_name': config_name,
        'label': scfg['label'],
        'architecture': f"{scfg['n_layer']}L/{scfg['n_embd']}D/{scfg['n_head']}H/{scfg['n_triadic_bits']}bits",
        'total_params': total_params,
        'total_params_M': round(total_params / 1e6, 1),
        'ppl': ppl,
        'final_loss': best_loss,
        'training_time_min': round(total_time / 60, 1),
        'training_config': {
            'steps': tcfg['steps'],
            'batch_size': tcfg['batch_size'],
            'lr': tcfg['lr'],
            'alpha': tcfg['alpha'],
            'align_weight': tcfg['align_weight'],
            'align_mode': tcfg['align_mode'],
            'entropy_weight': tcfg['entropy_weight'],
            'triadic_warmup_pct': tcfg['triadic_warmup_pct'],
            'seed': seed,
        },
        **triadic,
    }

    results_path = os.path.join(ckpt_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path}")

    # Free GPU memory
    del model, optimizer, scaler
    torch.cuda.empty_cache()

    return results


# ============================================================
# Aggregate and compare all scale points
# ============================================================

def print_comparison_table(new_results=None):
    """Print comparison table with known + new data points."""

    # Load any saved results
    loaded = {}
    for cfg_name in SCALE_CONFIGS:
        rpath = os.path.join(RESULTS_DIR, cfg_name, 'results.json')
        if os.path.exists(rpath):
            with open(rpath) as f:
                loaded[cfg_name] = json.load(f)

    # Override with fresh results if provided
    if new_results:
        for r in new_results:
            loaded[r['config_name']] = r

    if not loaded:
        print("  No results found. Run --config 25m or --all first.")
        return

    # Build full table
    rows = []
    for kp in KNOWN_POINTS:
        if kp['scale'] in ('small', 'base', 'large'):
            rows.append(kp)

    # Insert 25m
    if '25m' in loaded:
        r = loaded['25m']
        rows.append({
            'scale': '**25M**', 'params': f"{r['total_params_M']}M",
            'ppl': r['ppl'], 'gap': r['semantic_gap'],
            'analogy': r['analogy_verification'], 'dead_bits': r['dead_bits'],
        })

    # Insert 30m
    if '30m' in loaded:
        r = loaded['30m']
        rows.append({
            'scale': '**30M**', 'params': f"{r['total_params_M']}M",
            'ppl': r['ppl'], 'gap': r['semantic_gap'],
            'analogy': r['analogy_verification'], 'dead_bits': r['dead_bits'],
        })

    # Add xl at end
    rows.append(KNOWN_POINTS[-1])

    print(f"\n{'='*70}")
    print(f"  E5: SCALE INTERPOLATION -- RESULTS")
    print(f"{'='*70}")
    print()

    # Table header
    hdr = f"  {'Scale':>10s}  {'Params':>8s}  {'PPL':>8s}  {'Sem Gap':>10s}  {'Analogy':>8s}  {'Dead':>5s}"
    print(hdr)
    print(f"  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*5}")

    for row in rows:
        ppl_str = f"{row['ppl']:.2f}" if row['ppl'] is not None else "-"
        gap_str = f"{row['gap']:+.4f}" if row['gap'] is not None else "-"
        ana_str = f"{row['analogy']:.1%}" if row['analogy'] is not None else "-"
        dead_str = f"{row['dead_bits']}" if row['dead_bits'] is not None else "-"
        print(f"  {row['scale']:>10s}  {row['params']:>8s}  {ppl_str:>8s}  {gap_str:>10s}  {ana_str:>8s}  {dead_str:>5s}")

    # Analysis: is the transition smooth or sharp?
    gaps = []
    params = []
    for row in rows:
        if row['gap'] is not None:
            gaps.append(row['gap'])
            # Parse params string
            p_str = row['params'].replace('M', '').replace('*', '')
            params.append(float(p_str))

    if len(gaps) >= 4:
        print(f"\n  --- Transition Analysis ---")

        # Find zero crossing
        for i in range(len(gaps) - 1):
            if gaps[i] < 0 and gaps[i + 1] >= 0:
                # Linear interpolation for zero crossing
                frac = -gaps[i] / (gaps[i + 1] - gaps[i])
                crossing_params = params[i] + frac * (params[i + 1] - params[i])
                print(f"  Zero crossing: ~{crossing_params:.1f}M params (between {params[i]}M and {params[i+1]}M)")
                break

        # Compute gap deltas to detect sharpness
        print(f"\n  Gap deltas (scale-to-scale):")
        for i in range(len(gaps) - 1):
            delta = gaps[i + 1] - gaps[i]
            dp = params[i + 1] - params[i]
            rate = delta / dp if dp > 0 else 0
            print(f"    {params[i]:.1f}M -> {params[i+1]:.1f}M:  delta_gap = {delta:+.4f}  "
                  f"(rate = {rate:+.5f} per M)")

        # Verdict
        # If the rate is relatively constant, it's smooth; if one jump is much larger, it's sharp
        rates = []
        for i in range(len(gaps) - 1):
            dp = params[i + 1] - params[i]
            if dp > 0:
                rates.append((gaps[i + 1] - gaps[i]) / dp)
        if len(rates) >= 2:
            rate_std = np.std(rates)
            rate_mean = abs(np.mean(rates))
            cv = rate_std / rate_mean if rate_mean > 0 else float('inf')
            print(f"\n  Rate of change CV: {cv:.2f}")
            if cv < 0.5:
                print(f"  Verdict: SMOOTH crossover (consistent rate across scales)")
            elif cv < 1.0:
                print(f"  Verdict: GRADUAL transition (some acceleration)")
            else:
                print(f"  Verdict: SHARP phase transition (rate varies significantly)")

    # Detailed results for new points
    for cfg_name, r in sorted(loaded.items()):
        print(f"\n  --- {r['label']} Details ({r['architecture']}) ---")
        print(f"    Params:          {r['total_params']:,} ({r['total_params_M']}M)")
        print(f"    PPL:             {r['ppl']:.2f}")
        print(f"    Semantic gap:    {r['semantic_gap']:+.4f}")
        print(f"      Related sim:   {r['related_mean_sim']:.4f}")
        print(f"      Random sim:    {r['random_mean_sim']:.4f}")
        print(f"    Analogy:         {r['analogy_correct']}/{r['analogy_total']} ({r['analogy_verification']:.1%})")
        print(f"    Dead bits:       {r['dead_bits']}/{SCALE_CONFIGS[cfg_name]['n_triadic_bits']}")
        print(f"    Mean entropy:    {r['mean_entropy']:.4f}")
        print(f"    Ordering:        {'PASS' if r['ordering_correct'] else 'FAIL'} "
              f"(K-Q={r['king_queen_sim']:.3f}, K-D={r['king_dog_sim']:.3f})")
        print(f"    Training time:   {r['training_time_min']:.1f} min")

    # Save aggregate
    agg = {
        'experiment': 'E5_scale_interpolation',
        'description': 'Interpolation between large (15.9M) and xl (40M) to locate semantic gap transition',
        'configs': {k: v for k, v in SCALE_CONFIGS.items()},
        'results': {k: v for k, v in loaded.items()},
    }
    agg_path = os.path.join(RESULTS_DIR, 'aggregate.json')
    with open(agg_path, 'w') as f:
        json.dump(agg, f, indent=2, default=str)
    print(f"\n  Aggregate saved: {agg_path}")


# ============================================================
# Plotting
# ============================================================

def plot_scale_curve():
    """Plot semantic gap vs params across all known + new data points."""
    # Load results
    loaded = {}
    for cfg_name in SCALE_CONFIGS:
        rpath = os.path.join(RESULTS_DIR, cfg_name, 'results.json')
        if os.path.exists(rpath):
            with open(rpath) as f:
                loaded[cfg_name] = json.load(f)

    if not loaded:
        return

    # All data points
    all_points = [
        (0.8, -0.076, 'small', False),
        (5.8, -0.040, 'base', False),
        (15.9, -0.034, 'large', False),
        (40.0, +0.020, 'xl', False),
    ]
    for cfg_name, r in loaded.items():
        all_points.append((r['total_params_M'], r['semantic_gap'], r['label'], True))

    all_points.sort(key=lambda x: x[0])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Plot 1: Semantic Gap vs Scale ---
    ax = axes[0]
    for params, gap, label, is_new in all_points:
        color = 'red' if is_new else 'steelblue'
        marker = 'D' if is_new else 'o'
        size = 100 if is_new else 60
        ax.scatter(params, gap, c=color, s=size, marker=marker, zorder=5, edgecolors='black', linewidths=0.5)
        ax.annotate(label, (params, gap), textcoords='offset points',
                    xytext=(0, 10), ha='center', fontsize=8, fontweight='bold' if is_new else 'normal')

    # Connect with line
    pts = sorted(all_points, key=lambda x: x[0])
    ax.plot([p[0] for p in pts], [p[1] for p in pts], 'k--', alpha=0.3, linewidth=1)
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.set_xlabel('Parameters (M)', fontsize=11)
    ax.set_ylabel('Semantic Gap', fontsize=11)
    ax.set_title('Semantic Gap vs Model Scale', fontsize=12)
    ax.grid(True, alpha=0.3)

    # --- Plot 2: PPL vs Scale (only points with PPL data) ---
    ax = axes[1]
    ppl_points = [(40.0, 7.69, 'xl', False)]
    for cfg_name, r in loaded.items():
        ppl_points.append((r['total_params_M'], r['ppl'], r['label'], True))
    ppl_points.sort(key=lambda x: x[0])

    for params, ppl, label, is_new in ppl_points:
        color = 'red' if is_new else 'steelblue'
        marker = 'D' if is_new else 'o'
        size = 100 if is_new else 60
        ax.scatter(params, ppl, c=color, s=size, marker=marker, zorder=5, edgecolors='black', linewidths=0.5)
        ax.annotate(f"{label}\n{ppl:.2f}", (params, ppl), textcoords='offset points',
                    xytext=(0, 10), ha='center', fontsize=8, fontweight='bold' if is_new else 'normal')

    ax.plot([p[0] for p in ppl_points], [p[1] for p in ppl_points], 'k--', alpha=0.3, linewidth=1)
    ax.set_xlabel('Parameters (M)', fontsize=11)
    ax.set_ylabel('Perplexity', fontsize=11)
    ax.set_title('Perplexity vs Model Scale', fontsize=12)
    ax.grid(True, alpha=0.3)

    # --- Plot 3: Loss curves from training logs ---
    ax = axes[2]
    colors = {'25m': 'tab:orange', '30m': 'tab:red'}
    for cfg_name in SCALE_CONFIGS:
        csv_path = os.path.join(RESULTS_DIR, cfg_name, 'training_log.csv')
        if not os.path.exists(csv_path):
            continue
        steps_list, losses_list = [], []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                steps_list.append(int(row['step']))
                losses_list.append(float(row['loss']))

        # Smooth
        window = max(1, len(losses_list) // 100)
        if window > 1 and len(losses_list) > window:
            smoothed = np.convolve(losses_list, np.ones(window) / window, mode='valid')
            ax.plot(steps_list[window-1:], smoothed, color=colors.get(cfg_name, 'gray'),
                    linewidth=2, label=f"{SCALE_CONFIGS[cfg_name]['label']}", alpha=0.9)
        else:
            ax.plot(steps_list, losses_list, color=colors.get(cfg_name, 'gray'),
                    linewidth=1, label=f"{SCALE_CONFIGS[cfg_name]['label']}", alpha=0.7)

    ax.set_xlabel('Step', fontsize=11)
    ax.set_ylabel('Language Loss', fontsize=11)
    ax.set_title('Training Loss Curves', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle('E5: Scale Interpolation — Semantic Gap Transition (15.9M -> 40M)', fontsize=13)
    plt.tight_layout()

    plot_path = os.path.join(RESULTS_DIR, 'scale_interpolation.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved: {plot_path}")


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='E5: Scale Interpolation')
    parser.add_argument('--config', type=str, choices=['25m', '30m'], default=None,
                        help='Train a single scale config')
    parser.add_argument('--all', action='store_true',
                        help='Train both 25m and 30m sequentially')
    parser.add_argument('--aggregate-only', action='store_true',
                        help='Only print comparison table from existing results')
    args = parser.parse_args()

    if args.aggregate_only:
        print_comparison_table()
        plot_scale_curve()
        return

    if not args.config and not args.all:
        parser.print_help()
        print("\n  Error: specify --config 25m, --config 30m, or --all")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  E5: SCALE INTERPOLATION")
    print(f"  Goal: Locate the semantic gap transition between 15.9M and 40M")
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"{'='*70}")

    configs_to_run = ['25m', '30m'] if args.all else [args.config]
    all_results = []

    for i, cfg_name in enumerate(configs_to_run):
        print(f"\n  >>> Config {i+1}/{len(configs_to_run)}: {cfg_name}")
        results = train_config(cfg_name, device)
        all_results.append(results)

    # Print comparison table and plot
    print_comparison_table(all_results)
    plot_scale_curve()

    print(f"\n{'='*70}")
    print(f"  E5: SCALE INTERPOLATION COMPLETE")
    print(f"  Results: {RESULTS_DIR}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
