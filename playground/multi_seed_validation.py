"""
E1: Multi-Seed Validation — Run 15 config with 3 seeds for confidence intervals.

Trains the production config (12L/512D/8H/64bits, alpha=0.05, align=5.0 MSE,
entropy=1.0, 50K steps) with different random seeds and computes mean +/- std
for all key metrics: PPL, semantic gap, analogy verification, dead bits, entropy.

Each seed takes ~76 min on RTX 5060 Ti (total ~4h for 3 seeds).
Reuses Run 15 tokenizer for fair comparison (only model init + data shuffle differ).

Usage:
  python playground/multi_seed_validation.py                    # all 3 seeds sequential
  python playground/multi_seed_validation.py --seed 42          # single seed
  python playground/multi_seed_validation.py --seeds 42 123 777 # custom seeds
  python playground/multi_seed_validation.py --aggregate-only   # just aggregate existing results
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'TinyStories-train.txt')
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, 'checkpoints', 'torch_run15_strongalign', 'tokenizer.json')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'playground', 'results', 'multi_seed')
STORY_SEPARATOR = '<' + '|endoftext|' + '>'

# Run 15 production config
RUN15_CONFIG = dict(
    n_layer=12, n_embd=512, n_head=8, n_triadic_bits=64,
    block_size=256, dropout=0.1,
    lr=3e-4, alpha=0.05, entropy_weight=1.0, align_weight=5.0,
    align_mode='mse', triadic_warmup_pct=0.8,
    batch_size=64, steps=50000, stories=50000, vocab=4096,
)


class TextDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.tokens) - self.block_size - 1)

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.block_size + 1]
        return torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)


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


def load_and_tokenize(tokenizer, seed):
    """Load stories, shuffle with seed, tokenize."""
    with open(DATA_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()
    stories = [s.strip() for s in raw.split(STORY_SEPARATOR) if s.strip() and len(s.strip()) > 30]

    # Shuffle with experiment seed (NOT fixed 42)
    random.seed(seed)
    random.shuffle(stories)
    stories = stories[:RUN15_CONFIG['stories']]

    all_tokens = []
    for story in stories:
        ids = tokenizer.encode(story, add_special=True)
        all_tokens.extend(ids)
    return all_tokens


def compute_ppl(model, tokenizer, device, block_size, max_samples=200):
    """Compute perplexity on held-out validation stories."""
    with open(DATA_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()
    stories = [s.strip() for s in raw.split(STORY_SEPARATOR) if s.strip() and len(s.strip()) > 50]
    val_stories = stories[-max_samples:]  # last N as validation

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


def compute_triadic_metrics(model, tokenizer, device, n_bits):
    """Compute semantic gap, analogy verification, dead bits, entropy."""
    mapper = PrimeMapper(n_bits)

    concept_pairs = {
        'related': [("king","queen"),("dog","cat"),("happy","sad"),("mother","father"),
                    ("sun","moon"),("hot","cold"),("love","hate"),("big","small"),("bird","fish"),
                    ("doctor","hospital"),("teacher","school"),("princess","prince"),("old","young")],
        'unrelated': [("king","fish"),("dog","moon"),("happy","river"),("mother","blue"),
                      ("sun","cat"),("hot","queen"),("bird","school"),("love","tree"),("big","night")],
    }
    analogy_triples = [
        ("king","queen","man","woman"),("father","mother","brother","sister"),
        ("father","mother","son","daughter"),("dog","puppy","cat","kitten"),
        ("big","small","tall","short"),("hot","cold","day","night"),
        ("happy","sad","love","hate"),("princess","prince","queen","king"),
        ("bird","fly","fish","swim"),("old","young","big","small"),
        ("doctor","hospital","teacher","school"),("sun","day","moon","night"),
        ("red","blue","green","yellow"),
    ]

    all_words = set()
    for group in concept_pairs.values():
        for w1, w2 in group:
            all_words.update([w1, w2])
    for a, b, c, d in analogy_triples:
        all_words.update([a, b, c, d])

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

    # Semantic gap
    related_sims = [cosine(sigs[w1], sigs[w2]) for w1, w2 in concept_pairs['related'] if w1 in sigs and w2 in sigs]
    rand_sims = []
    words = list(sigs.keys())
    rng = np.random.RandomState(0)  # fixed for eval consistency
    for _ in range(200):
        i, j = rng.choice(len(words), 2, replace=False)
        rand_sims.append(cosine(sigs[words[i]], sigs[words[j]]))
    gap = float(np.mean(related_sims) - np.mean(rand_sims))

    # Analogy verification
    correct, total = 0, 0
    for a, b, c, d in analogy_triples:
        if not all(w in sigs for w in [a, b, c, d]):
            continue
        predicted = TriadicValidator.analogy(mapper.map(sigs[a]), mapper.map(sigs[b]), mapper.map(sigs[c]))
        if TriadicValidator.similarity(predicted, mapper.map(sigs[d])) > 0.3:
            correct += 1
        total += 1
    analogy_verif = correct / max(total, 1)

    # Bit entropy and dead bits
    all_projs = np.stack(list(sigs.values()))
    bit_means = (all_projs > 0).mean(axis=0)
    eps = 1e-7
    bit_entropy = -(bit_means * np.log2(bit_means + eps) + (1 - bit_means) * np.log2(1 - bit_means + eps))
    dead_bits = int((bit_entropy < 0.3).sum())
    mean_entropy = float(bit_entropy.mean())

    # Semantic ordering: king-queen vs king-dog
    kq = cosine(sigs['king'], sigs['queen']) if 'king' in sigs and 'queen' in sigs else 0
    kd = cosine(sigs['king'], sigs['dog']) if 'king' in sigs and 'dog' in sigs else 0
    ordering_correct = kq > kd

    return {
        'semantic_gap': gap,
        'analogy_verification': analogy_verif,
        'dead_bits': dead_bits,
        'mean_entropy': mean_entropy,
        'ordering_correct': ordering_correct,
        'king_queen_sim': kq,
        'king_dog_sim': kd,
    }


def train_one_seed(seed, device):
    """Train Run 15 config with a specific seed. Returns metrics dict."""
    cfg = RUN15_CONFIG
    ckpt_dir = os.path.join(RESULTS_DIR, f'seed_{seed}')
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  MULTI-SEED VALIDATION — Seed {seed}")
    print(f"{'='*70}")

    # Set all seeds
    set_all_seeds(seed)

    # Load tokenizer (shared across seeds)
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    actual_vocab = tokenizer.vocab_size
    print(f"  Tokenizer: {TOKENIZER_PATH} (vocab: {actual_vocab})")

    # Tokenize with this seed's shuffle
    print(f"  Tokenizing with seed {seed}...")
    all_tokens = load_and_tokenize(tokenizer, seed)
    print(f"  Tokens: {len(all_tokens):,}")

    # Init model (seed affects weight initialization)
    set_all_seeds(seed)  # reset before model init
    config = TriadicGPTConfig(
        vocab_size=actual_vocab, block_size=cfg['block_size'],
        n_layer=cfg['n_layer'], n_embd=cfg['n_embd'], n_head=cfg['n_head'],
        n_triadic_bits=cfg['n_triadic_bits'], dropout=cfg['dropout'],
    )
    model = TriadicGPT(config).to(device)
    total_params = model.num_params()
    print(f"  Model: {total_params:,} params ({cfg['n_layer']}L/{cfg['n_embd']}D/{cfg['n_head']}H/{cfg['n_triadic_bits']}bits)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=0.01, betas=(0.9, 0.95))
    dataset = TextDataset(all_tokens, cfg['block_size'])
    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, drop_last=True,
                            num_workers=0, generator=torch.Generator().manual_seed(seed))
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    triadic_warmup = int(cfg['steps'] * cfg['triadic_warmup_pct'])

    # CSV log
    csv_path = os.path.join(ckpt_dir, 'training_log.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['step', 'loss', 'tri_loss', 'lr', 'elapsed_s'])

    # Training loop
    print(f"  Training {cfg['steps']} steps (warmup triadic at {triadic_warmup})...")
    print("-" * 70)
    model.train()
    start_time = time.time()
    step = 0
    best_loss = float('inf')
    data_iter = iter(dataloader)

    while step < cfg['steps']:
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

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits, triadic_proj, lang_loss = model(x, targets=y)
            total_loss = lang_loss
            tri_loss_val = 0.0

            if step >= triadic_warmup:
                alpha_warmup_steps = int(cfg['steps'] * 0.2)
                alpha_factor = min(1.0, (step - triadic_warmup + 1) / alpha_warmup_steps)
                current_alpha = cfg['alpha'] * alpha_factor

                tri_loss = model.triadic_loss(
                    triadic_proj, entropy_weight=cfg['entropy_weight'],
                    input_ids=x, align_weight=cfg['align_weight'],
                    align_mode=cfg['align_mode']
                )
                total_loss = lang_loss + current_alpha * tri_loss
                tri_loss_val = tri_loss.item()

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        elapsed = time.time() - start_time
        csv_writer.writerow([step + 1, f'{lang_loss.item():.6f}', f'{tri_loss_val:.6f}', f'{lr_t:.8f}', f'{elapsed:.1f}'])

        if step % 500 == 0 or step == cfg['steps'] - 1:
            sps = (step + 1) / elapsed if elapsed > 0 else 0
            remaining = (cfg['steps'] - step - 1) / sps if sps > 0 else 0
            pct = (step + 1) / cfg['steps'] * 100
            eta = f"{remaining/60:.1f}m" if remaining >= 60 else f"{remaining:.0f}s"
            tri_str = f" | tri {tri_loss_val:.4f}" if step >= triadic_warmup else ""
            print(f"  seed {seed} | {pct:5.1f}% | step {step+1}/{cfg['steps']} | loss {lang_loss.item():.4f}{tri_str} | {sps:.1f} stp/s | ETA {eta}")

        # Save best
        if lang_loss.item() < best_loss:
            best_loss = lang_loss.item()

        step += 1

    csv_file.close()
    total_time = time.time() - start_time

    # Save final checkpoint
    model_tag = f"L{cfg['n_layer']}_D{cfg['n_embd']}_B{cfg['n_triadic_bits']}"
    best_path = os.path.join(ckpt_dir, f'model_{model_tag}_best.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': vars(config),
        'step': cfg['steps'],
        'loss': best_loss,
        'seed': seed,
    }, best_path)

    # Also save tokenizer copy for eval convenience
    tokenizer.save(os.path.join(ckpt_dir, 'tokenizer.json'))

    # Evaluate
    print(f"\n  --- Evaluation (seed {seed}) ---")
    ppl = compute_ppl(model, tokenizer, device, cfg['block_size'])
    print(f"  PPL: {ppl:.2f}")

    triadic = compute_triadic_metrics(model, tokenizer, device, cfg['n_triadic_bits'])
    print(f"  Semantic gap:      {triadic['semantic_gap']:+.4f}")
    print(f"  Analogy verif:     {triadic['analogy_verification']:.1%}")
    print(f"  Dead bits:         {triadic['dead_bits']}")
    print(f"  Mean entropy:      {triadic['mean_entropy']:.4f}")
    print(f"  Ordering (K-Q>K-D): {triadic['ordering_correct']} ({triadic['king_queen_sim']:.3f} vs {triadic['king_dog_sim']:.3f})")
    print(f"  Time:              {total_time/60:.1f} min")

    results = {
        'seed': seed,
        'ppl': ppl,
        'final_loss': best_loss,
        'training_time_min': total_time / 60,
        **triadic,
    }

    results_path = os.path.join(ckpt_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {results_path}")

    # Free GPU memory
    del model, optimizer, scaler
    torch.cuda.empty_cache()

    return results


def aggregate_results():
    """Load all seed results and compute mean +/- std."""
    all_results = []
    for d in sorted(os.listdir(RESULTS_DIR)):
        rpath = os.path.join(RESULTS_DIR, d, 'results.json')
        if os.path.exists(rpath):
            with open(rpath) as f:
                all_results.append(json.load(f))

    if len(all_results) < 2:
        print(f"  Only {len(all_results)} seed(s) found. Need >= 2 for statistics.")
        return all_results

    metrics = ['ppl', 'semantic_gap', 'analogy_verification', 'dead_bits', 'mean_entropy']
    print(f"\n{'='*70}")
    print(f"  MULTI-SEED AGGREGATION ({len(all_results)} seeds)")
    print(f"{'='*70}")
    print(f"  Seeds: {[r['seed'] for r in all_results]}")
    print()
    print(f"  {'Metric':>25s}  {'Mean':>10s}  {'Std':>10s}  {'Min':>10s}  {'Max':>10s}  {'Values'}")
    print(f"  {'─'*25}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*30}")

    summary = {}
    for m in metrics:
        vals = [r[m] for r in all_results]
        mean_v = np.mean(vals)
        std_v = np.std(vals, ddof=1) if len(vals) > 1 else 0
        min_v = min(vals)
        max_v = max(vals)
        fmt = '.4f' if m in ['semantic_gap', 'mean_entropy'] else '.2f' if m == 'ppl' else '.1%' if m == 'analogy_verification' else 'd'

        if m == 'analogy_verification':
            vals_str = ', '.join(f'{v:.1%}' for v in vals)
            print(f"  {m:>25s}  {mean_v:>10.1%}  {std_v:>10.1%}  {min_v:>10.1%}  {max_v:>10.1%}  [{vals_str}]")
        elif m == 'dead_bits':
            vals_str = ', '.join(f'{v}' for v in vals)
            print(f"  {m:>25s}  {mean_v:>10.1f}  {std_v:>10.1f}  {min_v:>10}  {max_v:>10}  [{vals_str}]")
        else:
            vals_str = ', '.join(f'{v:{fmt}}' for v in vals)
            print(f"  {m:>25s}  {mean_v:>10{fmt}}  {std_v:>10{fmt}}  {min_v:>10{fmt}}  {max_v:>10{fmt}}  [{vals_str}]")

        summary[m] = {'mean': float(mean_v), 'std': float(std_v), 'min': float(min_v), 'max': float(max_v), 'values': vals}

    # Ordering check
    ordering = [r.get('ordering_correct', False) for r in all_results]
    print(f"\n  Ordering (K-Q > K-D): {sum(ordering)}/{len(ordering)} seeds correct")

    # Training time
    times = [r.get('training_time_min', 0) for r in all_results]
    print(f"  Training time: {np.mean(times):.1f} +/- {np.std(times, ddof=1):.1f} min")

    # Compare to Run 15
    print(f"\n  --- vs Run 15 (reference) ---")
    print(f"  Run 15: PPL=7.69, gap=+0.020, analogy=69.2%, dead=15, entropy=0.749")
    print(f"  Seeds:  PPL={summary['ppl']['mean']:.2f}+/-{summary['ppl']['std']:.2f}, "
          f"gap={summary['semantic_gap']['mean']:+.4f}+/-{summary['semantic_gap']['std']:.4f}, "
          f"analogy={summary['analogy_verification']['mean']:.1%}+/-{summary['analogy_verification']['std']:.1%}, "
          f"dead={summary['dead_bits']['mean']:.1f}+/-{summary['dead_bits']['std']:.1f}, "
          f"entropy={summary['mean_entropy']['mean']:.4f}+/-{summary['mean_entropy']['std']:.4f}")

    # Save aggregate
    agg_path = os.path.join(RESULTS_DIR, 'aggregate.json')
    with open(agg_path, 'w') as f:
        json.dump({'n_seeds': len(all_results), 'seeds': [r['seed'] for r in all_results],
                   'summary': summary, 'per_seed': all_results}, f, indent=2)
    print(f"\n  Saved: {agg_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Multi-seed validation (E1)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 777],
                        help='Seeds to train (default: 42 123 777)')
    parser.add_argument('--seed', type=int, default=None, help='Train a single seed')
    parser.add_argument('--aggregate-only', action='store_true', help='Only aggregate existing results')
    args = parser.parse_args()

    if args.aggregate_only:
        aggregate_results()
        return

    seeds = [args.seed] if args.seed is not None else args.seeds
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  E1: MULTI-SEED VALIDATION")
    print(f"  Seeds: {seeds}")
    print(f"  Config: Run 15 (12L/512D/8H/64bits, alpha=0.05, align=5.0)")
    print(f"  Device: {device}")
    print(f"{'='*70}")

    all_results = []
    for i, seed in enumerate(seeds):
        print(f"\n  >>> SEED {i+1}/{len(seeds)}: {seed}")
        results = train_one_seed(seed, device)
        all_results.append(results)

    # Aggregate
    if len(seeds) > 1:
        aggregate_results()


if __name__ == '__main__':
    main()
