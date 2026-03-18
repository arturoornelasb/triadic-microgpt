"""
B2/B3 — XL Training Baselines.

Two critical baselines that a reviewer would demand:

  B2 (PURE_LANG): alpha=0, triadic head exists but gets NO gradient.
     Proves: triadic head has zero language cost.
     Expected: PPL ≈ Run 15 (7.69). Triadic metrics = random.

  B3 (FROZEN_RANDOM): alpha=0.05, triadic head weights FROZEN (random init).
     Proves: triadic TRAINING adds value at XL scale.
     Expected: If gap ≈ Run 15, training the head is useless.
               If gap << Run 15, training the head matters.

Both use Run 15's exact config (12L/512D/8H/64bits, 50K steps, batch 64).

Usage:
  python playground/xl_baselines.py --variant pure_lang      # B2 (~76 min)
  python playground/xl_baselines.py --variant frozen_random   # B3 (~76 min)
  python playground/xl_baselines.py --all                     # Both (~2.5h)
  python playground/xl_baselines.py --aggregate-only          # Compare table
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.torch_transformer import TriadicGPT, TriadicGPTConfig
try:
    from src.fast_tokenizer import FastBPETokenizer as BPETokenizer
except ImportError:
    from src.tokenizer import BPETokenizer
from src.triadic import PrimeMapper, TriadicValidator


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORY_SEPARATOR = '<' + '|endoftext|' + '>'

# Run 15 reference
RUN15_BASELINE = {
    'ppl': 7.69, 'semantic_gap': 0.020, 'dead_bits': 15,
    'entropy': 0.749, 'analogy_verification': 1.0,
}

VARIANTS = {
    'pure_lang': {
        'label': 'B2: Pure Language (alpha=0)',
        'description': 'No triadic loss — head exists but is never trained',
        'alpha': 0.0,
        'freeze_head': False,
    },
    'frozen_random': {
        'label': 'B3: Frozen Random Head',
        'description': 'Triadic loss (entropy+diversity only, no alignment) with frozen random head',
        'alpha': 0.05,
        'freeze_head': True,
        'align_weight': 0.0,  # NO alignment — prevents backbone from aligning to noise
    },
}


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
        return (torch.tensor(chunk[:-1], dtype=torch.long),
                torch.tensor(chunk[1:], dtype=torch.long))


# ============================================================
# Evaluation
# ============================================================

RELATED_PAIRS = [
    ("king", "queen"), ("dog", "cat"), ("happy", "sad"),
    ("mother", "father"), ("sun", "moon"), ("hot", "cold"),
    ("love", "hate"), ("big", "small"), ("bird", "fish"),
    ("doctor", "hospital"), ("teacher", "school"),
    ("princess", "prince"), ("old", "young"),
]

ANALOGY_QUADS = [
    ("king", "queen", "man", "woman"),
    ("father", "mother", "brother", "sister"),
    ("dog", "puppy", "cat", "kitten"),
    ("big", "small", "tall", "short"),
    ("hot", "cold", "day", "night"),
    ("happy", "sad", "love", "hate"),
]


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


@torch.no_grad()
def compute_perplexity(model, tokenizer, data_path, device, block_size, max_samples=200):
    model.eval()
    with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()
    stories = [s.strip() for s in raw.split(STORY_SEPARATOR)
               if s.strip() and len(s.strip()) > 50]
    val_stories = stories[-max_samples:]

    total_loss = 0.0
    total_tokens = 0
    for story in val_stories:
        ids = tokenizer.encode(story, add_special=True)
        if len(ids) < 3:
            continue
        ids = ids[:block_size + 1]
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
        _, _, loss = model(x, targets=y)
        total_loss += loss.item() * (len(ids) - 1)
        total_tokens += len(ids) - 1

    avg_loss = total_loss / max(total_tokens, 1)
    model.train()
    return math.exp(avg_loss), avg_loss


@torch.no_grad()
def evaluate_semantic(model, tokenizer, device, n_bits):
    model.eval()
    mapper = PrimeMapper(n_bits)

    all_words = set()
    for w1, w2 in RELATED_PAIRS:
        all_words.update([w1, w2])
    for a, b, c, d in ANALOGY_QUADS:
        all_words.update([a, b, c, d])

    sigs = {}
    for word in all_words:
        ids = tokenizer.encode(word, add_special=False)
        if ids:
            x = torch.tensor([ids], dtype=torch.long, device=device)
            _, proj, _ = model(x)
            sigs[word] = proj[0].mean(dim=0).cpu().numpy()

    # Semantic gap
    related_sims = [cosine(sigs[w1], sigs[w2])
                    for w1, w2 in RELATED_PAIRS if w1 in sigs and w2 in sigs]
    random_sims = []
    words = list(sigs.keys())
    rng = random.Random(42)
    for _ in range(200):
        i, j = rng.sample(range(len(words)), 2)
        random_sims.append(cosine(sigs[words[i]], sigs[words[j]]))
    semantic_gap = float(np.mean(related_sims) - np.mean(random_sims))

    # Analogy verification
    verified = 0
    total_analogies = 0
    for a, b, c, d in ANALOGY_QUADS:
        if not all(w in sigs for w in [a, b, c, d]):
            continue
        total_analogies += 1
        offset_ab = sigs[b] - sigs[a]
        offset_cd = sigs[d] - sigs[c]
        if cosine(offset_ab, offset_cd) > 0:
            verified += 1
    analogy_rate = verified / max(total_analogies, 1)

    # Bit stats
    all_projs = np.stack(list(sigs.values()))
    bit_means = (all_projs > 0).mean(axis=0)
    eps = 1e-7
    bit_entropy = -(bit_means * np.log2(bit_means + eps) +
                    (1 - bit_means) * np.log2(1 - bit_means + eps))
    dead_bits = int((bit_entropy < 0.3).sum())
    mean_entropy = float(bit_entropy.mean())

    # Ordering
    kq = cosine(sigs['king'], sigs['queen']) if 'king' in sigs and 'queen' in sigs else 0
    kd = cosine(sigs['king'], sigs['dog']) if 'king' in sigs and 'dog' in sigs else 0

    model.train()
    return {
        'semantic_gap': semantic_gap,
        'mean_bit_entropy': mean_entropy,
        'dead_bits': dead_bits,
        'analogy_verification': analogy_rate,
        'king_queen_sim': kq,
        'king_dog_sim': kd,
        'ordering_correct': kq > kd,
    }


# ============================================================
# Helpers
# ============================================================

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def progress_bar(current, total, width=30):
    pct = current / max(total, 1)
    filled = int(width * pct)
    return f"[{'#' * filled}{'-' * (width - filled)}] {pct:5.1%}"


# ============================================================
# Training
# ============================================================

def train_variant(variant_name, args):
    variant = VARIANTS[variant_name]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_bits = 64
    steps = args.steps
    block_size = args.block
    batch_size = args.batch_size
    alpha = variant['alpha']
    freeze_head = variant['freeze_head']

    # Output dirs
    results_dir = os.path.join(PROJECT_ROOT, 'playground', 'results',
                               'xl_baselines', variant_name)
    os.makedirs(results_dir, exist_ok=True)

    print()
    print("=" * 70)
    print(f"  {variant['label']}")
    print(f"  {variant['description']}")
    print("=" * 70)
    print(f"  Device:       {device}")
    if device.type == 'cuda':
        print(f"  GPU:          {torch.cuda.get_device_name(0)}")
    print(f"  Scale:        XL (12L/512D/8H/64bits, 40M params)")
    print(f"  Steps:        {steps}")
    print(f"  Alpha:        {alpha}")
    print(f"  Freeze head:  {freeze_head}")
    print(f"  Align weight: {variant.get('align_weight', args.align_weight)}")
    print(f"  Warmup:       {args.warmup_pct:.0%}")
    print()

    # Paths
    data_path = os.path.join(PROJECT_ROOT, 'data', 'TinyStories-train.txt')
    tokenizer_path = os.path.join(PROJECT_ROOT, 'checkpoints',
                                  'torch_run15_strongalign', 'tokenizer.json')

    # Tokenizer
    print("[1/5] Loading tokenizer (Run 15)...")
    tokenizer = BPETokenizer.load(tokenizer_path)
    vocab_size = tokenizer.vocab_size

    # Corpus
    print("[2/5] Tokenizing corpus...")
    with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()
    stories = [s.strip() for s in raw.split(STORY_SEPARATOR)
               if s.strip() and len(s.strip()) > 30]
    random.seed(42)
    random.shuffle(stories)
    stories = stories[:50000]

    all_tokens = []
    for story in stories:
        ids = tokenizer.encode(story, add_special=True)
        all_tokens.extend(ids)
    print(f"  {len(all_tokens):,} tokens from {len(stories)} stories")

    # Model
    print("[3/5] Initializing model...")
    config = TriadicGPTConfig(
        vocab_size=vocab_size, block_size=block_size,
        n_layer=12, n_embd=512, n_head=8,
        n_triadic_bits=n_bits, dropout=0.1,
    )
    model = TriadicGPT(config).to(device)
    total_params = model.num_params()
    print(f"  Parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    # Freeze triadic head if needed
    if freeze_head:
        for name, param in model.named_parameters():
            if 'triadic' in name:
                param.requires_grad = False
        frozen_count = sum(1 for n, p in model.named_parameters()
                          if 'triadic' in n and not p.requires_grad)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Frozen triadic params: {frozen_count} layers")
        print(f"  Trainable params: {trainable:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    dataset = TextDataset(all_tokens, block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            drop_last=True, num_workers=0)
    amp_dtype = torch.bfloat16
    use_scaler = False  # bfloat16 doesn't need loss scaling
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    triadic_warmup = int(steps * args.warmup_pct)

    # CSV log
    csv_path = os.path.join(results_dir, 'training_log.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['step', 'lang_loss', 'tri_loss', 'lr', 'entropy',
                         'dead_bits', 'elapsed_s'])

    # Training
    print()
    print(f"[4/5] Training for {steps} steps...")
    if alpha == 0:
        print("  Triadic loss: DISABLED (alpha=0)")
    else:
        print(f"  Triadic activation at step {triadic_warmup}")
        print(f"  Head weights: {'FROZEN' if freeze_head else 'trainable'}")
    print("-" * 70)

    model.train()
    data_iter = iter(dataloader)
    start_time = time.time()

    for step in range(steps):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)

        # LR schedule
        warmup_steps = min(500, steps // 10)
        if step < warmup_steps:
            lr_t = args.lr * (step + 1) / warmup_steps
        else:
            prog = (step - warmup_steps) / max(steps - warmup_steps, 1)
            lr_t = args.lr * max(0.1, 0.5 * (1.0 + math.cos(math.pi * prog)))
        for pg in optimizer.param_groups:
            pg['lr'] = lr_t

        # Forward
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=(device.type == 'cuda')):
            logits, triadic_proj, lang_loss = model(x, targets=y)
            total_loss = lang_loss
            tri_loss_val = 0.0

            if alpha > 0 and step >= triadic_warmup:
                alpha_warmup_steps = int(steps * 0.2)
                alpha_factor = min(1.0, (step - triadic_warmup + 1) / alpha_warmup_steps)
                current_alpha = alpha * alpha_factor

                # Use variant-specific align_weight if set (frozen_random = 0)
                effective_align = variant.get('align_weight', args.align_weight)
                tri_loss = model.triadic_loss(
                    triadic_proj,
                    entropy_weight=args.entropy_weight,
                    input_ids=x,
                    align_weight=effective_align,
                    align_mode='mse',
                )
                total_loss = lang_loss + current_alpha * tri_loss
                tri_loss_val = tri_loss.item()

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Logging
        if step % 50 == 0 or step == steps - 1:
            with torch.no_grad():
                flat = triadic_proj.reshape(-1, n_bits)
                bm = (flat > 0).float().mean(dim=0)
                eps = 1e-7
                ent = -(bm * (bm + eps).log2() + (1 - bm) * (1 - bm + eps).log2())
                mean_ent = ent.mean().item()
                dead = int((ent < 0.3).sum().item())

            elapsed = time.time() - start_time
            csv_writer.writerow([step + 1, f"{lang_loss.item():.4f}",
                                 f"{tri_loss_val:.4f}", f"{lr_t:.6f}",
                                 f"{mean_ent:.4f}", dead, f"{elapsed:.0f}"])
            csv_file.flush()

            if step % 500 == 0 or step == steps - 1:
                bar = progress_bar(step + 1, steps)
                speed = (step + 1) / max(elapsed, 1)
                eta_s = (steps - step - 1) / max(speed, 0.01)
                print(f"  {bar}  step {step+1:>6d}/{steps}  "
                      f"loss={lang_loss.item():.3f}  tri={tri_loss_val:.4f}  "
                      f"ent={mean_ent:.3f}  dead={dead}/64  "
                      f"ETA {format_time(eta_s)}  [{format_time(elapsed)}]")

    elapsed = time.time() - start_time
    csv_file.close()

    # Final eval
    print()
    print("[5/5] Final evaluation...")
    ppl, avg_loss = compute_perplexity(model, tokenizer, data_path, device, block_size)
    sem = evaluate_semantic(model, tokenizer, device, n_bits)

    print(f"    PPL: {ppl:.2f}")
    print(f"    Semantic gap: {sem['semantic_gap']:+.4f}")
    print(f"    Dead bits: {sem['dead_bits']}/{n_bits}")
    print(f"    Entropy: {sem['mean_bit_entropy']:.3f}")
    print(f"    Analogy verif: {sem['analogy_verification']:.1%}")
    print(f"    Ordering: {'correct' if sem['ordering_correct'] else 'WRONG'}")

    # Save
    results = {
        'experiment': f'xl_baseline_{variant_name}',
        'variant': variant_name,
        'label': variant['label'],
        'description': variant['description'],
        'config': '12L/512D/8H/64bits (XL, 40M params)',
        'alpha': alpha,
        'freeze_head': freeze_head,
        'steps': steps,
        'training_time_min': elapsed / 60,
        'ppl': ppl,
        'avg_loss': avg_loss,
        **sem,
    }
    results_path = os.path.join(results_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print()
    print("=" * 70)
    print(f"  {variant['label']} — RESULTS")
    print("=" * 70)
    print(f"  {'Metric':<25s} {'Result':>12s} {'Run 15':>12s}")
    print(f"  {'='*25} {'='*12} {'='*12}")
    print(f"  {'PPL':<25s} {ppl:>12.2f} {RUN15_BASELINE['ppl']:>12.2f}")
    print(f"  {'Semantic gap':<25s} {sem['semantic_gap']:>+12.4f} {RUN15_BASELINE['semantic_gap']:>+12.4f}")
    print(f"  {'Dead bits':<25s} {sem['dead_bits']:>12d} {RUN15_BASELINE['dead_bits']:>12d}")
    print(f"  {'Entropy':<25s} {sem['mean_bit_entropy']:>12.3f} {RUN15_BASELINE['entropy']:>12.3f}")
    print(f"  {'Analogy verif':<25s} {sem['analogy_verification']:>12.1%} {RUN15_BASELINE['analogy_verification']:>12.1%}")
    print(f"  {'Training time':<25s} {elapsed/60:>12.1f} {'N/A':>12s}")
    print("=" * 70)

    return results


# ============================================================
# Aggregate
# ============================================================

def aggregate_results():
    base_dir = os.path.join(PROJECT_ROOT, 'playground', 'results', 'xl_baselines')
    all_results = {}
    for v in VARIANTS:
        path = os.path.join(base_dir, v, 'results.json')
        if os.path.exists(path):
            with open(path) as f:
                all_results[v] = json.load(f)

    if not all_results:
        print("  No results found. Run --all or --variant first.")
        return

    print()
    print("=" * 80)
    print("  XL BASELINES — COMPARISON")
    print("=" * 80)

    cols = ['Metric', 'Run 15']
    for v in VARIANTS:
        if v in all_results:
            cols.append(VARIANTS[v]['label'].split(':')[0].strip())
    header = "  " + "".join(f"{c:>16s}" for c in cols)
    print(header)
    print("  " + "-" * (16 * len(cols)))

    metrics = [
        ('PPL', 'ppl', '.2f'),
        ('Semantic gap', 'semantic_gap', '+.4f'),
        ('Dead bits', 'dead_bits', 'd'),
        ('Entropy', 'mean_bit_entropy', '.3f'),
        ('Analogy verif', 'analogy_verification', '.1%'),
        ('Time (min)', 'training_time_min', '.1f'),
    ]

    run15_vals = {
        'ppl': 7.69, 'semantic_gap': 0.020, 'dead_bits': 15,
        'mean_bit_entropy': 0.749, 'analogy_verification': 1.0,
        'training_time_min': 76,
    }

    for label, key, fmt in metrics:
        row = f"  {label:>16s}"
        val = run15_vals.get(key, '-')
        if isinstance(val, int):
            row += f"{val:>16d}"
        elif isinstance(val, str):
            row += f"{val:>16s}"
        else:
            row += f"{format(val, fmt):>16s}"

        for v in VARIANTS:
            if v in all_results:
                val = all_results[v].get(key, '-')
                if val == '-' or val is None:
                    row += f"{'—':>16s}"
                elif fmt == 'd':
                    row += f"{int(val):>16d}"
                else:
                    row += f"{format(val, fmt):>16s}"
        print(row)

    # Verdicts
    print()
    if 'pure_lang' in all_results:
        ppl_diff = all_results['pure_lang']['ppl'] - RUN15_BASELINE['ppl']
        pct = ppl_diff / RUN15_BASELINE['ppl'] * 100
        print(f"  B2 verdict: Triadic head language cost = {ppl_diff:+.2f} PPL ({pct:+.1f}%)")

    if 'frozen_random' in all_results:
        gap_random = all_results['frozen_random']['semantic_gap']
        gap_run15 = RUN15_BASELINE['semantic_gap']
        if gap_random < gap_run15 * 0.5:
            print(f"  B3 verdict: Training the head MATTERS (random gap {gap_random:+.4f} << trained {gap_run15:+.4f})")
        else:
            print(f"  B3 verdict: WARNING — random head gap ({gap_random:+.4f}) close to trained ({gap_run15:+.4f})")

    # Save aggregate
    agg_path = os.path.join(base_dir, 'comparison.json')
    os.makedirs(base_dir, exist_ok=True)
    with open(agg_path, 'w') as f:
        json.dump({'run15': run15_vals, 'baselines': all_results}, f, indent=2)
    print(f"\n  Saved: {agg_path}")
    print("=" * 80)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='XL Training Baselines (B2/B3)')
    parser.add_argument('--variant', choices=list(VARIANTS.keys()),
                        help='Which baseline to train')
    parser.add_argument('--all', action='store_true',
                        help='Train both baselines sequentially')
    parser.add_argument('--aggregate-only', action='store_true',
                        help='Print comparison from saved results')

    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--block', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--entropy-weight', type=float, default=1.0)
    parser.add_argument('--align-weight', type=float, default=5.0)
    parser.add_argument('--warmup-pct', type=float, default=0.80)
    args = parser.parse_args()

    if args.aggregate_only:
        aggregate_results()
        return

    if args.all:
        print("*" * 70)
        print("  XL BASELINES — B2 (pure_lang) + B3 (frozen_random)")
        print("  Estimated: ~2.5h total")
        print("*" * 70)
        for v in VARIANTS:
            train_variant(v, args)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        aggregate_results()

    elif args.variant:
        train_variant(args.variant, args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
