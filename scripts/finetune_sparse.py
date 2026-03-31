"""
Fine-tune Run 15 with sparsity + subsumption loss.

Problem: Run 15 produces signatures with ~29.5/64 active bits (too dense).
  - Subsumption via divisibility requires ~6-10 active bits
  - SimLex-999 rho = -0.012 (indistinguishable from random)

Solution: Fine-tune triadic head with:
  1. L_sparse: target 8/64 active bits (activation rate = 0.125)
  2. L_sub: for known hypernym pairs, penalize bit-subset violations
  3. Keep language loss to prevent catastrophic forgetting
  4. Freeze transformer blocks, only train triadic_head + final layer norm

Usage:
  python scripts/finetune_sparse.py [--steps 5000] [--device cuda]
"""

import os
import sys
import json
import time
import math
import random

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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Hyperparameters ---
STEPS = 5000
BATCH_SIZE = 32
BLOCK_SIZE = 256
LR = 1e-4              # lower than original — fine-tuning
ALPHA = 0.10            # triadic weight
ENTROPY_WEIGHT = 0.5    # reduced from 1.0 (was pushing toward 50%)
ALIGN_WEIGHT = 5.0
SPARSITY_WEIGHT = 10.0  # heavy — this is the key fix
TARGET_ACTIVE_BITS = 8  # target: 8/64 = 12.5% activation
SUBSUMPTION_WEIGHT = 2.0

# --- Hypernym pairs for L_sub (word-level, tokenized at runtime) ---
HYPERNYM_WORDS = [
    ("animal", "dog"), ("animal", "cat"), ("animal", "bird"),
    ("animal", "fish"), ("animal", "horse"),
    ("person", "boy"), ("person", "girl"), ("person", "man"),
    ("person", "woman"), ("person", "king"), ("person", "queen"),
    ("feeling", "happy"), ("feeling", "sad"), ("feeling", "love"),
    ("food", "bread"), ("food", "milk"), ("food", "apple"),
    ("place", "city"), ("place", "school"), ("place", "house"),
    ("color", "red"), ("color", "blue"), ("color", "green"),
    ("nature", "fire"), ("nature", "water"), ("nature", "sun"),
    ("action", "run"), ("action", "walk"), ("action", "jump"),
]


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


def build_subsumption_map(tokenizer):
    """Build token_id -> [token_id] map for subsumption pairs.

    Since BPE produces multi-token words, we use the FIRST token
    as the representative. This is a simplification but sufficient
    for the gradient signal.
    """
    sub_map = {}
    for hyper_word, hypo_word in HYPERNYM_WORDS:
        hyper_ids = tokenizer.encode(hyper_word, add_special=False)
        hypo_ids = tokenizer.encode(hypo_word, add_special=False)
        if hyper_ids and hypo_ids:
            hyper_tid = hyper_ids[0]  # first BPE token
            hypo_tid = hypo_ids[0]
            if hyper_tid not in sub_map:
                sub_map[hyper_tid] = []
            sub_map[hyper_tid].append(hypo_tid)
    return sub_map


def evaluate_sparsity(model, tokenizer, device):
    """Quick check: how many bits are active per concept?"""
    test_words = [
        "dog", "cat", "king", "queen", "happy", "sad",
        "fire", "water", "run", "walk", "red", "blue",
        "animal", "person", "feeling", "food", "place",
    ]
    model.eval()
    mapper = PrimeMapper(model.config.n_triadic_bits)
    active_counts = []
    subsumption_hits = 0
    subsumption_total = 0

    sigs = {}
    with torch.no_grad():
        for word in test_words:
            ids = tokenizer.encode(word, add_special=False)
            if not ids:
                continue
            x = torch.tensor([ids], dtype=torch.long, device=device)
            _, proj, _ = model(x)
            sig = proj[0].mean(dim=0).cpu().numpy()
            bits = (sig > 0).astype(int)
            n_active = int(bits.sum())
            active_counts.append(n_active)
            prime = mapper.map(sig)
            sigs[word] = {'bits': bits, 'prime': prime, 'n_active': n_active}

    # Check subsumption
    for hyper, hypo in HYPERNYM_WORDS[:10]:
        if hyper in sigs and hypo in sigs:
            subsumption_total += 1
            p_hyper = sigs[hyper]['prime']
            p_hypo = sigs[hypo]['prime']
            if p_hypo % p_hyper == 0:
                subsumption_hits += 1

    mean_active = np.mean(active_counts) if active_counts else 0
    sub_rate = subsumption_hits / subsumption_total if subsumption_total > 0 else 0

    return {
        'mean_active_bits': float(mean_active),
        'min_active': int(min(active_counts)) if active_counts else 0,
        'max_active': int(max(active_counts)) if active_counts else 0,
        'subsumption_rate': sub_rate,
        'subsumption_hits': subsumption_hits,
        'subsumption_total': subsumption_total,
        'details': {w: sigs[w]['n_active'] for w in sigs},
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=STEPS)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--sparsity-weight', type=float, default=SPARSITY_WEIGHT)
    parser.add_argument('--target-bits', type=int, default=TARGET_ACTIVE_BITS)
    parser.add_argument('--subsumption-weight', type=float, default=SUBSUMPTION_WEIGHT)
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Load from this checkpoint instead of Run 15 (cascaded fine-tuning)')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    steps = args.steps
    sparsity_w = args.sparsity_weight
    target_bits = args.target_bits
    sub_w = args.subsumption_weight

    print("=" * 68)
    print("  SPARSE FINE-TUNING — Fix Signature Density")
    print("=" * 68)
    print(f"  Device: {device}")
    print(f"  Steps: {steps}")
    print(f"  Sparsity weight: {sparsity_w} (target: {target_bits}/{64} bits)")
    print(f"  Subsumption weight: {sub_w}")
    print()

    # --- Load checkpoint ---
    if args.resume_from:
        ckpt_path = args.resume_from
        ckpt_dir = os.path.dirname(ckpt_path)
        tok_path = os.path.join(ckpt_dir, 'tokenizer.json')
        if not os.path.exists(tok_path):
            tok_path = os.path.join(PROJECT_ROOT, 'checkpoints',
                                    'torch_run15_strongalign', 'tokenizer.json')
        print(f"  Resuming from: {ckpt_path}")
    else:
        ckpt_dir = os.path.join(PROJECT_ROOT, 'checkpoints', 'torch_run15_strongalign')
        ckpt_path = os.path.join(ckpt_dir, 'model_L12_D512_B64_best.pt')
        tok_path = os.path.join(ckpt_dir, 'tokenizer.json')

    print("[1/5] Loading model...")
    tokenizer = BPETokenizer.load(tok_path)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = checkpoint['config']
    config = TriadicGPTConfig(
        vocab_size=cfg['vocab_size'],
        block_size=cfg['block_size'],
        n_layer=cfg['n_layer'],
        n_embd=cfg['n_embd'],
        n_head=cfg['n_head'],
        n_triadic_bits=cfg['n_triadic_bits'],
        dropout=0.0,
    )
    model = TriadicGPT(config).to(device)
    state = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state, strict=False)
    print(f"  Loaded: {config.n_layer}L/{config.n_embd}D/{config.n_triadic_bits}bits")

    # --- Pre-finetune evaluation ---
    print()
    print("[2/5] Pre-finetune sparsity check...")
    pre_eval = evaluate_sparsity(model, tokenizer, device)
    print(f"  Mean active bits: {pre_eval['mean_active_bits']:.1f}/{config.n_triadic_bits}")
    print(f"  Range: [{pre_eval['min_active']}, {pre_eval['max_active']}]")
    print(f"  Subsumption: {pre_eval['subsumption_rate']:.1%} "
          f"({pre_eval['subsumption_hits']}/{pre_eval['subsumption_total']})")
    print(f"  Per-word: {pre_eval['details']}")

    # --- Freeze everything except triadic head ---
    print()
    # Unfreeze triadic head + last 2 transformer blocks + layer norms
    # 33K params (head only) was not enough — the linear projection
    # cannot transform dense features into sparse codes. Unfreezing
    # blocks 10-11 gives ~6M trainable params and lets the transformer
    # learn to produce features amenable to sparse projection.
    n_unfreeze = 2  # last N transformer blocks
    unfreeze_from = config.n_layer - n_unfreeze  # block index 10+
    print(f"[3/5] Unfreezing triadic_head + ln_f + blocks {unfreeze_from}-{config.n_layer-1}...")
    for name, param in model.named_parameters():
        if 'triadic_head' in name or 'ln_f' in name:
            param.requires_grad = True
        elif any(f'blocks.{i}.' in name for i in range(unfreeze_from, config.n_layer)):
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total:.1%})")

    # --- Load data ---
    print()
    print("[4/5] Loading data...")
    data_path = os.path.join(PROJECT_ROOT, 'data', 'TinyStories-train.txt')
    sep = '<' + '|endoftext|' + '>'
    with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()
    stories = [s.strip() for s in raw.split(sep) if s.strip() and len(s.strip()) > 30]
    random.seed(42)
    random.shuffle(stories)
    stories = stories[:5000]
    all_tokens = []
    for story in stories:
        all_tokens.extend(tokenizer.encode(story, add_special=True))
    print(f"  {len(stories)} stories, {len(all_tokens):,} tokens")

    dataset = TextDataset(all_tokens, BLOCK_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            drop_last=True, num_workers=0)

    # Build subsumption map
    sub_map = build_subsumption_map(tokenizer)
    print(f"  Subsumption pairs: {sum(len(v) for v in sub_map.values())} "
          f"({len(sub_map)} hypernyms)")

    # --- Fine-tune ---
    print()
    print(f"[5/5] Fine-tuning for {steps} steps...")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01)

    model.train()
    data_iter = iter(dataloader)
    history = {'step': [], 'lang_loss': [], 'tri_loss': [],
               'sparsity': [], 'active_bits': []}

    t0 = time.time()
    for step in range(steps):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        # LR schedule: cosine decay
        warmup = min(200, steps // 10)
        if step < warmup:
            lr_t = LR * (step + 1) / warmup
        else:
            progress = (step - warmup) / max(steps - warmup, 1)
            lr_t = LR * max(0.05, 0.5 * (1 + math.cos(math.pi * progress)))
        for pg in optimizer.param_groups:
            pg['lr'] = lr_t

        logits, triadic_proj, lang_loss = model(x, targets=y)

        tri_loss = model.triadic_loss(
            triadic_proj,
            entropy_weight=ENTROPY_WEIGHT,
            input_ids=x,
            align_weight=ALIGN_WEIGHT,
            align_mode='mse',
            sparsity_weight=sparsity_w,
            target_active_bits=target_bits,
            subsumption_weight=sub_w,
            subsumption_pairs=sub_map,
        )

        total_loss = lang_loss + ALPHA * tri_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 200 == 0 or step == steps - 1:
            with torch.no_grad():
                act_rate = (triadic_proj > 0).float().mean().item()
                n_active = act_rate * config.n_triadic_bits

            history['step'].append(step)
            history['lang_loss'].append(lang_loss.item())
            history['tri_loss'].append(tri_loss.item())
            history['sparsity'].append(act_rate)
            history['active_bits'].append(n_active)

            if step % 500 == 0:
                elapsed = time.time() - t0
                print(f"  step {step:5d}/{steps}  "
                      f"lang={lang_loss.item():.3f}  "
                      f"tri={tri_loss.item():.3f}  "
                      f"active={n_active:.1f}/{config.n_triadic_bits}  "
                      f"({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.0f}s")

    # --- Post-finetune evaluation ---
    print()
    print("=" * 68)
    print("  POST-FINETUNE EVALUATION")
    print("=" * 68)
    post_eval = evaluate_sparsity(model, tokenizer, device)
    print(f"  Mean active bits: {post_eval['mean_active_bits']:.1f}/{config.n_triadic_bits}"
          f"  (was {pre_eval['mean_active_bits']:.1f})")
    print(f"  Range: [{post_eval['min_active']}, {post_eval['max_active']}]")
    print(f"  Subsumption: {post_eval['subsumption_rate']:.1%} "
          f"({post_eval['subsumption_hits']}/{post_eval['subsumption_total']})"
          f"  (was {pre_eval['subsumption_rate']:.1%})")
    print(f"  Per-word: {post_eval['details']}")

    # --- Save (versioned to avoid overwriting) ---
    import datetime
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    version_tag = f"sp{int(sparsity_w)}_tb{target_bits}_sw{int(sub_w)}_{steps}s"
    save_dir = os.path.join(PROJECT_ROOT, 'checkpoints', f'sparse_{version_tag}')
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, 'model_sparse_best.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'vocab_size': config.vocab_size,
            'block_size': config.block_size,
            'n_layer': config.n_layer,
            'n_embd': config.n_embd,
            'n_head': config.n_head,
            'n_triadic_bits': config.n_triadic_bits,
        },
        'finetune_config': {
            'base': 'torch_run15_strongalign',
            'steps': steps,
            'sparsity_weight': sparsity_w,
            'target_active_bits': target_bits,
            'subsumption_weight': sub_w,
        },
        'pre_eval': pre_eval,
        'post_eval': post_eval,
    }, save_path)
    print(f"\n  Model saved: {save_path}")

    # Copy tokenizer
    import shutil
    tok_dest = os.path.join(save_dir, 'tokenizer.json')
    if not os.path.exists(tok_dest):
        shutil.copy2(tok_path, tok_dest)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].plot(history['step'], history['lang_loss'], 'b-', alpha=0.8)
    axes[0].set_title('Language Loss')
    axes[0].set_xlabel('Step')

    axes[1].plot(history['step'], history['active_bits'], 'r-', alpha=0.8)
    axes[1].axhline(y=target_bits, color='g', linestyle='--',
                     label=f'target={target_bits}')
    axes[1].axhline(y=pre_eval['mean_active_bits'], color='gray',
                     linestyle=':', label=f"pre={pre_eval['mean_active_bits']:.0f}")
    axes[1].set_title('Active Bits per Token')
    axes[1].set_xlabel('Step')
    axes[1].legend()

    axes[2].plot(history['step'], history['tri_loss'], 'purple', alpha=0.8)
    axes[2].set_title('Triadic Loss')
    axes[2].set_xlabel('Step')

    plt.suptitle(f'Sparse Fine-tuning: {pre_eval["mean_active_bits"]:.0f} -> '
                 f'{post_eval["mean_active_bits"]:.0f} active bits', fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'finetune_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {plot_path}")

    # Save results
    results = {
        'experiment': 'sparse_finetune',
        'steps': steps,
        'config': {
            'sparsity_weight': sparsity_w,
            'target_active_bits': target_bits,
            'subsumption_weight': sub_w,
            'entropy_weight': ENTROPY_WEIGHT,
            'align_weight': ALIGN_WEIGHT,
            'alpha': ALPHA,
            'lr': LR,
        },
        'pre_eval': pre_eval,
        'post_eval': post_eval,
        'history': history,
    }
    results_path = os.path.join(save_dir, 'finetune_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {results_path}")

    # Verdict
    print()
    print("=" * 68)
    improved = (post_eval['mean_active_bits'] < pre_eval['mean_active_bits'] * 0.5
                and post_eval['subsumption_rate'] > pre_eval['subsumption_rate'])
    if improved:
        print("  IMPROVED — now run benchmarks:")
    else:
        print("  PARTIAL — check results, may need more steps or weight tuning:")
    print(f"    python benchmarks/scripts/subsumption_benchmark.py \\")
    print(f"      --model {save_path} --version sparse-finetune")
    print(f"    python benchmarks/scripts/simlex_benchmark.py \\")
    print(f"      --model {save_path} --version sparse-finetune")
    print(f"    python benchmarks/scripts/analogy_benchmark.py \\")
    print(f"      --model {save_path} --version sparse-finetune")
    print("=" * 68)


if __name__ == '__main__':
    main()
