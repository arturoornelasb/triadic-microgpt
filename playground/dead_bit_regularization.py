"""
P1 — Dead Bit Regularization via L1 Penalty

~15 of 64 bits have entropy < 0.3 in Run 15 (wasted capacity).
This experiment adds an L1 penalty specifically targeting low-variance bits
to force them to activate and contribute meaningfully.

Compares: baseline (entropy reg only) vs L1 on dead bits vs L1 on all bits.
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
from src.triadic import PrimeMapper
try:
    from src.fast_tokenizer import FastBPETokenizer as BPETokenizer
except ImportError:
    from src.tokenizer import BPETokenizer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'playground', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

STEPS = 10000
BATCH_SIZE = 32
BLOCK_SIZE = 256
LR = 3e-4
ALPHA = 0.05
ENTROPY_WEIGHT = 1.0
ALIGN_WEIGHT = 5.0
TRIADIC_WARMUP_PCT = 0.25
N_LAYER = 6
N_EMBD = 256
N_HEAD = 8
N_BITS = 64


class TextDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.tokens) - self.block_size - 1)

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.block_size + 1]
        return torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)


def load_data(tokenizer, max_stories=5000):
    data_path = os.path.join(PROJECT_ROOT, 'data', 'TinyStories-train.txt')
    sep = '<' + '|endoftext|' + '>'
    with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()
    stories = [s.strip() for s in raw.split(sep) if s.strip() and len(s.strip()) > 30]
    random.seed(42)
    random.shuffle(stories)
    stories = stories[:max_stories]
    all_tokens = []
    for story in stories:
        all_tokens.extend(tokenizer.encode(story, add_special=True))
    print(f"  Loaded {len(stories)} stories, {len(all_tokens):,} tokens")
    return all_tokens


def compute_dead_bit_loss(triadic_proj, mode='targeted', dead_threshold=0.3):
    """
    Compute L1 penalty to revive dead bits.

    Args:
        triadic_proj: (B, T, n_bits)
        mode: 'targeted' = only penalize dead bits, 'global' = penalize all
        dead_threshold: entropy threshold below which a bit is considered dead
    """
    B, T, n_bits = triadic_proj.shape
    flat = triadic_proj.reshape(-1, n_bits)

    # Compute per-bit entropy
    bit_means = (flat > 0).float().mean(dim=0)  # P(bit=1)
    eps = 1e-7
    bit_entropy = -(bit_means * (bit_means + eps).log2() +
                    (1 - bit_means) * (1 - bit_means + eps).log2())

    if mode == 'targeted':
        # Only penalize bits with low entropy
        dead_mask = (bit_entropy < dead_threshold).float()  # 1 for dead, 0 for alive

        # For dead bits: penalize deviation from 0 mean (push toward 50/50)
        # |mean| should be 0 for balanced bits
        deviation = torch.abs(flat.mean(dim=0))  # How far from balanced
        loss = (deviation * dead_mask).sum() / (dead_mask.sum() + 1e-7)

    elif mode == 'global':
        # Penalize all bits proportional to their imbalance
        deviation = torch.abs(flat.mean(dim=0))
        loss = deviation.mean()

    elif mode == 'variance':
        # Directly maximize variance of each bit
        bit_var = flat.var(dim=0)
        # Ideal variance for uniform [-1,1] tanh output is ~0.33
        loss = F.relu(0.33 - bit_var).mean()

    return loss


def train_variant(model, tokenizer, all_tokens, device, label, l1_mode='none', l1_weight=0.5):
    """Train with optional dead-bit L1 regularization."""
    dataset = TextDataset(all_tokens, BLOCK_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
    amp_dtype = torch.bfloat16
    use_scaler = False  # bfloat16 doesn't need loss scaling
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
    triadic_warmup = int(STEPS * TRIADIC_WARMUP_PCT)

    model.train()
    data_iter = iter(dataloader)
    history = {'step': [], 'loss': [], 'tri_loss': [], 'l1_loss': [],
               'entropy': [], 'dead_bits': [], 'per_bit_entropy': []}

    t0 = time.time()
    for step in range(STEPS):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        warmup_steps = min(500, STEPS // 10)
        if step < warmup_steps:
            lr_t = LR * (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / max(STEPS - warmup_steps, 1)
            lr_t = LR * max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        for pg in optimizer.param_groups:
            pg['lr'] = lr_t

        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=(device.type == 'cuda')):
            logits, triadic_proj, lang_loss = model(x, targets=y)
            total_loss = lang_loss
            tri_loss_val = 0.0
            l1_loss_val = 0.0

            if step >= triadic_warmup:
                alpha_warmup = int(STEPS * 0.2)
                alpha_factor = min(1.0, (step - triadic_warmup + 1) / alpha_warmup)
                current_alpha = ALPHA * alpha_factor

                tri_loss = model.triadic_loss(
                    triadic_proj, entropy_weight=ENTROPY_WEIGHT,
                    input_ids=x, align_weight=ALIGN_WEIGHT, align_mode='mse'
                )
                total_loss = lang_loss + current_alpha * tri_loss
                tri_loss_val = tri_loss.item()

                # Dead bit L1 regularization
                if l1_mode != 'none':
                    l1_loss = compute_dead_bit_loss(triadic_proj, mode=l1_mode)
                    total_loss = total_loss + current_alpha * l1_weight * l1_loss
                    l1_loss_val = l1_loss.item()

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % 500 == 0 or step == STEPS - 1:
            with torch.no_grad():
                flat = triadic_proj.reshape(-1, triadic_proj.size(-1))
                bit_means = (flat > 0).float().mean(dim=0)
                eps = 1e-7
                ent = -(bit_means * (bit_means + eps).log2() +
                        (1 - bit_means) * (1 - bit_means + eps).log2())
                mean_ent = ent.mean().item()
                dead = int((ent < 0.3).sum().item())

            history['step'].append(step)
            history['loss'].append(lang_loss.item())
            history['tri_loss'].append(tri_loss_val)
            history['l1_loss'].append(l1_loss_val)
            history['entropy'].append(mean_ent)
            history['dead_bits'].append(dead)
            # Save full per-bit entropy at checkpoints
            if step % 2000 == 0:
                history['per_bit_entropy'].append(ent.cpu().tolist())
                elapsed = time.time() - t0
                print(f"  [{label}] step {step}/{STEPS}  loss={lang_loss.item():.3f}  "
                      f"tri={tri_loss_val:.4f}  l1={l1_loss_val:.4f}  "
                      f"dead={dead}/{N_BITS}  ({elapsed:.0f}s)")

    return history


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 64)
    print("  DEAD BIT REGULARIZATION EXPERIMENT")
    print("=" * 64)
    print(f"  Device: {device}")

    tok_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'torch_run15_strongalign', 'tokenizer.json')
    tokenizer = BPETokenizer.load(tok_path)
    vocab_size = tokenizer.vocab_size

    print("\nLoading data...")
    all_tokens = load_data(tokenizer)

    config = TriadicGPTConfig(
        vocab_size=vocab_size, block_size=BLOCK_SIZE,
        n_layer=N_LAYER, n_embd=N_EMBD, n_head=N_HEAD,
        n_triadic_bits=N_BITS, dropout=0.1,
    )

    variants = [
        ('baseline (entropy only)', 'none', 0.0),
        ('L1 targeted (dead bits)', 'targeted', 0.5),
        ('L1 global (all bits)', 'global', 0.5),
        ('L1 variance', 'variance', 0.5),
    ]

    all_results = {}
    for name, mode, weight in variants:
        print(f"\n{'─' * 64}")
        print(f"  Training: {name}")
        print(f"{'─' * 64}")
        model = TriadicGPT(config).to(device)
        print(f"  Params: {model.num_params():,}")
        hist = train_variant(model, tokenizer, all_tokens, device, name, l1_mode=mode, l1_weight=weight)
        all_results[name] = hist

    # Compare
    print("\n" + "=" * 64)
    print("  RESULTS")
    print("=" * 64)
    print(f"  {'Variant':>30s}  {'Loss':>8s}  {'Entropy':>8s}  {'Dead':>6s}")
    print(f"  {'─'*30}  {'─'*8}  {'─'*8}  {'─'*6}")
    for name, h in all_results.items():
        print(f"  {name:>30s}  {h['loss'][-1]:>8.3f}  {h['entropy'][-1]:>8.3f}  {h['dead_bits'][-1]:>6d}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ['blue', 'orange', 'green', 'red']

    for (name, h), color in zip(all_results.items(), colors):
        short = name.split('(')[0].strip()
        axes[0, 0].plot(h['step'], h['loss'], color=color, label=short, alpha=0.8)
        axes[0, 1].plot(h['step'], h['entropy'], color=color, label=short, alpha=0.8)
        axes[1, 0].plot(h['step'], h['dead_bits'], color=color, label=short, alpha=0.8)
        axes[1, 1].plot(h['step'], h['l1_loss'], color=color, label=short, alpha=0.8)

    for ax, title, ylabel in [
        (axes[0, 0], 'Language Loss', 'Loss'),
        (axes[0, 1], 'Mean Bit Entropy', 'Entropy'),
        (axes[1, 0], 'Dead Bits (< 0.3 entropy)', 'Count'),
        (axes[1, 1], 'L1 Regularization Loss', 'L1 Loss'),
    ]:
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Step')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Dead Bit Regularization: L1 Penalty Comparison', fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'dead_bit_regularization.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved: {plot_path}")

    # Per-bit entropy heatmap for final step of each variant
    fig, axes = plt.subplots(len(all_results), 1, figsize=(14, 3 * len(all_results)))
    if len(all_results) == 1:
        axes = [axes]
    for ax, ((name, h), color) in zip(axes, zip(all_results.items(), colors)):
        if h['per_bit_entropy']:
            data = np.array(h['per_bit_entropy'])
            im = ax.imshow(data, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
            ax.set_ylabel('Checkpoint')
            ax.set_xlabel('Bit Index')
            ax.set_title(f'{name} — Per-bit Entropy Over Training')
            plt.colorbar(im, ax=ax, label='Entropy')

    plt.tight_layout()
    heatmap_path = os.path.join(RESULTS_DIR, 'dead_bit_heatmap.png')
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Heatmap saved: {heatmap_path}")

    # Save results
    save_data = {
        'experiment': 'dead_bit_regularization',
        'config': f'{N_LAYER}L/{N_EMBD}D/{N_BITS}bits',
        'steps': STEPS,
    }
    for name, h in all_results.items():
        key = name.replace(' ', '_').replace('(', '').replace(')', '')
        save_data[key] = {
            'final_loss': h['loss'][-1],
            'final_entropy': h['entropy'][-1],
            'final_dead_bits': h['dead_bits'][-1],
        }

    results_path = os.path.join(RESULTS_DIR, 'dead_bit_regularization.json')
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"  Results saved: {results_path}")


if __name__ == '__main__':
    main()
