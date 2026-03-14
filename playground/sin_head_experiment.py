"""
P1 — Sinusoidal Triadic Head (from La Danza Cosmica, Cap. 7-9)

The book proposes that all opposites follow y(t) = A*sin(2*pi*f*t + phi).
Currently the triadic head uses tanh(Wx). This experiment replaces it
with sin(Wx + b) to test whether periodic activation captures
cyclic oppositions better.

Trains a small model (10K steps) and compares metrics vs tanh baseline.
"""

import os
import sys
import json
import time
import math
import random
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.torch_transformer import TriadicGPT, TriadicGPTConfig, TransformerBlock
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

# Training config — small model, fast iteration
STEPS = 10000
BATCH_SIZE = 32
BLOCK_SIZE = 256
LR = 3e-4
ALPHA = 0.05
ENTROPY_WEIGHT = 1.0
ALIGN_WEIGHT = 5.0
TRIADIC_WARMUP_PCT = 0.25

# Scale: base (5.8M params — fast but meaningful)
N_LAYER = 6
N_EMBD = 256
N_HEAD = 8
N_BITS = 64


class SinTriadicGPT(TriadicGPT):
    """TriadicGPT variant with sinusoidal activation on triadic head.

    Instead of tanh(Wx), uses sin(Wx + b) where b is a learnable phase bias.
    The sinusoidal activation has natural periodicity which may capture
    cyclic oppositions (hot/cold, love/hate) more naturally.
    """

    def __init__(self, config):
        super().__init__(config)
        # Add learnable phase bias (La Danza: phi in y = A*sin(wt + phi))
        self.triadic_phase = nn.Parameter(torch.zeros(config.n_triadic_bits))
        # Learnable frequency scale
        self.triadic_freq = nn.Parameter(torch.ones(config.n_triadic_bits))

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        assert T <= self.config.block_size

        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        tok_emb = self.wte(input_ids)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        # Sinusoidal triadic head: sin(freq * Wx + phase)
        raw_proj = self.triadic_head(x)
        triadic_proj = torch.sin(self.triadic_freq * raw_proj + self.triadic_phase)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, triadic_proj, loss


class TextDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.tokens) - self.block_size - 1)

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def load_data(tokenizer, max_stories=5000):
    """Load and tokenize TinyStories (subset for fast iteration)."""
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
        ids = tokenizer.encode(story, add_special=True)
        all_tokens.extend(ids)

    print(f"  Loaded {len(stories)} stories, {len(all_tokens):,} tokens")
    return all_tokens


def train_model(model, tokenizer, all_tokens, device, label, steps=STEPS):
    """Train a model and return metrics history."""
    dataset = TextDataset(all_tokens, BLOCK_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    triadic_warmup = int(steps * TRIADIC_WARMUP_PCT)

    model.train()
    data_iter = iter(dataloader)
    history = {'step': [], 'loss': [], 'tri_loss': [], 'entropy': [], 'active_bits': []}

    t0 = time.time()
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
            lr_t = LR * (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / max(steps - warmup_steps, 1)
            lr_t = LR * max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        for pg in optimizer.param_groups:
            pg['lr'] = lr_t

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits, triadic_proj, lang_loss = model(x, targets=y)
            total_loss = lang_loss
            tri_loss_val = 0.0

            if step >= triadic_warmup:
                alpha_warmup = int(steps * 0.2)
                alpha_factor = min(1.0, (step - triadic_warmup + 1) / alpha_warmup)
                current_alpha = ALPHA * alpha_factor

                tri_loss = model.triadic_loss(
                    triadic_proj, entropy_weight=ENTROPY_WEIGHT,
                    input_ids=x, align_weight=ALIGN_WEIGHT, align_mode='mse'
                )
                total_loss = lang_loss + current_alpha * tri_loss
                tri_loss_val = tri_loss.item()

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Log every 500 steps
        if step % 500 == 0 or step == steps - 1:
            with torch.no_grad():
                flat = triadic_proj.reshape(-1, triadic_proj.size(-1))
                bit_means = (flat > 0).float().mean(dim=0)
                entropy_per_bit = -(bit_means * (bit_means + 1e-7).log2() +
                                    (1 - bit_means) * (1 - bit_means + 1e-7).log2())
                mean_entropy = entropy_per_bit.mean().item()
                active = (entropy_per_bit > 0.3).sum().item()

            history['step'].append(step)
            history['loss'].append(lang_loss.item())
            history['tri_loss'].append(tri_loss_val)
            history['entropy'].append(mean_entropy)
            history['active_bits'].append(active)

            elapsed = time.time() - t0
            if step % 2000 == 0:
                print(f"  [{label}] step {step}/{steps}  loss={lang_loss.item():.3f}  "
                      f"tri={tri_loss_val:.4f}  ent={mean_entropy:.3f}  "
                      f"active={active}/{N_BITS}  ({elapsed:.0f}s)")

    return history


def evaluate_model(model, tokenizer, device, mapper):
    """Compute semantic metrics for a trained model."""
    concept_pairs = [
        ("king", "queen"), ("dog", "cat"), ("happy", "sad"),
        ("mother", "father"), ("sun", "moon"), ("big", "small"),
        ("hot", "cold"), ("love", "hate"), ("bird", "fish"),
    ]

    # Semantic gap: mean sim(related) - mean sim(random)
    related_sims = []
    random_sims = []

    model.eval()
    with torch.no_grad():
        sigs = {}
        all_words = list(set(w for pair in concept_pairs for w in pair))
        for word in all_words:
            ids = tokenizer.encode(word, add_special=False)
            if ids:
                x = torch.tensor([ids], dtype=torch.long, device=device)
                _, proj, _ = model(x)
                sigs[word] = proj[0].mean(dim=0).cpu().numpy()

        for w1, w2 in concept_pairs:
            if w1 in sigs and w2 in sigs:
                p1, p2 = sigs[w1], sigs[w2]
                sim = float(np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2) + 1e-10))
                related_sims.append(sim)

        words = list(sigs.keys())
        for _ in range(50):
            i, j = random.sample(range(len(words)), 2)
            p1, p2 = sigs[words[i]], sigs[words[j]]
            sim = float(np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2) + 1e-10))
            random_sims.append(sim)

    gap = np.mean(related_sims) - np.mean(random_sims)

    # Bit entropy
    all_projs = list(sigs.values())
    if all_projs:
        stacked = np.stack(all_projs)
        bit_means = (stacked > 0).mean(axis=0)
        eps = 1e-7
        entropy = -(bit_means * np.log2(bit_means + eps) +
                    (1 - bit_means) * np.log2(1 - bit_means + eps))
        dead_bits = int((entropy < 0.3).sum())
    else:
        entropy = np.zeros(N_BITS)
        dead_bits = N_BITS

    return {
        'semantic_gap': float(gap),
        'mean_related_sim': float(np.mean(related_sims)) if related_sims else 0,
        'mean_random_sim': float(np.mean(random_sims)) if random_sims else 0,
        'mean_bit_entropy': float(entropy.mean()),
        'dead_bits': dead_bits,
        'active_bits': N_BITS - dead_bits,
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 64)
    print("  SINUSOIDAL TRIADIC HEAD EXPERIMENT")
    print("  (La Danza Cosmica, Cap. 7-9: Wave Model)")
    print("=" * 64)
    print(f"  Device: {device}")
    print(f"  Model: {N_LAYER}L/{N_EMBD}D/{N_HEAD}H/{N_BITS} bits")
    print(f"  Steps: {STEPS}")

    # Load tokenizer from Run 15 (reuse for consistency)
    tok_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'torch_run15_strongalign', 'tokenizer.json')
    tokenizer = BPETokenizer.load(tok_path)
    vocab_size = tokenizer.vocab_size
    print(f"  Vocab: {vocab_size}")

    # Load data
    print("\nLoading data...")
    all_tokens = load_data(tokenizer)

    # Config
    config = TriadicGPTConfig(
        vocab_size=vocab_size, block_size=BLOCK_SIZE,
        n_layer=N_LAYER, n_embd=N_EMBD, n_head=N_HEAD,
        n_triadic_bits=N_BITS, dropout=0.1,
    )
    mapper = PrimeMapper(N_BITS)

    # Train TANH baseline
    print("\n" + "-" * 64)
    print("  Training TANH baseline...")
    print("-" * 64)
    model_tanh = TriadicGPT(config).to(device)
    print(f"  Params: {model_tanh.num_params():,}")
    hist_tanh = train_model(model_tanh, tokenizer, all_tokens, device, "TANH")
    eval_tanh = evaluate_model(model_tanh, tokenizer, device, mapper)

    # Train SIN variant
    print("\n" + "-" * 64)
    print("  Training SIN variant...")
    print("-" * 64)
    model_sin = SinTriadicGPT(config).to(device)
    print(f"  Params: {model_sin.num_params():,}")
    hist_sin = train_model(model_sin, tokenizer, all_tokens, device, "SIN")
    eval_sin = evaluate_model(model_sin, tokenizer, device, mapper)

    # Compare
    print("\n" + "=" * 64)
    print("  RESULTS COMPARISON")
    print("=" * 64)
    print(f"  {'Metric':>25s}  {'TANH':>10s}  {'SIN':>10s}  {'Delta':>10s}")
    print(f"  {'─'*25}  {'─'*10}  {'─'*10}  {'─'*10}")

    metrics = [
        ('Final loss', hist_tanh['loss'][-1], hist_sin['loss'][-1]),
        ('Final tri_loss', hist_tanh['tri_loss'][-1], hist_sin['tri_loss'][-1]),
        ('Final entropy', hist_tanh['entropy'][-1], hist_sin['entropy'][-1]),
        ('Active bits', hist_tanh['active_bits'][-1], hist_sin['active_bits'][-1]),
        ('Semantic gap', eval_tanh['semantic_gap'], eval_sin['semantic_gap']),
        ('Dead bits', eval_tanh['dead_bits'], eval_sin['dead_bits']),
    ]

    for name, v_tanh, v_sin in metrics:
        delta = v_sin - v_tanh
        sign = '+' if delta > 0 else ''
        print(f"  {name:>25s}  {v_tanh:>10.4f}  {v_sin:>10.4f}  {sign}{delta:>9.4f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(hist_tanh['step'], hist_tanh['loss'], 'b-', label='TANH', alpha=0.8)
    ax.plot(hist_sin['step'], hist_sin['loss'], 'r-', label='SIN', alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Language Loss')
    ax.set_title('Language Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(hist_tanh['step'], hist_tanh['tri_loss'], 'b-', label='TANH', alpha=0.8)
    ax.plot(hist_sin['step'], hist_sin['tri_loss'], 'r-', label='SIN', alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Triadic Loss')
    ax.set_title('Triadic Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(hist_tanh['step'], hist_tanh['entropy'], 'b-', label='TANH', alpha=0.8)
    ax.plot(hist_sin['step'], hist_sin['entropy'], 'r-', label='SIN', alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean Bit Entropy')
    ax.set_title('Entropy Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(hist_tanh['step'], hist_tanh['active_bits'], 'b-', label='TANH', alpha=0.8)
    ax.plot(hist_sin['step'], hist_sin['active_bits'], 'r-', label='SIN', alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Active Bits (entropy > 0.3)')
    ax.set_title('Active Bits')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Sinusoidal vs Tanh Triadic Head', fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'sin_head_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved: {plot_path}")

    # Save results
    results = {
        'experiment': 'sin_head_experiment',
        'source': 'La Danza Cosmica Cap. 7-9 — Wave Model',
        'config': f'{N_LAYER}L/{N_EMBD}D/{N_BITS}bits',
        'steps': STEPS,
        'tanh': {**{f'hist_{k}': v for k, v in hist_tanh.items()}, **eval_tanh},
        'sin': {**{f'hist_{k}': v for k, v in hist_sin.items()}, **eval_sin},
    }
    results_path = os.path.join(RESULTS_DIR, 'sin_head_experiment.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {results_path}")


if __name__ == '__main__':
    main()
