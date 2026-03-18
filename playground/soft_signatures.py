"""
P1 — Soft Triadic Signatures (from La Danza Cosmica, Cap. 14-16)

The book proposes quantum superposition: concepts exist in superposition
until context collapses them. Currently we hard-threshold at 0 (tanh > 0 = 1).

This experiment uses sigmoid (soft) during training and only discretizes
for evaluation. Hypothesis: soft signatures improve gradients, reduce dead
bits, and allow "superposition" of ambiguous concepts.

Compares: hard threshold (current) vs sigmoid-soft vs temperature-annealed.
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


class SoftTriadicGPT(TriadicGPT):
    """TriadicGPT with sigmoid activation (soft signatures).

    Uses sigmoid(Wx * temperature) where temperature starts low (soft)
    and anneals toward high (hard), mimicking quantum collapse.
    """

    def __init__(self, config, mode='sigmoid', anneal=False, final_temp=10.0):
        super().__init__(config)
        self.soft_mode = mode
        self.anneal = anneal
        self.final_temp = final_temp
        self.current_temp = 1.0  # Set externally during training

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
        raw = self.triadic_head(x)

        if self.soft_mode == 'sigmoid':
            # Sigmoid maps to [0,1]; shift to [-1,1] for compatibility
            triadic_proj = 2.0 * torch.sigmoid(raw * self.current_temp) - 1.0
        elif self.soft_mode == 'gumbel':
            # Straight-through Gumbel-Softmax (differentiable binary)
            # Convert raw to logits for Bernoulli
            probs = torch.sigmoid(raw * self.current_temp)
            if self.training:
                # Gumbel noise for exploration
                u = torch.rand_like(probs).clamp(1e-7, 1 - 1e-7)
                gumbel = -torch.log(-torch.log(u))
                hard = ((probs.log() + gumbel) > 0).float()
                # Straight-through: hard forward, soft backward
                triadic_proj = (hard - probs).detach() + probs
                triadic_proj = 2.0 * triadic_proj - 1.0
            else:
                triadic_proj = 2.0 * (probs > 0.5).float() - 1.0
        else:
            triadic_proj = torch.tanh(raw)

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


def train_variant(model, tokenizer, all_tokens, device, label, steps=STEPS, anneal=False):
    """Train and return history."""
    dataset = TextDataset(all_tokens, BLOCK_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
    amp_dtype = torch.bfloat16
    use_scaler = False  # bfloat16 doesn't need loss scaling
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
    triadic_warmup = int(steps * TRIADIC_WARMUP_PCT)

    model.train()
    data_iter = iter(dataloader)
    history = {'step': [], 'loss': [], 'tri_loss': [], 'entropy': [], 'dead_bits': [], 'temp': []}

    t0 = time.time()
    for step in range(steps):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        # Temperature annealing
        if anneal and hasattr(model, 'current_temp'):
            progress = step / steps
            model.current_temp = 1.0 + (model.final_temp - 1.0) * progress

        # LR schedule
        warmup_steps = min(500, steps // 10)
        if step < warmup_steps:
            lr_t = LR * (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / max(steps - warmup_steps, 1)
            lr_t = LR * max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        for pg in optimizer.param_groups:
            pg['lr'] = lr_t

        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=(device.type == 'cuda')):
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

        if step % 500 == 0 or step == steps - 1:
            with torch.no_grad():
                flat = triadic_proj.reshape(-1, triadic_proj.size(-1))
                bit_means = (flat > 0).float().mean(dim=0)
                eps = 1e-7
                ent = -(bit_means * (bit_means + eps).log2() +
                        (1 - bit_means) * (1 - bit_means + eps).log2())
                mean_ent = ent.mean().item()
                dead = int((ent < 0.3).sum().item())

            temp = getattr(model, 'current_temp', 1.0)
            history['step'].append(step)
            history['loss'].append(lang_loss.item())
            history['tri_loss'].append(tri_loss_val)
            history['entropy'].append(mean_ent)
            history['dead_bits'].append(dead)
            history['temp'].append(temp)

            if step % 2000 == 0:
                elapsed = time.time() - t0
                print(f"  [{label}] step {step}/{steps}  loss={lang_loss.item():.3f}  "
                      f"tri={tri_loss_val:.4f}  ent={mean_ent:.3f}  "
                      f"dead={dead}/{N_BITS}  temp={temp:.1f}  ({elapsed:.0f}s)")

    return history


def evaluate_semantics(model, tokenizer, device):
    """Quick semantic gap evaluation."""
    pairs = [("king", "queen"), ("dog", "cat"), ("happy", "sad"),
             ("mother", "father"), ("sun", "moon"), ("hot", "cold")]

    model.eval()
    sigs = {}
    with torch.no_grad():
        all_words = list(set(w for p in pairs for w in p))
        for word in all_words:
            ids = tokenizer.encode(word, add_special=False)
            if ids:
                x = torch.tensor([ids], dtype=torch.long, device=device)
                _, proj, _ = model(x)
                sigs[word] = proj[0].mean(dim=0).cpu().numpy()

    related, rand_sims = [], []
    for w1, w2 in pairs:
        if w1 in sigs and w2 in sigs:
            s = float(np.dot(sigs[w1], sigs[w2]) /
                      (np.linalg.norm(sigs[w1]) * np.linalg.norm(sigs[w2]) + 1e-10))
            related.append(s)

    words = list(sigs.keys())
    for _ in range(50):
        i, j = random.sample(range(len(words)), 2)
        s = float(np.dot(sigs[words[i]], sigs[words[j]]) /
                  (np.linalg.norm(sigs[words[i]]) * np.linalg.norm(sigs[words[j]]) + 1e-10))
        rand_sims.append(s)

    return {
        'semantic_gap': float(np.mean(related) - np.mean(rand_sims)),
        'mean_related': float(np.mean(related)) if related else 0,
        'mean_random': float(np.mean(rand_sims)) if rand_sims else 0,
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 64)
    print("  SOFT TRIADIC SIGNATURES EXPERIMENT")
    print("  (La Danza Cosmica, Cap. 14-16: Quantum Superposition)")
    print("=" * 64)
    print(f"  Device: {device}")

    tok_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'torch_run15_strongalign', 'tokenizer.json')
    tokenizer = BPETokenizer.load(tok_path)
    vocab_size = tokenizer.vocab_size
    print(f"  Vocab: {vocab_size}")

    print("\nLoading data...")
    all_tokens = load_data(tokenizer)

    config = TriadicGPTConfig(
        vocab_size=vocab_size, block_size=BLOCK_SIZE,
        n_layer=N_LAYER, n_embd=N_EMBD, n_head=N_HEAD,
        n_triadic_bits=N_BITS, dropout=0.1,
    )

    variants = {
        'tanh (baseline)': lambda: TriadicGPT(config).to(device),
        'sigmoid (soft)': lambda: SoftTriadicGPT(config, mode='sigmoid').to(device),
        'sigmoid+anneal': lambda: SoftTriadicGPT(config, mode='sigmoid', anneal=True, final_temp=10.0).to(device),
        'gumbel-softmax': lambda: SoftTriadicGPT(config, mode='gumbel', anneal=True, final_temp=5.0).to(device),
    }

    all_results = {}
    for name, create_fn in variants.items():
        print(f"\n{'─' * 64}")
        print(f"  Training: {name}")
        print(f"{'─' * 64}")
        model = create_fn()
        print(f"  Params: {model.num_params():,}")
        anneal = 'anneal' in name or 'gumbel' in name
        hist = train_variant(model, tokenizer, all_tokens, device, name, anneal=anneal)
        sem = evaluate_semantics(model, tokenizer, device)
        all_results[name] = {'history': hist, 'semantics': sem}

    # Compare
    print("\n" + "=" * 64)
    print("  RESULTS")
    print("=" * 64)
    print(f"  {'Variant':>20s}  {'Loss':>8s}  {'Entropy':>8s}  {'Dead':>6s}  {'Gap':>8s}")
    print(f"  {'─'*20}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*8}")
    for name, r in all_results.items():
        h = r['history']
        s = r['semantics']
        print(f"  {name:>20s}  {h['loss'][-1]:>8.3f}  {h['entropy'][-1]:>8.3f}  "
              f"{h['dead_bits'][-1]:>6d}  {s['semantic_gap']:>+8.4f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ['blue', 'orange', 'green', 'red']

    for i, ((name, r), color) in enumerate(zip(all_results.items(), colors)):
        h = r['history']
        axes[0, 0].plot(h['step'], h['loss'], color=color, label=name, alpha=0.8)
        axes[0, 1].plot(h['step'], h['tri_loss'], color=color, label=name, alpha=0.8)
        axes[1, 0].plot(h['step'], h['entropy'], color=color, label=name, alpha=0.8)
        axes[1, 1].plot(h['step'], h['dead_bits'], color=color, label=name, alpha=0.8)

    for ax, title, ylabel in [
        (axes[0, 0], 'Language Loss', 'Loss'),
        (axes[0, 1], 'Triadic Loss', 'Loss'),
        (axes[1, 0], 'Bit Entropy', 'Mean Entropy'),
        (axes[1, 1], 'Dead Bits', 'Count (entropy < 0.3)'),
    ]:
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Step')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Soft Signatures: Quantum Superposition for Triadic Head', fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'soft_signatures_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved: {plot_path}")

    # Save
    save_results = {
        'experiment': 'soft_signatures',
        'source': 'La Danza Cosmica Cap. 14-16 — Quantum Superposition',
        'config': f'{N_LAYER}L/{N_EMBD}D/{N_BITS}bits',
        'steps': STEPS,
    }
    for name, r in all_results.items():
        key = name.replace(' ', '_').replace('(', '').replace(')', '')
        save_results[key] = {
            'final_loss': r['history']['loss'][-1],
            'final_entropy': r['history']['entropy'][-1],
            'final_dead_bits': r['history']['dead_bits'][-1],
            **r['semantics'],
        }

    results_path = os.path.join(RESULTS_DIR, 'soft_signatures.json')
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"  Results saved: {results_path}")


if __name__ == '__main__':
    main()
