"""
P2 — Rule of Three Loss (from La Danza Cosmica, Cap. 25)

The Rule of Three: C4 = (a * C2 * C3) / (b * C1)

This experiment adds a new loss component that directly optimizes for
algebraic analogy quality. Given known analogy triples (A:B :: C:D),
the loss penalizes when Phi(B)-Phi(A)+Phi(C) is far from Phi(D)
in the triadic projection space.

Trains with: language + triadic + rule-of-three loss.
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
from src.triadic import PrimeMapper, TriadicValidator
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


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def progress_bar(current, total, width=25):
    pct = current / max(total, 1)
    filled = int(width * pct)
    return f"[{'█' * filled}{'░' * (width - filled)}] {pct:5.1%}"

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

# Rule-of-Three analogy triples for supervised signal
# Format: (A, B, C, D) where A:B :: C:D
ANALOGY_TRIPLES = [
    ("king", "queen", "man", "woman"),
    ("king", "queen", "boy", "girl"),
    ("father", "mother", "brother", "sister"),
    ("father", "mother", "son", "daughter"),
    ("dog", "puppy", "cat", "kitten"),
    ("big", "small", "tall", "short"),
    ("hot", "cold", "day", "night"),
    ("happy", "sad", "love", "hate"),
    ("doctor", "hospital", "teacher", "school"),
    ("sun", "day", "moon", "night"),
    ("princess", "prince", "queen", "king"),
    ("bird", "fly", "fish", "swim"),
    ("old", "young", "big", "small"),
    # Augmented (reverse direction)
    ("queen", "king", "woman", "man"),
    ("mother", "father", "sister", "brother"),
    ("small", "big", "short", "tall"),
]


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


def prepare_analogy_data(tokenizer, device):
    """Pre-encode analogy triples as token IDs for fast lookup during training."""
    analogy_tensors = []
    for a, b, c, d in ANALOGY_TRIPLES:
        ids = {}
        valid = True
        for label, word in [('a', a), ('b', b), ('c', c), ('d', d)]:
            encoded = tokenizer.encode(word, add_special=False)
            if not encoded:
                valid = False
                break
            ids[label] = torch.tensor(encoded, dtype=torch.long, device=device)
        if valid:
            analogy_tensors.append(ids)

    print(f"  Prepared {len(analogy_tensors)} valid analogy triples")
    return analogy_tensors


def compute_rule_of_three_loss(model, analogy_tensors, device):
    """
    Rule of Three loss: B - A + C should be close to D in triadic space.

    For each analogy (A:B :: C:D):
      predicted = Phi(B) - Phi(A) + Phi(C)
      loss = MSE(predicted, Phi(D))

    Also computes a "K-constant" loss:
      K = cos(Phi(A),Phi(B)) * cos(Phi(C),Phi(D)) / (cos(Phi(A),Phi(C)) * cos(Phi(B),Phi(D)))
      loss += |K - 1|  (K should be 1 for perfect analogies)
    """
    if not analogy_tensors:
        return torch.tensor(0.0, device=device)

    offset_losses = []
    k_losses = []

    for ids in analogy_tensors:
        # Get triadic projections for each word
        projs = {}
        for label in ['a', 'b', 'c', 'd']:
            x = ids[label].unsqueeze(0)  # (1, seq_len)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                _, triadic_proj, _ = model(x)
            projs[label] = triadic_proj[0].mean(dim=0)  # (n_bits,)

        # Offset loss: B - A + C ≈ D
        predicted = projs['b'] - projs['a'] + projs['c']
        offset_loss = F.mse_loss(predicted, projs['d'])
        offset_losses.append(offset_loss)

        # K-constant loss: K ≈ 1
        sim_ab = F.cosine_similarity(projs['a'].unsqueeze(0), projs['b'].unsqueeze(0))
        sim_cd = F.cosine_similarity(projs['c'].unsqueeze(0), projs['d'].unsqueeze(0))
        sim_ac = F.cosine_similarity(projs['a'].unsqueeze(0), projs['c'].unsqueeze(0))
        sim_bd = F.cosine_similarity(projs['b'].unsqueeze(0), projs['d'].unsqueeze(0))

        k = (sim_ab * sim_cd) / (sim_ac * sim_bd + 1e-7)
        k_loss = (k - 1.0).pow(2)
        k_losses.append(k_loss.squeeze())

    total_offset = torch.stack(offset_losses).mean()
    total_k = torch.stack(k_losses).mean()

    return total_offset + 0.1 * total_k


def train_model(model, tokenizer, all_tokens, device, label, analogy_tensors=None, r3_weight=0.0):
    """Train with optional Rule-of-Three loss."""
    dataset = TextDataset(all_tokens, BLOCK_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    triadic_warmup = int(STEPS * TRIADIC_WARMUP_PCT)

    model.train()
    data_iter = iter(dataloader)
    history = {'step': [], 'loss': [], 'tri_loss': [], 'r3_loss': [], 'entropy': []}

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

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits, triadic_proj, lang_loss = model(x, targets=y)
            total_loss = lang_loss
            tri_loss_val = 0.0
            r3_loss_val = 0.0

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

                # Rule-of-Three loss (every 5 steps to save compute)
                if r3_weight > 0 and analogy_tensors and step % 5 == 0:
                    r3_loss = compute_rule_of_three_loss(model, analogy_tensors, device)
                    total_loss = total_loss + current_alpha * r3_weight * r3_loss
                    r3_loss_val = r3_loss.item()

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % 200 == 0 or step == STEPS - 1:
            with torch.no_grad():
                flat = triadic_proj.reshape(-1, triadic_proj.size(-1))
                bit_means = (flat > 0).float().mean(dim=0)
                eps = 1e-7
                ent = -(bit_means * (bit_means + eps).log2() +
                        (1 - bit_means) * (1 - bit_means + eps).log2())

            history['step'].append(step)
            history['loss'].append(lang_loss.item())
            history['tri_loss'].append(tri_loss_val)
            history['r3_loss'].append(r3_loss_val)
            history['entropy'].append(ent.mean().item())

            elapsed = time.time() - t0
            speed = (step + 1) / max(elapsed, 1)
            eta_s = (STEPS - step - 1) / max(speed, 0.01)
            bar = progress_bar(step + 1, STEPS)
            print(f"  [{label:>9s}] {bar}  step {step:>5d}/{STEPS}  "
                  f"loss={lang_loss.item():.3f}  tri={tri_loss_val:.4f}  r3={r3_loss_val:.4f}  "
                  f"ETA {format_time(eta_s)}  [{format_time(elapsed)}]")

    return history


def evaluate_analogies(model, tokenizer, device, mapper):
    """Evaluate analogy quality for trained model."""
    model.eval()

    test_analogies = [
        ("king", "queen", "man", "woman"),
        ("father", "mother", "brother", "sister"),
        ("dog", "puppy", "cat", "kitten"),
        ("big", "small", "tall", "short"),
        ("hot", "cold", "day", "night"),
        ("happy", "sad", "love", "hate"),
    ]

    sigs = {}
    all_words = set(w for t in test_analogies for w in t)
    with torch.no_grad():
        for word in all_words:
            ids = tokenizer.encode(word, add_special=False)
            if ids:
                x = torch.tensor([ids], dtype=torch.long, device=device)
                _, proj, _ = model(x)
                sigs[word] = proj[0].mean(dim=0).cpu().numpy()

    results = []
    for a, b, c, d in test_analogies:
        if not all(w in sigs for w in [a, b, c, d]):
            continue

        # Vector offset analogy
        predicted = sigs[b] - sigs[a] + sigs[c]
        cos_sim = float(np.dot(predicted, sigs[d]) /
                        (np.linalg.norm(predicted) * np.linalg.norm(sigs[d]) + 1e-10))

        # Algebraic analogy
        phi_a = mapper.map(sigs[a])
        phi_b = mapper.map(sigs[b])
        phi_c = mapper.map(sigs[c])
        phi_d = mapper.map(sigs[d])
        alg_pred = TriadicValidator.analogy(phi_a, phi_b, phi_c)
        alg_sim = TriadicValidator.similarity(alg_pred, phi_d)

        # K constant
        def cosine(x, y):
            return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-10))

        sim_ab = cosine(sigs[a], sigs[b])
        sim_cd = cosine(sigs[c], sigs[d])
        sim_ac = cosine(sigs[a], sigs[c])
        sim_bd = cosine(sigs[b], sigs[d])
        k = (sim_ab * sim_cd) / (sim_ac * sim_bd + 1e-10)

        results.append({
            'analogy': f'{a}:{b}::{c}:{d}',
            'offset_cos': cos_sim,
            'algebraic_sim': float(alg_sim),
            'K': float(k),
        })

    return results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 64)
    print("  RULE OF THREE LOSS EXPERIMENT")
    print("  (La Danza Cosmica, Cap. 25)")
    print("=" * 64)
    print(f"  Device: {device}")

    tok_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'torch_run15_strongalign', 'tokenizer.json')
    tokenizer = BPETokenizer.load(tok_path)
    vocab_size = tokenizer.vocab_size

    print("\nLoading data...")
    all_tokens = load_data(tokenizer)

    print("\nPreparing analogy data...")
    analogy_tensors = prepare_analogy_data(tokenizer, device)

    config = TriadicGPTConfig(
        vocab_size=vocab_size, block_size=BLOCK_SIZE,
        n_layer=N_LAYER, n_embd=N_EMBD, n_head=N_HEAD,
        n_triadic_bits=N_BITS, dropout=0.1,
    )
    mapper = PrimeMapper(N_BITS)

    # Variant 1: Baseline (no R3 loss)
    print(f"\n{'─' * 64}")
    print("  Training: BASELINE (no Rule-of-Three)")
    print(f"{'─' * 64}")
    model_base = TriadicGPT(config).to(device)
    hist_base = train_model(model_base, tokenizer, all_tokens, device, "BASE")
    eval_base = evaluate_analogies(model_base, tokenizer, device, mapper)

    # Variant 2: With R3 loss (weight = 1.0)
    print(f"\n{'─' * 64}")
    print("  Training: R3 LOSS (weight=1.0)")
    print(f"{'─' * 64}")
    model_r3 = TriadicGPT(config).to(device)
    hist_r3 = train_model(model_r3, tokenizer, all_tokens, device, "R3",
                          analogy_tensors=analogy_tensors, r3_weight=1.0)
    eval_r3 = evaluate_analogies(model_r3, tokenizer, device, mapper)

    # Variant 3: With strong R3 loss (weight = 5.0)
    print(f"\n{'─' * 64}")
    print("  Training: STRONG R3 (weight=5.0)")
    print(f"{'─' * 64}")
    model_r3s = TriadicGPT(config).to(device)
    hist_r3s = train_model(model_r3s, tokenizer, all_tokens, device, "R3-STRONG",
                           analogy_tensors=analogy_tensors, r3_weight=5.0)
    eval_r3s = evaluate_analogies(model_r3s, tokenizer, device, mapper)

    # Compare
    print("\n" + "=" * 64)
    print("  ANALOGY RESULTS")
    print("=" * 64)

    for variant_name, eval_results in [("Baseline", eval_base), ("R3 (1.0)", eval_r3), ("R3 (5.0)", eval_r3s)]:
        print(f"\n  {variant_name}:")
        mean_offset = np.mean([r['offset_cos'] for r in eval_results])
        mean_alg = np.mean([r['algebraic_sim'] for r in eval_results])
        mean_k = np.mean([r['K'] for r in eval_results])
        print(f"    Mean offset cosine:    {mean_offset:.4f}")
        print(f"    Mean algebraic sim:    {mean_alg:.2%}")
        print(f"    Mean K-constant:       {mean_k:.4f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {'BASE': 'blue', 'R3': 'orange', 'R3-STRONG': 'red'}

    for label, hist, color in [('Baseline', hist_base, 'blue'),
                                ('R3 (1.0)', hist_r3, 'orange'),
                                ('R3 (5.0)', hist_r3s, 'red')]:
        axes[0, 0].plot(hist['step'], hist['loss'], color=color, label=label, alpha=0.8)
        axes[0, 1].plot(hist['step'], hist['tri_loss'], color=color, label=label, alpha=0.8)
        axes[1, 0].plot(hist['step'], hist['r3_loss'], color=color, label=label, alpha=0.8)
        axes[1, 1].plot(hist['step'], hist['entropy'], color=color, label=label, alpha=0.8)

    for ax, title, ylabel in [
        (axes[0, 0], 'Language Loss', 'Loss'),
        (axes[0, 1], 'Triadic Loss', 'Loss'),
        (axes[1, 0], 'Rule-of-Three Loss', 'Loss'),
        (axes[1, 1], 'Bit Entropy', 'Mean Entropy'),
    ]:
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Step')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Rule of Three Loss: Direct Analogy Optimization', fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'rule_of_three_loss.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved: {plot_path}")

    # Save
    save_data = {
        'experiment': 'rule_of_three_loss',
        'source': 'La Danza Cosmica Cap. 25',
        'config': f'{N_LAYER}L/{N_EMBD}D/{N_BITS}bits',
        'steps': STEPS,
        'n_analogy_triples': len(analogy_tensors),
        'baseline': {
            'final_loss': hist_base['loss'][-1],
            'analogies': eval_base,
        },
        'r3_1.0': {
            'final_loss': hist_r3['loss'][-1],
            'analogies': eval_r3,
        },
        'r3_5.0': {
            'final_loss': hist_r3s['loss'][-1],
            'analogies': eval_r3s,
        },
    }

    results_path = os.path.join(RESULTS_DIR, 'rule_of_three_loss.json')
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"  Results saved: {results_path}")


if __name__ == '__main__':
    main()
