"""
P1 — Random Baseline (rigorous control experiment)

Critical question: Is the triadic head actually learning anything useful,
or could a random projection achieve similar metrics?

This experiment trains 3 models:
  1. Normal TriadicGPT (trainable triadic head)
  2. Frozen random triadic head (projection is random, never updated)
  3. No triadic head (language-only ablation)

If (1) >> (2) on semantic metrics, the head is learning meaningful structure.
If (1) ~ (2), the head is just a random hash (still useful but not "learned").
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


def train_model(model, tokenizer, all_tokens, device, label, use_triadic=True):
    """Train model, optionally without triadic loss."""
    dataset = TextDataset(all_tokens, BLOCK_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01, betas=(0.9, 0.95)
    )
    amp_dtype = torch.bfloat16
    use_scaler = False  # bfloat16 doesn't need loss scaling
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
    triadic_warmup = int(STEPS * TRIADIC_WARMUP_PCT)

    model.train()
    data_iter = iter(dataloader)
    history = {'step': [], 'loss': [], 'tri_loss': [], 'entropy': [], 'dead_bits': []}

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
            if use_triadic and step >= triadic_warmup:
                alpha_warmup = int(STEPS * 0.2)
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

        if step % 500 == 0 or step == STEPS - 1:
            with torch.no_grad():
                flat = triadic_proj.reshape(-1, triadic_proj.size(-1))
                bit_means = (flat > 0).float().mean(dim=0)
                eps = 1e-7
                ent = -(bit_means * (bit_means + eps).log2() +
                        (1 - bit_means) * (1 - bit_means + eps).log2())

            history['step'].append(step)
            history['loss'].append(lang_loss.item())
            history['tri_loss'].append(tri_loss_val)
            history['entropy'].append(ent.mean().item())
            history['dead_bits'].append(int((ent < 0.3).sum().item()))

            if step % 2000 == 0:
                elapsed = time.time() - t0
                print(f"  [{label}] step {step}/{STEPS}  loss={lang_loss.item():.3f}  "
                      f"tri={tri_loss_val:.4f}  ent={ent.mean().item():.3f}  ({elapsed:.0f}s)")

    return history


def evaluate_semantic_quality(model, tokenizer, device):
    """Comprehensive semantic evaluation."""
    concept_groups = {
        'related': [
            ("king", "queen"), ("dog", "cat"), ("happy", "sad"),
            ("mother", "father"), ("sun", "moon"), ("hot", "cold"),
            ("love", "hate"), ("big", "small"), ("bird", "fish"),
        ],
        'unrelated': [
            ("king", "fish"), ("dog", "moon"), ("happy", "river"),
            ("mother", "blue"), ("sun", "cat"), ("hot", "queen"),
        ],
    }

    analogy_triples = [
        ("king", "queen", "man", "woman"),
        ("father", "mother", "brother", "sister"),
        ("dog", "puppy", "cat", "kitten"),
        ("big", "small", "tall", "short"),
    ]

    model.eval()
    mapper = PrimeMapper(N_BITS)

    # Get all signatures
    sigs = {}
    all_words = set()
    for group in concept_groups.values():
        for w1, w2 in group:
            all_words.add(w1)
            all_words.add(w2)
    for a, b, c, d in analogy_triples:
        all_words.update([a, b, c, d])

    with torch.no_grad():
        for word in all_words:
            ids = tokenizer.encode(word, add_special=False)
            if ids:
                x = torch.tensor([ids], dtype=torch.long, device=device)
                _, proj, _ = model(x)
                sigs[word] = proj[0].mean(dim=0).cpu().numpy()

    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    # Similarity metrics
    related_sims = []
    for w1, w2 in concept_groups['related']:
        if w1 in sigs and w2 in sigs:
            related_sims.append(cosine(sigs[w1], sigs[w2]))

    unrelated_sims = []
    for w1, w2 in concept_groups['unrelated']:
        if w1 in sigs and w2 in sigs:
            unrelated_sims.append(cosine(sigs[w1], sigs[w2]))

    # Random baseline similarity
    words = list(sigs.keys())
    random_sims = []
    for _ in range(100):
        i, j = random.sample(range(len(words)), 2)
        random_sims.append(cosine(sigs[words[i]], sigs[words[j]]))

    # Analogy accuracy (offset method)
    correct = 0
    total = 0
    for a, b, c, d in analogy_triples:
        if all(w in sigs for w in [a, b, c, d]):
            # B - A + C should be close to D
            predicted = sigs[b] - sigs[a] + sigs[c]
            # Find nearest among candidates
            best_sim = -1
            best_word = None
            for w in sigs:
                if w not in [a, b, c]:
                    s = cosine(predicted, sigs[w])
                    if s > best_sim:
                        best_sim = s
                        best_word = w
            if best_word == d:
                correct += 1
            total += 1

    # Algebraic analogy
    alg_correct = 0
    alg_total = 0
    for a, b, c, d in analogy_triples:
        if all(w in sigs for w in [a, b, c, d]):
            phi_a = mapper.map(sigs[a])
            phi_b = mapper.map(sigs[b])
            phi_c = mapper.map(sigs[c])
            phi_d = mapper.map(sigs[d])
            predicted = TriadicValidator.analogy(phi_a, phi_b, phi_c)
            sim = TriadicValidator.similarity(predicted, phi_d)
            if sim > 0.5:
                alg_correct += 1
            alg_total += 1

    # Bit entropy
    all_projs = np.stack(list(sigs.values()))
    bit_means = (all_projs > 0).mean(axis=0)
    eps = 1e-7
    bit_entropy = -(bit_means * np.log2(bit_means + eps) +
                    (1 - bit_means) * np.log2(1 - bit_means + eps))

    return {
        'semantic_gap': float(np.mean(related_sims) - np.mean(random_sims)),
        'related_vs_unrelated_gap': float(np.mean(related_sims) - np.mean(unrelated_sims)),
        'mean_related_sim': float(np.mean(related_sims)),
        'mean_unrelated_sim': float(np.mean(unrelated_sims)),
        'mean_random_sim': float(np.mean(random_sims)),
        'vector_analogy_acc': correct / max(total, 1),
        'algebraic_analogy_acc': alg_correct / max(alg_total, 1),
        'mean_bit_entropy': float(bit_entropy.mean()),
        'dead_bits': int((bit_entropy < 0.3).sum()),
        'unique_signatures': len(set(mapper.map(p) for p in all_projs)),
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 64)
    print("  RANDOM BASELINE CONTROL EXPERIMENT")
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

    results = {}

    # --- Variant 1: Normal triadic training ---
    print(f"\n{'─' * 64}")
    print("  Variant 1: NORMAL (trainable triadic head)")
    print(f"{'─' * 64}")
    model_normal = TriadicGPT(config).to(device)
    print(f"  Params: {model_normal.num_params():,}")
    hist_normal = train_model(model_normal, tokenizer, all_tokens, device, "NORMAL", use_triadic=True)
    eval_normal = evaluate_semantic_quality(model_normal, tokenizer, device)
    results['normal'] = {'history': hist_normal, 'eval': eval_normal}

    # --- Variant 2: Frozen random head ---
    print(f"\n{'─' * 64}")
    print("  Variant 2: FROZEN RANDOM (triadic head never updated)")
    print(f"{'─' * 64}")
    model_frozen = TriadicGPT(config).to(device)
    # Freeze triadic head
    model_frozen.triadic_head.weight.requires_grad_(False)
    trainable = sum(p.numel() for p in model_frozen.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,} (triadic head frozen)")
    hist_frozen = train_model(model_frozen, tokenizer, all_tokens, device, "FROZEN", use_triadic=False)
    eval_frozen = evaluate_semantic_quality(model_frozen, tokenizer, device)
    results['frozen_random'] = {'history': hist_frozen, 'eval': eval_frozen}

    # --- Variant 3: No triadic (language only) ---
    print(f"\n{'─' * 64}")
    print("  Variant 3: LANGUAGE ONLY (no triadic loss)")
    print(f"{'─' * 64}")
    model_ablation = TriadicGPT(config).to(device)
    hist_ablation = train_model(model_ablation, tokenizer, all_tokens, device, "LANG-ONLY", use_triadic=False)
    eval_ablation = evaluate_semantic_quality(model_ablation, tokenizer, device)
    results['language_only'] = {'history': hist_ablation, 'eval': eval_ablation}

    # --- Comparison ---
    print("\n" + "=" * 64)
    print("  RESULTS COMPARISON")
    print("=" * 64)

    print(f"\n  {'Metric':>30s}  {'Normal':>10s}  {'Frozen':>10s}  {'LangOnly':>10s}")
    print(f"  {'─'*30}  {'─'*10}  {'─'*10}  {'─'*10}")

    eval_keys = [
        'semantic_gap', 'related_vs_unrelated_gap', 'mean_related_sim',
        'mean_unrelated_sim', 'vector_analogy_acc', 'algebraic_analogy_acc',
        'mean_bit_entropy', 'dead_bits', 'unique_signatures',
    ]
    for key in eval_keys:
        vn = eval_normal.get(key, 0)
        vf = eval_frozen.get(key, 0)
        va = eval_ablation.get(key, 0)
        fmt = '.4f' if isinstance(vn, float) else 'd'
        print(f"  {key:>30s}  {vn:>10{fmt}}  {vf:>10{fmt}}  {va:>10{fmt}}")

    print(f"\n  Language loss:")
    for name, r in results.items():
        print(f"    {name:>20s}: {r['history']['loss'][-1]:.4f}")

    # Key question
    gap_normal = eval_normal['semantic_gap']
    gap_frozen = eval_frozen['semantic_gap']
    gap_ablation = eval_ablation['semantic_gap']

    print(f"\n  CONCLUSION:")
    if gap_normal > gap_frozen + 0.005:
        print(f"    Triadic head IS learning meaningful structure")
        print(f"    (gap: {gap_normal:+.4f} trained vs {gap_frozen:+.4f} frozen)")
    elif gap_normal > gap_frozen:
        print(f"    Triadic head shows MARGINAL improvement over random")
        print(f"    (gap: {gap_normal:+.4f} trained vs {gap_frozen:+.4f} frozen)")
    else:
        print(f"    Triadic head NOT learning beyond random projection")
        print(f"    (gap: {gap_normal:+.4f} trained vs {gap_frozen:+.4f} frozen)")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    variants = [('Normal', results['normal'], 'blue'),
                ('Frozen Random', results['frozen_random'], 'red'),
                ('Language Only', results['language_only'], 'gray')]

    for name, r, color in variants:
        h = r['history']
        axes[0, 0].plot(h['step'], h['loss'], color=color, label=name, alpha=0.8)
        axes[0, 1].plot(h['step'], h['entropy'], color=color, label=name, alpha=0.8)
        axes[1, 0].plot(h['step'], h['dead_bits'], color=color, label=name, alpha=0.8)

    for ax, title, ylabel in [
        (axes[0, 0], 'Language Loss', 'Loss'),
        (axes[0, 1], 'Bit Entropy', 'Mean Entropy'),
        (axes[1, 0], 'Dead Bits', 'Count'),
    ]:
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Step')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Bar chart of semantic metrics
    ax = axes[1, 1]
    metrics_to_plot = ['semantic_gap', 'related_vs_unrelated_gap', 'mean_bit_entropy']
    x_pos = np.arange(len(metrics_to_plot))
    width = 0.25
    for i, (name, r, color) in enumerate(variants):
        vals = [r['eval'][m] for m in metrics_to_plot]
        ax.bar(x_pos + i * width, vals, width, label=name, color=color, alpha=0.7)
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([m.replace('_', '\n') for m in metrics_to_plot], fontsize=8)
    ax.set_title('Semantic Quality Metrics')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Random Baseline Control: Is the Triadic Head Learning?', fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'random_baseline.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved: {plot_path}")

    # Save
    save_data = {
        'experiment': 'random_baseline',
        'config': f'{N_LAYER}L/{N_EMBD}D/{N_BITS}bits',
        'steps': STEPS,
    }
    for name, r in results.items():
        save_data[name] = {
            'final_loss': r['history']['loss'][-1],
            **r['eval'],
        }

    results_path = os.path.join(RESULTS_DIR, 'random_baseline.json')
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"  Results saved: {results_path}")


if __name__ == '__main__':
    main()
