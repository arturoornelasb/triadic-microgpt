"""
R3 + Entropy Guard — Fix R3's entropy collapse.

R3 loss generalizes to held-out analogies (0.999 offset cosine) but causes
64/64 dead bits. Can we prevent the collapse by cranking entropy_weight?

5 variants: baseline, R3 with entropy 1x, 5x, 10x, 20x.
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
ALIGN_WEIGHT = 5.0
TRIADIC_WARMUP_PCT = 0.25
N_LAYER = 6
N_EMBD = 256
N_HEAD = 8
N_BITS = 64
R3_WEIGHT = 5.0

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
    ("queen", "king", "woman", "man"),
    ("mother", "father", "sister", "brother"),
    ("small", "big", "short", "tall"),
]

HELD_OUT_ANALOGIES = [
    ("boy", "girl", "man", "woman"),
    ("dog", "cat", "puppy", "kitten"),
    ("red", "blue", "green", "yellow"),
    ("morning", "evening", "day", "night"),
]


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def progress_bar(current, total, width=25):
    pct = current / max(total, 1)
    filled = int(width * pct)
    return f"[{'█' * filled}{'░' * (width - filled)}] {pct:5.1%}"


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
    tensors = []
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
            tensors.append(ids)
    print(f"  Analogy triples: {len(tensors)}")
    return tensors


def compute_r3_loss(model, analogy_tensors, device):
    if not analogy_tensors:
        return torch.tensor(0.0, device=device)
    losses = []
    for ids in analogy_tensors:
        projs = {}
        for label in ['a', 'b', 'c', 'd']:
            x = ids[label].unsqueeze(0)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                _, proj, _ = model(x)
            projs[label] = proj[0].mean(dim=0)
        predicted = projs['b'] - projs['a'] + projs['c']
        losses.append(F.mse_loss(predicted, projs['d']))
    return torch.stack(losses).mean()


def evaluate(model, tokenizer, device):
    model.eval()
    mapper = PrimeMapper(N_BITS)

    all_words = set()
    for a, b, c, d in ANALOGY_TRIPLES + HELD_OUT_ANALOGIES:
        all_words.update([a, b, c, d])
    all_words.update(["tree", "river", "mountain", "stone", "cloud", "table"])

    projs = {}
    with torch.no_grad():
        for word in all_words:
            ids = tokenizer.encode(word, add_special=False)
            if ids:
                x = torch.tensor([ids], dtype=torch.long, device=device)
                _, proj, _ = model(x)
                projs[word] = proj[0].mean(dim=0).cpu().numpy()

    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    # Analogies
    def eval_ana(triples):
        results = []
        for a, b, c, d in triples:
            if not all(w in projs for w in [a, b, c, d]):
                continue
            pred = projs[b] - projs[a] + projs[c]
            results.append(cosine(pred, projs[d]))
        return results

    ana_train = eval_ana(ANALOGY_TRIPLES[:13])
    ana_test = eval_ana(HELD_OUT_ANALOGIES)

    # Semantic gap
    related = [("king","queen"),("dog","cat"),("happy","sad"),("mother","father"),
               ("sun","moon"),("hot","cold"),("love","hate"),("big","small")]
    rel_sims = [cosine(projs[a], projs[b]) for a, b in related if a in projs and b in projs]
    rand_sims = []
    wlist = list(projs.keys())
    for _ in range(200):
        i, j = random.sample(range(len(wlist)), 2)
        rand_sims.append(cosine(projs[wlist[i]], projs[wlist[j]]))
    gap = np.mean(rel_sims) - np.mean(rand_sims) if rel_sims else 0

    # Bits
    all_p = np.stack(list(projs.values()))
    bm = (all_p > 0).mean(axis=0)
    eps = 1e-7
    bent = -(bm * np.log2(bm + eps) + (1 - bm) * np.log2(1 - bm + eps))
    dead = int((bent < 0.3).sum())

    return {
        'gap': float(gap),
        'ana_train': float(np.mean(ana_train)) if ana_train else 0,
        'ana_test': float(np.mean(ana_test)) if ana_test else 0,
        'ana_test_detail': ana_test,
        'dead': dead,
        'entropy': float(bent.mean()),
    }


def train_model(model, tokenizer, all_tokens, device, label,
                analogy_tensors, entropy_weight):
    dataset = TextDataset(all_tokens, BLOCK_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    triadic_warmup = int(STEPS * TRIADIC_WARMUP_PCT)

    model.train()
    data_iter = iter(dataloader)
    history = {'step': [], 'loss': [], 'tri': [], 'r3': []}

    t0 = time.time()
    for step in range(STEPS):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)

        ws = min(500, STEPS // 10)
        if step < ws:
            lr_t = LR * (step + 1) / ws
        else:
            prog = (step - ws) / max(STEPS - ws, 1)
            lr_t = LR * max(0.1, 0.5 * (1.0 + math.cos(math.pi * prog)))
        for pg in optimizer.param_groups:
            pg['lr'] = lr_t

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits, triadic_proj, lang_loss = model(x, targets=y)
            total_loss = lang_loss
            tri_v, r3_v = 0.0, 0.0

            if step >= triadic_warmup:
                aw = int(STEPS * 0.2)
                af = min(1.0, (step - triadic_warmup + 1) / aw)
                ca = ALPHA * af

                tri_loss = model.triadic_loss(triadic_proj, entropy_weight=entropy_weight,
                                              input_ids=x, align_weight=ALIGN_WEIGHT, align_mode='mse')
                total_loss = lang_loss + ca * tri_loss
                tri_v = tri_loss.item()

                if step % 5 == 0 and analogy_tensors:
                    r3_loss = compute_r3_loss(model, analogy_tensors, device)
                    total_loss = total_loss + ca * R3_WEIGHT * r3_loss
                    r3_v = r3_loss.item()

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % 200 == 0 or step == STEPS - 1:
            history['step'].append(step)
            history['loss'].append(lang_loss.item())
            history['tri'].append(tri_v)
            history['r3'].append(r3_v)

            elapsed = time.time() - t0
            speed = (step + 1) / max(elapsed, 1)
            eta_s = (STEPS - step - 1) / max(speed, 0.01)
            bar = progress_bar(step + 1, STEPS)
            print(f"  [{label:>10s}] {bar}  step {step:>5d}/{STEPS}  "
                  f"loss={lang_loss.item():.3f}  r3={r3_v:.4f}  "
                  f"ETA {format_time(eta_s)}  [{format_time(elapsed)}]")

    return history


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("  R3 + ENTROPY GUARD EXPERIMENT")
    print("  Can stronger entropy regularization prevent R3's bit collapse?")
    print("=" * 70)
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU:    {torch.cuda.get_device_name(0)}")

    tok_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'torch_run15_strongalign', 'tokenizer.json')
    tokenizer = BPETokenizer.load(tok_path)

    print("\nLoading data...")
    all_tokens = load_data(tokenizer)
    print("\nPreparing analogies...")
    analogy_tensors = prepare_analogy_data(tokenizer, device)

    config = TriadicGPTConfig(
        vocab_size=tokenizer.vocab_size, block_size=BLOCK_SIZE,
        n_layer=N_LAYER, n_embd=N_EMBD, n_head=N_HEAD,
        n_triadic_bits=N_BITS, dropout=0.1,
    )

    # Baseline (no R3) + R3 with increasing entropy weight
    variants = [
        ("No R3 (1x)", False, 1.0),
        ("R3 + ent 1x", True, 1.0),
        ("R3 + ent 5x", True, 5.0),
        ("R3 + ent 10x", True, 10.0),
        ("R3 + ent 20x", True, 20.0),
    ]

    all_results = {}
    all_histories = {}

    for name, use_r3, ent_w in variants:
        print(f"\n{'─' * 70}")
        print(f"  Training: {name}")
        print(f"{'─' * 70}")

        model = TriadicGPT(config).to(device)
        hist = train_model(model, tokenizer, all_tokens, device, name,
                           analogy_tensors=analogy_tensors if use_r3 else None,
                           entropy_weight=ent_w)
        ev = evaluate(model, tokenizer, device)
        all_results[name] = ev
        all_histories[name] = hist

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  R3 + ENTROPY GUARD RESULTS")
    print("=" * 70)

    print(f"\n  {'Variant':>14s}  {'Loss':>6s}  {'Gap':>8s}  {'Dead':>4s}  {'Ent':>6s}  "
          f"{'Ana(tr)':>8s}  {'Ana(te)':>8s}")
    print(f"  {'─'*14}  {'─'*6}  {'─'*8}  {'─'*4}  {'─'*6}  {'─'*8}  {'─'*8}")

    for name, _, _ in variants:
        ev = all_results[name]
        h = all_histories[name]
        print(f"  {name:>14s}  {h['loss'][-1]:>6.3f}  {ev['gap']:>+8.4f}  "
              f"{ev['dead']:>4d}  {ev['entropy']:>6.3f}  "
              f"{ev['ana_train']:>8.4f}  {ev['ana_test']:>8.4f}")

    print(f"\n  Held-out Analogies Detail:")
    for name, _, _ in variants:
        ev = all_results[name]
        if ev['ana_test_detail']:
            vals = ev['ana_test_detail']
            print(f"    {name:>14s}: {' | '.join(f'{v:.3f}' for v in vals)}  mean={np.mean(vals):.4f}")

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = ['blue', 'red', 'orange', 'green', 'purple']

    for i, (name, _, _) in enumerate(variants):
        h = all_histories[name]
        axes[0].plot(h['step'], h['loss'], color=colors[i], label=name, alpha=0.8)
        axes[1].plot(h['step'], h['r3'], color=colors[i], label=name, alpha=0.8)

    # Bar chart: dead bits vs analogy quality
    names = [v[0] for v in variants]
    dead_bits = [all_results[n]['dead'] for n in names]
    ana_test = [all_results[n]['ana_test'] for n in names]

    x = np.arange(len(names))
    w = 0.35
    axes[2].bar(x - w/2, [d/64 for d in dead_bits], w, label='Dead bits (frac)', color='lightcoral')
    axes[2].bar(x + w/2, ana_test, w, label='Ana test cos', color='steelblue')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([n.replace('R3 + ', '') for n in names], rotation=45, ha='right', fontsize=7)
    axes[2].set_ylim(0, 1.1)
    axes[2].legend(fontsize=8)
    axes[2].set_title('Dead Bits vs Analogy Quality')

    for ax, title in [(axes[0], 'Language Loss'), (axes[1], 'R3 Loss')]:
        ax.set_title(title)
        ax.set_xlabel('Step')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle('R3 + Entropy Guard: Can We Prevent Bit Collapse?', fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'r3_entropy_guard.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved: {plot_path}")

    # ── Save ──
    save_data = {
        'experiment': 'r3_entropy_guard',
        'config': f'{N_LAYER}L/{N_EMBD}D/{N_BITS}bits',
        'steps': STEPS,
        'r3_weight': R3_WEIGHT,
        'variants': {name: {'use_r3': use_r3, 'entropy_weight': ent_w,
                            'final_loss': all_histories[name]['loss'][-1],
                            'eval': all_results[name]}
                     for name, use_r3, ent_w in variants},
    }
    results_path = os.path.join(RESULTS_DIR, 'r3_entropy_guard.json')
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"  Results saved: {results_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
