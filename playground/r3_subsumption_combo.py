"""
P2 — R3 + Subsumption Combo

Combines the two successful playground findings:
1. Rule-of-Three loss → perfect analogy algebra (K=1.0)
2. Subsumption loss → 91.7% held-out hypernym containment

Question: do they compound or interfere?

4 variants: baseline, R3-only, Sub-only, R3+Sub combined.
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


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def progress_bar(current, total, width=25):
    pct = current / max(total, 1)
    filled = int(width * pct)
    return f"[{'█' * filled}{'░' * (width - filled)}] {pct:5.1%}"


# ── Data ──

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

# Held-out analogies (NOT in training set)
HELD_OUT_ANALOGIES = [
    ("boy", "girl", "man", "woman"),
    ("dog", "cat", "puppy", "kitten"),
    ("red", "blue", "green", "yellow"),
    ("morning", "evening", "day", "night"),
]

HYPERNYM_PAIRS = {
    "animal": ["dog", "cat", "bird", "fish", "horse", "rabbit", "bear", "mouse", "lion"],
    "person": ["king", "queen", "doctor", "teacher", "princess", "prince", "boy", "girl"],
    "feeling": ["happy", "sad", "love", "hate", "angry", "scared"],
    "food": ["apple", "cake", "bread", "candy", "cookie"],
    "color": ["red", "blue", "green", "yellow", "pink", "purple"],
    "place": ["school", "hospital", "house", "garden", "forest", "beach", "park"],
    "time": ["day", "night", "morning", "evening"],
}

HELD_OUT_HYPERNYMS = {
    "animal": ["tiger", "frog", "deer"],
    "person": ["man", "woman", "baby"],
    "food": ["pizza", "milk", "egg"],
    "place": ["castle", "farm", "river"],
}


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
    print(f"  Analogy triples (train): {len(tensors)}")
    return tensors


def prepare_subsumption_data(tokenizer, device, pairs_dict):
    sub_pairs = []
    for hypernym, hyponyms in pairs_dict.items():
        hyper_ids = tokenizer.encode(hypernym, add_special=False)
        if not hyper_ids:
            continue
        hyper_tensor = torch.tensor(hyper_ids, dtype=torch.long, device=device)
        for hyponym in hyponyms:
            hypo_ids = tokenizer.encode(hyponym, add_special=False)
            if not hypo_ids:
                continue
            sub_pairs.append({
                'hypernym': hypernym, 'hyponym': hyponym,
                'hyper_ids': hyper_tensor,
                'hypo_ids': torch.tensor(hypo_ids, dtype=torch.long, device=device),
            })
    print(f"  Subsumption pairs: {len(sub_pairs)}")
    return sub_pairs


# ── Losses ──

def compute_r3_loss(model, analogy_tensors, device):
    if not analogy_tensors:
        return torch.tensor(0.0, device=device)
    losses = []
    for ids in analogy_tensors:
        projs = {}
        for label in ['a', 'b', 'c', 'd']:
            x = ids[label].unsqueeze(0)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                _, proj, _ = model(x)
            projs[label] = proj[0].mean(dim=0)
        predicted = projs['b'] - projs['a'] + projs['c']
        losses.append(F.mse_loss(predicted, projs['d']))
    return torch.stack(losses).mean()


def compute_sub_loss(model, sub_pairs, device):
    if not sub_pairs:
        return torch.tensor(0.0, device=device)
    losses = []
    for pair in sub_pairs:
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
            _, proj_h, _ = model(pair['hyper_ids'].unsqueeze(0))
            _, proj_y, _ = model(pair['hypo_ids'].unsqueeze(0))
        h = proj_h[0].mean(dim=0)
        y = proj_y[0].mean(dim=0)
        losses.append(F.relu(h - y).mean())
    return torch.stack(losses).mean()


# ── Evaluation ──

def get_projections(model, tokenizer, device, words):
    model.eval()
    sigs = {}
    projs = {}
    mapper = PrimeMapper(N_BITS)
    with torch.no_grad():
        for word in words:
            ids = tokenizer.encode(word, add_special=False)
            if not ids:
                continue
            x = torch.tensor([ids], dtype=torch.long, device=device)
            _, proj, _ = model(x)
            p = proj[0].mean(dim=0).cpu().numpy()
            projs[word] = p
            sigs[word] = mapper.map(p)
    return projs, sigs


def evaluate_all(model, tokenizer, device):
    """Comprehensive evaluation: analogies (train+test), subsumption (train+test), semantic gap."""
    mapper = PrimeMapper(N_BITS)

    # Collect all words
    all_words = set()
    for a, b, c, d in ANALOGY_TRIPLES + HELD_OUT_ANALOGIES:
        all_words.update([a, b, c, d])
    for pairs in [HYPERNYM_PAIRS, HELD_OUT_HYPERNYMS]:
        for h, hypos in pairs.items():
            all_words.add(h)
            all_words.update(hypos)

    # Extra words for semantic gap
    extra = ["river", "tree", "mountain", "car", "table", "window", "stone", "cloud"]
    all_words.update(extra)

    projs, sigs = get_projections(model, tokenizer, device, all_words)

    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    # ── Analogies ──
    def eval_analogies(triples, label):
        results = []
        for a, b, c, d in triples:
            if not all(w in projs for w in [a, b, c, d]):
                continue
            pred = projs[b] - projs[a] + projs[c]
            offset_cos = cosine(pred, projs[d])
            phi_pred = TriadicValidator.analogy(sigs[a], sigs[b], sigs[c])
            alg_sim = TriadicValidator.similarity(phi_pred, sigs[d])

            sim_ab = cosine(projs[a], projs[b])
            sim_cd = cosine(projs[c], projs[d])
            sim_ac = cosine(projs[a], projs[c])
            sim_bd = cosine(projs[b], projs[d])
            k = (sim_ab * sim_cd) / (sim_ac * sim_bd + 1e-10)

            results.append({'analogy': f'{a}:{b}::{c}:{d}',
                            'offset_cos': offset_cos, 'alg_sim': float(alg_sim), 'K': float(k)})
        return results

    ana_train = eval_analogies(ANALOGY_TRIPLES[:13], "train")  # first 13 unique
    ana_test = eval_analogies(HELD_OUT_ANALOGIES, "test")

    # ── Subsumption ──
    def eval_subsumption(pairs_dict):
        total, hits, inheritances = 0, 0, []
        for hyper, hypos in pairs_dict.items():
            if hyper not in projs:
                continue
            hbits = (projs[hyper] > 0).astype(int)
            for hypo in hypos:
                if hypo not in projs:
                    continue
                total += 1
                ybits = (projs[hypo] > 0).astype(int)
                if sigs[hypo] != 0 and sigs[hyper] != 0 and sigs[hypo] % sigs[hyper] == 0:
                    hits += 1
                ha = hbits.sum()
                inheritances.append((hbits * ybits).sum() / max(ha, 1))
        return {'rate': hits / max(total, 1), 'inheritance': float(np.mean(inheritances)) if inheritances else 0,
                'total': total, 'hits': hits}

    sub_train = eval_subsumption(HYPERNYM_PAIRS)
    sub_test = eval_subsumption(HELD_OUT_HYPERNYMS)

    # ── Semantic gap ──
    related_pairs = [("king", "queen"), ("dog", "cat"), ("happy", "sad"),
                     ("mother", "father"), ("sun", "moon"), ("hot", "cold"),
                     ("love", "hate"), ("big", "small"), ("bird", "fish")]
    related_sims = [cosine(projs[a], projs[b]) for a, b in related_pairs if a in projs and b in projs]
    rand_sims = []
    wlist = list(projs.keys())
    for _ in range(200):
        i, j = random.sample(range(len(wlist)), 2)
        rand_sims.append(cosine(projs[wlist[i]], projs[wlist[j]]))
    gap = np.mean(related_sims) - np.mean(rand_sims) if related_sims else 0

    # ── Bit stats ──
    all_p = np.stack(list(projs.values()))
    bm = (all_p > 0).mean(axis=0)
    eps = 1e-7
    bent = -(bm * np.log2(bm + eps) + (1 - bm) * np.log2(1 - bm + eps))
    dead = int((bent < 0.3).sum())

    return {
        'analogy_train': ana_train,
        'analogy_test': ana_test,
        'sub_train': sub_train,
        'sub_test': sub_test,
        'semantic_gap': float(gap),
        'dead_bits': dead,
        'mean_entropy': float(bent.mean()),
    }


# ── Training ──

def train_model(model, tokenizer, all_tokens, device, label,
                analogy_tensors=None, sub_pairs=None,
                r3_weight=0.0, sub_weight=0.0):
    dataset = TextDataset(all_tokens, BLOCK_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
    amp_dtype = torch.bfloat16
    use_scaler = False  # bfloat16 doesn't need loss scaling
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
    triadic_warmup = int(STEPS * TRIADIC_WARMUP_PCT)

    model.train()
    data_iter = iter(dataloader)
    history = {'step': [], 'loss': [], 'tri': [], 'r3': [], 'sub': []}

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

        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=(device.type == 'cuda')):
            logits, triadic_proj, lang_loss = model(x, targets=y)
            total_loss = lang_loss
            tri_v, r3_v, sub_v = 0.0, 0.0, 0.0

            if step >= triadic_warmup:
                aw = int(STEPS * 0.2)
                af = min(1.0, (step - triadic_warmup + 1) / aw)
                ca = ALPHA * af

                tri_loss = model.triadic_loss(triadic_proj, entropy_weight=ENTROPY_WEIGHT,
                                              input_ids=x, align_weight=ALIGN_WEIGHT, align_mode='mse')
                total_loss = lang_loss + ca * tri_loss
                tri_v = tri_loss.item()

                if step % 5 == 0:
                    if r3_weight > 0 and analogy_tensors:
                        r3_loss = compute_r3_loss(model, analogy_tensors, device)
                        total_loss = total_loss + ca * r3_weight * r3_loss
                        r3_v = r3_loss.item()
                    if sub_weight > 0 and sub_pairs:
                        s_loss = compute_sub_loss(model, sub_pairs, device)
                        total_loss = total_loss + ca * sub_weight * s_loss
                        sub_v = s_loss.item()

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
            history['sub'].append(sub_v)

            elapsed = time.time() - t0
            speed = (step + 1) / max(elapsed, 1)
            eta_s = (STEPS - step - 1) / max(speed, 0.01)
            bar = progress_bar(step + 1, STEPS)
            print(f"  [{label:>10s}] {bar}  step {step:>5d}/{STEPS}  "
                  f"loss={lang_loss.item():.3f}  r3={r3_v:.4f}  sub={sub_v:.4f}  "
                  f"ETA {format_time(eta_s)}  [{format_time(elapsed)}]")

    return history


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("  R3 + SUBSUMPTION COMBO EXPERIMENT")
    print("  Do analogy and subsumption losses compound or interfere?")
    print("=" * 70)
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU:    {torch.cuda.get_device_name(0)}")

    tok_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'torch_run15_strongalign', 'tokenizer.json')
    tokenizer = BPETokenizer.load(tok_path)

    print("\nLoading data...")
    all_tokens = load_data(tokenizer)

    print("\nPreparing structured data...")
    analogy_tensors = prepare_analogy_data(tokenizer, device)
    sub_train = prepare_subsumption_data(tokenizer, device, HYPERNYM_PAIRS)
    sub_test = prepare_subsumption_data(tokenizer, device, HELD_OUT_HYPERNYMS)

    config = TriadicGPTConfig(
        vocab_size=tokenizer.vocab_size, block_size=BLOCK_SIZE,
        n_layer=N_LAYER, n_embd=N_EMBD, n_head=N_HEAD,
        n_triadic_bits=N_BITS, dropout=0.1,
    )

    variants = [
        ("Baseline",  0.0, 0.0),
        ("R3 only",   5.0, 0.0),
        ("Sub only",  0.0, 5.0),
        ("R3+Sub",    5.0, 5.0),
    ]

    all_results = {}
    all_histories = {}

    for name, r3w, subw in variants:
        print(f"\n{'─' * 70}")
        print(f"  Training: {name} (r3={r3w}, sub={subw})")
        print(f"{'─' * 70}")

        model = TriadicGPT(config).to(device)
        hist = train_model(model, tokenizer, all_tokens, device, name,
                           analogy_tensors=analogy_tensors if r3w > 0 else None,
                           sub_pairs=sub_train if subw > 0 else None,
                           r3_weight=r3w, sub_weight=subw)
        ev = evaluate_all(model, tokenizer, device)
        all_results[name] = ev
        all_histories[name] = hist
        model.train()  # reset for next

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  COMBO RESULTS")
    print("=" * 70)

    header = (f"  {'Variant':>12s}  {'Loss':>6s}  {'Gap':>7s}  {'Dead':>4s}  "
              f"{'Ana(tr)':>8s}  {'Ana(te)':>8s}  {'K(tr)':>6s}  {'K(te)':>6s}  "
              f"{'Sub(tr)':>8s}  {'Sub(te)':>8s}")
    print(header)
    print(f"  {'─'*12}  {'─'*6}  {'─'*7}  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*8}  {'─'*8}")

    for name in [v[0] for v in variants]:
        ev = all_results[name]
        h = all_histories[name]

        ana_tr_cos = np.mean([a['offset_cos'] for a in ev['analogy_train']]) if ev['analogy_train'] else 0
        ana_te_cos = np.mean([a['offset_cos'] for a in ev['analogy_test']]) if ev['analogy_test'] else 0
        k_tr = np.mean([a['K'] for a in ev['analogy_train']]) if ev['analogy_train'] else 0
        k_te = np.mean([a['K'] for a in ev['analogy_test']]) if ev['analogy_test'] else 0

        print(f"  {name:>12s}  {h['loss'][-1]:>6.3f}  {ev['semantic_gap']:>+7.4f}  {ev['dead_bits']:>4d}  "
              f"{ana_tr_cos:>8.4f}  {ana_te_cos:>8.4f}  {k_tr:>6.3f}  {k_te:>6.3f}  "
              f"{ev['sub_train']['rate']:>7.1%}  {ev['sub_test']['rate']:>7.1%}")

    # ── Detail: held-out analogies ──
    print(f"\n  Held-out Analogies (never in training):")
    for name in [v[0] for v in variants]:
        ev = all_results[name]
        if ev['analogy_test']:
            cos_vals = [a['offset_cos'] for a in ev['analogy_test']]
            print(f"    {name:>12s}: {' | '.join(f'{c:.3f}' for c in cos_vals)}  "
                  f"mean={np.mean(cos_vals):.4f}")

    # ── Plot ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {'Baseline': 'blue', 'R3 only': 'orange', 'Sub only': 'green', 'R3+Sub': 'red'}

    for name in [v[0] for v in variants]:
        h = all_histories[name]
        c = colors[name]
        axes[0, 0].plot(h['step'], h['loss'], color=c, label=name, alpha=0.8)
        axes[0, 1].plot(h['step'], h['r3'], color=c, label=name, alpha=0.8)
        axes[1, 0].plot(h['step'], h['sub'], color=c, label=name, alpha=0.8)

    # Bar chart for final metrics
    names = [v[0] for v in variants]
    sub_test_rates = [all_results[n]['sub_test']['rate'] for n in names]
    ana_test_means = [np.mean([a['offset_cos'] for a in all_results[n]['analogy_test']])
                      if all_results[n]['analogy_test'] else 0 for n in names]

    x_pos = np.arange(len(names))
    width = 0.35
    axes[1, 1].bar(x_pos - width/2, sub_test_rates, width, label='Sub test', color='steelblue')
    axes[1, 1].bar(x_pos + width/2, ana_test_means, width, label='Ana test cos', color='coral')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(names, fontsize=8)
    axes[1, 1].set_ylim(0, 1.1)
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_title('Held-Out Performance')

    for ax, title, ylabel in [
        (axes[0, 0], 'Language Loss', 'Loss'),
        (axes[0, 1], 'R3 Loss', 'Loss'),
        (axes[1, 0], 'Subsumption Loss', 'Loss'),
    ]:
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Step')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('R3 + Subsumption Combo: Do They Compound?', fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'r3_subsumption_combo.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved: {plot_path}")

    # ── Save ──
    save_data = {
        'experiment': 'r3_subsumption_combo',
        'config': f'{N_LAYER}L/{N_EMBD}D/{N_BITS}bits',
        'steps': STEPS,
        'variants': {name: {'r3_weight': r3w, 'sub_weight': subw,
                            'final_loss': all_histories[name]['loss'][-1],
                            'eval': all_results[name]}
                     for name, r3w, subw in variants},
    }
    results_path = os.path.join(RESULTS_DIR, 'r3_subsumption_combo.json')
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"  Results saved: {results_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
