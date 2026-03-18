"""
Curriculum: Sub → R3 — Two-phase training.

Phase 1: Train with subsumption loss to establish hierarchical structure.
Phase 2: Fine-tune with R3 loss + high entropy to add analogy structure
          on top of the established hierarchy.

Hypothesis: sequential training avoids the interference seen when
combining both losses simultaneously.
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

PHASE1_STEPS = 7000
PHASE2_STEPS = 3000
TOTAL_STEPS = PHASE1_STEPS + PHASE2_STEPS
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
    return tensors


def prepare_sub_data(tokenizer, device, pairs_dict):
    sub_pairs = []
    for hyper, hypos in pairs_dict.items():
        hyper_ids = tokenizer.encode(hyper, add_special=False)
        if not hyper_ids:
            continue
        ht = torch.tensor(hyper_ids, dtype=torch.long, device=device)
        for hypo in hypos:
            hypo_ids = tokenizer.encode(hypo, add_special=False)
            if not hypo_ids:
                continue
            sub_pairs.append({'hyper_ids': ht,
                              'hypo_ids': torch.tensor(hypo_ids, dtype=torch.long, device=device)})
    return sub_pairs


def compute_r3_loss(model, analogy_tensors, device):
    losses = []
    for ids in analogy_tensors:
        projs = {}
        for label in ['a', 'b', 'c', 'd']:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                _, proj, _ = model(ids[label].unsqueeze(0))
            projs[label] = proj[0].mean(dim=0)
        predicted = projs['b'] - projs['a'] + projs['c']
        losses.append(F.mse_loss(predicted, projs['d']))
    return torch.stack(losses).mean()


def compute_sub_loss(model, sub_pairs, device):
    losses = []
    for pair in sub_pairs:
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
            _, ph, _ = model(pair['hyper_ids'].unsqueeze(0))
            _, py, _ = model(pair['hypo_ids'].unsqueeze(0))
        losses.append(F.relu(ph[0].mean(dim=0) - py[0].mean(dim=0)).mean())
    return torch.stack(losses).mean()


def evaluate_all(model, tokenizer, device):
    model.eval()
    mapper = PrimeMapper(N_BITS)

    all_words = set()
    for a, b, c, d in ANALOGY_TRIPLES + HELD_OUT_ANALOGIES:
        all_words.update([a, b, c, d])
    for pairs in [HYPERNYM_PAIRS, HELD_OUT_HYPERNYMS]:
        for h, hypos in pairs.items():
            all_words.add(h)
            all_words.update(hypos)
    all_words.update(["tree", "river", "mountain", "stone", "cloud"])

    projs, sigs = {}, {}
    with torch.no_grad():
        for word in all_words:
            ids = tokenizer.encode(word, add_special=False)
            if ids:
                x = torch.tensor([ids], dtype=torch.long, device=device)
                _, proj, _ = model(x)
                p = proj[0].mean(dim=0).cpu().numpy()
                projs[word] = p
                sigs[word] = mapper.map(p)

    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    # Analogies
    def eval_ana(triples):
        return [cosine(projs[b]-projs[a]+projs[c], projs[d])
                for a,b,c,d in triples if all(w in projs for w in [a,b,c,d])]

    ana_train = eval_ana(ANALOGY_TRIPLES[:13])
    ana_test = eval_ana(HELD_OUT_ANALOGIES)

    # Subsumption
    def eval_sub(pairs_dict):
        total, hits = 0, 0
        for hyper, hypos in pairs_dict.items():
            if hyper not in sigs:
                continue
            for hypo in hypos:
                if hypo not in sigs:
                    continue
                total += 1
                if sigs[hypo] != 0 and sigs[hyper] != 0 and sigs[hypo] % sigs[hyper] == 0:
                    hits += 1
        return hits / max(total, 1)

    # Semantic gap
    related = [("king","queen"),("dog","cat"),("happy","sad"),("mother","father"),
               ("sun","moon"),("hot","cold"),("love","hate"),("big","small")]
    rel_sims = [cosine(projs[a], projs[b]) for a, b in related if a in projs and b in projs]
    rand_sims = []
    wlist = list(projs.keys())
    for _ in range(200):
        i, j = random.sample(range(len(wlist)), 2)
        rand_sims.append(cosine(projs[wlist[i]], projs[wlist[j]]))

    # Bits
    all_p = np.stack(list(projs.values()))
    bm = (all_p > 0).mean(axis=0)
    eps = 1e-7
    bent = -(bm * np.log2(bm + eps) + (1 - bm) * np.log2(1 - bm + eps))

    return {
        'gap': float(np.mean(rel_sims) - np.mean(rand_sims)),
        'ana_train': float(np.mean(ana_train)) if ana_train else 0,
        'ana_test': float(np.mean(ana_test)) if ana_test else 0,
        'ana_test_detail': ana_test,
        'sub_train': eval_sub(HYPERNYM_PAIRS),
        'sub_test': eval_sub(HELD_OUT_HYPERNYMS),
        'dead': int((bent < 0.3).sum()),
        'entropy': float(bent.mean()),
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("  CURRICULUM: SUB → R3")
    print("  Phase 1: Subsumption (hierarchy) → Phase 2: R3 (analogies)")
    print("=" * 70)
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU:    {torch.cuda.get_device_name(0)}")
    print(f"  Phase 1: {PHASE1_STEPS} steps (Sub loss)")
    print(f"  Phase 2: {PHASE2_STEPS} steps (R3 loss + high entropy)")

    tok_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'torch_run15_strongalign', 'tokenizer.json')
    tokenizer = BPETokenizer.load(tok_path)

    print("\nLoading data...")
    all_tokens = load_data(tokenizer)
    analogy_tensors = prepare_analogy_data(tokenizer, device)
    sub_train = prepare_sub_data(tokenizer, device, HYPERNYM_PAIRS)
    print(f"  Analogies: {len(analogy_tensors)}, Sub pairs: {len(sub_train)}")

    config = TriadicGPTConfig(
        vocab_size=tokenizer.vocab_size, block_size=BLOCK_SIZE,
        n_layer=N_LAYER, n_embd=N_EMBD, n_head=N_HEAD,
        n_triadic_bits=N_BITS, dropout=0.1,
    )

    # ── Train 3 models for comparison ──
    results = {}
    histories = {}

    for variant_name, phases in [
        ("Sub only", [
            {'steps': TOTAL_STEPS, 'sub_w': 5.0, 'r3_w': 0.0, 'ent_w': 1.0, 'label': 'Sub only'}
        ]),
        ("R3 only", [
            {'steps': TOTAL_STEPS, 'sub_w': 0.0, 'r3_w': 5.0, 'ent_w': 1.0, 'label': 'R3 only'}
        ]),
        ("Sub→R3", [
            {'steps': PHASE1_STEPS, 'sub_w': 5.0, 'r3_w': 0.0, 'ent_w': 1.0, 'label': 'Sub→R3 P1'},
            {'steps': PHASE2_STEPS, 'sub_w': 0.0, 'r3_w': 5.0, 'ent_w': 10.0, 'label': 'Sub→R3 P2'},
        ]),
    ]:
        print(f"\n{'─' * 70}")
        print(f"  Variant: {variant_name}")
        print(f"{'─' * 70}")

        model = TriadicGPT(config).to(device)
        dataset = TextDataset(all_tokens, BLOCK_SIZE)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
        amp_dtype = torch.bfloat16
        use_scaler = False  # bfloat16 doesn't need loss scaling
        scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

        full_history = {'step': [], 'loss': [], 'tri': [], 'r3': [], 'sub': []}
        global_step = 0
        t0 = time.time()

        for phase in phases:
            data_iter = iter(dataloader)
            triadic_warmup = int(TOTAL_STEPS * TRIADIC_WARMUP_PCT)

            for local_step in range(phase['steps']):
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    x, y = next(data_iter)
                x, y = x.to(device), y.to(device)

                ws = min(500, TOTAL_STEPS // 10)
                if global_step < ws:
                    lr_t = LR * (global_step + 1) / ws
                else:
                    prog = (global_step - ws) / max(TOTAL_STEPS - ws, 1)
                    lr_t = LR * max(0.1, 0.5 * (1.0 + math.cos(math.pi * prog)))
                for pg in optimizer.param_groups:
                    pg['lr'] = lr_t

                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=(device.type == 'cuda')):
                    logits, triadic_proj, lang_loss = model(x, targets=y)
                    total_loss = lang_loss
                    tri_v, r3_v, sub_v = 0.0, 0.0, 0.0

                    if global_step >= triadic_warmup:
                        aw = int(TOTAL_STEPS * 0.2)
                        af = min(1.0, (global_step - triadic_warmup + 1) / aw)
                        ca = ALPHA * af

                        tri_loss = model.triadic_loss(triadic_proj, entropy_weight=phase['ent_w'],
                                                      input_ids=x, align_weight=ALIGN_WEIGHT, align_mode='mse')
                        total_loss = lang_loss + ca * tri_loss
                        tri_v = tri_loss.item()

                        if global_step % 5 == 0:
                            if phase['r3_w'] > 0:
                                r3_loss = compute_r3_loss(model, analogy_tensors, device)
                                total_loss = total_loss + ca * phase['r3_w'] * r3_loss
                                r3_v = r3_loss.item()
                            if phase['sub_w'] > 0:
                                s_loss = compute_sub_loss(model, sub_train, device)
                                total_loss = total_loss + ca * phase['sub_w'] * s_loss
                                sub_v = s_loss.item()

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                if global_step % 200 == 0 or global_step == TOTAL_STEPS - 1:
                    full_history['step'].append(global_step)
                    full_history['loss'].append(lang_loss.item())
                    full_history['tri'].append(tri_v)
                    full_history['r3'].append(r3_v)
                    full_history['sub'].append(sub_v)

                    elapsed = time.time() - t0
                    speed = (global_step + 1) / max(elapsed, 1)
                    eta_s = (TOTAL_STEPS - global_step - 1) / max(speed, 0.01)
                    bar = progress_bar(global_step + 1, TOTAL_STEPS)
                    print(f"  [{phase['label']:>12s}] {bar}  step {global_step:>5d}/{TOTAL_STEPS}  "
                          f"loss={lang_loss.item():.3f}  r3={r3_v:.4f}  sub={sub_v:.4f}  "
                          f"ETA {format_time(eta_s)}  [{format_time(elapsed)}]")

                global_step += 1

        ev = evaluate_all(model, tokenizer, device)
        results[variant_name] = ev
        histories[variant_name] = full_history

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  CURRICULUM RESULTS")
    print("=" * 70)

    print(f"\n  {'Variant':>10s}  {'Loss':>6s}  {'Gap':>8s}  {'Dead':>4s}  "
          f"{'Ana(tr)':>8s}  {'Ana(te)':>8s}  {'Sub(tr)':>8s}  {'Sub(te)':>8s}")
    print(f"  {'─'*10}  {'─'*6}  {'─'*8}  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

    for name in ["Sub only", "R3 only", "Sub→R3"]:
        ev = results[name]
        h = histories[name]
        print(f"  {name:>10s}  {h['loss'][-1]:>6.3f}  {ev['gap']:>+8.4f}  {ev['dead']:>4d}  "
              f"{ev['ana_train']:>8.4f}  {ev['ana_test']:>8.4f}  "
              f"{ev['sub_train']:>7.1%}  {ev['sub_test']:>7.1%}")

    print(f"\n  Held-out Analogies:")
    for name in ["Sub only", "R3 only", "Sub→R3"]:
        ev = results[name]
        if ev['ana_test_detail']:
            vals = ev['ana_test_detail']
            print(f"    {name:>10s}: {' | '.join(f'{v:.3f}' for v in vals)}  mean={np.mean(vals):.4f}")

    # ── Plot ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {'Sub only': 'green', 'R3 only': 'red', 'Sub→R3': 'purple'}

    for name in ["Sub only", "R3 only", "Sub→R3"]:
        h = histories[name]
        c = colors[name]
        axes[0, 0].plot(h['step'], h['loss'], color=c, label=name, alpha=0.8)
        axes[0, 1].plot(h['step'], h['r3'], color=c, label=name, alpha=0.8)
        axes[1, 0].plot(h['step'], h['sub'], color=c, label=name, alpha=0.8)

    # Phase transition marker for curriculum
    axes[0, 0].axvline(x=PHASE1_STEPS, color='gray', linestyle='--', alpha=0.5, label='Phase switch')
    axes[0, 1].axvline(x=PHASE1_STEPS, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].axvline(x=PHASE1_STEPS, color='gray', linestyle='--', alpha=0.5)

    # Bar chart
    names = ["Sub only", "R3 only", "Sub→R3"]
    x = np.arange(3)
    w = 0.2
    axes[1, 1].bar(x - w, [results[n]['sub_test'] for n in names], w, label='Sub test', color='steelblue')
    axes[1, 1].bar(x, [results[n]['ana_test'] for n in names], w, label='Ana test', color='coral')
    axes[1, 1].bar(x + w, [1 - results[n]['dead']/64 for n in names], w, label='Bit health', color='seagreen')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(names, fontsize=9)
    axes[1, 1].set_ylim(0, 1.1)
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_title('Final Metrics Comparison')

    for ax, title in [(axes[0,0], 'Language Loss'), (axes[0,1], 'R3 Loss'), (axes[1,0], 'Sub Loss')]:
        ax.set_title(title)
        ax.set_xlabel('Step')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Curriculum: Sub → R3 (Sequential Training)', fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'curriculum_sub_r3.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved: {plot_path}")

    # Save
    save_data = {
        'experiment': 'curriculum_sub_r3',
        'config': f'{N_LAYER}L/{N_EMBD}D/{N_BITS}bits',
        'phase1_steps': PHASE1_STEPS,
        'phase2_steps': PHASE2_STEPS,
        'variants': {name: {'final_loss': histories[name]['loss'][-1], 'eval': results[name]}
                     for name in names},
    }
    results_path = os.path.join(RESULTS_DIR, 'curriculum_sub_r3.json')
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"  Results saved: {results_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
