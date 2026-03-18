"""
P2 — Subsumption Loss (Linea 4: Recuperacion de Subsumption)

Current problem: Subsumption = 0% at k=64.
The book proposes compound concepts should CONTAIN the primes of their hypernyms.

This experiment adds a loss that forces hyponyms to inherit hypernym bits:
  If "animal" has bit_i active, then "dog" must also have bit_i active.
  Loss = mean(relu(proj_hyper - proj_hypo)) over all bits.

This is a differentiable proxy for the algebraic condition: Phi(hypo) % Phi(hyper) == 0.

Trains with: language + triadic + subsumption loss.
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

# WordNet-style hypernym pairs: (hypernym, [hyponyms])
# The hypernym's active bits should be a SUBSET of each hyponym's active bits.
HYPERNYM_PAIRS = {
    "animal": ["dog", "cat", "bird", "fish", "horse", "rabbit", "bear", "mouse", "lion"],
    "person": ["king", "queen", "doctor", "teacher", "princess", "prince", "boy", "girl"],
    "feeling": ["happy", "sad", "love", "hate", "angry", "scared"],
    "food": ["apple", "cake", "bread", "candy", "cookie"],
    "color": ["red", "blue", "green", "yellow", "pink", "purple"],
    "place": ["school", "hospital", "house", "garden", "forest", "beach", "park"],
    "time": ["day", "night", "morning", "evening"],
}

# Held-out pairs for testing generalization (NOT used during training)
HELD_OUT_PAIRS = {
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


def prepare_subsumption_data(tokenizer, device, pairs_dict):
    """Pre-encode hypernym-hyponym pairs as token tensors."""
    sub_pairs = []
    skipped = []
    for hypernym, hyponyms in pairs_dict.items():
        hyper_ids = tokenizer.encode(hypernym, add_special=False)
        if not hyper_ids:
            skipped.append(hypernym)
            continue
        hyper_tensor = torch.tensor(hyper_ids, dtype=torch.long, device=device)

        for hyponym in hyponyms:
            hypo_ids = tokenizer.encode(hyponym, add_special=False)
            if not hypo_ids:
                skipped.append(hyponym)
                continue
            hypo_tensor = torch.tensor(hypo_ids, dtype=torch.long, device=device)
            sub_pairs.append({
                'hypernym': hypernym,
                'hyponym': hyponym,
                'hyper_ids': hyper_tensor,
                'hypo_ids': hypo_tensor,
            })

    if skipped:
        print(f"  Skipped (not in vocab): {skipped}")
    print(f"  Prepared {len(sub_pairs)} hypernym-hyponym pairs")
    return sub_pairs


def compute_subsumption_loss(model, sub_pairs, device):
    """
    Subsumption loss: hypernym bits should be a subset of hyponym bits.

    For each (hypernym H, hyponym Y):
      loss += mean(relu(proj_H - proj_Y))

    This penalizes when a hypernym bit is active but the corresponding
    hyponym bit is not. It's a differentiable proxy for Phi(Y) % Phi(H) == 0.
    """
    if not sub_pairs:
        return torch.tensor(0.0, device=device)

    losses = []
    for pair in sub_pairs:
        hyper_x = pair['hyper_ids'].unsqueeze(0)
        hypo_x = pair['hypo_ids'].unsqueeze(0)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
            _, proj_hyper, _ = model(hyper_x)
            _, proj_hypo, _ = model(hypo_x)

        # Mean-pool over sequence length → (n_bits,)
        h = proj_hyper[0].mean(dim=0)
        y = proj_hypo[0].mean(dim=0)

        # Penalize when hypernym projection exceeds hyponym projection
        # relu(H - Y): only active when H > Y (hypernym bit active but hyponym bit isn't)
        loss = F.relu(h - y).mean()
        losses.append(loss)

    return torch.stack(losses).mean()


def evaluate_subsumption(model, tokenizer, device, pairs_dict, mapper, label=""):
    """Measure actual subsumption recall on hypernym-hyponym pairs."""
    model.eval()

    # Get projections for all words
    all_words = set()
    for hyper, hypos in pairs_dict.items():
        all_words.add(hyper)
        all_words.update(hypos)

    sigs = {}
    projs = {}
    with torch.no_grad():
        for word in all_words:
            ids = tokenizer.encode(word, add_special=False)
            if not ids:
                continue
            x = torch.tensor([ids], dtype=torch.long, device=device)
            _, proj, _ = model(x)
            proj_np = proj[0].mean(dim=0).cpu().numpy()
            projs[word] = proj_np
            sigs[word] = mapper.map(proj_np)

    # Measure subsumption
    results = []
    total_pairs = 0
    subsumes_count = 0
    bit_inheritance_scores = []

    for hypernym, hyponyms in pairs_dict.items():
        if hypernym not in sigs:
            continue
        hyper_bits = (projs[hypernym] > 0).astype(int)

        for hyponym in hyponyms:
            if hyponym not in sigs:
                continue
            hypo_bits = (projs[hyponym] > 0).astype(int)
            total_pairs += 1

            # Algebraic subsumption: Phi(hypo) % Phi(hyper) == 0
            is_subsumes = TriadicValidator.subsumes(sigs[hyponym], sigs[hypernym])
            if is_subsumes:
                subsumes_count += 1

            # Bit inheritance: what fraction of hypernym's active bits are also active in hyponym
            hyper_active = hyper_bits.sum()
            if hyper_active > 0:
                inherited = (hyper_bits * hypo_bits).sum()
                inheritance_rate = inherited / hyper_active
            else:
                inheritance_rate = 1.0
            bit_inheritance_scores.append(float(inheritance_rate))

            results.append({
                'pair': f'{hypernym}->{hyponym}',
                'subsumes': bool(is_subsumes),
                'bit_inheritance': float(inheritance_rate),
                'hyper_active_bits': int(hyper_active),
                'shared_bits': int((hyper_bits * hypo_bits).sum()),
            })

    subsumption_rate = subsumes_count / max(total_pairs, 1)
    mean_inheritance = np.mean(bit_inheritance_scores) if bit_inheritance_scores else 0.0

    if label:
        print(f"\n  [{label}] Subsumption Results:")
        print(f"    Algebraic subsumption: {subsumes_count}/{total_pairs} ({subsumption_rate:.1%})")
        print(f"    Mean bit inheritance:  {mean_inheritance:.1%}")

        # Per-hypernym breakdown
        for hypernym in pairs_dict:
            pair_results = [r for r in results if r['pair'].startswith(f'{hypernym}->')]
            if pair_results:
                h_inherit = np.mean([r['bit_inheritance'] for r in pair_results])
                h_sub = sum(r['subsumes'] for r in pair_results)
                print(f"      {hypernym:>10s}: inheritance={h_inherit:.0%}  "
                      f"subsumption={h_sub}/{len(pair_results)}")

    return {
        'subsumption_rate': float(subsumption_rate),
        'mean_bit_inheritance': float(mean_inheritance),
        'total_pairs': total_pairs,
        'details': results,
    }


def train_model(model, tokenizer, all_tokens, device, label,
                sub_pairs=None, sub_weight=0.0):
    """Train with optional subsumption loss."""
    dataset = TextDataset(all_tokens, BLOCK_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
    amp_dtype = torch.bfloat16
    use_scaler = False  # bfloat16 doesn't need loss scaling
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
    triadic_warmup = int(STEPS * TRIADIC_WARMUP_PCT)

    model.train()
    data_iter = iter(dataloader)
    history = {'step': [], 'loss': [], 'tri_loss': [], 'sub_loss': [], 'entropy': []}

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
            sub_loss_val = 0.0

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

                # Subsumption loss (every 5 steps to save compute)
                if sub_weight > 0 and sub_pairs and step % 5 == 0:
                    sub_loss = compute_subsumption_loss(model, sub_pairs, device)
                    total_loss = total_loss + current_alpha * sub_weight * sub_loss
                    sub_loss_val = sub_loss.item()

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
            history['sub_loss'].append(sub_loss_val)
            history['entropy'].append(ent.mean().item())

            elapsed = time.time() - t0
            speed = (step + 1) / max(elapsed, 1)
            eta_s = (STEPS - step - 1) / max(speed, 0.01)
            bar = progress_bar(step + 1, STEPS)
            print(f"  [{label:>9s}] {bar}  step {step:>5d}/{STEPS}  "
                  f"loss={lang_loss.item():.3f}  tri={tri_loss_val:.4f}  sub={sub_loss_val:.4f}  "
                  f"ETA {format_time(eta_s)}  [{format_time(elapsed)}]")

    return history


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("  SUBSUMPTION LOSS EXPERIMENT")
    print("  (Linea 4: Bit Inheritance for Hypernym-Hyponym Pairs)")
    print("=" * 70)
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU:    {torch.cuda.get_device_name(0)}")

    tok_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'torch_run15_strongalign', 'tokenizer.json')
    tokenizer = BPETokenizer.load(tok_path)
    vocab_size = tokenizer.vocab_size

    print("\nLoading data...")
    all_tokens = load_data(tokenizer)

    print("\nPreparing subsumption pairs (training)...")
    train_pairs = prepare_subsumption_data(tokenizer, device, HYPERNYM_PAIRS)
    print("\nPreparing held-out pairs (test)...")
    test_pairs_data = prepare_subsumption_data(tokenizer, device, HELD_OUT_PAIRS)

    config = TriadicGPTConfig(
        vocab_size=vocab_size, block_size=BLOCK_SIZE,
        n_layer=N_LAYER, n_embd=N_EMBD, n_head=N_HEAD,
        n_triadic_bits=N_BITS, dropout=0.1,
    )
    mapper = PrimeMapper(N_BITS)

    # ── Variant 1: Baseline (no subsumption loss) ──
    print(f"\n{'─' * 70}")
    print("  Training: BASELINE (no subsumption loss)")
    print(f"{'─' * 70}")
    model_base = TriadicGPT(config).to(device)
    hist_base = train_model(model_base, tokenizer, all_tokens, device, "BASE")
    eval_base_train = evaluate_subsumption(model_base, tokenizer, device, HYPERNYM_PAIRS, mapper, "BASE-train")
    eval_base_test = evaluate_subsumption(model_base, tokenizer, device, HELD_OUT_PAIRS, mapper, "BASE-test")

    # ── Variant 2: Subsumption loss (weight=1.0) ──
    print(f"\n{'─' * 70}")
    print("  Training: SUBSUMPTION (weight=1.0)")
    print(f"{'─' * 70}")
    model_sub1 = TriadicGPT(config).to(device)
    hist_sub1 = train_model(model_sub1, tokenizer, all_tokens, device, "SUB-1.0",
                            sub_pairs=train_pairs, sub_weight=1.0)
    eval_sub1_train = evaluate_subsumption(model_sub1, tokenizer, device, HYPERNYM_PAIRS, mapper, "SUB1-train")
    eval_sub1_test = evaluate_subsumption(model_sub1, tokenizer, device, HELD_OUT_PAIRS, mapper, "SUB1-test")

    # ── Variant 3: Strong subsumption loss (weight=5.0) ──
    print(f"\n{'─' * 70}")
    print("  Training: STRONG SUBSUMPTION (weight=5.0)")
    print(f"{'─' * 70}")
    model_sub5 = TriadicGPT(config).to(device)
    hist_sub5 = train_model(model_sub5, tokenizer, all_tokens, device, "SUB-5.0",
                            sub_pairs=train_pairs, sub_weight=5.0)
    eval_sub5_train = evaluate_subsumption(model_sub5, tokenizer, device, HYPERNYM_PAIRS, mapper, "SUB5-train")
    eval_sub5_test = evaluate_subsumption(model_sub5, tokenizer, device, HELD_OUT_PAIRS, mapper, "SUB5-test")

    # ── Comparison ──
    print("\n" + "=" * 70)
    print("  SUBSUMPTION RESULTS")
    print("=" * 70)
    print(f"\n  {'Variant':>15s}  {'Lang Loss':>10s}  {'Sub(train)':>11s}  {'Inherit(train)':>15s}  "
          f"{'Sub(test)':>10s}  {'Inherit(test)':>14s}")
    print(f"  {'─'*15}  {'─'*10}  {'─'*11}  {'─'*15}  {'─'*10}  {'─'*14}")

    for name, hist, ev_tr, ev_te in [
        ("Baseline",  hist_base, eval_base_train, eval_base_test),
        ("Sub (1.0)",  hist_sub1, eval_sub1_train, eval_sub1_test),
        ("Sub (5.0)",  hist_sub5, eval_sub5_train, eval_sub5_test),
    ]:
        print(f"  {name:>15s}  {hist['loss'][-1]:>10.3f}  "
              f"{ev_tr['subsumption_rate']:>10.1%}  {ev_tr['mean_bit_inheritance']:>14.1%}  "
              f"{ev_te['subsumption_rate']:>10.1%}  {ev_te['mean_bit_inheritance']:>13.1%}")

    # ── Plot ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for label, hist, color in [('Baseline', hist_base, 'blue'),
                                ('Sub (1.0)', hist_sub1, 'orange'),
                                ('Sub (5.0)', hist_sub5, 'red')]:
        axes[0, 0].plot(hist['step'], hist['loss'], color=color, label=label, alpha=0.8)
        axes[0, 1].plot(hist['step'], hist['tri_loss'], color=color, label=label, alpha=0.8)
        axes[1, 0].plot(hist['step'], hist['sub_loss'], color=color, label=label, alpha=0.8)
        axes[1, 1].plot(hist['step'], hist['entropy'], color=color, label=label, alpha=0.8)

    for ax, title, ylabel in [
        (axes[0, 0], 'Language Loss', 'Loss'),
        (axes[0, 1], 'Triadic Loss', 'Loss'),
        (axes[1, 0], 'Subsumption Loss', 'Loss'),
        (axes[1, 1], 'Bit Entropy', 'Mean Entropy'),
    ]:
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Step')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Subsumption Loss: Bit Inheritance for Hypernym-Hyponym Pairs', fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'subsumption_loss.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved: {plot_path}")

    # ── Save ──
    save_data = {
        'experiment': 'subsumption_loss',
        'source': 'Linea 4 — Recuperacion de Subsumption',
        'config': f'{N_LAYER}L/{N_EMBD}D/{N_BITS}bits',
        'steps': STEPS,
        'n_train_pairs': len(train_pairs),
        'n_test_pairs': len(test_pairs_data),
        'baseline': {
            'final_loss': hist_base['loss'][-1],
            'train_eval': eval_base_train,
            'test_eval': eval_base_test,
        },
        'sub_1.0': {
            'final_loss': hist_sub1['loss'][-1],
            'train_eval': eval_sub1_train,
            'test_eval': eval_sub1_test,
        },
        'sub_5.0': {
            'final_loss': hist_sub5['loss'][-1],
            'train_eval': eval_sub5_train,
            'test_eval': eval_sub5_test,
        },
    }

    results_path = os.path.join(RESULTS_DIR, 'subsumption_loss.json')
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"  Results saved: {results_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
