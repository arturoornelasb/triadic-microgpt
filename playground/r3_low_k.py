"""
E7 — R3 Loss at Low k (k=6, 8, 12)

R3 (Rule-of-Three) loss was proven DEAD at k=64 across three separate experiments
(P7 combo, P10 entropy guard, P11 curriculum) — it collapses to 64/64 dead bits
every time regardless of entropy regularization.

Hypothesis: R3 may work at the Engine's original regime (k=6-12) where the
combinatorial space is small enough for offset parallelism to emerge without
the curse of dimensionality that kills it at k=64.

Design:
  - BASE scale (6L/256D/8H, ~5.8M params) for speed
  - k in {6, 8, 12}
  - 3 variants per k: baseline, R3 weight=1.0, R3 weight=5.0
  - 10K steps each (~5 min)
  - R3 loss = 1 - cosine_similarity(offset_ab, offset_cd)

Key metrics:
  - Dead bits / entropy (the failure mode at k=64)
  - R3 train accuracy (offset cosine > 0.9 on training triples)
  - R3 test accuracy (held-out triples)
  - Semantic gap
  - Language loss
"""

import os, sys, csv, json, math, time, random, argparse, numpy as np, torch
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
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'playground', 'results', 'r3_low_k_v2')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# Constants
# ============================================================

STEPS = 10000
BATCH_SIZE = 32
BLOCK_SIZE = 256
LR = 3e-4
ALPHA = 0.05
ENTROPY_WEIGHT = 1.0
ALIGN_WEIGHT = 3.0
ALIGN_MODE = 'mse'
TRIADIC_WARMUP_PCT = 0.25
N_LAYER = 6
N_EMBD = 256
N_HEAD = 8
MAX_STORIES = 5000

K_VALUES = [6, 8, 12]

# 16 training analogy triples
TRAIN_TRIPLES = [
    ("king", "queen", "man", "woman"),
    ("father", "mother", "brother", "sister"),
    ("father", "mother", "son", "daughter"),
    ("dog", "puppy", "cat", "kitten"),
    ("big", "small", "tall", "short"),
    ("hot", "cold", "day", "night"),
    ("happy", "sad", "love", "hate"),
    ("princess", "prince", "queen", "king"),
    ("bird", "fly", "fish", "swim"),
    ("old", "young", "big", "small"),
    ("doctor", "hospital", "teacher", "school"),
    ("sun", "day", "moon", "night"),
    ("red", "blue", "green", "yellow"),
    ("husband", "wife", "boy", "girl"),
    ("teacher", "student", "doctor", "patient"),
    ("morning", "night", "summer", "winter"),
]

# 4 held-out test triples (no word overlap with TRAIN_TRIPLES)
TEST_TRIPLES = [
    ("uncle", "aunt", "grandpa", "grandma"),
    ("black", "white", "dark", "light"),
    ("up", "down", "left", "right"),
    ("cake", "sweet", "lemon", "sour"),
]


# ============================================================
# Helpers
# ============================================================

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def progress_bar(current, total, width=25):
    pct = current / max(total, 1)
    filled = int(width * pct)
    return f"[{'#' * filled}{'-' * (width - filled)}] {pct:5.1%}"


# ============================================================
# Dataset
# ============================================================

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


def load_data(tokenizer, max_stories=MAX_STORIES):
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


# ============================================================
# Analogy data preparation
# ============================================================

def prepare_analogy_tensors(triples, tokenizer, device):
    """Pre-encode analogy triples as token IDs."""
    tensors = []
    skipped = []
    for a, b, c, d in triples:
        ids = {}
        valid = True
        for label, word in [('a', a), ('b', b), ('c', c), ('d', d)]:
            encoded = tokenizer.encode(word, add_special=False)
            if not encoded:
                valid = False
                skipped.append(word)
                break
            ids[label] = torch.tensor(encoded, dtype=torch.long, device=device)
        if valid:
            tensors.append((ids, f"{a}:{b}::{c}:{d}"))
    if skipped:
        print(f"  WARNING: skipped words not in vocabulary: {set(skipped)}")
    return tensors


def get_word_projection(model, word_ids, device):
    """Get the mean triadic projection for a word (token IDs tensor)."""
    x = word_ids.unsqueeze(0)  # (1, seq_len)
    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
        _, triadic_proj, _ = model(x)
    return triadic_proj[0].mean(dim=0)  # (n_bits,)


# ============================================================
# R3 Loss: offset cosine similarity
# ============================================================

def compute_r3_loss(model, analogy_tensors, device, n_sample=8):
    """
    R3 loss for a batch of sampled analogy triples.

    For triple (a, b, c, d):
      offset_ab = P(b) - P(a)
      offset_cd = P(d) - P(c)
      L = 1 - cosine_similarity(offset_ab, offset_cd)

    Samples min(n_sample, len(triples)) each call for efficiency.
    """
    if not analogy_tensors:
        return torch.tensor(0.0, device=device)

    # Sample a subset each call
    if len(analogy_tensors) > n_sample:
        batch = random.sample(analogy_tensors, n_sample)
    else:
        batch = analogy_tensors

    losses = []
    for ids, _label in batch:
        pa = get_word_projection(model, ids['a'], device)
        pb = get_word_projection(model, ids['b'], device)
        pc = get_word_projection(model, ids['c'], device)
        pd = get_word_projection(model, ids['d'], device)

        offset_ab = pb - pa
        offset_cd = pd - pc

        cos = F.cosine_similarity(offset_ab.unsqueeze(0), offset_cd.unsqueeze(0))
        losses.append(1.0 - cos.squeeze())

    return torch.stack(losses).mean()


# ============================================================
# Training loop
# ============================================================

def train_variant(model, tokenizer, all_tokens, device, label, n_bits,
                  analogy_tensors=None, r3_weight=0.0):
    """Train one variant, return history dict."""
    dataset = TextDataset(all_tokens, BLOCK_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            drop_last=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01,
                                  betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    triadic_warmup = int(STEPS * TRIADIC_WARMUP_PCT)

    model.train()
    data_iter = iter(dataloader)
    history = {'step': [], 'loss': [], 'tri': [], 'r3': [], 'entropy': []}

    t0 = time.time()
    for step in range(STEPS):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        # Cosine LR with warmup
        warmup_steps = min(500, STEPS // 10)
        if step < warmup_steps:
            lr_t = LR * (step + 1) / warmup_steps
        else:
            prog = (step - warmup_steps) / max(STEPS - warmup_steps, 1)
            lr_t = LR * max(0.1, 0.5 * (1.0 + math.cos(math.pi * prog)))
        for pg in optimizer.param_groups:
            pg['lr'] = lr_t

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits, triadic_proj, lang_loss = model(x, targets=y)
            total_loss = lang_loss
            tri_v, r3_v = 0.0, 0.0

            if step >= triadic_warmup:
                alpha_warmup = int(STEPS * 0.2)
                alpha_factor = min(1.0, (step - triadic_warmup + 1) / alpha_warmup)
                current_alpha = ALPHA * alpha_factor

                tri_loss = model.triadic_loss(
                    triadic_proj, entropy_weight=ENTROPY_WEIGHT,
                    input_ids=x, align_weight=ALIGN_WEIGHT,
                    align_mode=ALIGN_MODE,
                )
                total_loss = lang_loss + current_alpha * tri_loss
                tri_v = tri_loss.item()

                # R3 loss every 5 steps (sample from training triples)
                if r3_weight > 0 and analogy_tensors and step % 5 == 0:
                    r3_loss = compute_r3_loss(model, analogy_tensors, device)
                    total_loss = total_loss + current_alpha * r3_weight * r3_loss
                    r3_v = r3_loss.item()

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Logging
        if step % 200 == 0 or step == STEPS - 1:
            with torch.no_grad():
                flat = triadic_proj.reshape(-1, n_bits)
                bm = (flat > 0).float().mean(dim=0)
                eps = 1e-7
                ent = -(bm * (bm + eps).log2() + (1 - bm) * (1 - bm + eps).log2())

            history['step'].append(step)
            history['loss'].append(lang_loss.item())
            history['tri'].append(tri_v)
            history['r3'].append(r3_v)
            history['entropy'].append(ent.mean().item())

            elapsed = time.time() - t0
            speed = (step + 1) / max(elapsed, 1)
            eta_s = (STEPS - step - 1) / max(speed, 0.01)
            bar = progress_bar(step + 1, STEPS)
            print(f"  [{label:>14s}] {bar}  step {step:>5d}/{STEPS}  "
                  f"loss={lang_loss.item():.3f}  tri={tri_v:.4f}  r3={r3_v:.4f}  "
                  f"ent={ent.mean().item():.3f}  "
                  f"ETA {format_time(eta_s)}  [{format_time(elapsed)}]")

    elapsed = time.time() - t0
    print(f"  [{label:>14s}] Done in {format_time(elapsed)}")
    return history


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(model, tokenizer, device, n_bits, train_tensors, test_tensors):
    """Evaluate a trained model on all metrics."""
    model.eval()
    mapper = PrimeMapper(n_bits)

    # Collect all words from train + test triples, plus extra for semantic gap
    all_words = set()
    for triples in [TRAIN_TRIPLES, TEST_TRIPLES]:
        for a, b, c, d in triples:
            all_words.update([a, b, c, d])
    extra_words = ["tree", "river", "mountain", "stone", "cloud", "table",
                   "car", "house", "water", "fire", "child", "baby"]
    all_words.update(extra_words)

    # Get projections for all words
    projs = {}
    with torch.no_grad():
        for word in all_words:
            ids = tokenizer.encode(word, add_special=False)
            if ids:
                x = torch.tensor([ids], dtype=torch.long, device=device)
                _, proj, _ = model(x)
                projs[word] = proj[0].mean(dim=0).cpu().numpy()

    def cosine(a, b):
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom < 1e-10:
            return 0.0
        return float(np.dot(a, b) / denom)

    # --- R3 accuracy: offset cosine > 0.9 ---
    def eval_r3(triples):
        """Return list of offset cosines and accuracy (fraction > 0.9)."""
        cosines = []
        for a, b, c, d in triples:
            if not all(w in projs for w in [a, b, c, d]):
                continue
            offset_ab = projs[b] - projs[a]
            offset_cd = projs[d] - projs[c]
            cos = cosine(offset_ab, offset_cd)
            cosines.append(cos)
        acc = sum(1 for c in cosines if c > 0.9) / max(len(cosines), 1)
        return cosines, acc

    r3_train_cos, r3_train_acc = eval_r3(TRAIN_TRIPLES)
    r3_test_cos, r3_test_acc = eval_r3(TEST_TRIPLES)

    # --- Semantic gap ---
    related_pairs = [
        ("king", "queen"), ("dog", "cat"), ("happy", "sad"),
        ("mother", "father"), ("sun", "moon"), ("hot", "cold"),
        ("love", "hate"), ("big", "small"),
    ]
    rel_sims = [cosine(projs[a], projs[b])
                for a, b in related_pairs if a in projs and b in projs]
    rand_sims = []
    wlist = list(projs.keys())
    rng = random.Random(42)
    for _ in range(200):
        i, j = rng.sample(range(len(wlist)), 2)
        rand_sims.append(cosine(projs[wlist[i]], projs[wlist[j]]))
    gap = float(np.mean(rel_sims) - np.mean(rand_sims)) if rel_sims else 0.0

    # --- Bit entropy and dead bits ---
    if projs:
        all_p = np.stack(list(projs.values()))
        bm = (all_p > 0).astype(float).mean(axis=0)
        eps = 1e-7
        bent = -(bm * np.log2(bm + eps) + (1 - bm) * np.log2(1 - bm + eps))
        dead = int((bent < 0.3).sum())
        mean_entropy = float(bent.mean())
    else:
        dead = n_bits
        mean_entropy = 0.0

    return {
        'dead_bits': dead,
        'entropy': mean_entropy,
        'r3_train_acc': r3_train_acc,
        'r3_train_cos': r3_train_cos,
        'r3_train_cos_mean': float(np.mean(r3_train_cos)) if r3_train_cos else 0.0,
        'r3_test_acc': r3_test_acc,
        'r3_test_cos': r3_test_cos,
        'r3_test_cos_mean': float(np.mean(r3_test_cos)) if r3_test_cos else 0.0,
        'semantic_gap': gap,
    }


# ============================================================
# Single k run
# ============================================================

def run_single_k(k, tokenizer, all_tokens, device, train_tensors, test_tensors):
    """Train 3 variants for a single k value, return results dict."""
    print(f"\n{'=' * 70}")
    print(f"  k = {k} bits  |  BASE scale ({N_LAYER}L/{N_EMBD}D/{N_HEAD}H)")
    print(f"{'=' * 70}")

    config = TriadicGPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_triadic_bits=k,
        dropout=0.1,
    )

    variants = [
        ("baseline", 0.0),
        ("r3_w1", 1.0),
        ("r3_w5", 5.0),
    ]

    results = {}

    for var_name, r3_weight in variants:
        label = f"k={k} {var_name}"
        nice_name = {
            'baseline': 'Baseline (no R3)',
            'r3_w1': 'R3 weight=1.0',
            'r3_w5': 'R3 weight=5.0',
        }[var_name]

        print(f"\n{'~' * 70}")
        print(f"  {nice_name}  (k={k})")
        print(f"{'~' * 70}")

        model = TriadicGPT(config).to(device)
        params = model.num_params()
        print(f"  Parameters: {params:,}")

        hist = train_variant(
            model, tokenizer, all_tokens, device, label, n_bits=k,
            analogy_tensors=train_tensors if r3_weight > 0 else None,
            r3_weight=r3_weight,
        )

        ev = evaluate_model(model, tokenizer, device, k, train_tensors, test_tensors)
        ev['final_loss'] = hist['loss'][-1]
        ev['final_entropy_hist'] = hist['entropy'][-1]
        ev['params'] = params
        ev['history'] = hist

        results[var_name] = ev

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    return results


# ============================================================
# Aggregate and report
# ============================================================

def print_comparison_table(all_k_results):
    """Print the final comparison table across all k values."""
    print("\n" + "=" * 100)
    print("  E7: R3 LOSS AT LOW k — COMPARISON TABLE")
    print("=" * 100)

    header = (f"  {'k':>3s}  {'Variant':>14s}  {'Dead':>4s}/{'>4s'}  {'Entropy':>7s}  "
              f"{'R3 Tr':>6s}  {'R3 Te':>6s}  {'Sem Gap':>8s}  {'Lang Loss':>9s}")
    print(header)
    sep = f"  {'---':>3s}  {'-' * 14}  {'-' * 9}  {'-' * 7}  {'-' * 6}  {'-' * 6}  {'-' * 8}  {'-' * 9}"
    print(sep)

    variant_labels = {
        'baseline': 'Baseline',
        'r3_w1': 'R3 w=1.0',
        'r3_w5': 'R3 w=5.0',
    }

    for k in sorted(all_k_results.keys()):
        for var_name in ['baseline', 'r3_w1', 'r3_w5']:
            ev = all_k_results[k][var_name]
            print(f"  {k:>3d}  {variant_labels[var_name]:>14s}  "
                  f"{ev['dead_bits']:>4d}/{k:<4d}  "
                  f"{ev['entropy']:>7.3f}  "
                  f"{ev['r3_train_acc']:>5.0%}  "
                  f"{ev['r3_test_acc']:>5.0%}  "
                  f"{ev['semantic_gap']:>+8.4f}  "
                  f"{ev['final_loss']:>9.4f}")
        print(sep)

    # Summary insight
    print("\n  Key question: Does R3 avoid dead-bit collapse at low k?")
    for k in sorted(all_k_results.keys()):
        base_dead = all_k_results[k]['baseline']['dead_bits']
        r3w5_dead = all_k_results[k]['r3_w5']['dead_bits']
        r3w5_test = all_k_results[k]['r3_w5']['r3_test_acc']
        status = "ALIVE" if r3w5_dead < k * 0.5 else "DEAD"
        print(f"    k={k:>2d}: baseline dead={base_dead}/{k}, "
              f"R3(5.0) dead={r3w5_dead}/{k} [{status}], "
              f"test acc={r3w5_test:.0%}")


def plot_results(all_k_results):
    """Generate comparison plots."""
    k_values = sorted(all_k_results.keys())
    n_k = len(k_values)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors = {'baseline': 'steelblue', 'r3_w1': 'orange', 'r3_w5': 'crimson'}
    labels = {'baseline': 'Baseline', 'r3_w1': 'R3 w=1.0', 'r3_w5': 'R3 w=5.0'}

    # Row 1: Training curves per k
    for col, k in enumerate(k_values):
        ax = axes[0, col]
        for var_name in ['baseline', 'r3_w1', 'r3_w5']:
            hist = all_k_results[k][var_name]['history']
            ax.plot(hist['step'], hist['loss'], color=colors[var_name],
                    label=labels[var_name], alpha=0.8, linewidth=1.2)
        ax.set_title(f'Language Loss (k={k})')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Row 2, Col 0: Dead bits bar chart
    ax = axes[1, 0]
    x = np.arange(n_k)
    width = 0.25
    for i, var_name in enumerate(['baseline', 'r3_w1', 'r3_w5']):
        dead = [all_k_results[k][var_name]['dead_bits'] for k in k_values]
        total = k_values
        frac = [d / t for d, t in zip(dead, total)]
        ax.bar(x + i * width, frac, width, label=labels[var_name],
               color=colors[var_name], alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'k={k}' for k in k_values])
    ax.set_ylabel('Dead Bits (fraction)')
    ax.set_title('Dead Bits by k and Variant')
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    # Row 2, Col 1: R3 test accuracy bar chart
    ax = axes[1, 1]
    for i, var_name in enumerate(['baseline', 'r3_w1', 'r3_w5']):
        acc = [all_k_results[k][var_name]['r3_test_acc'] for k in k_values]
        ax.bar(x + i * width, acc, width, label=labels[var_name],
               color=colors[var_name], alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'k={k}' for k in k_values])
    ax.set_ylabel('R3 Test Accuracy')
    ax.set_title('Held-out Analogy Accuracy (offset cos > 0.9)')
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    # Row 2, Col 2: Semantic gap bar chart
    ax = axes[1, 2]
    for i, var_name in enumerate(['baseline', 'r3_w1', 'r3_w5']):
        gaps = [all_k_results[k][var_name]['semantic_gap'] for k in k_values]
        ax.bar(x + i * width, gaps, width, label=labels[var_name],
               color=colors[var_name], alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'k={k}' for k in k_values])
    ax.set_ylabel('Semantic Gap')
    ax.set_title('Semantic Gap by k and Variant')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('E7: R3 Loss at Low k  (6L/256D, 10K steps, BASE scale)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    plot_path = os.path.join(RESULTS_DIR, 'r3_low_k_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved: {plot_path}")
    return plot_path


def save_results(all_k_results):
    """Save results JSON (strip non-serializable history arrays)."""
    save_data = {
        'experiment': 'E7_r3_low_k',
        'description': 'R3 loss at k=6,8,12 — does it avoid the dead-bit collapse seen at k=64?',
        'config': {
            'scale': 'base',
            'arch': f'{N_LAYER}L/{N_EMBD}D/{N_HEAD}H',
            'steps': STEPS,
            'batch_size': BATCH_SIZE,
            'alpha': ALPHA,
            'entropy_weight': ENTROPY_WEIGHT,
            'align_weight': ALIGN_WEIGHT,
            'align_mode': ALIGN_MODE,
            'triadic_warmup_pct': TRIADIC_WARMUP_PCT,
        },
        'train_triples': [f"{a}:{b}::{c}:{d}" for a, b, c, d in TRAIN_TRIPLES],
        'test_triples': [f"{a}:{b}::{c}:{d}" for a, b, c, d in TEST_TRIPLES],
        'results': {},
    }

    variant_labels = {'baseline': 'Baseline', 'r3_w1': 'R3 w=1.0', 'r3_w5': 'R3 w=5.0'}

    for k in sorted(all_k_results.keys()):
        k_data = {}
        for var_name in ['baseline', 'r3_w1', 'r3_w5']:
            ev = all_k_results[k][var_name]
            k_data[variant_labels[var_name]] = {
                'dead_bits': ev['dead_bits'],
                'total_bits': k,
                'entropy': ev['entropy'],
                'r3_train_acc': ev['r3_train_acc'],
                'r3_train_cos_mean': ev['r3_train_cos_mean'],
                'r3_test_acc': ev['r3_test_acc'],
                'r3_test_cos_mean': ev['r3_test_cos_mean'],
                'r3_test_cos_detail': ev['r3_test_cos'],
                'semantic_gap': ev['semantic_gap'],
                'lang_loss': ev['final_loss'],
                'params': ev['params'],
            }
        save_data['results'][f'k={k}'] = k_data

    results_path = os.path.join(RESULTS_DIR, 'r3_low_k_results.json')
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"  Results saved: {results_path}")
    return results_path


def save_csv_summary(all_k_results):
    """Save a flat CSV for easy comparison."""
    csv_path = os.path.join(RESULTS_DIR, 'r3_low_k_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['k', 'variant', 'dead_bits', 'total_bits', 'entropy',
                         'r3_train_acc', 'r3_test_acc', 'r3_train_cos_mean',
                         'r3_test_cos_mean', 'semantic_gap', 'lang_loss'])
        variant_labels = {'baseline': 'Baseline', 'r3_w1': 'R3 w=1.0', 'r3_w5': 'R3 w=5.0'}
        for k in sorted(all_k_results.keys()):
            for var_name in ['baseline', 'r3_w1', 'r3_w5']:
                ev = all_k_results[k][var_name]
                writer.writerow([
                    k, variant_labels[var_name],
                    ev['dead_bits'], k,
                    f"{ev['entropy']:.4f}",
                    f"{ev['r3_train_acc']:.4f}",
                    f"{ev['r3_test_acc']:.4f}",
                    f"{ev['r3_train_cos_mean']:.4f}",
                    f"{ev['r3_test_cos_mean']:.4f}",
                    f"{ev['semantic_gap']:.4f}",
                    f"{ev['final_loss']:.4f}",
                ])
    print(f"  CSV saved: {csv_path}")
    return csv_path


# ============================================================
# Aggregate-only mode
# ============================================================

def aggregate_only():
    """Re-load saved results and regenerate table/plot."""
    results_path = os.path.join(RESULTS_DIR, 'r3_low_k_results.json')
    if not os.path.exists(results_path):
        print(f"  ERROR: No results found at {results_path}")
        print("  Run with --all or --k first.")
        return

    with open(results_path, 'r') as f:
        data = json.load(f)

    print("\n  Loaded saved results.")
    print(f"  Config: {data['config']}")

    # Reconstruct the format expected by print_comparison_table
    all_k_results = {}
    variant_map = {'Baseline': 'baseline', 'R3 w=1.0': 'r3_w1', 'R3 w=5.0': 'r3_w5'}

    for k_key, k_data in data['results'].items():
        k = int(k_key.split('=')[1])
        all_k_results[k] = {}
        for nice_name, var_name in variant_map.items():
            if nice_name in k_data:
                ev = k_data[nice_name]
                all_k_results[k][var_name] = {
                    'dead_bits': ev['dead_bits'],
                    'entropy': ev['entropy'],
                    'r3_train_acc': ev['r3_train_acc'],
                    'r3_train_cos_mean': ev.get('r3_train_cos_mean', 0),
                    'r3_test_acc': ev['r3_test_acc'],
                    'r3_test_cos': ev.get('r3_test_cos_detail', []),
                    'r3_test_cos_mean': ev.get('r3_test_cos_mean', 0),
                    'semantic_gap': ev['semantic_gap'],
                    'final_loss': ev['lang_loss'],
                    'params': ev.get('params', 0),
                    # Dummy history for table (no plot in aggregate-only)
                    'history': {'step': [], 'loss': [], 'tri': [], 'r3': [], 'entropy': []},
                }

    print_comparison_table(all_k_results)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='E7: R3 Loss at Low k (k=6, 8, 12)')
    parser.add_argument('--k', type=int, default=None,
                        help='Single k value to test (6, 8, or 12)')
    parser.add_argument('--all', action='store_true',
                        help='Run all k values: 6, 8, 12')
    parser.add_argument('--aggregate-only', action='store_true',
                        help='Re-load saved results and regenerate table/plot')
    args = parser.parse_args()

    if args.aggregate_only:
        aggregate_only()
        return

    if not args.all and args.k is None:
        parser.error("Specify --k INT or --all")

    k_list = K_VALUES if args.all else [args.k]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("  E7: R3 LOSS AT LOW k")
    print("  Hypothesis: R3 works at k=6-12 (Engine regime) but dies at k=64")
    print("=" * 70)
    print(f"  Device:  {device}")
    if device.type == 'cuda':
        print(f"  GPU:     {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:    {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  k values: {k_list}")
    print(f"  Variants: baseline, R3 w=1.0, R3 w=5.0")
    print(f"  Steps:   {STEPS} per variant")
    est_min = len(k_list) * 3 * 5
    print(f"  Estimated time: ~{est_min} min ({len(k_list)} k x 3 variants x ~5 min)")

    # Load tokenizer
    tok_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'torch_run15_strongalign', 'tokenizer.json')
    if not os.path.exists(tok_path):
        # Fallback
        tok_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'torch', 'tokenizer.json')
    print(f"\n  Tokenizer: {tok_path}")
    tokenizer = BPETokenizer.load(tok_path)
    print(f"  Vocab: {tokenizer.vocab_size}")

    # Load data
    print("\nLoading data...")
    all_tokens = load_data(tokenizer)

    # Prepare analogy tensors
    print("\nPreparing analogy triples...")
    train_tensors = prepare_analogy_tensors(TRAIN_TRIPLES, tokenizer, device)
    test_tensors = prepare_analogy_tensors(TEST_TRIPLES, tokenizer, device)
    print(f"  Train: {len(train_tensors)}/{len(TRAIN_TRIPLES)} valid")
    print(f"  Test:  {len(test_tensors)}/{len(TEST_TRIPLES)} valid")

    # Run experiments
    all_k_results = {}

    # Try to load partial results (for resuming)
    results_path = os.path.join(RESULTS_DIR, 'r3_low_k_results.json')
    if os.path.exists(results_path):
        try:
            with open(results_path, 'r') as f:
                existing = json.load(f)
            print(f"\n  Found existing results at {results_path}")
            # We will overwrite with fresh runs for the requested k values
        except Exception:
            pass

    for k in k_list:
        results = run_single_k(k, tokenizer, all_tokens, device,
                               train_tensors, test_tensors)
        all_k_results[k] = results

    # Print table
    print_comparison_table(all_k_results)

    # Plot (only if we have at least 2 k values for meaningful comparison,
    # but plot even with 1 for single-k runs)
    if all_k_results:
        plot_results(all_k_results)

    # Save
    save_results(all_k_results)
    save_csv_summary(all_k_results)

    print("\n" + "=" * 70)
    print("  E7 COMPLETE")
    print(f"  Results: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
