"""
P3 — Phase-Aware Position Encoding (La Danza Cosmica, Cap. 7-9, 14-16)

The book defines that every opposition follows y(t) = A sin(2πft + φ),
where φ captures the "perspective of the observer". Each attention head
can represent a different observer with a different phase.

This experiment replaces learned position embeddings with sinusoidal
encodings that have learnable per-head phase parameters:
  pos_enc(pos, h, k) = sin(pos * freq[k] + φ[h, k])

3 variants:
1. Baseline: standard learned position embeddings (nn.Embedding)
2. Sinusoidal (fixed): standard sin/cos positional encoding (no learning)
3. Phase-Aware: sinusoidal + learnable phase φ per attention head
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

from src.torch_transformer import TriadicGPT, TriadicGPTConfig, CausalSelfAttention, TransformerBlock
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


# ── Sinusoidal Position Encoding (fixed, no learning) ──

class SinusoidalPositionEncoding(nn.Module):
    """Standard sinusoidal position encoding (Vaswani et al. 2017). Not learned."""

    def __init__(self, block_size, n_embd):
        super().__init__()
        pe = torch.zeros(block_size, n_embd)
        position = torch.arange(0, block_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embd, 2, dtype=torch.float) * -(math.log(10000.0) / n_embd))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, positions):
        return self.pe[positions]


# ── Phase-Aware Position Encoding (learnable phase per head) ──

class PhasePositionEncoding(nn.Module):
    """Sinusoidal encoding with learnable phase per attention head.

    Each head h gets its own phase vector φ_h of shape (head_dim,).
    The encoding for position p at head h, dimension k is:
        enc(p, h, k) = sin(p * freq[k] + φ[h, k])

    The phases are projected back to n_embd via a linear layer so
    they can be added to the standard token embedding path.
    """

    def __init__(self, block_size, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.n_embd = n_embd

        # Fixed frequencies (same as standard sinusoidal)
        freq = torch.exp(torch.arange(0, self.head_dim, dtype=torch.float) * -(math.log(10000.0) / self.head_dim))
        self.register_buffer('freq', freq)  # (head_dim,)

        # Learnable phase per head: initialized to spread heads across [0, 2π)
        # Each head starts with a different phase offset, inspired by the book's
        # "5 constants" — different perspectives on the same position.
        init_phases = torch.linspace(0, 2 * math.pi * (1 - 1/n_head), n_head)  # (n_head,)
        self.phase = nn.Parameter(init_phases.unsqueeze(1).expand(n_head, self.head_dim).clone())  # (n_head, head_dim)

        # Learnable amplitude per head (initialized to 1.0)
        self.amplitude = nn.Parameter(torch.ones(n_head, 1))  # (n_head, 1)

        # Project from (n_head * head_dim) back to n_embd
        self.proj = nn.Linear(n_head * self.head_dim, n_embd, bias=False)

    def forward(self, positions):
        """
        Args:
            positions: (T,) long tensor of position indices
        Returns:
            (T, n_embd) position encoding
        """
        T = positions.shape[0]
        pos_float = positions.float().unsqueeze(1)  # (T, 1)

        # Compute per-head sinusoidal with phase: (T, n_head, head_dim)
        # freq: (head_dim,), phase: (n_head, head_dim)
        angles = pos_float.unsqueeze(1) * self.freq.unsqueeze(0).unsqueeze(0) + self.phase.unsqueeze(0)
        # angles: (T, n_head, head_dim)

        encoded = torch.sin(angles) * self.amplitude.unsqueeze(0)  # (T, n_head, head_dim)

        # Flatten heads and project to n_embd
        encoded_flat = encoded.reshape(T, self.n_head * self.head_dim)  # (T, n_head * head_dim)
        return self.proj(encoded_flat)  # (T, n_embd)


# ── Model Variants ──

class SinusoidalGPT(TriadicGPT):
    """TriadicGPT with fixed sinusoidal position encoding."""

    def __init__(self, config):
        super().__init__(config)
        # Replace learned wpe with fixed sinusoidal
        self.wpe = SinusoidalPositionEncoding(config.block_size, config.n_embd)


class PhaseGPT(TriadicGPT):
    """TriadicGPT with phase-aware position encoding (learnable phase per head)."""

    def __init__(self, config):
        super().__init__(config)
        # Replace learned wpe with phase-aware sinusoidal
        self.wpe = PhasePositionEncoding(config.block_size, config.n_embd, config.n_head)


# ── Data ──

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


# ── Evaluation ──

def evaluate_model(model, tokenizer, device):
    model.eval()
    mapper = PrimeMapper(N_BITS)

    concept_pairs = {
        'related': [
            ("king", "queen"), ("dog", "cat"), ("happy", "sad"),
            ("mother", "father"), ("sun", "moon"), ("hot", "cold"),
            ("love", "hate"), ("big", "small"), ("bird", "fish"),
            ("doctor", "hospital"), ("teacher", "school"),
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
        ("hot", "cold", "day", "night"),
        ("happy", "sad", "love", "hate"),
    ]

    all_words = set()
    for group in concept_pairs.values():
        for w1, w2 in group:
            all_words.update([w1, w2])
    for a, b, c, d in analogy_triples:
        all_words.update([a, b, c, d])
    all_words.update(["tree", "river", "mountain", "stone", "cloud", "table"])

    sigs = {}
    projs = {}
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

    # Semantic gap
    rel_sims = [cosine(projs[a], projs[b]) for a, b in concept_pairs['related'] if a in projs and b in projs]
    rand_sims = []
    wlist = list(projs.keys())
    for _ in range(200):
        i, j = random.sample(range(len(wlist)), 2)
        rand_sims.append(cosine(projs[wlist[i]], projs[wlist[j]]))
    gap = np.mean(rel_sims) - np.mean(rand_sims) if rel_sims else 0

    # Analogies
    ana_correct = 0
    ana_total = 0
    ana_offsets = []
    for a, b, c, d in analogy_triples:
        if not all(w in projs for w in [a, b, c, d]):
            continue
        pred = projs[b] - projs[a] + projs[c]
        cos = cosine(pred, projs[d])
        ana_offsets.append(cos)
        phi_pred = TriadicValidator.analogy(sigs[a], sigs[b], sigs[c])
        if TriadicValidator.similarity(phi_pred, sigs[d]) > 0.3:
            ana_correct += 1
        ana_total += 1

    # Bit stats
    all_p = np.stack(list(projs.values()))
    bm = (all_p > 0).mean(axis=0)
    eps = 1e-7
    bent = -(bm * np.log2(bm + eps) + (1 - bm) * np.log2(1 - bm + eps))
    dead = int((bent < 0.3).sum())

    return {
        'semantic_gap': float(gap),
        'mean_offset_cos': float(np.mean(ana_offsets)) if ana_offsets else 0,
        'analogy_verif': ana_correct / max(ana_total, 1),
        'dead_bits': dead,
        'mean_entropy': float(bent.mean()),
    }


# ── Training ──

def train_model(model, tokenizer, all_tokens, device, label):
    dataset = TextDataset(all_tokens, BLOCK_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    triadic_warmup = int(STEPS * TRIADIC_WARMUP_PCT)

    model.train()
    data_iter = iter(dataloader)
    history = {'step': [], 'loss': [], 'tri': [], 'entropy': []}

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
            tri_v = 0.0

            if step >= triadic_warmup:
                aw = int(STEPS * 0.2)
                af = min(1.0, (step - triadic_warmup + 1) / aw)
                ca = ALPHA * af
                tri_loss = model.triadic_loss(triadic_proj, entropy_weight=ENTROPY_WEIGHT,
                                              input_ids=x, align_weight=ALIGN_WEIGHT, align_mode='mse')
                total_loss = lang_loss + ca * tri_loss
                tri_v = tri_loss.item()

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % 200 == 0 or step == STEPS - 1:
            with torch.no_grad():
                flat = triadic_proj.reshape(-1, triadic_proj.size(-1))
                bm = (flat > 0).float().mean(dim=0)
                eps = 1e-7
                ent = -(bm * (bm + eps).log2() + (1 - bm) * (1 - bm + eps).log2())

            history['step'].append(step)
            history['loss'].append(lang_loss.item())
            history['tri'].append(tri_v)
            history['entropy'].append(ent.mean().item())

            elapsed = time.time() - t0
            speed = (step + 1) / max(elapsed, 1)
            eta_s = (STEPS - step - 1) / max(speed, 0.01)
            bar = progress_bar(step + 1, STEPS)
            print(f"  [{label:>12s}] {bar}  step {step:>5d}/{STEPS}  "
                  f"loss={lang_loss.item():.3f}  tri={tri_v:.4f}  "
                  f"ETA {format_time(eta_s)}  [{format_time(elapsed)}]")

    return history


# ── Phase Analysis ──

def analyze_phases(model, label):
    """Analyze learned phase parameters if they exist."""
    if not hasattr(model.wpe, 'phase'):
        return None

    phase = model.wpe.phase.detach().cpu().numpy()  # (n_head, head_dim)
    amplitude = model.wpe.amplitude.detach().cpu().numpy().flatten()  # (n_head,)

    # Phase spread: how different are the heads' phases?
    phase_means = phase.mean(axis=1)  # mean phase per head
    phase_spread = phase_means.std()

    # Phase change from initialization
    init_phases = np.linspace(0, 2 * np.pi * (1 - 1/N_HEAD), N_HEAD)
    phase_delta = np.abs(phase_means - init_phases).mean()

    print(f"\n  [{label}] Phase Analysis:")
    print(f"    Amplitude per head: {', '.join(f'{a:.3f}' for a in amplitude)}")
    print(f"    Mean phase per head: {', '.join(f'{p:.2f}' for p in phase_means)}")
    print(f"    Phase spread (std): {phase_spread:.4f}")
    print(f"    Phase delta from init: {phase_delta:.4f} rad")

    return {
        'amplitude': amplitude.tolist(),
        'mean_phase_per_head': phase_means.tolist(),
        'phase_spread': float(phase_spread),
        'phase_delta_from_init': float(phase_delta),
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("  PHASE-AWARE POSITION ENCODING EXPERIMENT")
    print("  (La Danza Cosmica, Cap. 7-9: Perspective of the Observer)")
    print("=" * 70)
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU:    {torch.cuda.get_device_name(0)}")

    tok_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'torch_run15_strongalign', 'tokenizer.json')
    tokenizer = BPETokenizer.load(tok_path)

    print("\nLoading data...")
    all_tokens = load_data(tokenizer)

    config = TriadicGPTConfig(
        vocab_size=tokenizer.vocab_size, block_size=BLOCK_SIZE,
        n_layer=N_LAYER, n_embd=N_EMBD, n_head=N_HEAD,
        n_triadic_bits=N_BITS, dropout=0.1,
    )

    variants = [
        ("Learned", TriadicGPT),
        ("Sinusoidal", SinusoidalGPT),
        ("Phase-Aware", PhaseGPT),
    ]

    all_results = {}
    all_histories = {}
    all_phases = {}

    for name, model_cls in variants:
        print(f"\n{'─' * 70}")
        print(f"  Training: {name}")
        print(f"{'─' * 70}")

        model = model_cls(config).to(device)
        n_params = model.num_params()
        print(f"  Parameters: {n_params:,}")

        hist = train_model(model, tokenizer, all_tokens, device, name)
        ev = evaluate_model(model, tokenizer, device)
        phase_info = analyze_phases(model, name)

        all_results[name] = ev
        all_histories[name] = hist
        if phase_info:
            all_phases[name] = phase_info

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  PHASE ATTENTION RESULTS")
    print("=" * 70)

    print(f"\n  {'Variant':>14s}  {'Loss':>6s}  {'Gap':>8s}  {'Dead':>4s}  {'Entropy':>8s}  "
          f"{'Offset':>7s}  {'Ana':>5s}")
    print(f"  {'─'*14}  {'─'*6}  {'─'*8}  {'─'*4}  {'─'*8}  {'─'*7}  {'─'*5}")

    for name in [v[0] for v in variants]:
        ev = all_results[name]
        h = all_histories[name]
        print(f"  {name:>14s}  {h['loss'][-1]:>6.3f}  {ev['semantic_gap']:>+8.4f}  "
              f"{ev['dead_bits']:>4d}  {ev['mean_entropy']:>8.4f}  "
              f"{ev['mean_offset_cos']:>7.4f}  {ev['analogy_verif']:>5.1%}")

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = {'Learned': 'blue', 'Sinusoidal': 'green', 'Phase-Aware': 'red'}

    for name in [v[0] for v in variants]:
        h = all_histories[name]
        c = colors[name]
        axes[0].plot(h['step'], h['loss'], color=c, label=name, alpha=0.8)
        axes[1].plot(h['step'], h['tri'], color=c, label=name, alpha=0.8)
        axes[2].plot(h['step'], h['entropy'], color=c, label=name, alpha=0.8)

    for ax, title, ylabel in [
        (axes[0], 'Language Loss', 'Loss'),
        (axes[1], 'Triadic Loss', 'Loss'),
        (axes[2], 'Bit Entropy', 'Mean Entropy'),
    ]:
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Step')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Phase-Aware Position Encoding: Observer Perspective', fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'phase_attention.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved: {plot_path}")

    # Phase heatmap if available
    if all_phases:
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        phase_data = all_phases['Phase-Aware']
        ax2.bar(range(N_HEAD), phase_data['amplitude'], color='steelblue', alpha=0.7, label='Amplitude')
        ax2.set_xlabel('Head')
        ax2.set_ylabel('Amplitude')
        ax2.set_title('Learned Phase Amplitudes per Attention Head')
        ax2.set_xticks(range(N_HEAD))
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        phase_plot = os.path.join(RESULTS_DIR, 'phase_attention_heads.png')
        plt.savefig(phase_plot, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Phase plot saved: {phase_plot}")

    # ── Save ──
    save_data = {
        'experiment': 'phase_attention',
        'source': 'La Danza Cosmica Cap. 7-9',
        'config': f'{N_LAYER}L/{N_EMBD}D/{N_HEAD}H/{N_BITS}bits',
        'steps': STEPS,
        'variants': {name: {'final_loss': all_histories[name]['loss'][-1],
                            'eval': all_results[name],
                            'phase_analysis': all_phases.get(name)}
                     for name in [v[0] for v in variants]},
    }
    results_path = os.path.join(RESULTS_DIR, 'phase_attention.json')
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"  Results saved: {results_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
