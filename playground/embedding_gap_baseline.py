"""
B1 — Embedding Semantic Gap Baseline (0 GPU, eval only).

Measures semantic gap on RAW wte embeddings from Run 15 to answer:
"Does the triadic head add semantic structure, or do the embeddings
already have it?"

If embedding gap ≈ triadic gap, the head is just copying embedding structure.
If embedding gap << triadic gap, the head is adding genuine algebraic value.

Usage:
  python playground/embedding_gap_baseline.py
"""

import os
import sys
import json
import random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.torch_transformer import TriadicGPT, TriadicGPTConfig
try:
    from src.fast_tokenizer import FastBPETokenizer as BPETokenizer
except ImportError:
    from src.tokenizer import BPETokenizer
from src.triadic import PrimeMapper, TriadicValidator


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Same concept pairs used in ALL semantic gap evaluations
RELATED_PAIRS = [
    ("king", "queen"), ("dog", "cat"), ("happy", "sad"),
    ("mother", "father"), ("sun", "moon"), ("hot", "cold"),
    ("love", "hate"), ("big", "small"), ("bird", "fish"),
    ("doctor", "hospital"), ("teacher", "school"),
    ("princess", "prince"), ("old", "young"),
]

UNRELATED_PAIRS = [
    ("king", "fish"), ("dog", "moon"), ("happy", "river"),
    ("mother", "blue"), ("sun", "cat"), ("hot", "queen"),
    ("bird", "school"), ("love", "tree"), ("big", "night"),
]

ANALOGY_QUADS = [
    ("king", "queen", "man", "woman"),
    ("father", "mother", "brother", "sister"),
    ("dog", "puppy", "cat", "kitten"),
    ("big", "small", "tall", "short"),
    ("hot", "cold", "day", "night"),
    ("happy", "sad", "love", "hate"),
]


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


@torch.no_grad()
def get_representations(model, tokenizer, device, words):
    """Get both wte embeddings and triadic projections for a list of words."""
    model.eval()
    embeddings = {}
    projections = {}

    for word in words:
        ids = tokenizer.encode(word, add_special=False)
        if not ids:
            continue

        x = torch.tensor([ids], dtype=torch.long, device=device)

        # Raw wte embedding (mean-pool if multi-token)
        wte = model.wte(x)  # (1, seq_len, n_embd)
        emb = wte[0].mean(dim=0).cpu().numpy()  # (n_embd,)
        embeddings[word] = emb

        # Triadic projection
        _, proj, _ = model(x)
        proj_np = proj[0].mean(dim=0).cpu().numpy()  # (n_bits,)
        projections[word] = proj_np

    return embeddings, projections


def compute_gap(representations, label):
    """Compute semantic gap from a dict of word -> vector."""
    related_sims = [cosine(representations[w1], representations[w2])
                    for w1, w2 in RELATED_PAIRS
                    if w1 in representations and w2 in representations]

    random_sims = []
    words = list(representations.keys())
    rng = random.Random(42)
    for _ in range(200):
        i, j = rng.sample(range(len(words)), 2)
        random_sims.append(cosine(representations[words[i]], representations[words[j]]))

    gap = float(np.mean(related_sims) - np.mean(random_sims))
    print(f"  [{label}] related={np.mean(related_sims):.4f}, "
          f"random={np.mean(random_sims):.4f}, gap={gap:+.4f}")
    return {
        'gap': gap,
        'mean_related': float(np.mean(related_sims)),
        'mean_random': float(np.mean(random_sims)),
        'n_related': len(related_sims),
        'n_random': len(random_sims),
    }


def compute_analogy_verification(representations, mapper, label, use_primes=False):
    """Compute analogy verification rate."""
    verified = 0
    total = 0

    for a, b, c, d in ANALOGY_QUADS:
        if not all(w in representations for w in [a, b, c, d]):
            continue
        total += 1

        if use_primes:
            sigs = {w: mapper.map(representations[w]) for w in [a, b, c, d]}
            # Algebraic: Phi(b)/Phi(a) * Phi(c) ≈ Phi(d)
            # Verification: offset similarity
            va, vb, vc, vd = [representations[w] for w in [a, b, c, d]]
            offset_ab = vb - va
            offset_cd = vd - vc
            sim = cosine(offset_ab, offset_cd)
            if sim > 0:
                verified += 1
        else:
            # Vector arithmetic: b - a + c ≈ d
            va, vb, vc, vd = [representations[w] for w in [a, b, c, d]]
            offset_ab = vb - va
            offset_cd = vd - vc
            sim = cosine(offset_ab, offset_cd)
            if sim > 0:
                verified += 1

    rate = verified / max(total, 1)
    print(f"  [{label}] Analogy verification: {verified}/{total} ({rate:.1%})")
    return {'verification_rate': rate, 'verified': verified, 'total': total}


def compute_bit_stats(projections, n_bits):
    """Dead bits and entropy for triadic projections."""
    all_projs = np.stack(list(projections.values()))
    bit_means = (all_projs > 0).mean(axis=0)
    eps = 1e-7
    bit_entropy = -(bit_means * np.log2(bit_means + eps) +
                    (1 - bit_means) * np.log2(1 - bit_means + eps))
    dead_bits = int((bit_entropy < 0.3).sum())
    return {
        'dead_bits': dead_bits,
        'mean_entropy': float(bit_entropy.mean()),
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths
    ckpt_path = os.path.join(PROJECT_ROOT, 'checkpoints',
                             'torch_run15_strongalign', 'model_L12_D512_B64_best.pt')
    tokenizer_path = os.path.join(PROJECT_ROOT, 'checkpoints',
                                  'torch_run15_strongalign', 'tokenizer.json')

    print("=" * 70)
    print("  B1: EMBEDDING SEMANTIC GAP BASELINE")
    print("  Question: Does the triadic head add structure beyond embeddings?")
    print("=" * 70)
    print(f"  Device: {device}")
    print(f"  Checkpoint: Run 15 (v1.4-strongalign)")
    print()

    # Load model
    print("[1/3] Loading model...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt['config'] if 'config' in ckpt else ckpt.get('model_config', {})
    config = TriadicGPTConfig(
        vocab_size=cfg['vocab_size'],
        block_size=cfg['block_size'],
        n_layer=cfg['n_layer'],
        n_embd=cfg['n_embd'],
        n_head=cfg['n_head'],
        n_triadic_bits=cfg['n_triadic_bits'],
        dropout=0.0,
    )
    model = TriadicGPT(config).to(device)
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    n_bits = config.n_triadic_bits
    print(f"  Loaded: {config.n_layer}L/{config.n_embd}D/{n_bits}bits")

    # Load tokenizer
    tokenizer = BPETokenizer.load(tokenizer_path)
    mapper = PrimeMapper(n_bits)

    # Collect all words
    all_words = set()
    for w1, w2 in RELATED_PAIRS + UNRELATED_PAIRS:
        all_words.update([w1, w2])
    for a, b, c, d in ANALOGY_QUADS:
        all_words.update([a, b, c, d])

    # Get representations
    print()
    print("[2/3] Extracting representations...")
    embeddings, projections = get_representations(model, tokenizer, device, all_words)
    print(f"  Words encoded: {len(embeddings)}/{len(all_words)}")
    print(f"  Embedding dim: {config.n_embd}")
    print(f"  Triadic dim: {n_bits}")

    # Compute metrics
    print()
    print("[3/3] Computing metrics...")
    print()
    print("  --- Semantic Gap ---")
    emb_gap = compute_gap(embeddings, f"Embedding ({config.n_embd}D)")
    tri_gap = compute_gap(projections, f"Triadic ({n_bits}D)")

    print()
    print("  --- Analogy Verification (offset cosine > 0) ---")
    emb_analogy = compute_analogy_verification(embeddings, mapper, f"Embedding ({config.n_embd}D)")
    tri_analogy = compute_analogy_verification(projections, mapper, f"Triadic ({n_bits}D)", use_primes=True)

    print()
    print("  --- Triadic Bit Stats ---")
    bit_stats = compute_bit_stats(projections, n_bits)
    print(f"  Dead bits: {bit_stats['dead_bits']}/{n_bits}")
    print(f"  Mean entropy: {bit_stats['mean_entropy']:.3f}")

    # Summary
    print()
    print("=" * 70)
    print("  B1: RESULTS — Embedding vs Triadic")
    print("=" * 70)
    print(f"  {'Metric':<30s} {'Embedding':>14s} {'Triadic':>14s} {'Delta':>14s}")
    print(f"  {'='*30} {'='*14} {'='*14} {'='*14}")
    print(f"  {'Semantic gap':<30s} {emb_gap['gap']:>+14.4f} {tri_gap['gap']:>+14.4f} {tri_gap['gap']-emb_gap['gap']:>+14.4f}")
    print(f"  {'Related similarity':<30s} {emb_gap['mean_related']:>14.4f} {tri_gap['mean_related']:>14.4f}")
    print(f"  {'Random similarity':<30s} {emb_gap['mean_random']:>14.4f} {tri_gap['mean_random']:>14.4f}")
    print(f"  {'Analogy verification':<30s} {emb_analogy['verification_rate']:>14.1%} {tri_analogy['verification_rate']:>14.1%}")
    print(f"  {'Dimensionality':<30s} {config.n_embd:>14d} {n_bits:>14d} {'(8x compression)':>14s}")

    # Net gap
    net_gap = tri_gap['gap'] - emb_gap['gap']
    print()
    print(f"  {'NET SEMANTIC GAP':<30s} {net_gap:>+14.4f}")
    print(f"    (triadic gap {tri_gap['gap']:+.4f} minus embedding gap {emb_gap['gap']:+.4f})")

    # Verdict
    print()
    if net_gap > 0.005:
        verdict = "TRIADIC ADDS VALUE — net gap positive"
    elif abs(net_gap) <= 0.005:
        verdict = "INCONCLUSIVE — net gap within noise"
    else:
        verdict = "WARNING — embeddings have HIGHER gap than triadic"
    print(f"  VERDICT: {verdict}")
    print("=" * 70)

    # Save
    results_dir = os.path.join(PROJECT_ROOT, 'playground', 'results')
    os.makedirs(results_dir, exist_ok=True)
    results = {
        'experiment': 'embedding_gap_baseline',
        'description': 'B1: Semantic gap on raw wte embeddings vs triadic projections',
        'checkpoint': 'Run 15 (v1.4-strongalign)',
        'embedding_dim': config.n_embd,
        'triadic_dim': n_bits,
        'n_words': len(embeddings),
        'embedding': {
            'semantic_gap': emb_gap,
            'analogy_verification': emb_analogy,
        },
        'triadic': {
            'semantic_gap': tri_gap,
            'analogy_verification': tri_analogy,
            'bit_stats': bit_stats,
        },
        'net_semantic_gap': net_gap,
        'gross_triadic_gap': tri_gap['gap'],
        'gross_embedding_gap': emb_gap['gap'],
        'verdict': verdict,
    }
    results_path = os.path.join(results_dir, 'embedding_gap_baseline.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results: {results_path}")


if __name__ == '__main__':
    main()
