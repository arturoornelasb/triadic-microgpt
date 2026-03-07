"""
Bit Entropy Benchmark — Measures triadic head health.

Diagnoses triadic collapse by computing per-bit entropy across a vocabulary
of concepts. Healthy triadic heads have H > 0.8; collapsed heads have H ≈ 0.

Usage:
  python benchmarks/scripts/bit_entropy.py \
    --model checkpoints/torch/model_best.pt \
    --tokenizer checkpoints/torch/tokenizer.json

Outputs:
  - benchmarks/results/v{VER}_bit_entropy_{DATE}.json
  - benchmarks/figures/bit_entropy_heatmap.png
"""

import os
import sys
import json
import argparse
import math
from datetime import date

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.evaluate import load_model
from src.triadic import PrimeMapper


# WordNet-derived concept vocabulary for testing
CONCEPTS = [
    "king", "queen", "man", "woman", "boy", "girl", "prince", "princess",
    "dog", "cat", "bird", "fish", "horse", "cow", "pig", "sheep",
    "doctor", "nurse", "teacher", "student", "lawyer", "judge", "scientist",
    "happy", "sad", "angry", "afraid", "brave", "kind", "cruel",
    "fire", "water", "earth", "air", "sun", "moon", "star", "cloud",
    "house", "car", "tree", "flower", "river", "mountain", "ocean", "forest",
    "red", "blue", "green", "big", "small", "fast", "slow", "old", "young",
    "mother", "father", "brother", "sister", "friend", "enemy",
    "food", "drink", "bread", "milk", "apple", "cake",
    "morning", "night", "summer", "winter", "spring", "rain", "snow",
    "love", "hate", "fear", "hope", "joy", "pain", "peace", "war",
    "book", "pen", "door", "window", "table", "chair", "bed", "lamp",
    "city", "village", "school", "church", "hospital", "garden", "park",
    "music", "dance", "song", "game", "story", "dream", "magic",
    "run", "walk", "swim", "fly", "jump", "climb", "fall", "sleep",
]


def compute_projections(model, tokenizer, concepts, device):
    """Get triadic projections for all concepts."""
    projections = []
    primes = []
    mapper = PrimeMapper(model.config.n_triadic_bits)

    for concept in concepts:
        ids = tokenizer.encode(concept, add_special=False)
        if not ids:
            projections.append(None)
            primes.append(None)
            continue

        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            _, triadic_proj, _ = model(x)

        proj = triadic_proj[0].mean(dim=0).cpu().numpy()
        projections.append(proj)
        primes.append(mapper.map(proj))

    return projections, primes


def compute_bit_entropy(projections):
    """Compute per-bit entropy from projection values."""
    valid = [p for p in projections if p is not None]
    if not valid:
        return np.zeros(1), 0.0

    matrix = np.stack(valid)  # (N, n_bits)
    # Convert tanh outputs to probabilities: p = (tanh + 1) / 2
    probs = (matrix + 1.0) / 2.0

    # Per-bit entropy: H = -p*log2(p) - (1-p)*log2(1-p)
    eps = 1e-10
    entropy = -(probs * np.log2(probs + eps) + (1 - probs) * np.log2(1 - probs + eps))
    per_bit_entropy = entropy.mean(axis=0)  # (n_bits,)
    mean_entropy = per_bit_entropy.mean()

    return per_bit_entropy, float(mean_entropy)


def compute_signature_diversity(primes):
    """Count unique prime signatures."""
    valid = [p for p in primes if p is not None]
    unique = len(set(valid))
    total = len(valid)
    return unique, total, unique / total if total > 0 else 0.0


def plot_heatmap(projections, concepts, output_path):
    """Plot concept × bit activation heatmap."""
    valid_pairs = [(c, p) for c, p in zip(concepts, projections) if p is not None]
    if len(valid_pairs) < 2:
        return

    labels = [c for c, _ in valid_pairs]
    matrix = np.stack([p for _, p in valid_pairs])

    # Show first 50 concepts max for readability
    n_show = min(50, len(labels))
    matrix = matrix[:n_show]
    labels = labels[:n_show]

    fig, ax = plt.subplots(figsize=(16, max(8, n_show * 0.25)))
    im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_yticks(range(n_show))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('Triadic Bit Index')
    ax.set_ylabel('Concept')
    ax.set_title('Triadic Projection Heatmap (tanh outputs)')
    plt.colorbar(im, ax=ax, label='Activation')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Heatmap saved: {output_path}")


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print()
    print("=" * 64)
    print("  BIT ENTROPY BENCHMARK")
    print("=" * 64)
    print(f"  Model: {args.model}")
    print(f"  Concepts: {len(CONCEPTS)}")
    print()

    # Load model
    model, tokenizer, config = load_model(args.model, args.tokenizer, device)
    print(f"  Config: {config.n_layer}L/{config.n_embd}D/{config.n_head}H/{config.n_triadic_bits}bits")
    print()

    # Compute projections
    print("  Computing triadic projections...")
    projections, primes = compute_projections(model, tokenizer, CONCEPTS, device)

    # Bit entropy
    per_bit_entropy, mean_entropy = compute_bit_entropy(projections)
    print(f"  Mean bit entropy: {mean_entropy:.4f} (target: > 0.8)")
    print(f"  Min bit entropy:  {per_bit_entropy.min():.4f}")
    print(f"  Max bit entropy:  {per_bit_entropy.max():.4f}")

    # Signature diversity
    unique, total, diversity = compute_signature_diversity(primes)
    print(f"  Unique signatures: {unique}/{total} ({diversity:.1%})")

    # Collapse detection
    collapsed = mean_entropy < 0.3
    if collapsed:
        print(f"  WARNING: TRIADIC COLLAPSE DETECTED (entropy {mean_entropy:.4f} << 0.8)")
    else:
        print(f"  Triadic head appears healthy.")

    # Concept pair samples
    print()
    print("  Sample concept pairs:")
    mapper = PrimeMapper(config.n_triadic_bits)
    from src.triadic import TriadicValidator
    validator = TriadicValidator()
    pairs = [("king", "queen"), ("dog", "cat"), ("king", "dog"), ("happy", "sad"), ("fire", "water")]
    for a, b in pairs:
        ia = CONCEPTS.index(a) if a in CONCEPTS else -1
        ib = CONCEPTS.index(b) if b in CONCEPTS else -1
        if ia >= 0 and ib >= 0 and primes[ia] and primes[ib]:
            sim = validator.similarity(primes[ia], primes[ib])
            print(f"    {a:>10s} ↔ {b:<10s}  sim={sim:.2%}  primes_equal={primes[ia]==primes[ib]}")

    # Save results
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    results_dir = os.path.join(project_root, 'benchmarks', 'results')
    figures_dir = os.path.join(project_root, 'benchmarks', 'figures')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    version = args.version
    today = date.today().isoformat()

    result = {
        "benchmark": "bit_entropy",
        "version": version,
        "date": today,
        "model_checkpoint": args.model,
        "model_config": f"{config.n_layer}L/{config.n_embd}D/{config.n_head}H/{config.n_triadic_bits}bits",
        "num_concepts": len(CONCEPTS),
        "metrics": {
            "mean_bit_entropy": mean_entropy,
            "min_bit_entropy": float(per_bit_entropy.min()),
            "max_bit_entropy": float(per_bit_entropy.max()),
            "per_bit_entropy": per_bit_entropy.tolist(),
            "unique_signatures": unique,
            "total_concepts": total,
            "signature_diversity": diversity,
            "collapsed": collapsed,
        },
    }

    result_path = os.path.join(results_dir, f"{version}_bit_entropy_{today}.json")
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved: {result_path}")

    # Plot heatmap
    heatmap_path = os.path.join(figures_dir, "bit_entropy_heatmap.png")
    plot_heatmap(projections, CONCEPTS, heatmap_path)

    # Plot per-bit entropy bar chart
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(len(per_bit_entropy)), per_bit_entropy, color='steelblue', alpha=0.8)
    ax.axhline(y=0.8, color='green', linestyle='--', label='Target (0.8)')
    ax.axhline(y=mean_entropy, color='red', linestyle='-', label=f'Mean ({mean_entropy:.3f})')
    ax.set_xlabel('Bit Index')
    ax.set_ylabel('Entropy (bits)')
    ax.set_title('Per-Bit Entropy of Triadic Projections')
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    entropy_path = os.path.join(figures_dir, "per_bit_entropy.png")
    plt.savefig(entropy_path, dpi=150)
    plt.close()
    print(f"  Entropy chart saved: {entropy_path}")

    print()
    print("=" * 64)
    verdict = "PASS" if not collapsed and diversity > 0.8 else "FAIL"
    print(f"  VERDICT: {verdict}")
    print(f"  Entropy: {mean_entropy:.4f} {'>=0.8' if mean_entropy >= 0.8 else '<0.8 FAIL'}")
    print(f"  Diversity: {diversity:.1%} {'>=80%' if diversity >= 0.8 else '<80% FAIL'}")
    print("=" * 64)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bit Entropy Benchmark')
    parser.add_argument('--model', required=True, help='Model checkpoint path')
    parser.add_argument('--tokenizer', default=None, help='Tokenizer path (auto-detected if omitted)')
    parser.add_argument('--version', default='v1.1', help='Version tag for results')
    args = parser.parse_args()

    if args.tokenizer is None:
        args.tokenizer = os.path.join(os.path.dirname(args.model), 'tokenizer.json')

    main(args)
