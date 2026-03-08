"""
Experiment 10 -- Evaluation & Benchmark.

Runs the same metrics as engine_comparison.py on the GPT-2+Triadic model
and prints a comparison table with:
  1. GPT-2+Triadic (this experiment)
  2. TriadicGPT from-scratch (Run 15 results)
  3. Engine PCA (Experiment 9 results)

Usage:
  python experiment10/src/evaluate.py \\
    --checkpoint experiment10/checkpoints/phase_2_(unfreeze_last_layers)_final.pt
"""

import os
import sys
import math
import json
import time
import argparse

import numpy as np
import torch

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from model import GPT2TriadicModel
from src.triadic import PrimeMapper


# ============================================================
# Concept sets (identical to engine_comparison.py)
# ============================================================

CONCEPTS = [
    "king", "queen", "man", "woman", "boy", "girl", "prince", "princess",
    "dog", "cat", "bird", "fish", "horse", "cow", "pig", "sheep",
    "doctor", "nurse", "teacher", "student", "lawyer", "judge", "scientist",
    "happy", "sad", "angry", "afraid", "brave", "kind", "cruel",
    "fire", "water", "earth", "air", "sun", "moon", "star", "cloud",
    "house", "tree", "flower", "river", "mountain", "ocean", "forest",
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

RELATED_PAIRS = [
    ("king", "queen"), ("dog", "cat"), ("happy", "sad"),
    ("mother", "father"), ("boy", "girl"), ("brother", "sister"),
    ("love", "hate"), ("sun", "moon"), ("fire", "water"),
    ("run", "walk"), ("big", "small"), ("doctor", "nurse"),
]

UNRELATED_PAIRS = [
    ("king", "dog"), ("queen", "fish"), ("happy", "table"),
    ("mother", "river"), ("boy", "cloud"), ("brother", "cake"),
    ("love", "chair"), ("sun", "pen"), ("fire", "girl"),
    ("run", "lamp"), ("big", "milk"), ("doctor", "star"),
]

HYPERNYM_PAIRS = [
    ("animal", "dog"), ("animal", "cat"), ("animal", "bird"), ("animal", "fish"),
    ("person", "king"), ("person", "queen"), ("person", "boy"), ("person", "girl"),
    ("person", "mother"), ("person", "father"), ("person", "doctor"), ("person", "teacher"),
]

ANALOGIES = [
    ("king", "queen", "man", "woman"),
    ("king", "queen", "boy", "girl"),
    ("king", "queen", "father", "mother"),
    ("man", "woman", "boy", "girl"),
    ("happy", "sad", "love", "hate"),
    ("happy", "sad", "peace", "war"),
    ("big", "small", "fast", "slow"),
    ("big", "small", "old", "young"),
    ("fire", "water", "sun", "moon"),
    ("morning", "night", "summer", "winter"),
    ("dog", "cat", "horse", "cow"),
    ("doctor", "hospital", "teacher", "school"),
]

EXTRA_CONCEPTS = ["animal", "hospital", "castle", "person"]


# ============================================================
# Metric computation
# ============================================================

def prime_factors(n):
    """Return set of prime factors of n."""
    if n <= 1:
        return set()
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors


def jaccard_factors(a, b):
    fa = prime_factors(a)
    fb = prime_factors(b)
    if not fa and not fb:
        return 1.0
    union = fa | fb
    return len(fa & fb) / len(union) if union else 0.0


def compute_metrics(concept_primes, n_bits):
    """Compute all metrics from {concept: prime_composite} dict."""
    import sympy
    results = {}
    plane_primes = [sympy.prime(i + 1) for i in range(n_bits)]

    # Bit entropy + unique signatures
    bit_matrix = []
    valid_concepts = []
    for concept in CONCEPTS:
        if concept not in concept_primes:
            continue
        p = concept_primes[concept]
        bits = [1 if p % pp == 0 else 0 for pp in plane_primes]
        bit_matrix.append(bits)
        valid_concepts.append(concept)

    if not bit_matrix:
        return {'error': 'no valid concepts'}

    bit_matrix = np.array(bit_matrix, dtype=float)
    p_bits = bit_matrix.mean(axis=0)
    p_bits = np.clip(p_bits, 1e-10, 1 - 1e-10)
    entropy = -p_bits * np.log2(p_bits) - (1 - p_bits) * np.log2(1 - p_bits)
    results['bit_entropy'] = float(entropy.mean())

    sigs = set(tuple(row) for row in bit_matrix.astype(int))
    results['unique_sigs'] = len(sigs)
    results['total_concepts'] = len(valid_concepts)
    results['unique_pct'] = len(sigs) / len(valid_concepts) * 100

    # Semantic ordering
    related_sims = []
    for a, b in RELATED_PAIRS:
        if a in concept_primes and b in concept_primes:
            related_sims.append(jaccard_factors(concept_primes[a], concept_primes[b]))
    unrelated_sims = []
    for a, b in UNRELATED_PAIRS:
        if a in concept_primes and b in concept_primes:
            unrelated_sims.append(jaccard_factors(concept_primes[a], concept_primes[b]))

    results['mean_related_sim'] = float(np.mean(related_sims)) if related_sims else 0.0
    results['mean_unrelated_sim'] = float(np.mean(unrelated_sims)) if unrelated_sims else 0.0
    results['semantic_gap'] = results['mean_related_sim'] - results['mean_unrelated_sim']

    # Subsumption
    sub_tp, sub_total, sub_fp, sub_fp_total = 0, 0, 0, 0
    for hyper, hypo in HYPERNYM_PAIRS:
        if hyper in concept_primes and hypo in concept_primes:
            sub_total += 1
            if concept_primes[hypo] % concept_primes[hyper] == 0:
                sub_tp += 1
    for a, b in UNRELATED_PAIRS[:10]:
        if a in concept_primes and b in concept_primes:
            sub_fp_total += 1
            if concept_primes[b] % concept_primes[a] == 0:
                sub_fp += 1

    results['subsumption_recall'] = sub_tp / sub_total if sub_total > 0 else 0.0
    results['subsumption_fpr'] = sub_fp / sub_fp_total if sub_fp_total > 0 else 0.0

    # Analogy verification
    all_vocab = set(CONCEPTS + EXTRA_CONCEPTS)
    vocab_primes = {c: concept_primes[c] for c in all_vocab if c in concept_primes}

    verif_correct, verif_total = 0, 0
    for a, b, c, d in ANALOGIES:
        if any(x not in concept_primes for x in [a, b, c, d]):
            continue
        verif_total += 1
        phi_a, phi_b, phi_c, phi_d = [concept_primes[x] for x in [a, b, c, d]]
        shared = math.gcd(phi_a, phi_b)
        transform = phi_b // shared if shared > 0 else phi_b
        target = (phi_c * transform) // math.gcd(phi_c, transform)

        exclude = {a, b, c}
        candidates = [(w, p) for w, p in vocab_primes.items() if w not in exclude]
        d_sim = jaccard_factors(target, phi_d)
        median_sim = np.median([jaccard_factors(target, p) for _, p in candidates])
        if d_sim > median_sim:
            verif_correct += 1

    results['analogy_verification'] = verif_correct / verif_total if verif_total > 0 else 0.0
    results['analogy_total'] = verif_total

    return results


# ============================================================
# Model loading and concept projection
# ============================================================

def load_model(checkpoint_path, device):
    """Load GPT-2+Triadic model from checkpoint."""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = ckpt.get('gpt2_model_name', 'gpt2')
    n_bits = ckpt.get('n_triadic_bits', 64)

    print(f"  Loading base model: {model_name}")
    gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2TriadicModel(gpt2, n_triadic_bits=n_bits)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"  Loaded checkpoint: step {ckpt.get('step', '?')}, "
          f"phase: {ckpt.get('phase', '?')}, "
          f"lang_loss: {ckpt.get('lang_loss', '?')}")

    return model, tokenizer, n_bits


def project_concepts(model, tokenizer, n_bits, device):
    """Project all concepts through the model and return prime composites."""
    mapper = PrimeMapper(n_bits)
    all_concepts = list(set(CONCEPTS + EXTRA_CONCEPTS))
    concept_primes = {}

    # Warmup
    dummy = tokenizer.encode("warmup")
    x = torch.tensor([dummy], dtype=torch.long, device=device)
    with torch.no_grad():
        model(x)

    start = time.perf_counter()
    for concept in all_concepts:
        ids = tokenizer.encode(concept)
        if not ids:
            continue
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            _, triadic_proj, _ = model(x)
        # Average across tokens, then map to prime
        proj = triadic_proj[0].mean(dim=0).cpu().numpy()
        concept_primes[concept] = mapper.map(proj)
    elapsed = time.perf_counter() - start

    ms_per_concept = elapsed / len(all_concepts) * 1000
    return concept_primes, ms_per_concept


# ============================================================
# Comparison table
# ============================================================

# Baselines from our previous experiments (hardcoded for comparison)
BASELINE_TRIADICGPT = {
    'method': 'TriadicGPT (from-scratch)',
    'bit_entropy': 0.680,
    'unique_pct': 100.0,
    'semantic_gap': 0.020,
    'analogy_verification': 0.667,
    'subsumption_recall': 0.0,
    'ms_per_concept': 5.23,
}

BASELINE_ENGINE_PCA = {
    'method': 'Engine PCA',
    'bit_entropy': 0.947,
    'unique_pct': 100.0,
    'semantic_gap': 0.136,
    'analogy_verification': 0.917,
    'subsumption_recall': 0.0,
    'ms_per_concept': 0.92,
}


def print_comparison(gpt2_results, baselines):
    """Print comparison table."""
    all_results = [gpt2_results] + baselines
    labels = [r['method'] for r in all_results]
    short = [l[:18] for l in labels]

    print()
    print("=" * 86)
    print("  EXPERIMENT 10 - Comparison Table")
    print("=" * 86)

    header = f"  {'Metric':<22}" + "".join(f" {s:>18}" for s in short)
    print(header)
    print("  " + "-" * (22 + 18 * len(short)))

    rows = [
        ("Bit Entropy", 'bit_entropy', '.3f'),
        ("Unique Sigs (%)", 'unique_pct', '.1f'),
        ("Semantic Gap", 'semantic_gap', '+.3f'),
        ("Analogy Verif", 'analogy_verification', '.1%'),
        ("Subsumption Recall", 'subsumption_recall', '.1%'),
        ("Speed (ms/concept)", 'ms_per_concept', '.2f'),
    ]

    for label, key, fmt in rows:
        vals = []
        for r in all_results:
            v = r.get(key, 0)
            vals.append(f"{v:{fmt}}")
        print(f"  {label:<22}" + "".join(f" {v:>18}" for v in vals))

    print("=" * 86)

    # Key comparison
    gpt2_gap = gpt2_results['semantic_gap']
    scratch_gap = BASELINE_TRIADICGPT['semantic_gap']
    engine_gap = BASELINE_ENGINE_PCA['semantic_gap']

    print()
    print("  Key findings:")
    if gpt2_gap > scratch_gap:
        improvement = gpt2_gap / scratch_gap if scratch_gap > 0 else float('inf')
        print(f"    GPT-2+Triadic gap ({gpt2_gap:+.3f}) > TriadicGPT from-scratch ({scratch_gap:+.3f})")
        print(f"    -> {improvement:.1f}x improvement from pre-trained embeddings")
    else:
        print(f"    GPT-2+Triadic gap ({gpt2_gap:+.3f}) <= TriadicGPT from-scratch ({scratch_gap:+.3f})")
        print(f"    -> Pre-trained embeddings did NOT help. Investigate triadic loss formulation.")

    if gpt2_gap > engine_gap:
        print(f"    GPT-2+Triadic EXCEEDS Engine PCA ({engine_gap:+.3f}) -- breakthrough result!")
    elif gpt2_gap > engine_gap * 0.5:
        print(f"    GPT-2+Triadic approaches Engine PCA ({engine_gap:+.3f}) -- significant progress.")
    else:
        print(f"    GPT-2+Triadic still below Engine PCA ({engine_gap:+.3f})")
        print(f"    -> Gap closed: {(gpt2_gap - scratch_gap) / (engine_gap - scratch_gap) * 100:.0f}% of the way")

    print("=" * 86)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Experiment 10 Evaluation')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--output', default='experiment10/results/experiment10_results.json')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print()
    print("=" * 86)
    print("  EXPERIMENT 10 - Evaluation: GPT-2 + Triadic Head")
    print("=" * 86)

    # Load model
    model, tokenizer, n_bits = load_model(args.checkpoint, device)

    # Project concepts
    print(f"\n  Projecting {len(set(CONCEPTS + EXTRA_CONCEPTS))} concepts...")
    concept_primes, ms_per_concept = project_concepts(model, tokenizer, n_bits, device)
    print(f"  Done: {ms_per_concept:.2f} ms/concept")

    # Compute metrics
    print("  Computing metrics...")
    metrics = compute_metrics(concept_primes, n_bits)
    metrics['ms_per_concept'] = ms_per_concept
    metrics['method'] = 'GPT-2+Triadic (transfer)'
    metrics['n_bits'] = n_bits
    metrics['n_params'] = model.num_params()
    metrics['n_params_trainable'] = model.num_params(trainable_only=True)

    print(f"\n  Results:")
    print(f"    Bit Entropy:          {metrics['bit_entropy']:.3f}")
    print(f"    Unique Signatures:    {metrics['unique_pct']:.1f}%")
    print(f"    Semantic Gap:         {metrics['semantic_gap']:+.3f}")
    print(f"    Analogy Verification: {metrics['analogy_verification']:.1%}")
    print(f"    Subsumption Recall:   {metrics['subsumption_recall']:.1%}")
    print(f"    Speed:                {metrics['ms_per_concept']:.2f} ms/concept")

    # Comparison table
    print_comparison(metrics, [BASELINE_TRIADICGPT, BASELINE_ENGINE_PCA])

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\n  Results saved: {args.output}")

    # Generate a few text samples
    print("\n  Text generation samples:")
    prompts = ["The king", "A happy dog", "Once upon a time"]
    for prompt in prompts:
        ids = tokenizer.encode(prompt)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=50, temperature=0.7)
        text = tokenizer.decode(output[0].cpu().tolist()).split('\n')[0][:200]
        print(f"    > {text}")


if __name__ == '__main__':
    main()
