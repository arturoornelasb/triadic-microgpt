"""
Engine Comparison — TriadicGPT (end-to-end) vs Triadic Engine (post-hoc).

Experiment 9: Full Table 7 comparison across all projection methods.
  1. TriadicGPT: single forward pass (end-to-end triadic bits)
  2. Engine (Random LSH): random hyperplanes
  3. Engine (PCA): principal component directions
  4. Engine (Consensus): multi-seed voting, stable primes only
  5. Engine (Contrastive): hypernym-trained hyperplanes

Metrics compared:
  - Bit entropy
  - Unique signatures
  - Semantic ordering (related vs unrelated similarity gap)
  - Analogy verification rate
  - Subsumption recall & FPR
  - Inference speed (ms per concept)

Usage:
  python benchmarks/scripts/engine_comparison.py \
    --model checkpoints/torch_run15_strongalign/model_L12_D512_B64_best.pt \
    --tokenizer checkpoints/torch_runXL/tokenizer.json \
    --engine-path ../Triadic-Neurosymbolic-Engine
"""

import os
import sys
import json
import math
import time
import argparse
from datetime import date
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.evaluate import load_model
from src.triadic import PrimeMapper, prime_factors


# ============================================================
# Shared concept sets
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

# Extra concepts needed for analogies/subsumption
EXTRA_CONCEPTS = ["animal", "hospital", "castle"]


# ============================================================
# Metric computation (shared across methods)
# ============================================================

def compute_metrics(concept_primes, n_bits, label=""):
    """Compute all metrics from a dict of {concept: prime_composite}."""
    results = {}

    # 1. Bit entropy — reconstruct bits from primes
    import sympy
    plane_primes = [sympy.prime(i + 1) for i in range(n_bits)]
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

    # 2. Unique signatures
    sigs = set(tuple(row) for row in bit_matrix.astype(int))
    results['unique_sigs'] = len(sigs)
    results['total_concepts'] = len(valid_concepts)
    results['unique_pct'] = len(sigs) / len(valid_concepts) * 100

    # 3. Semantic ordering
    def jaccard_factors(a, b):
        fa = set(prime_factors(a))
        fb = set(prime_factors(b))
        if not fa and not fb:
            return 1.0
        union = fa | fb
        return len(fa & fb) / len(union) if union else 0.0

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

    # 4. Subsumption
    sub_tp = 0
    sub_total = 0
    sub_fp = 0
    sub_fp_total = 0

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

    # 5. Analogy verification
    all_vocab = set(CONCEPTS + EXTRA_CONCEPTS)
    vocab_primes = {c: concept_primes[c] for c in all_vocab if c in concept_primes}

    verif_correct = 0
    verif_total = 0
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
# MicroGPT method
# ============================================================

def run_microgpt(model_path, tokenizer_path, device):
    """Run TriadicGPT end-to-end: single forward pass."""
    model, tokenizer, config = load_model(model_path, tokenizer_path, device)
    n_bits = config.n_triadic_bits
    mapper = PrimeMapper(n_bits)

    all_concepts = list(set(CONCEPTS + EXTRA_CONCEPTS))
    concept_primes = {}

    # Warmup
    dummy = tokenizer.encode("hello", add_special=False)
    x = torch.tensor([dummy], dtype=torch.long, device=device)
    with torch.no_grad():
        model(x)

    # Timed run
    start = time.perf_counter()
    for concept in all_concepts:
        ids = tokenizer.encode(concept, add_special=False)
        if not ids:
            continue
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            _, triadic_proj, _ = model(x)
        proj = triadic_proj[0].mean(dim=0).cpu().numpy()
        concept_primes[concept] = mapper.map(proj)
    elapsed = time.perf_counter() - start

    ms_per_concept = elapsed / len(all_concepts) * 1000
    metrics = compute_metrics(concept_primes, n_bits)
    metrics['ms_per_concept'] = ms_per_concept
    metrics['method'] = 'TriadicGPT (end-to-end)'
    metrics['n_bits'] = n_bits
    metrics['n_params'] = model.num_params()

    del model
    torch.cuda.empty_cache()
    return metrics


# ============================================================
# Engine method (post-hoc)
# ============================================================

def run_engine(engine_path, projection="pca", n_bits=64):
    """Run Triadic Engine: embed with sentence-transformers + LSH projection."""
    # Add Engine to path
    engine_src = os.path.join(engine_path, 'src')
    if engine_src not in sys.path:
        sys.path.insert(0, engine_src)

    from neurosym.encoder import ContinuousEncoder, DiscreteMapper

    all_concepts = list(set(CONCEPTS + EXTRA_CONCEPTS))

    # Warmup encoder
    encoder = ContinuousEncoder('all-MiniLM-L6-v2')
    _ = encoder.encode(["hello"])

    # Timed run: embed
    start_embed = time.perf_counter()
    embeddings = encoder.encode(all_concepts)
    elapsed_embed = time.perf_counter() - start_embed

    # Timed run: project
    hypernym_pairs = HYPERNYM_PAIRS if projection == "contrastive" else None
    mapper = DiscreteMapper(
        n_bits=n_bits,
        seed=42,
        projection=projection,
        hypernym_pairs=hypernym_pairs,
    )

    start_project = time.perf_counter()
    concept_primes = mapper.fit_transform(all_concepts, embeddings)
    elapsed_project = time.perf_counter() - start_project

    elapsed_total = elapsed_embed + elapsed_project
    ms_per_concept = elapsed_total / len(all_concepts) * 1000

    metrics = compute_metrics(concept_primes, n_bits)
    metrics['ms_per_concept'] = ms_per_concept
    metrics['ms_embed'] = elapsed_embed / len(all_concepts) * 1000
    metrics['ms_project'] = elapsed_project / len(all_concepts) * 1000
    metrics['method'] = f'Engine ({projection})'
    metrics['n_bits'] = n_bits
    metrics['encoder'] = 'all-MiniLM-L6-v2'

    return metrics


# ============================================================
# Main
# ============================================================

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print()
    print("=" * 86)
    print("  EXPERIMENT 9 — Full Projection Comparison (Table 7)")
    print("  TriadicGPT (end-to-end) vs Engine (Random / PCA / Consensus / Contrastive)")
    print("=" * 86)
    print()

    results = []

    def log_result(r):
        print(f"  Entropy: {r['bit_entropy']:.3f}  Unique: {r['unique_pct']:.1f}%  "
              f"Gap: {r['semantic_gap']:+.3f}  Verif: {r['analogy_verification']:.1%}  "
              f"Speed: {r['ms_per_concept']:.2f}ms")

    # 1. TriadicGPT
    print("[1/5] TriadicGPT (end-to-end, single forward pass)...")
    mgpt = run_microgpt(args.model, args.tokenizer, device)
    results.append(mgpt)
    log_result(mgpt)

    # 2. Engine (Random)
    print()
    print("[2/5] Engine (Random LSH, post-hoc)...")
    engine_rand = run_engine(args.engine_path, projection="random", n_bits=64)
    results.append(engine_rand)
    log_result(engine_rand)

    # 3. Engine (PCA)
    print()
    print("[3/5] Engine (PCA projection, post-hoc)...")
    engine_pca = run_engine(args.engine_path, projection="pca", n_bits=64)
    results.append(engine_pca)
    log_result(engine_pca)

    # 4. Engine (Consensus)
    print()
    print("[4/5] Engine (Consensus, multi-seed voting)...")
    engine_cons = run_engine(args.engine_path, projection="consensus", n_bits=64)
    results.append(engine_cons)
    log_result(engine_cons)

    # 5. Engine (Contrastive)
    print()
    print("[5/5] Engine (Contrastive, hypernym-trained)...")
    engine_ctr = run_engine(args.engine_path, projection="contrastive", n_bits=64)
    results.append(engine_ctr)
    log_result(engine_ctr)

    # Comparison table
    print()
    print("=" * 86)
    print("  TABLE 7 — Full Projection Comparison (Experiment 9)")
    print("=" * 86)

    method_labels = [r['method'] for r in results]
    short_labels = ['TriadicGPT', 'Random', 'PCA', 'Consensus', 'Contrastive']

    header = f"  {'Metric':<22}" + "".join(f" {s:>13}" for s in short_labels)
    print(header)
    print("  " + "-" * (22 + 13 * len(short_labels)))

    rows = [
        ("Bit Entropy", 'bit_entropy', '.3f'),
        ("Unique Sigs (%)", 'unique_pct', '.1f'),
        ("Semantic Gap", 'semantic_gap', '+.3f'),
        ("Related Sim", 'mean_related_sim', '.3f'),
        ("Unrelated Sim", 'mean_unrelated_sim', '.3f'),
        ("Subsumption Recall", 'subsumption_recall', '.1%'),
        ("Subsumption FPR", 'subsumption_fpr', '.1%'),
        ("Analogy Verif", 'analogy_verification', '.1%'),
        ("Speed (ms/concept)", 'ms_per_concept', '.2f'),
    ]

    for label, key, fmt in rows:
        vals = []
        for r in results:
            v = r.get(key, 0)
            vals.append(f"{v:{fmt}}")
        print(f"  {label:<22}" + "".join(f" {v:>13}" for v in vals))

    # Key observations
    print()
    print("  Key observations:")
    print(f"    TriadicGPT: self-contained, single forward pass, no external embeddings")
    print(f"    Engine modes all use sentence-transformers ({engine_pca.get('encoder', 'N/A')})")

    sub_recalls = [(r['method'], r.get('subsumption_recall', 0)) for r in results]
    best_sub = max(sub_recalls, key=lambda x: x[1])
    if best_sub[1] > 0:
        print(f"    Best subsumption: {best_sub[0]} ({best_sub[1]:.1%})")
        if 'ontrastive' in best_sub[0]:
            print(f"    NOTE: Contrastive uses hypernym pairs for training — tests on same pairs")

    gaps = [(r['method'], r.get('semantic_gap', 0)) for r in results]
    best_gap = max(gaps, key=lambda x: x[1])
    print(f"    Best semantic gap: {best_gap[0]} ({best_gap[1]:+.3f})")

    print("=" * 86)

    # Save results
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    results_dir = os.path.join(project_root, 'benchmarks', 'results')
    os.makedirs(results_dir, exist_ok=True)

    today = date.today().isoformat()
    version = args.version

    output = {
        "benchmark": "engine_comparison",
        "version": version,
        "date": today,
        "model_checkpoint": args.model,
        "engine_path": args.engine_path,
        "n_concepts": len(CONCEPTS),
        "results": results,
    }

    result_path = os.path.join(results_dir, f"{version}_engine_comparison_{today}.json")
    with open(result_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved: {result_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Engine Comparison Benchmark')
    parser.add_argument('--model', required=True, help='TriadicGPT checkpoint')
    parser.add_argument('--tokenizer', default=None)
    parser.add_argument('--engine-path', default='../Triadic-Neurosymbolic-Engine',
                        help='Path to Triadic-Neurosymbolic-Engine repo')
    parser.add_argument('--version', default='v4.0-experiment9')
    args = parser.parse_args()

    if args.tokenizer is None:
        args.tokenizer = os.path.join(os.path.dirname(args.model), 'tokenizer.json')

    main(args)
