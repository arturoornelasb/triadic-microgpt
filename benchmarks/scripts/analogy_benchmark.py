"""
Analogy Benchmark — Semantic Analogy via Prime Algebra.

Tests whether prime-factor representations support analogical reasoning:
  "King is to Queen as Man is to Woman"
  Phi(?) = lcm(Phi(King), Phi(Woman)) / gcd(Phi(King), Phi(Man))

The paper reports 2-10% top-1 accuracy for the parent library. MicroGPT's
end-to-end training should match or exceed this.

Metrics:
  - Top-1 Accuracy: correct concept retrieved by prime algebra
  - Top-5 Accuracy: correct concept in top-5 by prime similarity
  - Verification Accuracy: given the 4 concepts, does the algebraic
    relationship hold (looser criterion using similarity ranking)?

Usage:
  python benchmarks/scripts/analogy_benchmark.py \
    --model checkpoints/torch_run15_strongalign/model_L12_D512_B64_best.pt \
    --tokenizer checkpoints/torch_runXL/tokenizer.json \
    --version v1.4-strongalign
"""

import os
import sys
import json
import math
import argparse
from datetime import date

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.evaluate import load_model
from src.triadic import PrimeMapper, TriadicValidator, prime_factors


# ============================================================
# Analogy test sets
# Format: (A, B, C, D) where "A is to B as C is to D"
# ============================================================

ANALOGIES = [
    # Gender
    ("king", "queen", "man", "woman"),
    ("king", "queen", "boy", "girl"),
    ("king", "queen", "prince", "princess"),
    ("king", "queen", "father", "mother"),
    ("king", "queen", "brother", "sister"),
    ("man", "woman", "boy", "girl"),
    ("man", "woman", "father", "mother"),
    ("man", "woman", "brother", "sister"),
    ("father", "mother", "brother", "sister"),
    ("prince", "princess", "boy", "girl"),
    # Opposites
    ("happy", "sad", "love", "hate"),
    ("happy", "sad", "peace", "war"),
    ("happy", "sad", "hope", "fear"),
    ("big", "small", "fast", "slow"),
    ("big", "small", "old", "young"),
    ("fast", "slow", "old", "young"),
    ("fire", "water", "sun", "moon"),
    ("morning", "night", "summer", "winter"),
    ("friend", "enemy", "peace", "war"),
    ("love", "hate", "peace", "war"),
    # Domain transfer
    ("dog", "cat", "horse", "cow"),
    ("dog", "cat", "bird", "fish"),
    ("doctor", "hospital", "teacher", "school"),
    ("king", "castle", "judge", "church"),
    # Action pairs
    ("run", "walk", "swim", "float"),
    ("jump", "fall", "climb", "sleep"),
]

# Invalid analogies: semantically incoherent pairs (negative controls)
# If the model scores these similarly to valid analogies, the algebraic
# analogy mechanism is not capturing compositional structure.
# Kill criterion K4: if gap(valid - invalid) < 10pp, degrade claim.
INVALID_ANALOGIES = [
    ("king", "dog", "queen", "cat"),
    ("hot", "table", "cold", "chair"),
    ("happy", "river", "sad", "mountain"),
    ("mother", "blue", "father", "green"),
    ("sun", "bread", "moon", "milk"),
    ("fire", "school", "water", "hospital"),
    ("bird", "lamp", "fish", "bed"),
    ("love", "car", "hate", "tree"),
    ("big", "church", "small", "garden"),
    ("dog", "morning", "cat", "night"),
    ("run", "cake", "walk", "apple"),
    ("king", "rain", "queen", "snow"),
    ("prince", "ocean", "princess", "forest"),
    ("happy", "door", "sad", "window"),
    ("doctor", "fish", "nurse", "bird"),
    ("fast", "moon", "slow", "star"),
    ("brother", "flower", "sister", "cloud"),
    ("old", "swim", "young", "fly"),
    ("friend", "lamp", "enemy", "chair"),
    ("morning", "dog", "night", "cat"),
    ("fire", "pen", "water", "book"),
    ("love", "table", "hate", "door"),
    ("summer", "judge", "winter", "doctor"),
    ("big", "music", "small", "dance"),
    ("king", "milk", "queen", "bread"),
]

# Vocabulary pool for retrieval (must include all D answers + distractors)
VOCAB_POOL = [
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
    "castle", "float",
]


def compute_projections(model, tokenizer, concepts, device):
    """Get triadic projections and prime composites for all concepts."""
    results = {}
    mapper = PrimeMapper(model.config.n_triadic_bits)

    for concept in concepts:
        if concept in results:
            continue
        ids = tokenizer.encode(concept, add_special=False)
        if not ids:
            continue

        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            _, triadic_proj, _ = model(x)

        proj = triadic_proj[0].mean(dim=0).cpu().numpy()
        prime = mapper.map(proj)
        results[concept] = {
            'projection': proj,
            'prime': prime,
        }

    return results


def analogy_by_prime_algebra(phi_a, phi_b, phi_c):
    """
    Compute the analogy target: A:B :: C:?

    Method: ? = lcm(Phi(B), Phi(C)) / gcd(Phi(A), lcm(Phi(B), Phi(C)))

    Intuition: B has the features of A transformed. C needs the same transform.
    The "transform" is the features B has that A doesn't, applied to C.

    Simplified: ? = Phi(C) * (Phi(B) / gcd(Phi(A), Phi(B)))
    i.e., take C's features and add the B-specific features (not shared with A).
    """
    # Features unique to B (the "transform")
    shared_ab = math.gcd(phi_a, phi_b)
    transform = phi_b // shared_ab if shared_ab > 0 else phi_b

    # Apply transform to C
    target = (phi_c * transform) // math.gcd(phi_c, transform)
    return target


def similarity_by_factors(phi_a, phi_b):
    """Jaccard similarity of prime factor sets."""
    fa = set(prime_factors(phi_a))
    fb = set(prime_factors(phi_b))
    if not fa and not fb:
        return 1.0
    union = fa | fb
    return len(fa & fb) / len(union) if union else 0.0


def evaluate_analogies(concept_data, analogies, vocab_pool):
    """Evaluate analogies using prime algebra."""
    results = []
    top1_correct = 0
    top5_correct = 0
    verification_correct = 0
    total = 0

    # Pre-compute vocab primes
    vocab_primes = {}
    for w in vocab_pool:
        if w in concept_data:
            vocab_primes[w] = concept_data[w]['prime']

    for a, b, c, d in analogies:
        if any(x not in concept_data for x in [a, b, c, d]):
            continue
        if d not in vocab_primes:
            continue

        total += 1
        phi_a = concept_data[a]['prime']
        phi_b = concept_data[b]['prime']
        phi_c = concept_data[c]['prime']
        phi_d = concept_data[d]['prime']

        # Compute algebraic target
        target = analogy_by_prime_algebra(phi_a, phi_b, phi_c)

        # Rank all vocab words by similarity to target
        # Exclude A, B, C from candidates
        exclude = {a, b, c}
        candidates = [(w, p) for w, p in vocab_primes.items() if w not in exclude]
        ranked = sorted(candidates, key=lambda wp: -similarity_by_factors(target, wp[1]))

        top_words = [w for w, _ in ranked[:5]]
        top1 = ranked[0][0] if ranked else None

        is_top1 = (top1 == d)
        is_top5 = (d in top_words)

        if is_top1:
            top1_correct += 1
        if is_top5:
            top5_correct += 1

        # Verification: is D more similar to target than a random unrelated word?
        d_sim = similarity_by_factors(target, phi_d)
        median_sim = np.median([similarity_by_factors(target, p) for _, p in candidates])
        if d_sim > median_sim:
            verification_correct += 1

        results.append({
            'analogy': f"{a}:{b}::{c}:{d}",
            'target_prime': target,
            'correct_answer': d,
            'top1': top1,
            'top5': top_words,
            'is_top1': is_top1,
            'is_top5': is_top5,
            'd_similarity': d_sim,
            'median_similarity': float(median_sim),
            'above_median': d_sim > median_sim,
        })

    return {
        'total': total,
        'top1_correct': top1_correct,
        'top5_correct': top5_correct,
        'verification_correct': verification_correct,
        'top1_accuracy': top1_correct / total if total > 0 else 0.0,
        'top5_accuracy': top5_correct / total if total > 0 else 0.0,
        'verification_accuracy': verification_correct / total if total > 0 else 0.0,
        'details': results,
    }


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print()
    print("=" * 68)
    print("  ANALOGY BENCHMARK — Prime Algebra Reasoning")
    print("=" * 68)
    print(f"  Model: {args.model}")
    print()

    model, tokenizer, config = load_model(args.model, args.tokenizer, device)
    print(f"  Config: {config.n_layer}L/{config.n_embd}D/{config.n_head}H/{config.n_triadic_bits}bits")

    # Collect all concepts
    all_concepts = set(VOCAB_POOL)
    for a, b, c, d in ANALOGIES:
        all_concepts.update([a, b, c, d])

    print(f"  Vocabulary pool: {len(all_concepts)} concepts")
    print(f"  Analogy tests: {len(ANALOGIES)}")
    print()

    # Compute projections
    print("[1/2] Computing triadic projections...")
    concept_data = compute_projections(model, tokenizer, all_concepts, device)
    print(f"  Encoded: {len(concept_data)}/{len(all_concepts)}")

    # Evaluate valid analogies
    print()
    print("[2/3] Evaluating valid analogies...")
    eval_result = evaluate_analogies(concept_data, ANALOGIES, VOCAB_POOL)

    print(f"  Valid analogies tested: {eval_result['total']}")
    print()
    print(f"  Top-1 Accuracy:        {eval_result['top1_accuracy']:.1%} ({eval_result['top1_correct']}/{eval_result['total']})")
    print(f"  Top-5 Accuracy:        {eval_result['top5_accuracy']:.1%} ({eval_result['top5_correct']}/{eval_result['total']})")
    print(f"  Verification (>median): {eval_result['verification_accuracy']:.1%} ({eval_result['verification_correct']}/{eval_result['total']})")

    # Evaluate invalid analogies (negative controls)
    print()
    print("[3/3] Evaluating INVALID analogies (negative controls)...")
    eval_invalid = evaluate_analogies(concept_data, INVALID_ANALOGIES, VOCAB_POOL)

    print(f"  Invalid analogies tested: {eval_invalid['total']}")
    print()
    print(f"  Top-1 Accuracy:        {eval_invalid['top1_accuracy']:.1%} ({eval_invalid['top1_correct']}/{eval_invalid['total']})")
    print(f"  Top-5 Accuracy:        {eval_invalid['top5_accuracy']:.1%} ({eval_invalid['top5_correct']}/{eval_invalid['total']})")
    print(f"  Verification (>median): {eval_invalid['verification_accuracy']:.1%} ({eval_invalid['verification_correct']}/{eval_invalid['total']})")

    # Compositionality gap (valid - invalid)
    gap_verif = (eval_result['verification_accuracy'] - eval_invalid['verification_accuracy']) * 100
    gap_top5 = (eval_result['top5_accuracy'] - eval_invalid['top5_accuracy']) * 100
    print()
    print(f"  COMPOSITIONALITY GAP:")
    print(f"    Verification gap: {gap_verif:+.1f}pp  {'PASS' if gap_verif >= 10 else 'FAIL (kill K4)'}")
    print(f"    Top-5 gap:        {gap_top5:+.1f}pp")

    # Show examples
    print()
    print("  Sample analogies:")
    for d in eval_result['details'][:10]:
        mark = "OK" if d['is_top1'] else ("top5" if d['is_top5'] else "MISS")
        print(f"    {d['analogy']:<30} predicted={d['top1']:<12} [{mark}]  sim={d['d_similarity']:.2f}")

    # Save results
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    results_dir = os.path.join(project_root, 'benchmarks', 'results')
    os.makedirs(results_dir, exist_ok=True)

    version = args.version
    today = date.today().isoformat()

    result = {
        "benchmark": "analogy",
        "version": version,
        "date": today,
        "model_checkpoint": args.model,
        "model_config": f"{config.n_layer}L/{config.n_embd}D/{config.n_head}H/{config.n_triadic_bits}bits",
        "n_valid_analogies": eval_result['total'],
        "n_invalid_analogies": eval_invalid['total'],
        "n_vocab_pool": len(VOCAB_POOL),
        "valid_metrics": {
            "top1_accuracy": eval_result['top1_accuracy'],
            "top5_accuracy": eval_result['top5_accuracy'],
            "verification_accuracy": eval_result['verification_accuracy'],
            "top1_correct": eval_result['top1_correct'],
            "top5_correct": eval_result['top5_correct'],
        },
        "invalid_metrics": {
            "top1_accuracy": eval_invalid['top1_accuracy'],
            "top5_accuracy": eval_invalid['top5_accuracy'],
            "verification_accuracy": eval_invalid['verification_accuracy'],
            "top1_correct": eval_invalid['top1_correct'],
            "top5_correct": eval_invalid['top5_correct'],
        },
        "compositionality_gap_pp": {
            "verification": gap_verif,
            "top5": gap_top5,
            "kill_k4": gap_verif < 10,
        },
        "valid_details": eval_result['details'],
        "invalid_details": eval_invalid['details'],
    }

    result_path = os.path.join(results_dir, f"{version}_analogy_{today}.json")
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Results saved: {result_path}")

    # Verdict
    print()
    print("=" * 68)
    print(f"  VALID ANALOGIES:")
    print(f"    Top-1:         {eval_result['top1_accuracy']:.1%}  (paper baseline: 2-10%)  {'PASS' if eval_result['top1_accuracy'] >= 0.02 else 'BELOW'}")
    print(f"    Top-5:         {eval_result['top5_accuracy']:.1%}  (target: > 25%)  {'PASS' if eval_result['top5_accuracy'] >= 0.25 else 'BELOW'}")
    print(f"    Verification:  {eval_result['verification_accuracy']:.1%}  (target: > 50%)  {'PASS' if eval_result['verification_accuracy'] >= 0.50 else 'BELOW'}")
    print(f"  NEGATIVE CONTROLS:")
    print(f"    Invalid verif: {eval_invalid['verification_accuracy']:.1%}  (should be LOW)")
    print(f"  KILL CRITERION K4:")
    print(f"    Gap:           {gap_verif:+.1f}pp  (threshold: >= 10pp)  {'PASS' if gap_verif >= 10 else 'FAIL'}")
    print("=" * 68)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analogy Benchmark')
    parser.add_argument('--model', required=True)
    parser.add_argument('--tokenizer', default=None)
    parser.add_argument('--version', default='v1.4-strongalign')
    args = parser.parse_args()

    if args.tokenizer is None:
        args.tokenizer = os.path.join(os.path.dirname(args.model), 'tokenizer.json')

    main(args)
