"""
Subsumption Benchmark — Taxonomic Consistency via Prime Divisibility.

Tests whether the triadic head preserves WordNet-like taxonomy:
  If "dog" is-a "animal", then Phi(animal) | Phi(dog)  (subsumption)

This is the CORE algebraic property that distinguishes prime representations
from cosine similarity: subsumption is exact, verifiable, and compositional.

Metrics:
  - Subsumption Recall: % of true hypernym pairs where Phi(hyper) | Phi(hypo)
  - Subsumption FPR: % of unrelated pairs falsely showing subsumption
  - Taxonomic F1: harmonic mean of precision and recall
  - Jaccard similarity: shared factors / total factors for related vs unrelated

Usage:
  python benchmarks/scripts/subsumption_benchmark.py \
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
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.evaluate import load_model
from src.triadic import PrimeMapper, TriadicValidator


# ============================================================
# Curated WordNet-like taxonomy pairs
# Format: (hypernym, hyponym) — "hyponym is-a hypernym"
# All words must be in the model's vocabulary (common English)
# ============================================================

HYPERNYM_PAIRS = [
    # Animals
    ("animal", "dog"), ("animal", "cat"), ("animal", "bird"),
    ("animal", "fish"), ("animal", "horse"), ("animal", "cow"),
    ("animal", "pig"), ("animal", "sheep"),
    # People
    ("person", "boy"), ("person", "girl"), ("person", "man"),
    ("person", "woman"), ("person", "king"), ("person", "queen"),
    ("person", "prince"), ("person", "princess"),
    # Professions (person -> profession)
    ("person", "doctor"), ("person", "nurse"), ("person", "teacher"),
    ("person", "student"), ("person", "judge"), ("person", "scientist"),
    # Family
    ("person", "mother"), ("person", "father"),
    ("person", "brother"), ("person", "sister"),
    # Royalty chain
    ("royalty", "king"), ("royalty", "queen"),
    ("royalty", "prince"), ("royalty", "princess"),
    # Emotions
    ("feeling", "happy"), ("feeling", "sad"), ("feeling", "angry"),
    ("feeling", "afraid"), ("feeling", "brave"),
    ("feeling", "love"), ("feeling", "hate"), ("feeling", "fear"),
    ("feeling", "hope"), ("feeling", "joy"), ("feeling", "pain"),
    # Nature
    ("nature", "fire"), ("nature", "water"), ("nature", "earth"),
    ("nature", "air"), ("nature", "sun"), ("nature", "moon"),
    ("nature", "star"), ("nature", "cloud"),
    ("nature", "river"), ("nature", "mountain"), ("nature", "ocean"),
    ("nature", "forest"), ("nature", "rain"), ("nature", "snow"),
    # Food
    ("food", "bread"), ("food", "milk"), ("food", "apple"), ("food", "cake"),
    # Furniture
    ("furniture", "table"), ("furniture", "chair"),
    ("furniture", "bed"), ("furniture", "lamp"),
    # Color (attribute chain)
    ("color", "red"), ("color", "blue"), ("color", "green"),
    # Place
    ("place", "city"), ("place", "village"), ("place", "school"),
    ("place", "church"), ("place", "hospital"),
    ("place", "garden"), ("place", "park"),
    ("place", "house"),
    # Actions
    ("action", "run"), ("action", "walk"), ("action", "swim"),
    ("action", "fly"), ("action", "jump"), ("action", "climb"),
    ("action", "fall"), ("action", "sleep"),
    # Weather/Time
    ("time", "morning"), ("time", "night"),
    ("time", "summer"), ("time", "winter"), ("time", "spring"),
]

# Unrelated pairs — concepts with NO taxonomic relationship
# Used to measure false positive rate
UNRELATED_PAIRS = [
    ("dog", "table"), ("cat", "bread"), ("king", "river"),
    ("happy", "chair"), ("fire", "nurse"), ("bird", "cake"),
    ("mother", "cloud"), ("red", "horse"), ("school", "pig"),
    ("doctor", "rain"), ("boy", "lamp"), ("ocean", "angry"),
    ("tree", "judge"), ("music", "cow"), ("book", "sun"),
    ("door", "fish"), ("window", "sad"), ("pen", "moon"),
    ("car", "brave"), ("flower", "milk"), ("dream", "bed"),
    ("game", "fear"), ("song", "village"), ("dance", "snow"),
    ("magic", "apple"), ("war", "green"), ("peace", "sheep"),
    ("story", "blue"), ("park", "hate"), ("church", "swim"),
    ("garden", "fall"), ("hospital", "fly"), ("city", "sleep"),
    ("morning", "cake"), ("night", "chair"), ("summer", "lamp"),
    ("winter", "nurse"), ("spring", "judge"), ("run", "bread"),
    ("walk", "star"), ("jump", "milk"), ("climb", "red"),
    ("pain", "table"), ("hope", "dog"), ("joy", "river"),
    ("love", "mountain"), ("fear", "apple"), ("hate", "bird"),
    ("queen", "rain"), ("prince", "bread"), ("princess", "cow"),
    ("father", "cloud"),
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
        bits = mapper.get_bits(proj)
        results[concept] = {
            'projection': proj,
            'prime': prime,
            'bits': bits,
            'n_active': sum(bits),
        }

    return results


def evaluate_subsumption(concept_data, hypernym_pairs, unrelated_pairs):
    """Evaluate subsumption accuracy on hypernym and unrelated pairs."""
    validator = TriadicValidator()

    # True pairs (hypernym, hyponym) — expect Phi(hyper) | Phi(hypo)
    tp, fn, true_total = 0, 0, 0
    true_pair_details = []
    for hyper, hypo in hypernym_pairs:
        if hyper not in concept_data or hypo not in concept_data:
            continue
        true_total += 1
        phi_hyper = concept_data[hyper]['prime']
        phi_hypo = concept_data[hypo]['prime']
        subsumes = validator.subsumes(phi_hypo, phi_hyper)
        similarity = validator.similarity(phi_hyper, phi_hypo)
        if subsumes:
            tp += 1
        else:
            fn += 1
        true_pair_details.append({
            'hypernym': hyper, 'hyponym': hypo,
            'subsumes': subsumes, 'similarity': similarity,
            'phi_hyper': phi_hyper, 'phi_hypo': phi_hypo,
        })

    # Unrelated pairs — expect NO subsumption
    fp, tn, unrel_total = 0, 0, 0
    false_pair_details = []
    for a, b in unrelated_pairs:
        if a not in concept_data or b not in concept_data:
            continue
        unrel_total += 1
        phi_a = concept_data[a]['prime']
        phi_b = concept_data[b]['prime']
        # Check both directions
        sub_ab = validator.subsumes(phi_a, phi_b)
        sub_ba = validator.subsumes(phi_b, phi_a)
        false_subsumption = sub_ab or sub_ba
        similarity = validator.similarity(phi_a, phi_b)
        if false_subsumption:
            fp += 1
        else:
            tn += 1
        false_pair_details.append({
            'concept_a': a, 'concept_b': b,
            'subsumes_ab': sub_ab, 'subsumes_ba': sub_ba,
            'similarity': similarity,
        })

    # Metrics
    recall = tp / true_total if true_total > 0 else 0.0
    fpr = fp / unrel_total if unrel_total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Theoretical FP rate: (3/4)^k for k active bits
    # Under random independent bits, P(A divides B) = product of P(bit_i(A)=1 implies bit_i(B)=1)
    # For each active bit in A: P(that bit is also active in B) = activation_rate ~ 0.5
    # But for subsumption: every factor of A must also be in B.
    # Expected FP = (3/4)^k_mean where k_mean = mean active bits
    active_bits_list = []
    for d in true_pair_details + false_pair_details:
        for key in ['hypernym', 'concept_a']:
            if key in d and key.replace('hypernym', '').replace('concept_a', '') == '' :
                concept_name = d.get('hypernym', d.get('concept_a', ''))
                if concept_name in concept_data:
                    active_bits_list.append(concept_data[concept_name].get('n_active', 0))
    k_mean = float(np.mean(active_bits_list)) if active_bits_list else 0.0
    fp_theoretical = (3.0 / 4.0) ** k_mean if k_mean > 0 else 1.0
    fp_ratio = fpr / fp_theoretical if fp_theoretical > 0 else float('inf')

    # Similarity distributions
    true_sims = [d['similarity'] for d in true_pair_details]
    false_sims = [d['similarity'] for d in false_pair_details]

    return {
        'true_positives': tp,
        'false_negatives': fn,
        'false_positives': fp,
        'true_negatives': tn,
        'true_total': true_total,
        'unrelated_total': unrel_total,
        'recall': recall,
        'fpr': fpr,
        'precision': precision,
        'f1': f1,
        'mean_related_similarity': float(np.mean(true_sims)) if true_sims else 0.0,
        'mean_unrelated_similarity': float(np.mean(false_sims)) if false_sims else 0.0,
        'fp_theoretical': fp_theoretical,
        'fp_ratio': fp_ratio,
        'k_mean_active_bits': k_mean,
        'true_pair_details': true_pair_details[:20],
        'false_pair_details': false_pair_details[:20],
    }


def evaluate_shared_factors(concept_data, hypernym_pairs, unrelated_pairs):
    """Analyze shared prime factor patterns between related and unrelated pairs."""
    from src.triadic import prime_factors

    related_shared = []
    unrelated_shared = []

    for hyper, hypo in hypernym_pairs:
        if hyper not in concept_data or hypo not in concept_data:
            continue
        f_hyper = set(prime_factors(concept_data[hyper]['prime']))
        f_hypo = set(prime_factors(concept_data[hypo]['prime']))
        if f_hyper or f_hypo:
            jaccard = len(f_hyper & f_hypo) / len(f_hyper | f_hypo) if (f_hyper | f_hypo) else 0
            related_shared.append(jaccard)

    for a, b in unrelated_pairs:
        if a not in concept_data or b not in concept_data:
            continue
        f_a = set(prime_factors(concept_data[a]['prime']))
        f_b = set(prime_factors(concept_data[b]['prime']))
        if f_a or f_b:
            jaccard = len(f_a & f_b) / len(f_a | f_b) if (f_a | f_b) else 0
            unrelated_shared.append(jaccard)

    return {
        'mean_related_jaccard': float(np.mean(related_shared)) if related_shared else 0.0,
        'mean_unrelated_jaccard': float(np.mean(unrelated_shared)) if unrelated_shared else 0.0,
        'jaccard_gap': float(np.mean(related_shared) - np.mean(unrelated_shared)) if related_shared and unrelated_shared else 0.0,
        'n_related': len(related_shared),
        'n_unrelated': len(unrelated_shared),
    }


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print()
    print("=" * 68)
    print("  SUBSUMPTION BENCHMARK — Taxonomic Consistency")
    print("=" * 68)
    print(f"  Model: {args.model}")
    print()

    model, tokenizer, config = load_model(args.model, args.tokenizer, device)
    print(f"  Config: {config.n_layer}L/{config.n_embd}D/{config.n_head}H/{config.n_triadic_bits}bits")

    # Collect all unique concepts
    all_concepts = set()
    for h, p in HYPERNYM_PAIRS:
        all_concepts.add(h)
        all_concepts.add(p)
    for a, b in UNRELATED_PAIRS:
        all_concepts.add(a)
        all_concepts.add(b)

    print(f"  Concepts: {len(all_concepts)}")
    print()

    # Compute projections
    print("[1/3] Computing triadic projections...")
    concept_data = compute_projections(model, tokenizer, all_concepts, device)
    print(f"  Successfully encoded: {len(concept_data)}/{len(all_concepts)}")

    # Subsumption evaluation
    print()
    print("[2/3] Evaluating subsumption...")
    subsumption = evaluate_subsumption(concept_data, HYPERNYM_PAIRS, UNRELATED_PAIRS)
    print(f"  Hypernym pairs tested: {subsumption['true_total']}")
    print(f"  Unrelated pairs tested: {subsumption['unrelated_total']}")
    print()
    print(f"  Subsumption Recall:    {subsumption['recall']:.1%} ({subsumption['true_positives']}/{subsumption['true_total']})")
    print(f"  Subsumption FPR:       {subsumption['fpr']:.1%} ({subsumption['false_positives']}/{subsumption['unrelated_total']})")
    print(f"  FP theoretical (3/4)^k: {subsumption['fp_theoretical']:.3%} (k_mean={subsumption['k_mean_active_bits']:.1f})")
    print(f"  FP ratio (obs/theo):   {subsumption['fp_ratio']:.2f}  {'ABOVE CHANCE' if subsumption['fp_ratio'] > 1.2 else 'AT CHANCE (kill K6)'}")
    print(f"  Precision:             {subsumption['precision']:.1%}")
    print(f"  Taxonomic F1:          {subsumption['f1']:.3f}")
    print()
    print(f"  Mean related sim:      {subsumption['mean_related_similarity']:.3f}")
    print(f"  Mean unrelated sim:    {subsumption['mean_unrelated_similarity']:.3f}")

    # Show some examples
    print()
    print("  Sample hypernym pairs:")
    for d in subsumption['true_pair_details'][:8]:
        mark = "OK" if d['subsumes'] else "MISS"
        print(f"    {d['hypernym']:>10} > {d['hyponym']:<12} sim={d['similarity']:.1%}  [{mark}]")

    print()
    print("  Sample unrelated pairs:")
    for d in subsumption['false_pair_details'][:5]:
        mark = "FP!" if d['subsumes_ab'] or d['subsumes_ba'] else "ok"
        print(f"    {d['concept_a']:>10} ~ {d['concept_b']:<12} sim={d['similarity']:.1%}  [{mark}]")

    # Shared factor analysis
    print()
    print("[3/3] Analyzing shared prime factors...")
    factors = evaluate_shared_factors(concept_data, HYPERNYM_PAIRS, UNRELATED_PAIRS)
    print(f"  Mean Jaccard (related):   {factors['mean_related_jaccard']:.3f}")
    print(f"  Mean Jaccard (unrelated): {factors['mean_unrelated_jaccard']:.3f}")
    print(f"  Jaccard gap:              {factors['jaccard_gap']:+.3f}")

    # Save results
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    results_dir = os.path.join(project_root, 'benchmarks', 'results')
    os.makedirs(results_dir, exist_ok=True)

    version = args.version
    today = date.today().isoformat()

    result = {
        "benchmark": "subsumption",
        "version": version,
        "date": today,
        "model_checkpoint": args.model,
        "model_config": f"{config.n_layer}L/{config.n_embd}D/{config.n_head}H/{config.n_triadic_bits}bits",
        "n_hypernym_pairs": subsumption['true_total'],
        "n_unrelated_pairs": subsumption['unrelated_total'],
        "metrics": {
            "recall": subsumption['recall'],
            "fpr": subsumption['fpr'],
            "precision": subsumption['precision'],
            "f1": subsumption['f1'],
            "true_positives": subsumption['true_positives'],
            "false_negatives": subsumption['false_negatives'],
            "false_positives": subsumption['false_positives'],
            "true_negatives": subsumption['true_negatives'],
            "mean_related_similarity": subsumption['mean_related_similarity'],
            "mean_unrelated_similarity": subsumption['mean_unrelated_similarity'],
            "fp_theoretical": subsumption['fp_theoretical'],
            "fp_ratio_obs_over_theo": subsumption['fp_ratio'],
            "k_mean_active_bits": subsumption['k_mean_active_bits'],
            "mean_related_jaccard": factors['mean_related_jaccard'],
            "mean_unrelated_jaccard": factors['mean_unrelated_jaccard'],
            "jaccard_gap": factors['jaccard_gap'],
        },
        "sample_true_pairs": subsumption['true_pair_details'],
        "sample_false_pairs": subsumption['false_pair_details'],
    }

    result_path = os.path.join(results_dir, f"{version}_subsumption_{today}.json")
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Results saved: {result_path}")

    # Verdict
    print()
    print("=" * 68)
    target_recall = 0.60
    target_fpr = 0.05
    target_f1 = 0.50
    print(f"  Recall:    {subsumption['recall']:.1%}  (target > {target_recall:.0%}) {'PASS' if subsumption['recall'] > target_recall else 'BELOW'}")
    print(f"  FPR:       {subsumption['fpr']:.1%}  (target < {target_fpr:.0%}) {'PASS' if subsumption['fpr'] < target_fpr else 'ABOVE'}")
    print(f"  F1:        {subsumption['f1']:.3f}  (target > {target_f1:.2f}) {'PASS' if subsumption['f1'] > target_f1 else 'BELOW'}")
    print(f"  Jaccard:   {factors['jaccard_gap']:+.3f}  (target > 0) {'PASS' if factors['jaccard_gap'] > 0 else 'NEGATIVE'}")
    print("=" * 68)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Subsumption Benchmark')
    parser.add_argument('--model', required=True)
    parser.add_argument('--tokenizer', default=None)
    parser.add_argument('--version', default='v1.4-strongalign')
    args = parser.parse_args()

    if args.tokenizer is None:
        args.tokenizer = os.path.join(os.path.dirname(args.model), 'tokenizer.json')

    main(args)
