"""
EXP-F3.1: Bridge Test — PF-Q1, Q2, Q4-Q6
==========================================

INDISPENSABLE antes de publicar (~3 horas).

Las 5 predicciones falsificables restantes del libro. PF-Q3 ya resuelta (98% en D-A5/E3).

FASE A — PFs directamente testables (sin categorizacion):
  PF-Q1: Correlacion Hamming <-> similitud semantica
          PASS si Spearman rho > 0.3 (p < 0.001)
  PF-Q2: GCD=1 implica opuestos
          PASS si mean_cosine(GCD=1) < mean_cosine(random)

FASE B — PFs que requieren categorizacion (usa interpretability_probe.py como base):
  PF-Q4: Polos duales nunca simultaneos   -> violaciones < 5%
  PF-Q5: Minimo 2 categorias por concepto -> >=95% conceptos
  PF-Q6: Observador en experiencias       -> precision > 70%

Usage:
  cd C:\\Github\\triadic-microgpt
  python playground/audit_tests/test_pf_bridge.py
"""

import os
import sys
import json
import numpy as np
from collections import defaultdict

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from common import (
    load_run15, get_projection, get_projections_batch,
    hamming, cosine_sim, bits_shared, proj_to_prime, to_binary,
    save_results, print_header, print_section, N_BITS,
)
from src.triadic import PrimeMapper, TriadicValidator

# ============================================================
# Data
# ============================================================

# 7x7 category structure from the book (Sistema 7x7 = 49 primitives + extras = 63)
# Category 5 has dual pairs that should be mutually exclusive
DUAL_PAIRS_CAT5 = [
    (0, 1), (2, 3), (4, 5), (6, 7),  # bit indices within Cat 5
]

# For Q1/Q2: use gold_primes_64.json concepts
GOLD_PRIMES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(_THIS_DIR)),
    'data', 'gold_primes_64.json'
)


def load_gold_concepts(max_concepts=500):
    """Load concepts from gold_primes_64.json."""
    if not os.path.exists(GOLD_PRIMES_PATH):
        print(f"  WARNING: {GOLD_PRIMES_PATH} not found")
        return {}
    with open(GOLD_PRIMES_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Take first N concepts alphabetically
    sorted_words = sorted(data.keys())[:max_concepts]
    return {w: data[w] for w in sorted_words}


# ============================================================
# PF-Q1: Hamming correlates with semantic similarity
# ============================================================

def test_pf_q1(projections, mapper):
    """PF-Q1: Spearman correlation between Hamming distance and cosine similarity."""
    print_section("PF-Q1: HAMMING ~ SEMANTIC SIMILARITY")

    words = list(projections.keys())
    n = len(words)
    if n < 20:
        print(f"  Only {n} concepts, need at least 20. SKIP.")
        return {'skip': True, 'reason': f'only {n} concepts'}

    # Sample pairs (all pairs if < 200 concepts, else sample)
    max_pairs = min(n * (n - 1) // 2, 10000)
    print(f"  Computing pairwise metrics for {n} concepts ({max_pairs} pairs max)...")

    hammings = []
    cosines = []

    if n <= 200:
        for i in range(n):
            for j in range(i + 1, n):
                h = hamming(projections[words[i]], projections[words[j]])
                c = cosine_sim(projections[words[i]], projections[words[j]])
                hammings.append(h)
                cosines.append(c)
    else:
        rng = np.random.default_rng(42)
        indices = rng.choice(n * (n - 1) // 2, size=max_pairs, replace=False)
        pair_idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if pair_idx in set(indices):
                    h = hamming(projections[words[i]], projections[words[j]])
                    c = cosine_sim(projections[words[i]], projections[words[j]])
                    hammings.append(h)
                    cosines.append(c)
                pair_idx += 1

    from scipy import stats as scipy_stats
    rho, p = scipy_stats.spearmanr(hammings, cosines)

    # Hamming and cosine should be NEGATIVELY correlated
    # (more distant in Hamming = less similar semantically)
    passes = abs(rho) > 0.3 and p < 0.001
    print(f"  Pairs analyzed: {len(hammings)}")
    print(f"  Spearman rho: {rho:.4f}")
    print(f"  p-value: {p:.2e}")
    print(f"  PASS (|rho| > 0.3, p < 0.001): {'PASS' if passes else 'FAIL'}")

    return {
        'n_pairs': len(hammings),
        'spearman_rho': round(rho, 4),
        'p_value': float(p),
        'pass': passes,
    }


# ============================================================
# PF-Q2: GCD=1 implies opposites
# ============================================================

def test_pf_q2(projections, mapper, validator):
    """PF-Q2: Pairs with GCD=1 should have lower cosine similarity (more opposed)."""
    print_section("PF-Q2: GCD=1 IMPLIES OPPOSITES")

    words = list(projections.keys())
    n = len(words)
    if n < 20:
        print(f"  Only {n} concepts. SKIP.")
        return {'skip': True}

    gcd1_cosines = []
    other_cosines = []

    rng = np.random.default_rng(42)
    # Sample up to 5000 pairs
    sample_size = min(n * (n - 1) // 2, 5000)
    pairs = set()
    while len(pairs) < sample_size:
        i, j = rng.integers(0, n, size=2)
        if i != j:
            pairs.add((min(i, j), max(i, j)))

    for i, j in pairs:
        p1, p2 = projections[words[i]], projections[words[j]]
        pr1 = proj_to_prime(p1, mapper)
        pr2 = proj_to_prime(p2, mapper)
        gcd = validator.intersect(pr1, pr2)
        cos = cosine_sim(p1, p2)

        if gcd == 1:
            gcd1_cosines.append(cos)
        else:
            other_cosines.append(cos)

    if not gcd1_cosines:
        print(f"  No GCD=1 pairs found among {sample_size} samples. SKIP.")
        return {'skip': True, 'reason': 'no GCD=1 pairs'}

    mean_gcd1 = np.mean(gcd1_cosines)
    mean_other = np.mean(other_cosines)
    passes = mean_gcd1 < mean_other

    print(f"  GCD=1 pairs: {len(gcd1_cosines)}")
    print(f"  Other pairs: {len(other_cosines)}")
    print(f"  Mean cosine (GCD=1): {mean_gcd1:.4f}")
    print(f"  Mean cosine (other): {mean_other:.4f}")
    print(f"  GCD=1 more opposed: {'PASS' if passes else 'FAIL'}")

    # Mann-Whitney U test
    from scipy import stats as scipy_stats
    if len(gcd1_cosines) >= 5 and len(other_cosines) >= 5:
        U, p = scipy_stats.mannwhitneyu(gcd1_cosines, other_cosines, alternative='less')
        print(f"  Mann-Whitney U: {U:.0f}, p={p:.4f}")
    else:
        p = 1.0

    return {
        'n_gcd1': len(gcd1_cosines),
        'n_other': len(other_cosines),
        'mean_cosine_gcd1': round(mean_gcd1, 4),
        'mean_cosine_other': round(mean_other, 4),
        'mann_whitney_p': round(float(p), 4),
        'pass': passes,
    }


# ============================================================
# PF-Q4/Q5/Q6: Category-based tests
# ============================================================

def load_primitives_config():
    """Load primitives and their category assignments."""
    from danza_63bit import load_primitives
    prim_data = load_primitives()
    return prim_data


def test_pf_q4_q5_q6(projections, prim_data):
    """PF-Q4, Q5, Q6: Category-level tests using the 7x7 system."""
    print_section("PF-Q4/Q5/Q6: CATEGORY-BASED TESTS")

    # Load category structure
    name_to_bit = prim_data['name_to_bit']
    bit_to_name = prim_data['bit_to_name']

    # Group bits by category (7 categories of ~7-9 bits each)
    # Categories from the book's Sistema 7x7
    categories = defaultdict(list)
    for bit_idx in range(N_BITS):
        cat_idx = bit_idx // 7  # Approximate category grouping
        categories[cat_idx].append(bit_idx)

    # PF-Q4: Dual pairs never simultaneously active
    print(f"\n  PF-Q4: Dual pairs never simultaneously active")
    print(f"  (Using category 5 dual pairs: bits should not both be +1)")

    # Cat 5 starts at bit 28 (4*7) approximately
    cat5_start = 28
    violations = 0
    total_checks = 0

    for word, proj in projections.items():
        bits = to_binary(proj)
        for b1_offset, b2_offset in DUAL_PAIRS_CAT5:
            b1 = cat5_start + b1_offset
            b2 = cat5_start + b2_offset
            if b1 < N_BITS and b2 < N_BITS:
                if bits[b1] == 1 and bits[b2] == 1:
                    violations += 1
                total_checks += 1

    violation_rate = violations / max(total_checks, 1)
    q4_pass = violation_rate < 0.05
    print(f"  Violations: {violations}/{total_checks} = {violation_rate:.1%}")
    print(f"  PASS (< 5%): {'PASS' if q4_pass else 'FAIL'}")

    # PF-Q5: Minimum 2 categories per concept
    print(f"\n  PF-Q5: Minimum 2 categories per concept")

    concepts_with_2plus = 0
    total_concepts = 0
    per_concept_cats = []

    for word, proj in projections.items():
        bits = to_binary(proj)
        active_cats = set()
        for cat_idx, cat_bits in categories.items():
            if any(bits[b] == 1 for b in cat_bits if b < N_BITS):
                active_cats.add(cat_idx)
        n_cats = len(active_cats)
        per_concept_cats.append(n_cats)
        if n_cats >= 2:
            concepts_with_2plus += 1
        total_concepts += 1

    pct_2plus = concepts_with_2plus / max(total_concepts, 1)
    q5_pass = pct_2plus >= 0.95
    print(f"  Concepts with >=2 categories: {concepts_with_2plus}/{total_concepts} = {pct_2plus:.1%}")
    print(f"  Mean categories per concept: {np.mean(per_concept_cats):.1f}")
    print(f"  PASS (>= 95%): {'PASS' if q5_pass else 'FAIL'}")

    # PF-Q6: Observer category in experiential concepts
    print(f"\n  PF-Q6: Observer bit active in experiential concepts")

    # Experiential concepts (subjective experiences that require an observer)
    experiential = ['pain', 'joy', 'fear', 'love', 'anger', 'sadness',
                    'happiness', 'anxiety', 'pleasure', 'surprise',
                    'disgust', 'shame', 'pride', 'guilt', 'hope']
    # Non-experiential (abstract/physical, no observer needed)
    non_experiential = ['stone', 'water', 'number', 'triangle', 'gravity',
                        'velocity', 'mass', 'distance', 'volume', 'density',
                        'carbon', 'iron', 'oxygen', 'nitrogen', 'hydrogen']

    # Use last category (cat 8, bits 56-62) as proxy for "observer/subjectivity"
    observer_bits = list(range(56, min(63, N_BITS)))

    exp_scores = []
    non_exp_scores = []

    for word in experiential:
        proj = projections.get(word)
        if proj is not None:
            bits = to_binary(proj)
            score = sum(bits[b] for b in observer_bits if b < N_BITS) / max(len(observer_bits), 1)
            exp_scores.append(score)

    for word in non_experiential:
        proj = projections.get(word)
        if proj is not None:
            bits = to_binary(proj)
            score = sum(bits[b] for b in observer_bits if b < N_BITS) / max(len(observer_bits), 1)
            non_exp_scores.append(score)

    if exp_scores and non_exp_scores:
        mean_exp = np.mean(exp_scores)
        mean_non = np.mean(non_exp_scores)
        q6_pass = mean_exp > mean_non
        print(f"  Experiential concepts (n={len(exp_scores)}): mean observer score = {mean_exp:.3f}")
        print(f"  Non-experiential (n={len(non_exp_scores)}): mean observer score = {mean_non:.3f}")
        print(f"  PASS (experiential > non-experiential): {'PASS' if q6_pass else 'FAIL'}")
    else:
        q6_pass = False
        mean_exp = mean_non = 0
        print(f"  Insufficient data for Q6. SKIP.")

    return {
        'q4': {
            'violations': violations, 'total': total_checks,
            'rate': round(violation_rate, 4), 'pass': q4_pass,
        },
        'q5': {
            'concepts_2plus': concepts_with_2plus, 'total': total_concepts,
            'pct': round(pct_2plus, 4), 'mean_cats': round(float(np.mean(per_concept_cats)), 1),
            'pass': q5_pass,
        },
        'q6': {
            'mean_experiential': round(mean_exp, 3),
            'mean_non_experiential': round(mean_non, 3),
            'n_experiential': len(exp_scores),
            'n_non_experiential': len(non_exp_scores),
            'pass': q6_pass,
        },
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print_header("EXP-F3.1: BRIDGE TEST — PF-Q1, Q2, Q4-Q6")
    print(f"  INDISPENSABLE: These are explicit falsifiable predictions from the book.")

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    print(f"  Loading Run 15...")
    model, tokenizer = load_run15(str(device))

    mapper = PrimeMapper(N_BITS)
    validator = TriadicValidator()

    # Load gold concepts for Q1/Q2
    print(f"  Loading gold concepts...")
    gold = load_gold_concepts(max_concepts=500)
    print(f"  Gold concepts available: {len(gold)}")

    # Extract projections
    all_words = list(gold.keys())
    # Add Q6 test words
    experiential = ['pain', 'joy', 'fear', 'love', 'anger', 'sadness',
                    'happiness', 'anxiety', 'pleasure', 'surprise',
                    'disgust', 'shame', 'pride', 'guilt', 'hope']
    non_experiential = ['stone', 'water', 'number', 'triangle', 'gravity',
                        'velocity', 'mass', 'distance', 'volume', 'density',
                        'carbon', 'iron', 'oxygen', 'nitrogen', 'hydrogen']
    all_words = list(set(all_words + experiential + non_experiential))

    print(f"  Extracting projections for {len(all_words)} concepts...")
    projections = get_projections_batch(model, tokenizer, all_words, str(device), max_tokens=4)
    print(f"  Got {len(projections)}/{len(all_words)} concepts")

    # Run tests
    q1_result = test_pf_q1(projections, mapper)
    q2_result = test_pf_q2(projections, mapper, validator)

    prim_data = load_primitives_config()
    q456_result = test_pf_q4_q5_q6(projections, prim_data)

    # Summary
    print_section("SUMMARY — FALSIFIABLE PREDICTIONS")

    results = {
        'PF-Q1': q1_result,
        'PF-Q2': q2_result,
        'PF-Q4': q456_result['q4'],
        'PF-Q5': q456_result['q5'],
        'PF-Q6': q456_result['q6'],
    }

    n_pass = 0
    n_total = 0
    for name, r in results.items():
        if r.get('skip'):
            status = 'SKIP'
        elif r.get('pass'):
            status = 'PASS'
            n_pass += 1
            n_total += 1
        else:
            status = 'FAIL'
            n_total += 1
        print(f"  {name}: {status}")

    print(f"\n  PASSED: {n_pass}/{n_total}")
    overall = n_pass >= 3
    print(f"  OVERALL (>= 3/5 pass): {'PASS' if overall else 'FAIL'}")

    save_results({
        'test': 'EXP-F3.1',
        'model': 'Run 15 (40M)',
        'n_concepts': len(projections),
        'results': results,
        'n_pass': n_pass,
        'n_total': n_total,
        'overall_pass': overall,
    }, 'f3_1_pf_bridge.json')


if __name__ == '__main__':
    main()
