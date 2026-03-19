"""
EXP-F3.4: Test Ciego de Asignacion de Primos
=============================================

INDISPENSABLE antes de publicar (~2 horas).

Claim (Cap. 33): El autor reconoce que necesita validacion con
asignaciones no elegidas por el. Responde la objecion de cherry-picking.

Metodo:
  1. Tomar los 51 quads de D-A5 como benchmark
  2. Crear 3 asignaciones alternativas de primos:
     a) RANDOM: primos aleatorios a los 49 primitivos
     b) FREQUENCY: por frecuencia de uso (primitivo mas comun = primo mas chico)
     c) SEMANTIC: por similitud en embedding space
  3. Para cada asignacion: recalcular Phi(concepto), ejecutar R3, medir accuracy
  4. Repetir RANDOM 100 veces para distribucion nula

Hipotesis:
  - Si RANDOM ~= original -> R3 funciona por estructura, no por asignacion
  - Si RANDOM << original -> la asignacion importa (posible cherry-picking)
  - Si SEMANTIC ~= original > RANDOM -> buena asignacion = similitud semantica

Usage:
  cd C:\\Github\\triadic-microgpt
  python playground/audit_tests/test_blind_prime_assignment.py
"""

import os
import sys
import json
import math
import numpy as np
from collections import Counter

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from common import (
    load_run15, get_projections_batch, to_binary,
    save_results, print_header, print_section, N_BITS,
)
from src.triadic import PrimeMapper, TriadicValidator, nth_prime, prime_factors


# ============================================================
# Load quads and anchors
# ============================================================

def load_quads_and_anchors():
    """Load bootstrap quads and anchor data from danza_bootstrap."""
    from danza_bootstrap import BOOTSTRAP_QUADS, HOLDOUT_INFO
    from danza_63bit import load_primitives, load_anchors

    prim_data = load_primitives()
    all_anchors, _ = load_anchors(prim_data)
    return BOOTSTRAP_QUADS, all_anchors, prim_data


# ============================================================
# Prime assignment strategies
# ============================================================

def original_assignment(n_bits):
    """Original: bit i -> (i+1)th prime. This is the author's assignment."""
    return [nth_prime(i + 1) for i in range(n_bits)]


def random_assignment(n_bits, rng):
    """Random: shuffle the first n_bits primes."""
    primes = [nth_prime(i + 1) for i in range(n_bits)]
    rng.shuffle(primes)
    return primes


def frequency_assignment(projections, n_bits):
    """Frequency: most common active bit -> smallest prime."""
    # Count how often each bit is active across all concepts
    bit_counts = np.zeros(n_bits)
    for proj in projections.values():
        bits = to_binary(proj)
        bit_counts += bits[:n_bits]

    # Sort bits by frequency (descending)
    sorted_bits = np.argsort(-bit_counts)

    # Assign primes: most frequent bit gets smallest prime
    primes = [0] * n_bits
    available = [nth_prime(i + 1) for i in range(n_bits)]
    for rank, bit_idx in enumerate(sorted_bits):
        primes[bit_idx] = available[rank]

    return primes


def semantic_assignment(projections, n_bits):
    """Semantic: bits that co-activate -> adjacent primes."""
    # Compute co-activation matrix
    n_concepts = len(projections)
    bit_vectors = np.zeros((n_concepts, n_bits))
    for i, proj in enumerate(projections.values()):
        bit_vectors[i] = to_binary(proj)[:n_bits]

    # Correlation matrix between bits
    corr = np.corrcoef(bit_vectors.T)
    corr = np.nan_to_num(corr, nan=0.0)

    # Greedy ordering: start from bit 0, always pick most correlated next
    ordered = [0]
    remaining = set(range(1, n_bits))
    while remaining:
        last = ordered[-1]
        best = max(remaining, key=lambda b: corr[last, b])
        ordered.append(best)
        remaining.remove(best)

    # Assign primes in order
    primes = [0] * n_bits
    available = [nth_prime(i + 1) for i in range(n_bits)]
    for rank, bit_idx in enumerate(ordered):
        primes[bit_idx] = available[rank]

    return primes


# ============================================================
# R3 with custom prime assignment
# ============================================================

def compute_composite(proj, prime_list):
    """Map projection to composite using a custom prime assignment."""
    bits = to_binary(proj)
    composite = 1
    for i, prime in enumerate(prime_list):
        if i < len(bits) and bits[i] == 1:
            composite *= prime
    return composite


def r3_analogy(a, b, c):
    """A:B :: C:? using algebraic R3."""
    shared = math.gcd(a, b)
    only_a = a // shared
    only_b = b // shared
    c_reduced = c // math.gcd(c, only_a)
    target = (c_reduced * only_b) // math.gcd(c_reduced, only_b)
    return target


def evaluate_r3_with_assignment(quads, projections, prime_list, all_anchors):
    """Run R3 on all quads with a given prime assignment. Return accuracy."""
    correct = 0
    total = 0

    for quad in quads:
        if len(quad) < 4:
            continue
        a_word, b_word, c_word, d_word = quad[:4]

        # Check all words have projections
        if any(w not in projections for w in [a_word, b_word, c_word, d_word]):
            continue

        # Compute composites with this assignment
        a_prime = compute_composite(projections[a_word], prime_list)
        b_prime = compute_composite(projections[b_word], prime_list)
        c_prime = compute_composite(projections[c_word], prime_list)
        d_prime_gold = compute_composite(projections[d_word], prime_list)

        if a_prime == 1 or b_prime == 1 or c_prime == 1:
            continue

        # Predict D
        d_predicted = r3_analogy(a_prime, b_prime, c_prime)

        # Check: does predicted match gold?
        if d_predicted == d_prime_gold:
            correct += 1
        total += 1

    return correct / max(total, 1), correct, total


# ============================================================
# MAIN
# ============================================================

def main():
    print_header("EXP-F3.4: BLIND PRIME ASSIGNMENT TEST")
    print(f"  Answers: 'Are the prime assignments cherry-picked?'")

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    print(f"  Loading Run 15...")
    model, tokenizer = load_run15(str(device))

    print(f"  Loading quads and anchors...")
    quads, all_anchors, prim_data = load_quads_and_anchors()
    print(f"  Quads: {len(quads)}, Anchors: {len(all_anchors)}")

    # Extract projections for all anchor words
    all_words = list(all_anchors.keys())
    print(f"  Extracting projections for {len(all_words)} concepts...")
    projections = get_projections_batch(model, tokenizer, all_words, str(device), max_tokens=4)
    print(f"  Got {len(projections)}/{len(all_words)} concepts")

    # ============================================================
    # Test each assignment strategy
    # ============================================================

    print_section("ASSIGNMENT STRATEGIES")

    # 1. Original
    orig_primes = original_assignment(N_BITS)
    orig_acc, orig_correct, orig_total = evaluate_r3_with_assignment(
        quads, projections, orig_primes, all_anchors)
    print(f"  ORIGINAL:  {orig_acc:.1%} ({orig_correct}/{orig_total})")

    # 2. Frequency-based
    freq_primes = frequency_assignment(projections, N_BITS)
    freq_acc, freq_correct, freq_total = evaluate_r3_with_assignment(
        quads, projections, freq_primes, all_anchors)
    print(f"  FREQUENCY: {freq_acc:.1%} ({freq_correct}/{freq_total})")

    # 3. Semantic-based
    sem_primes = semantic_assignment(projections, N_BITS)
    sem_acc, sem_correct, sem_total = evaluate_r3_with_assignment(
        quads, projections, sem_primes, all_anchors)
    print(f"  SEMANTIC:  {sem_acc:.1%} ({sem_correct}/{sem_total})")

    # 4. Random (100 trials)
    print_section("RANDOM ASSIGNMENT (100 trials)")

    n_random = 100
    random_accs = []
    rng = np.random.default_rng(42)

    for trial in range(n_random):
        rand_primes = random_assignment(N_BITS, rng)
        acc, _, _ = evaluate_r3_with_assignment(
            quads, projections, rand_primes, all_anchors)
        random_accs.append(acc)
        if (trial + 1) % 25 == 0:
            print(f"    Trial {trial + 1}/{n_random}: mean={np.mean(random_accs):.1%} "
                  f"std={np.std(random_accs):.1%}")

    random_mean = np.mean(random_accs)
    random_std = np.std(random_accs)
    random_max = np.max(random_accs)
    random_min = np.min(random_accs)

    print(f"\n  RANDOM:    mean={random_mean:.1%} std={random_std:.1%} "
          f"range=[{random_min:.1%}, {random_max:.1%}]")

    # ============================================================
    # Analysis
    # ============================================================

    print_section("ANALYSIS")

    # Z-score of original vs random distribution
    if random_std > 0:
        z_score = (orig_acc - random_mean) / random_std
    else:
        z_score = float('inf') if orig_acc > random_mean else 0
    print(f"  Original vs Random z-score: {z_score:.2f}")

    # Percentile of original in random distribution
    percentile = sum(1 for a in random_accs if a >= orig_acc) / n_random * 100
    print(f"  Original percentile in random: {percentile:.0f}% of randoms are >= original")

    # Interpretation
    print(f"\n  INTERPRETATION:")
    if abs(orig_acc - random_mean) < 2 * random_std:
        interpretation = "STRUCTURE"
        print(f"    Original ~ Random -> R3 works by STRUCTURE, not assignment")
        print(f"    The prime assignment is NOT cherry-picked")
        cherry_picked = False
    elif orig_acc > random_mean + 2 * random_std:
        interpretation = "ASSIGNMENT_MATTERS"
        print(f"    Original >> Random -> The assignment DOES matter")
        if sem_acc > random_mean + random_std:
            print(f"    Semantic assignment also works -> Good assignment = semantic similarity")
            cherry_picked = False
        else:
            print(f"    WARNING: Possible cherry-picking of prime assignments")
            cherry_picked = True
    else:
        interpretation = "ORIGINAL_WORSE"
        print(f"    Original < Random -> Assignment is suboptimal (unexpected)")
        cherry_picked = False

    # Comparison table
    print_section("COMPARISON TABLE")
    print(f"  {'Strategy':<16} {'Accuracy':>10} {'vs Original':>12} {'vs Random':>12}")
    print(f"  {'-'*16} {'-'*10} {'-'*12} {'-'*12}")
    print(f"  {'Original':<16} {orig_acc:>10.1%} {'—':>12} {orig_acc - random_mean:>+12.1%}")
    print(f"  {'Frequency':<16} {freq_acc:>10.1%} {freq_acc - orig_acc:>+12.1%} {freq_acc - random_mean:>+12.1%}")
    print(f"  {'Semantic':<16} {sem_acc:>10.1%} {sem_acc - orig_acc:>+12.1%} {sem_acc - random_mean:>+12.1%}")
    print(f"  {'Random (mean)':<16} {random_mean:>10.1%} {random_mean - orig_acc:>+12.1%} {'—':>12}")

    # Overall
    print_section("SUMMARY")
    overall = not cherry_picked
    print(f"  Cherry-picking detected: {'YES' if cherry_picked else 'NO'}")
    print(f"  OVERALL: {'PASS' if overall else 'FAIL'}")

    save_results({
        'test': 'EXP-F3.4',
        'model': 'Run 15 (40M)',
        'n_quads': len(quads),
        'n_concepts': len(projections),
        'original': {'accuracy': round(orig_acc, 4), 'correct': orig_correct, 'total': orig_total},
        'frequency': {'accuracy': round(freq_acc, 4), 'correct': freq_correct, 'total': freq_total},
        'semantic': {'accuracy': round(sem_acc, 4), 'correct': sem_correct, 'total': sem_total},
        'random': {
            'mean': round(random_mean, 4), 'std': round(random_std, 4),
            'min': round(random_min, 4), 'max': round(random_max, 4),
            'n_trials': n_random,
        },
        'z_score': round(z_score, 2),
        'percentile': round(percentile, 1),
        'interpretation': interpretation,
        'cherry_picked': cherry_picked,
        'overall_pass': overall,
    }, 'f3_4_blind_prime_assignment.json')


if __name__ == '__main__':
    main()
