"""
EXP-F2.5: Enantiodromia en Firmas
==================================

Hilo tematico: H5 (Caps. 9, 18, 22)

Claim (Cap. 18): Los extremos se convierten en su opuesto.
"El exceso de un polo produce al otro."

Test: Los conceptos "extremos" tienen firmas MAS CERCANAS a su opuesto
que las versiones moderadas del mismo concepto.

  Hipotesis: Hamming(extremo, opuesto) < Hamming(moderado, opuesto)
  -> Los extremos "contienen la semilla" de su opuesto (comparten mas primos)

Criterio: >= 6/8 pares confirman la hipotesis.

Usage:
  cd C:\\Github\\triadic-microgpt
  python playground/audit_tests/test_enantiodromia.py
"""

import os
import sys
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from common import (
    load_run15, get_projections_batch, hamming, bits_shared,
    proj_to_prime, save_results, print_header, print_section,
    N_BITS,
)
from src.triadic import PrimeMapper, TriadicValidator

# ============================================================
# Dataset: (extreme, opposite, moderate_version)
# ============================================================

ENANTIODROMIA_TRIPLES = [
    # (extreme, opposite, moderate)
    ('tyranny', 'freedom', 'authority'),
    ('fanaticism', 'tolerance', 'dedication'),
    ('obsession', 'indifference', 'interest'),
    ('perfectionism', 'acceptance', 'diligence'),
    ('recklessness', 'caution', 'boldness'),
    ('greed', 'generosity', 'ambition'),
    ('pride', 'humility', 'confidence'),
    ('rage', 'calm', 'irritation'),
]

# Additional pairs to test with more nuance
EXTRA_TRIPLES = [
    ('starvation', 'abundance', 'hunger'),
    ('isolation', 'connection', 'solitude'),
    ('chaos', 'order', 'spontaneity'),
    ('worship', 'contempt', 'admiration'),
]


def main():
    print_header("EXP-F2.5: ENANTIODROMIA IN SIGNATURES")

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    print(f"  Loading Run 15...")
    model, tokenizer = load_run15(str(device))

    mapper = PrimeMapper(N_BITS)
    validator = TriadicValidator()

    # Collect all words
    all_words = set()
    for extreme, opposite, moderate in ENANTIODROMIA_TRIPLES + EXTRA_TRIPLES:
        all_words.update([extreme, opposite, moderate])

    print(f"  Extracting projections for {len(all_words)} concepts...")
    projections = get_projections_batch(model, tokenizer, list(all_words), str(device), max_tokens=4)
    print(f"  Got {len(projections)}/{len(all_words)} concepts")

    # Main test
    print_section("ENANTIODROMIA TEST")
    print(f"  Hypothesis: H(extreme, opposite) < H(moderate, opposite)")
    print(f"  -> Extremes are CLOSER to their opposite than moderates\n")

    print(f"  {'Extreme':<16} {'Opposite':<16} {'Moderate':<14} "
          f"{'H(ext,opp)':>10} {'H(mod,opp)':>10} {'delta':>7} {'Result':>8}")
    print(f"  {'-'*16} {'-'*16} {'-'*14} {'-'*10} {'-'*10} {'-'*7} {'-'*8}")

    results = []
    for extreme, opposite, moderate in ENANTIODROMIA_TRIPLES:
        p_ext = projections.get(extreme)
        p_opp = projections.get(opposite)
        p_mod = projections.get(moderate)

        if p_ext is None or p_opp is None or p_mod is None:
            missing = [w for w in [extreme, opposite, moderate]
                       if projections.get(w) is None]
            print(f"  {extreme:<16} {opposite:<16} {moderate:<14} SKIP (missing: {missing})")
            continue

        h_ext_opp = hamming(p_ext, p_opp)
        h_mod_opp = hamming(p_mod, p_opp)
        delta = h_ext_opp - h_mod_opp
        passes = h_ext_opp < h_mod_opp

        # Also compute shared bits for deeper analysis
        shared_ext = bits_shared(p_ext, p_opp)
        shared_mod = bits_shared(p_mod, p_opp)

        pr_ext = proj_to_prime(p_ext, mapper)
        pr_opp = proj_to_prime(p_opp, mapper)
        pr_mod = proj_to_prime(p_mod, mapper)
        gcd_ext = validator.intersect(pr_ext, pr_opp)
        gcd_mod = validator.intersect(pr_mod, pr_opp)

        r = {
            'extreme': extreme, 'opposite': opposite, 'moderate': moderate,
            'h_extreme_opposite': h_ext_opp,
            'h_moderate_opposite': h_mod_opp,
            'delta': delta,
            'pass': passes,
            'shared_extreme': round(shared_ext, 3),
            'shared_moderate': round(shared_mod, 3),
            'gcd_extreme': gcd_ext,
            'gcd_moderate': gcd_mod,
        }
        results.append(r)

        print(f"  {extreme:<16} {opposite:<16} {moderate:<14} "
              f"{h_ext_opp:>10} {h_mod_opp:>10} {delta:>+7} "
              f"{'PASS' if passes else 'FAIL':>8}")

    # Extra triples
    print_section("EXTRA TRIPLES (additional validation)")
    extra_results = []
    for extreme, opposite, moderate in EXTRA_TRIPLES:
        p_ext = projections.get(extreme)
        p_opp = projections.get(opposite)
        p_mod = projections.get(moderate)

        if p_ext is None or p_opp is None or p_mod is None:
            continue

        h_ext_opp = hamming(p_ext, p_opp)
        h_mod_opp = hamming(p_mod, p_opp)
        delta = h_ext_opp - h_mod_opp
        passes = h_ext_opp < h_mod_opp

        extra_results.append({
            'extreme': extreme, 'opposite': opposite, 'moderate': moderate,
            'h_extreme_opposite': h_ext_opp,
            'h_moderate_opposite': h_mod_opp,
            'delta': delta,
            'pass': passes,
        })

        print(f"  {extreme:<16} {opposite:<16} {moderate:<14} "
              f"{h_ext_opp:>10} {h_mod_opp:>10} {delta:>+7} "
              f"{'PASS' if passes else 'FAIL':>8}")

    # Summary
    print_section("SUMMARY")
    n_pass = sum(1 for r in results if r['pass'])
    n_total = len(results)
    print(f"  Core triples: {n_pass}/{n_total} confirmed")

    if extra_results:
        n_extra_pass = sum(1 for r in extra_results if r['pass'])
        print(f"  Extra triples: {n_extra_pass}/{len(extra_results)} confirmed")

    overall = n_pass >= 6 if n_total >= 8 else n_pass / max(n_total, 1) >= 0.75
    print(f"\n  OVERALL (>=6/8 or >=75%): {'PASS' if overall else 'FAIL'}")

    # Deeper analysis: do extremes share MORE primes with their opposite?
    if results:
        shared_ext_vals = [r['shared_extreme'] for r in results]
        shared_mod_vals = [r['shared_moderate'] for r in results]
        print(f"\n  Mean shared bits with opposite:")
        print(f"    Extremes:  {np.mean(shared_ext_vals):.3f}")
        print(f"    Moderates: {np.mean(shared_mod_vals):.3f}")
        ext_more_shared = np.mean(shared_ext_vals) > np.mean(shared_mod_vals)
        print(f"    Extremes share MORE: {'YES' if ext_more_shared else 'NO'}")

    save_results({
        'test': 'EXP-F2.5',
        'model': 'Run 15 (40M)',
        'core_results': results,
        'extra_results': extra_results,
        'n_pass': n_pass,
        'n_total': n_total,
        'overall_pass': overall,
    }, 'f2_5_enantiodromia.json')


if __name__ == '__main__':
    main()
