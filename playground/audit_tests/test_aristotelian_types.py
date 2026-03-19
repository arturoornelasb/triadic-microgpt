"""
EXP-F2.2: Tipos Aristotelicos en Firmas
========================================

Hilo tematico: H1 (opuestos -> algebra, Cap. 11)

Claim: Los 4 tipos de oposicion aristotelica producen patrones de firma distintos:
  - Contrarios: espectro continuo (hot/cold) -> alta simetria, muchos bits invertidos
  - Contradictorios: binario sin intermedio (alive/dead) -> maxima diferencia
  - Privativos: presencia/ausencia (sight/blindness) -> asimetricos
  - Relativos: mutuamente definitorios (parent/child) -> alta superposicion

Criterio: ANOVA o Kruskal-Wallis p<0.05 en al menos 2 de 4 metricas.

Usage:
  cd C:\\Github\\triadic-microgpt
  python playground/audit_tests/test_aristotelian_types.py
"""

import os
import sys
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from common import (
    load_run15, get_projections_batch, hamming, bits_shared,
    proj_to_prime, to_binary, save_results, print_header, print_section,
    N_BITS,
)
from src.triadic import PrimeMapper, TriadicValidator

# ============================================================
# Dataset: 40 pairs, 10 per type
# ============================================================

CONTRARIES = [
    ('hot', 'cold'), ('love', 'hate'), ('tall', 'short'),
    ('fast', 'slow'), ('bright', 'dark'), ('heavy', 'light'),
    ('loud', 'quiet'), ('rich', 'poor'), ('young', 'old'),
    ('happy', 'sad'),
]

CONTRADICTORIES = [
    ('alive', 'dead'), ('true', 'false'), ('present', 'absent'),
    ('even', 'odd'), ('married', 'single'), ('guilty', 'innocent'),
    ('legal', 'illegal'), ('open', 'closed'), ('visible', 'invisible'),
    ('possible', 'impossible'),
]

PRIVATIVES = [
    ('sight', 'blindness'), ('sound', 'silence'),
    ('knowledge', 'ignorance'), ('hope', 'despair'),
    ('health', 'disease'), ('wealth', 'poverty'),
    ('freedom', 'captivity'), ('order', 'chaos'),
    ('trust', 'distrust'), ('courage', 'cowardice'),
]

RELATIVES = [
    ('parent', 'child'), ('teacher', 'student'),
    ('buyer', 'seller'), ('doctor', 'patient'),
    ('predator', 'prey'), ('cause', 'effect'),
    ('question', 'answer'), ('host', 'guest'),
    ('employer', 'employee'), ('leader', 'follower'),
]

ALL_TYPES = {
    'contraries': CONTRARIES,
    'contradictories': CONTRADICTORIES,
    'privatives': PRIVATIVES,
    'relatives': RELATIVES,
}


def analyze_pair(w1, w2, projections, mapper, validator):
    """Compute all metrics for a pair."""
    p1, p2 = projections.get(w1), projections.get(w2)
    if p1 is None or p2 is None:
        return None

    b1, b2 = to_binary(p1), to_binary(p2)
    pr1, pr2 = proj_to_prime(p1, mapper), proj_to_prime(p2, mapper)

    # Hamming distance
    h = hamming(p1, p2)

    # Bits inverted (where one is 1 and other is 0, counted both ways)
    inverted = int(np.sum((b1 == 1) & (b2 == 0)) + np.sum((b1 == 0) & (b2 == 1)))

    # Shared bits (Jaccard on active bits)
    shared = bits_shared(p1, p2)

    # GCD (shared prime factors)
    gcd = validator.intersect(pr1, pr2)
    sim = validator.similarity(pr1, pr2)

    # Asymmetry: difference in number of active bits
    n_active_1 = int(b1.sum())
    n_active_2 = int(b2.sum())
    asymmetry = abs(n_active_1 - n_active_2) / max(n_active_1, n_active_2, 1)

    return {
        'pair': f"{w1}/{w2}",
        'hamming': h,
        'inverted': inverted,
        'shared': round(shared, 3),
        'similarity': round(sim, 3),
        'asymmetry': round(asymmetry, 3),
        'active_1': n_active_1,
        'active_2': n_active_2,
    }


def main():
    print_header("EXP-F2.2: ARISTOTELIAN OPPOSITION TYPES")

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    print(f"  Loading Run 15...")
    model, tokenizer = load_run15(str(device))

    mapper = PrimeMapper(N_BITS)
    validator = TriadicValidator()

    # Collect all words
    all_words = set()
    for pairs in ALL_TYPES.values():
        for w1, w2 in pairs:
            all_words.update([w1, w2])

    print(f"  Extracting projections for {len(all_words)} concepts...")
    projections = get_projections_batch(model, tokenizer, list(all_words), str(device), max_tokens=4)
    print(f"  Got {len(projections)}/{len(all_words)} concepts")

    # Analyze each type
    type_metrics = {}

    for type_name, pairs in ALL_TYPES.items():
        print_section(f"{type_name.upper()} ({len(pairs)} pairs)")

        metrics = []
        print(f"  {'Pair':<24} {'Ham':>5} {'Inv':>5} {'Shared':>7} {'Sim':>5} {'Asym':>6}")
        print(f"  {'-'*24} {'-'*5} {'-'*5} {'-'*7} {'-'*5} {'-'*6}")

        for w1, w2 in pairs:
            r = analyze_pair(w1, w2, projections, mapper, validator)
            if r is None:
                print(f"  {w1}/{w2:<20} SKIP")
                continue
            metrics.append(r)
            print(f"  {r['pair']:<24} {r['hamming']:>5} {r['inverted']:>5} "
                  f"{r['shared']:>7.3f} {r['similarity']:>5.3f} {r['asymmetry']:>6.3f}")

        if metrics:
            hammings = [m['hamming'] for m in metrics]
            inverteds = [m['inverted'] for m in metrics]
            shareds = [m['shared'] for m in metrics]
            asymmetries = [m['asymmetry'] for m in metrics]

            print(f"\n  Means: Hamming={np.mean(hammings):.1f} Inverted={np.mean(inverteds):.1f} "
                  f"Shared={np.mean(shareds):.3f} Asymmetry={np.mean(asymmetries):.3f}")

            type_metrics[type_name] = {
                'pairs': metrics,
                'means': {
                    'hamming': round(float(np.mean(hammings)), 1),
                    'inverted': round(float(np.mean(inverteds)), 1),
                    'shared': round(float(np.mean(shareds)), 3),
                    'asymmetry': round(float(np.mean(asymmetries)), 3),
                },
                'stds': {
                    'hamming': round(float(np.std(hammings)), 1),
                    'inverted': round(float(np.std(inverteds)), 1),
                    'shared': round(float(np.std(shareds)), 3),
                    'asymmetry': round(float(np.std(asymmetries)), 3),
                },
            }

    # Statistical tests
    print_section("STATISTICAL COMPARISON")

    # Kruskal-Wallis for each metric
    from scipy import stats as scipy_stats

    metric_names = ['hamming', 'inverted', 'shared', 'asymmetry']
    stat_results = {}
    n_significant = 0

    for metric in metric_names:
        groups = []
        for type_name in ['contraries', 'contradictories', 'privatives', 'relatives']:
            if type_name in type_metrics:
                vals = [m[metric] for m in type_metrics[type_name]['pairs']]
                groups.append(vals)

        if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
            H, p = scipy_stats.kruskal(*groups)
            significant = p < 0.05
            if significant:
                n_significant += 1
            stat_results[metric] = {'H': round(H, 2), 'p': round(p, 4), 'significant': significant}
            print(f"  {metric:<12}: H={H:.2f}, p={p:.4f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''}")
        else:
            print(f"  {metric:<12}: insufficient data for test")

    # Hypothesis checks from the book
    print_section("HYPOTHESIS CHECKS (Cap. 11)")

    checks = []
    if 'contraries' in type_metrics and 'contradictories' in type_metrics:
        # Contraries should have more inverted bits than contradictories
        c_inv = type_metrics['contraries']['means']['inverted']
        d_inv = type_metrics['contradictories']['means']['inverted']
        check = c_inv > d_inv
        checks.append(('Contraries have more inverted bits than contradictories', check))
        print(f"  Contraries inverted ({c_inv:.1f}) > Contradictories ({d_inv:.1f}): {'PASS' if check else 'FAIL'}")

    if 'privatives' in type_metrics:
        # Privatives should be most asymmetric
        p_asym = type_metrics['privatives']['means']['asymmetry']
        others = [type_metrics[t]['means']['asymmetry'] for t in type_metrics if t != 'privatives']
        if others:
            check = p_asym > max(others)
            checks.append(('Privatives are most asymmetric', check))
            print(f"  Privatives asymmetry ({p_asym:.3f}) > max others ({max(others):.3f}): {'PASS' if check else 'FAIL'}")

    if 'relatives' in type_metrics:
        # Relatives should share the most bits
        r_shared = type_metrics['relatives']['means']['shared']
        others = [type_metrics[t]['means']['shared'] for t in type_metrics if t != 'relatives']
        if others:
            check = r_shared > max(others)
            checks.append(('Relatives share the most bits', check))
            print(f"  Relatives shared ({r_shared:.3f}) > max others ({max(others):.3f}): {'PASS' if check else 'FAIL'}")

    if 'contradictories' in type_metrics:
        # Contradictories should have highest Hamming
        d_ham = type_metrics['contradictories']['means']['hamming']
        others = [type_metrics[t]['means']['hamming'] for t in type_metrics if t != 'contradictories']
        if others:
            check = d_ham > max(others)
            checks.append(('Contradictories have highest Hamming', check))
            print(f"  Contradictories Hamming ({d_ham:.1f}) > max others ({max(others):.1f}): {'PASS' if check else 'FAIL'}")

    # Overall
    print_section("SUMMARY")
    print(f"  Kruskal-Wallis significant: {n_significant}/4 metrics")
    print(f"  Hypothesis checks: {sum(1 for _, c in checks if c)}/{len(checks)}")
    overall = n_significant >= 2
    print(f"  OVERALL (>=2 significant metrics): {'PASS' if overall else 'FAIL'}")

    save_results({
        'test': 'EXP-F2.2',
        'model': 'Run 15 (40M)',
        'type_metrics': type_metrics,
        'kruskal_wallis': stat_results,
        'hypothesis_checks': [{'check': c, 'pass': p} for c, p in checks],
        'n_significant': n_significant,
        'overall_pass': overall,
    }, 'f2_2_aristotelian_types.json')


if __name__ == '__main__':
    main()
