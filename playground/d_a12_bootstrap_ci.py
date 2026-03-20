"""
D-A12: Bootstrap Confidence Intervals for Multi-Quad and Bootstrap Results.

Computes 95% CIs via percentile bootstrap (10,000 resamples) for:
  1. Multi-quad ensemble accuracy (14 concepts)
  2. Ensemble vs single-best delta
  3. R3 algebraic improvement over direct encoding (D-A5)
  4. R3 vs CTRL direct encoding comparison
  5. Effect size vs shuffled baseline

Output: JSON + console summary.
"""

import json
import os
import numpy as np

CKPT = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'danza_bootstrap_xl')
N_BOOT = 10_000
SEED = 42


def bootstrap_ci(data, n_boot=N_BOOT, ci=0.95, seed=SEED):
    """Percentile bootstrap CI for the mean."""
    rng = np.random.RandomState(seed)
    data = np.array(data)
    n = len(data)
    means = np.array([rng.choice(data, size=n, replace=True).mean() for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(means, [100 * alpha, 100 * (1 - alpha)])
    return {
        'mean': float(data.mean()),
        'std': float(data.std(ddof=1)),
        'ci_lo': float(lo),
        'ci_hi': float(hi),
        'n': int(n),
    }


def bootstrap_ci_diff(a, b, n_boot=N_BOOT, ci=0.95, seed=SEED):
    """Bootstrap CI for the difference of means (a - b)."""
    rng = np.random.RandomState(seed)
    a, b = np.array(a), np.array(b)
    diffs = []
    for _ in range(n_boot):
        a_s = rng.choice(a, size=len(a), replace=True)
        b_s = rng.choice(b, size=len(b), replace=True)
        diffs.append(a_s.mean() - b_s.mean())
    diffs = np.array(diffs)
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(diffs, [100 * alpha, 100 * (1 - alpha)])
    return {
        'mean_diff': float(np.mean(a) - np.mean(b)),
        'ci_lo': float(lo),
        'ci_hi': float(hi),
        'p_positive': float((diffs > 0).mean()),
    }


def main():
    # --- Load data ---
    with open(os.path.join(CKPT, 'multi_quad_results.json')) as f:
        mq = json.load(f)
    with open(os.path.join(CKPT, 'bootstrap_results.json')) as f:
        bs = json.load(f)
    with open(os.path.join(CKPT, 'negative_baselines.json')) as f:
        neg = json.load(f)

    # --- 1. Multi-quad ensemble accuracy (14 concepts) ---
    concepts_mq = {k: v for k, v in mq.items() if k != '_summary'}
    ensemble_accs = [v['ensemble_acc'] for v in concepts_mq.values()]
    single_bests = [v['single_best'] for v in concepts_mq.values()]
    deltas = [v['delta'] for v in concepts_mq.values()]

    ci_ensemble = bootstrap_ci(ensemble_accs)
    ci_single_best = bootstrap_ci(single_bests)
    ci_delta = bootstrap_ci(deltas)

    # --- 2. Per-quad level (all 47+ individual quads) ---
    all_quad_accs = []
    for v in concepts_mq.values():
        for q in v['per_quad']:
            all_quad_accs.append(q['accuracy'])
    ci_per_quad = bootstrap_ci(all_quad_accs)

    # --- 3. R3 algebraic improvement (D-A5 bootstrap_results) ---
    r3_concepts = {k: v for k, v in bs.items() if v.get('type') == 'R3'}
    ctrl_concepts = {k: v for k, v in bs.items() if v.get('type') == 'CTRL'}

    r3_direct = [v['direct_acc'] for v in r3_concepts.values()]
    r3_best = [v['best_r3_acc'] for v in r3_concepts.values()]
    r3_improve = [v['algebraic_improvement'] for v in r3_concepts.values()]
    ctrl_direct = [v['direct_acc'] for v in ctrl_concepts.values()]

    ci_r3_direct = bootstrap_ci(r3_direct)
    ci_r3_best = bootstrap_ci(r3_best)
    ci_r3_improve = bootstrap_ci(r3_improve)
    ci_ctrl_direct = bootstrap_ci(ctrl_direct)

    # R3 vs CTRL direct encoding
    ci_r3_vs_ctrl = bootstrap_ci_diff(r3_direct, ctrl_direct)

    # --- 4. Effect size vs shuffled baseline ---
    # We only have summary stats for shuffled, so compute analytic CI
    real_mean = neg['real_r3_mean']
    shuffled_mean = neg['shuffled_mean']
    shuffled_std = neg['shuffled_std']
    effect_size = neg['effect_size']

    # Bootstrap CI on real R3 accuracies vs trivial baseline
    trivial = mq['_summary']['trivial_baseline']
    above_trivial = [a - trivial for a in ensemble_accs]
    ci_above_trivial = bootstrap_ci(above_trivial)

    # --- 5. Bits correct distribution ---
    bits_correct = [v['bits_correct'] for v in concepts_mq.values()]
    bits_wrong = [v['bits_wrong'] for v in concepts_mq.values()]
    ci_bits = bootstrap_ci(bits_correct)

    # --- Compile results ---
    results = {
        'multi_quad_ensemble': {
            'description': '95% CI on mean ensemble accuracy (14 concepts, multi-quad)',
            **ci_ensemble,
        },
        'multi_quad_single_best': {
            'description': '95% CI on mean single-best accuracy',
            **ci_single_best,
        },
        'multi_quad_delta': {
            'description': '95% CI on ensemble minus single-best (negative = ensemble worse)',
            **ci_delta,
        },
        'per_quad_accuracy': {
            'description': f'95% CI on individual quad accuracy ({len(all_quad_accs)} quads)',
            **ci_per_quad,
        },
        'r3_direct_encoding': {
            'description': '95% CI on R3 concepts direct encoding (D-A5, 14 concepts)',
            **ci_r3_direct,
        },
        'r3_algebraic_best': {
            'description': '95% CI on R3 best algebraic prediction',
            **ci_r3_best,
        },
        'r3_algebraic_improvement': {
            'description': '95% CI on algebraic improvement over direct (R3 - direct)',
            **ci_r3_improve,
        },
        'ctrl_direct_encoding': {
            'description': '95% CI on CTRL concepts direct encoding (9 concepts)',
            **ci_ctrl_direct,
        },
        'r3_vs_ctrl_direct': {
            'description': '95% CI on difference (R3 direct - CTRL direct)',
            **ci_r3_vs_ctrl,
        },
        'above_trivial_baseline': {
            'description': '95% CI on ensemble accuracy minus trivial baseline (90.2%)',
            'trivial_baseline': trivial,
            **ci_above_trivial,
        },
        'bits_correct': {
            'description': '95% CI on bits correct per concept (out of 63)',
            **ci_bits,
        },
        'negative_baselines': {
            'description': 'Effect size vs shuffled baseline (from negative_baselines.json)',
            'real_r3_mean': real_mean,
            'shuffled_mean': shuffled_mean,
            'shuffled_std': shuffled_std,
            'cohens_d': effect_size,
            'gap': real_mean - shuffled_mean,
        },
        'metadata': {
            'n_bootstrap': N_BOOT,
            'ci_level': 0.95,
            'seed': SEED,
            'method': 'percentile bootstrap',
        },
    }

    # --- Save ---
    out_path = os.path.join(CKPT, 'd_a12_confidence_intervals.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # --- Print summary ---
    print("=" * 70)
    print("D-A12: Bootstrap Confidence Intervals (95%, 10K resamples)")
    print("=" * 70)

    def fmt(r, pct=True):
        m = r['mean'] * (100 if pct else 1)
        lo = r['ci_lo'] * (100 if pct else 1)
        hi = r['ci_hi'] * (100 if pct else 1)
        u = '%' if pct else ''
        return f"{m:.2f}{u}  [{lo:.2f}, {hi:.2f}]"

    print(f"\n--- Multi-Quad (14 concepts, 3-5 quads each) ---")
    print(f"  Ensemble accuracy:    {fmt(ci_ensemble)}")
    print(f"  Single-best accuracy: {fmt(ci_single_best)}")
    print(f"  Delta (ens - best):   {fmt(ci_delta)}")
    print(f"  Per-quad accuracy:    {fmt(ci_per_quad)} (n={len(all_quad_accs)} quads)")
    print(f"  Bits correct:         {fmt(ci_bits, pct=False)} / 63")

    print(f"\n--- D-A5 Bootstrap (14 R3 + 9 CTRL) ---")
    print(f"  R3 direct encoding:   {fmt(ci_r3_direct)}")
    print(f"  R3 algebraic best:    {fmt(ci_r3_best)}")
    print(f"  R3 improvement:       {fmt(ci_r3_improve)}")
    print(f"  CTRL direct:          {fmt(ci_ctrl_direct)}")
    print(f"  R3 vs CTRL (direct):  {ci_r3_vs_ctrl['mean_diff']*100:.2f}%  [{ci_r3_vs_ctrl['ci_lo']*100:.2f}, {ci_r3_vs_ctrl['ci_hi']*100:.2f}]  (P(>0)={ci_r3_vs_ctrl['p_positive']:.3f})")

    print(f"\n--- vs Baselines ---")
    print(f"  Trivial baseline:     {trivial*100:.1f}%")
    print(f"  Above trivial:        {fmt(ci_above_trivial)}")
    print(f"  Shuffled baseline:    {shuffled_mean*100:.2f}% +/- {shuffled_std*100:.2f}%")
    print(f"  Cohen's d:            {effect_size:.2f}")
    print(f"  Real - shuffled:      {(real_mean - shuffled_mean)*100:.2f}%")

    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
