"""
Compare key metrics across all major runs for paper integration.

Usage:
  python playground/compare_all_runs.py
"""

import os
import json
import csv

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_json(path):
    full = os.path.join(PROJECT, path)
    if not os.path.exists(full):
        return None
    with open(full, encoding='utf-8') as f:
        return json.load(f)


def get_training_final(ckpt_dir):
    """Get final metrics from training_log.csv."""
    path = os.path.join(PROJECT, ckpt_dir, 'training_log.csv')
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    last = rows[-1]
    return {k: float(v) for k, v in last.items()}


def main():
    print("=" * 90)
    print("  COMPARISON TABLE — All Major Runs")
    print("=" * 90)

    # Run 15 (production, 64-bit)
    r15_lang = load_json('benchmarks/results/v1.4-strongalign_language_quality_2026-03-07.json')
    r15_sub = load_json('benchmarks/results/v1.4-strongalign_subsumption_2026-03-07.json')
    r15_entropy = load_json('benchmarks/results/v1.4-strongalign_bit_entropy_2026-03-07.json')
    r15_analogy = load_json('benchmarks/results/v1.4-strongalign_analogy_2026-03-07.json')

    # P15 (49-bit structured)
    p15 = load_json('playground/results/concept_gpt_49bit.json')

    # D-A5 Bootstrap (63-bit, in progress or complete)
    da5_base = load_json('checkpoints/danza_bootstrap_base/bootstrap_results.json')
    da5_xl_log = get_training_final('checkpoints/danza_bootstrap_xl')
    da5_xl_results = load_json('checkpoints/danza_bootstrap_xl/bootstrap_results.json')

    # Header
    print(f"\n  {'Metric':35s} {'Run 15 (64b)':>14s} {'P15 XL (49b)':>14s} {'D-A5 Base':>14s} {'D-A5 XL':>14s}")
    print(f"  {'-'*35} {'-'*14} {'-'*14} {'-'*14} {'-'*14}")

    # Architecture
    print(f"  {'Bits':35s} {'64':>14s} {'49':>14s} {'63':>14s} {'63':>14s}")
    print(f"  {'Params':35s} {'40M':>14s} {'40M':>14s} {'5M':>14s} {'40M':>14s}")
    print(f"  {'Steps':35s} {'50K':>14s} {'50K':>14s} {'100':>14s} {'50K':>14s}")

    # Language
    r15_loss = '0.946'
    p15_loss = f"{p15['final_lang_loss']:.3f}" if p15 else '?'
    da5b_loss = '?'
    da5x_loss = f"{da5_xl_log['lang_loss']:.3f}" if da5_xl_log else '?'
    print(f"  {'Lang loss':35s} {r15_loss:>14s} {p15_loss:>14s} {da5b_loss:>14s} {da5x_loss:>14s}")

    # Dead bits
    r15_dead = '~15'
    p15_dead = f"{p15['dead_bits']}" if p15 else '?'
    da5b_dead = '61'
    da5x_dead = f"{int(da5_xl_log['dead_bits'])}" if da5_xl_log else '?'
    print(f"  {'Dead bits':35s} {r15_dead:>14s} {p15_dead:>14s} {da5b_dead:>14s} {da5x_dead:>14s}")

    # Subsumption
    r15_sub_v = '73.5%/92%'
    p15_sub = f"{p15['subsumption_test']:.1%}" if p15 else '?'
    print(f"  {'Subsumption test':35s} {r15_sub_v:>14s} {p15_sub:>14s} {'N/A':>14s} {'N/A':>14s}")

    # Primary accuracy (P15 specific)
    p15_acc = f"{p15['primary_accuracy']:.1%}" if p15 else '?'
    print(f"  {'Primary accuracy (49-class)':35s} {'N/A':>14s} {p15_acc:>14s} {'N/A':>14s} {'N/A':>14s}")

    # Bit accuracy
    p15_supacc = f"{p15['sup_train_acc']:.1%}/{p15['sup_test_acc']:.1%}" if p15 else '?'
    print(f"  {'Sup acc (train/test)':35s} {'N/A':>14s} {p15_supacc:>14s} {'N/A':>14s} {'N/A':>14s}")

    # D-A5 specific
    if da5_xl_log:
        da5x_bit_train = f"{da5_xl_log['bit_acc_train']:.1%}"
        da5x_bit_hold = f"{da5_xl_log['bit_acc_holdout']:.1%}"
    else:
        da5x_bit_train = da5x_bit_hold = '?'

    # D-A5 base
    import numpy as np
    if da5_base:
        r3_items = [(k, v) for k, v in da5_base.items() if v.get('type') == 'R3']
        ctrl_items = [(k, v) for k, v in da5_base.items() if v.get('type') == 'CTRL']
        da5b_r3d = f"{np.mean([v['direct_acc'] for _, v in r3_items]):.1%}"
        da5b_r3a = f"{np.mean([max(v['best_r3_acc'], v.get('ensemble_acc', 0)) for _, v in r3_items if v['best_r3_acc'] > 0]):.1%}"
    else:
        da5b_r3d = da5b_r3a = '?'

    print(f"  {'Bit acc train (63-bit)':35s} {'N/A':>14s} {'N/A':>14s} {'89.5%':>14s} {da5x_bit_train:>14s}")
    print(f"  {'Bit acc holdout (63-bit)':35s} {'N/A':>14s} {'N/A':>14s} {'89.7%':>14s} {da5x_bit_hold:>14s}")
    print(f"  {'Trivial baseline':35s} {'N/A':>14s} {'N/A':>14s} {'90.2%':>14s} {'90.2%':>14s}")
    print(f"  {'R3 direct mean':35s} {'N/A':>14s} {'N/A':>14s} {da5b_r3d:>14s} {'PENDING':>14s}")
    print(f"  {'R3 algebraic mean':35s} {'N/A':>14s} {'N/A':>14s} {da5b_r3a:>14s} {'PENDING':>14s}")

    # D-A5 XL results if available
    if da5_xl_results:
        r3x = [(k, v) for k, v in da5_xl_results.items() if v.get('type') == 'R3']
        ctrlx = [(k, v) for k, v in da5_xl_results.items() if v.get('type') == 'CTRL']
        if r3x:
            r3xd = np.mean([v['direct_acc'] for _, v in r3x])
            r3xa = np.mean([max(v['best_r3_acc'], v.get('ensemble_acc', 0)) for _, v in r3x if v['best_r3_acc'] > 0])
            print(f"\n  D-A5 XL RESULTS AVAILABLE:")
            print(f"    R3 direct: {r3xd:.1%}")
            print(f"    R3 algebraic: {r3xa:.1%}")
            print(f"    Algebraic lift: {r3xa - r3xd:+.1%}")

    print(f"\n  {'KEY DIFFERENCES':35s}")
    print(f"  {'-'*70}")
    print(f"  Run 15: Arbitrary 64 bits, emergent from training. No named primitives.")
    print(f"  P15:    49 bits = 7x7 named primitives. Supervised + subsumption.")
    print(f"  D-A5:   63 bits = inventario primitivos. Supervised 24 anchors, predict 23.")
    print(f"  {'':35s}")
    print(f"  Run 15 proves: end-to-end learning of algebraic structure is possible")
    print(f"  P15 proves:    named primitives are learnable (86%) but don't compose (17%)")
    print(f"  D-A5 tests:    algebraic prediction from partial supervision")
    print()


if __name__ == '__main__':
    main()
