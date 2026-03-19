"""
Run all audit tests sequentially.

Usage:
  cd C:\\Github\\triadic-microgpt

  # Run all tests
  python playground/audit_tests/run_all.py

  # Run only indispensable tests
  python playground/audit_tests/run_all.py --indispensable

  # Run only valuable tests
  python playground/audit_tests/run_all.py --valuable

  # Run a specific test
  python playground/audit_tests/run_all.py --test f2.1
"""

import os
import sys
import argparse
import importlib
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Test registry
PREFLIGHT = [
    ('f0',   'test_data_validation',         'Data Validation (pre-flight)'),
]

INDISPENSABLE = [
    ('f3.1', 'test_pf_bridge',              'PF Bridge Q1/Q2/Q4-Q6 (~3h)'),
    ('f3.4', 'test_blind_prime_assignment',  'Blind Prime Assignment (~2h)'),
    ('f4.4', 'test_d_a13_eval',             'D-A13 355M Evaluation (~2h)'),
]

VALUABLE = [
    ('f2.1', 'test_indifference_and_false_opposites', 'Indifference + False Opposites (~1h)'),
    ('f2.2', 'test_aristotelian_types',               'Aristotelian Types (~1h)'),
    ('f2.5', 'test_enantiodromia',                    'Enantiodromia (~30min)'),
]

ALL_TESTS = PREFLIGHT + INDISPENSABLE + VALUABLE


def run_test(module_name, test_name):
    """Import and run a single test module."""
    print(f"\n{'#' * 70}")
    print(f"# RUNNING: {test_name}")
    print(f"{'#' * 70}")

    start = time.time()
    try:
        mod = importlib.import_module(module_name)
        mod.main()
        elapsed = time.time() - start
        print(f"\n  Completed in {elapsed:.0f}s")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  FAILED after {elapsed:.0f}s: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Run audit tests')
    parser.add_argument('--indispensable', action='store_true',
                        help='Run only indispensable tests (F3.1, F3.4, F4.4)')
    parser.add_argument('--valuable', action='store_true',
                        help='Run only valuable tests (F2.1, F2.2, F2.5)')
    parser.add_argument('--test', type=str, default=None,
                        help='Run a specific test by ID (e.g., f2.1, f3.1)')
    args = parser.parse_args()

    if args.test:
        tests = [(tid, mod, name) for tid, mod, name in ALL_TESTS
                 if tid == args.test.lower()]
        if not tests:
            print(f"Unknown test: {args.test}")
            print(f"Available: {', '.join(t[0] for t in ALL_TESTS)}")
            sys.exit(1)
    elif args.indispensable:
        tests = INDISPENSABLE
    elif args.valuable:
        tests = VALUABLE
    else:
        tests = ALL_TESTS

    print(f"=" * 70)
    print(f"  AUDIT TEST RUNNER")
    print(f"  Tests to run: {len(tests)}")
    for tid, _, name in tests:
        print(f"    [{tid.upper()}] {name}")
    print(f"=" * 70)

    results = {}
    for tid, module_name, name in tests:
        success = run_test(module_name, name)
        results[tid] = success

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  FINAL SUMMARY")
    print(f"{'=' * 70}")
    for tid, success in results.items():
        status = 'OK' if success else 'FAILED'
        print(f"  [{tid.upper()}] {status}")

    n_ok = sum(1 for v in results.values() if v)
    print(f"\n  {n_ok}/{len(results)} tests completed successfully")


if __name__ == '__main__':
    main()
