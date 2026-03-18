"""
Monitor D-A5 Bootstrap training progress.
Run periodically to check how the XL training is going.

Usage:
  python playground/monitor_bootstrap.py
  python playground/monitor_bootstrap.py --watch 30   # refresh every 30s
"""

import os
import sys
import csv
import time
import argparse

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(_PROJECT, 'checkpoints', 'danza_bootstrap_xl')
LOG = os.path.join(CKPT, 'training_log.csv')
RESULTS = os.path.join(CKPT, 'bootstrap_results.json')

TOTAL_STEPS = 50000
TRIADIC_START_FRAC = 0.5  # triadic warmup at 50% = step 25000


def read_log():
    if not os.path.exists(LOG):
        return []
    with open(LOG, 'r') as f:
        return list(csv.DictReader(f))


def show_status():
    rows = read_log()
    if not rows:
        print("No training log found.")
        return False

    last = rows[-1]
    step = int(float(last['step']))
    pct = step / TOTAL_STEPS * 100

    # Estimate time remaining (very rough)
    triadic_active = float(last['tri_loss']) > 0

    print(f"\r  Step {step:,}/{TOTAL_STEPS:,} ({pct:.0f}%) | "
          f"loss={float(last['loss']):.4f} | "
          f"bit_train={float(last['bit_acc_train']):.1%} | "
          f"bit_hold={float(last['bit_acc_holdout']):.1%} | "
          f"dead={int(float(last['dead_bits']))}/63 | "
          f"tri={'ON' if triadic_active else 'off'}", end='')

    if triadic_active:
        print(f" | sup={float(last['sup_loss']):.3f} sub={float(last['sub_loss']):.3f}", end='')

    # Check if done
    if os.path.exists(RESULTS):
        print(f"\n  ** RESULTS READY: {RESULTS} **")
        print(f"  Run: python playground/analyze_bootstrap.py")
        return True

    if step >= TOTAL_STEPS:
        print(f"\n  ** TRAINING DONE - waiting for predict phase **")
        return True

    print()
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', type=int, default=0,
                        help='Refresh interval in seconds (0=single check)')
    args = parser.parse_args()

    if args.watch > 0:
        print(f"Monitoring D-A5 XL (refresh every {args.watch}s, Ctrl+C to stop)\n")
        try:
            while True:
                done = show_status()
                if done:
                    break
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        show_status()


if __name__ == '__main__':
    main()
