"""
Bits Sweep — Runs 22-26 sequentially.
Trains XL models with k=128, 48, 32, 16, 8 bits.
Total estimated time: ~6 hours.

Usage:
  python scripts/run_bits_sweep.py
"""

import subprocess
import sys
import os
import time
from datetime import datetime

PYTHON = sys.executable
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RUNS = [
    # {"name": "Run 22", "bits": 128, "dir": "checkpoints/torch_run22_bits128"},  # DONE
    {"name": "Run 23", "bits": 48,  "dir": "checkpoints/torch_run23_bits48"},
    {"name": "Run 24", "bits": 32,  "dir": "checkpoints/torch_run24_bits32"},
    {"name": "Run 25", "bits": 16,  "dir": "checkpoints/torch_run25_bits16"},
    {"name": "Run 26", "bits": 8,   "dir": "checkpoints/torch_run26_bits8"},
]

SHARED_ARGS = [
    "--scale", "xl",
    "--steps", "50000",
    "--alpha", "0.05",
    "--entropy-weight", "1.0",
    "--align-weight", "5.0",
    "--triadic-warmup-pct", "0.25",
    "--no-distill",
    "--tokenizer", "checkpoints/torch_runXL/tokenizer.json",
    "--save-every", "10000",
    "--print-every", "200",
]

LOG_FILE = os.path.join(BASE_DIR, "bits_sweep_log.txt")


def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + "\n")


def main():
    os.chdir(BASE_DIR)

    log("=" * 60)
    log("BITS SWEEP — Runs 22-26 (k=128, 48, 32, 16, 8)")
    log(f"Estimated total time: ~6 hours")
    log(f"Log file: {LOG_FILE}")
    log("=" * 60)

    total_start = time.time()
    completed = []
    failed = []

    for run in RUNS:
        log(f"\n{'='*60}")
        log(f"STARTING: {run['name']} (k={run['bits']} bits)")
        log(f"Checkpoint dir: {run['dir']}")
        log(f"{'='*60}")

        cmd = [
            PYTHON, "src/torch_train.py",
            "--override-bits", str(run['bits']),
            "--checkpoint-dir", run['dir'],
        ] + SHARED_ARGS

        run_start = time.time()
        try:
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=7200,  # 2h max per run
                env=env,
            )

            run_time = time.time() - run_start

            # Save individual run output
            output_file = os.path.join(run['dir'], 'training_output.txt')
            os.makedirs(run['dir'], exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"=== {run['name']} (k={run['bits']}) ===\n")
                f.write(f"Exit code: {result.returncode}\n")
                f.write(f"Time: {run_time:.1f}s ({run_time/60:.1f} min)\n\n")
                f.write("=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr)

            if result.returncode == 0:
                log(f"COMPLETED: {run['name']} in {run_time/60:.1f} min")
                # Extract final loss from last lines
                for line in result.stdout.split('\n')[-20:]:
                    if 'Final loss' in line:
                        log(f"  {line.strip()}")
                    if 'Parameters' in line or 'params' in line.lower():
                        log(f"  {line.strip()}")
                completed.append(run['name'])
            else:
                log(f"FAILED: {run['name']} (exit code {result.returncode})")
                log(f"  Last stderr: {result.stderr[-200:]}")
                failed.append(run['name'])

        except subprocess.TimeoutExpired:
            log(f"TIMEOUT: {run['name']} exceeded 2 hour limit")
            failed.append(run['name'])
        except Exception as e:
            log(f"ERROR: {run['name']} — {e}")
            failed.append(run['name'])

    total_time = time.time() - total_start

    log(f"\n{'='*60}")
    log(f"BITS SWEEP COMPLETE")
    log(f"Total time: {total_time/3600:.1f} hours ({total_time/60:.0f} min)")
    log(f"Completed: {len(completed)}/5 — {', '.join(completed)}")
    if failed:
        log(f"Failed: {len(failed)}/5 — {', '.join(failed)}")
    log(f"{'='*60}")

    # Summary
    log("\nCheckpoint locations:")
    for run in RUNS:
        status = "OK" if run['name'] in completed else "FAIL"
        log(f"  [{status}] {run['dir']}/ (k={run['bits']})")


if __name__ == '__main__':
    main()
