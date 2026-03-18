"""
D-A11: Negative Baselines — Shuffled labels + random projections.

Establishes rigorous lower bounds for D-A5 results:
1. Shuffled gold labels: permute gold targets across concepts → algebraic predictions should fail
2. Random projections: replace model projections with random vectors → bound chance performance
3. Majority-class verification: recompute the 90.2% trivial baseline from scratch

If D-A5's R3 algebraic result (90.7%) is meaningful, these baselines should all score lower.

CPU-only: loads existing D-A5 checkpoint, no training required.

Usage:
  python playground/negative_baselines.py
  python playground/negative_baselines.py --checkpoint checkpoints/danza_bootstrap_xl/
  python playground/negative_baselines.py --n-shuffles 100
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from collections import defaultdict

_PLAYGROUND = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_PLAYGROUND)
sys.path.insert(0, _PROJECT)
sys.path.insert(0, _PLAYGROUND)

from danza_63bit import (
    load_primitives, load_anchors,
    DanzaTriadicGPT,
    ANCHOR_TRANSLATIONS, SKIP_ANCHORS,
    N_BITS, STORY_SEPARATOR,
)
from danza_bootstrap import (
    TRAIN_CONCEPTS, HOLDOUT_INFO, BOOTSTRAP_QUADS,
    get_split, get_holdout_type,
)
from src.torch_transformer import TriadicGPT, TriadicGPTConfig
try:
    from src.fast_tokenizer import FastBPETokenizer as BPETokenizer
except ImportError:
    from src.tokenizer import BPETokenizer


# ============================================================
# Baseline 1: Majority-class (trivial)
# ============================================================

def compute_majority_baseline(holdout_anchors, all_anchors, device):
    """Per-bit majority class prediction = trivial baseline."""
    # Compute majority from ALL anchors (train + holdout) gold labels
    all_targets = []
    for w, data in all_anchors.items():
        all_targets.append((data['target'] > 0).float())

    stacked = torch.stack(all_targets)  # (N, 63)
    per_bit_mean = stacked.mean(dim=0)  # fraction of 1s per bit
    majority_pred = (per_bit_mean > 0.5).float()  # predict majority class per bit

    # Evaluate on holdout only
    holdout_accs = []
    for w, data in holdout_anchors.items():
        gold = (data['target'] > 0).float().to(device)
        acc = (majority_pred.to(device) == gold).float().mean().item()
        holdout_accs.append(acc)

    # Bit distribution analysis
    always_on = (per_bit_mean > 0.95).sum().item()
    always_off = (per_bit_mean < 0.05).sum().item()
    variable = N_BITS - always_on - always_off

    return {
        'mean_accuracy': float(np.mean(holdout_accs)),
        'std_accuracy': float(np.std(holdout_accs)),
        'per_concept': holdout_accs,
        'always_on_bits': int(always_on),
        'always_off_bits': int(always_off),
        'variable_bits': int(variable),
        'majority_pred': majority_pred,
    }


# ============================================================
# Baseline 2: Shuffled gold labels
# ============================================================

@torch.no_grad()
def compute_shuffled_baseline(model, tokenizer, train_anchors, holdout_anchors, device, n_shuffles=100):
    """Permute gold targets across concepts and re-run algebraic prediction.

    If R3 algebra exploits genuine structure, shuffled labels should degrade performance.
    """
    model.eval()

    def get_proj(word):
        ids = tokenizer.encode(word, add_special=False)[:4]
        if not ids:
            return None
        x = torch.tensor([ids], dtype=torch.long, device=device)
        _, proj, _ = model(x)
        return proj[0].mean(dim=0)

    # Get real projections for all relevant words
    proj_cache = {}
    for w in set(train_anchors.keys()) | set(holdout_anchors.keys()):
        p = get_proj(w)
        if p is not None:
            proj_cache[w] = p

    # Collect holdout gold targets
    holdout_words = sorted(holdout_anchors.keys())
    holdout_targets = {w: (holdout_anchors[w]['target'] > 0).float().to(device) for w in holdout_words}

    # Real R3 accuracy (for comparison)
    real_r3_accs = []
    for a, b, c, d in BOOTSTRAP_QUADS:
        if a not in proj_cache or b not in proj_cache or c not in proj_cache or d not in proj_cache:
            continue
        predicted = proj_cache[c] + (proj_cache[b] - proj_cache[a])
        pred_bits = (predicted > 0).float()
        gold_bits = holdout_targets.get(d)
        if gold_bits is not None:
            acc = (pred_bits == gold_bits).float().mean().item()
            real_r3_accs.append(acc)

    real_mean = float(np.mean(real_r3_accs)) if real_r3_accs else 0

    # Shuffled trials
    rng = np.random.RandomState(42)
    shuffled_means = []

    for trial in range(n_shuffles):
        # Permute gold targets across holdout concepts
        perm = rng.permutation(len(holdout_words))
        shuffled_targets = {}
        for i, w in enumerate(holdout_words):
            shuffled_targets[w] = holdout_targets[holdout_words[perm[i]]]

        # Re-evaluate R3 with shuffled targets
        trial_accs = []
        for a, b, c, d in BOOTSTRAP_QUADS:
            if a not in proj_cache or b not in proj_cache or c not in proj_cache:
                continue
            if d not in shuffled_targets:
                continue
            predicted = proj_cache[c] + (proj_cache[b] - proj_cache[a])
            pred_bits = (predicted > 0).float()
            gold_bits = shuffled_targets[d]
            acc = (pred_bits == gold_bits).float().mean().item()
            trial_accs.append(acc)

        if trial_accs:
            shuffled_means.append(float(np.mean(trial_accs)))

    # p-value: fraction of shuffled trials >= real accuracy
    p_value = float(np.mean([s >= real_mean for s in shuffled_means])) if shuffled_means else 1.0

    return {
        'real_r3_mean': real_mean,
        'shuffled_mean': float(np.mean(shuffled_means)) if shuffled_means else 0,
        'shuffled_std': float(np.std(shuffled_means)) if shuffled_means else 0,
        'shuffled_max': float(np.max(shuffled_means)) if shuffled_means else 0,
        'shuffled_min': float(np.min(shuffled_means)) if shuffled_means else 0,
        'p_value': p_value,
        'n_shuffles': n_shuffles,
        'effect_size': (real_mean - np.mean(shuffled_means)) / (np.std(shuffled_means) + 1e-8)
            if shuffled_means else 0,
    }


# ============================================================
# Baseline 3: Random projections
# ============================================================

@torch.no_grad()
def compute_random_projection_baseline(holdout_anchors, device, n_trials=100):
    """Replace model projections with random vectors and measure R3 accuracy.

    This bounds the chance level for algebraic prediction when projections have
    no semantic content.
    """
    holdout_words = sorted(holdout_anchors.keys())
    holdout_targets = {w: (holdout_anchors[w]['target'] > 0).float().to(device) for w in holdout_words}

    # Get train word list (any word that appears as A, B, or C in quads)
    quad_words = set()
    for a, b, c, d in BOOTSTRAP_QUADS:
        quad_words.update([a, b, c, d])

    rng = np.random.RandomState(42)
    trial_means = []

    for trial in range(n_trials):
        # Random projections for all words
        random_projs = {}
        for w in quad_words:
            random_projs[w] = torch.randn(N_BITS, device=device)

        # R3 with random projections
        trial_accs = []
        for a, b, c, d in BOOTSTRAP_QUADS:
            if d not in holdout_targets:
                continue
            predicted = random_projs[c] + (random_projs[b] - random_projs[a])
            pred_bits = (predicted > 0).float()
            gold_bits = holdout_targets[d]
            acc = (pred_bits == gold_bits).float().mean().item()
            trial_accs.append(acc)

        if trial_accs:
            trial_means.append(float(np.mean(trial_accs)))

    return {
        'random_mean': float(np.mean(trial_means)) if trial_means else 0,
        'random_std': float(np.std(trial_means)) if trial_means else 0,
        'random_max': float(np.max(trial_means)) if trial_means else 0,
        'random_min': float(np.min(trial_means)) if trial_means else 0,
        'n_trials': n_trials,
    }


# ============================================================
# Baseline 4: Per-concept majority (not just global majority)
# ============================================================

def compute_per_concept_majority(train_anchors, holdout_anchors, device):
    """Train-set majority per bit, evaluate on holdout.

    Slightly different from global majority: uses only train anchors to compute
    per-bit majority, then evaluates on holdout. This tests whether training
    distribution statistics alone explain holdout accuracy.
    """
    train_targets = []
    for w, data in train_anchors.items():
        train_targets.append((data['target'] > 0).float())

    stacked = torch.stack(train_targets).to(device)
    per_bit_mean = stacked.mean(dim=0)
    majority_pred = (per_bit_mean > 0.5).float()

    holdout_accs = []
    for w, data in holdout_anchors.items():
        gold = (data['target'] > 0).float().to(device)
        acc = (majority_pred == gold).float().mean().item()
        holdout_accs.append(acc)

    return {
        'mean_accuracy': float(np.mean(holdout_accs)),
        'std_accuracy': float(np.std(holdout_accs)),
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='D-A11: Negative Baselines')
    parser.add_argument('--checkpoint', type=str,
                       default=os.path.join(_PROJECT, 'checkpoints', 'danza_bootstrap_xl'),
                       help='D-A5 checkpoint directory')
    parser.add_argument('--n-shuffles', type=int, default=100,
                       help='Number of shuffled label permutations')
    parser.add_argument('--n-random', type=int, default=100,
                       help='Number of random projection trials')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu recommended)')
    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print(f"  D-A11: NEGATIVE BASELINES")
    print(f"{'=' * 70}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Shuffles: {args.n_shuffles}")
    print(f"  Random trials: {args.n_random}")

    device = torch.device(args.device)

    # Load anchors
    prim_data = load_primitives()
    all_anchors, _skipped = load_anchors(prim_data)
    train_anchors, holdout_anchors = get_split(all_anchors)
    print(f"  Train: {len(train_anchors)} | Holdout: {len(holdout_anchors)}")

    # --- Baseline 1: Majority class ---
    print(f"\n  {'─' * 60}")
    print(f"  BASELINE 1: Majority-Class Prediction")
    print(f"  {'─' * 60}")

    majority = compute_majority_baseline(holdout_anchors, all_anchors, device)
    print(f"  Mean accuracy: {majority['mean_accuracy']:.1%} +/- {majority['std_accuracy']:.1%}")
    print(f"  Always-ON bits:  {majority['always_on_bits']}")
    print(f"  Always-OFF bits: {majority['always_off_bits']}")
    print(f"  Variable bits:   {majority['variable_bits']}")

    # --- Baseline 4: Train-only majority ---
    print(f"\n  {'─' * 60}")
    print(f"  BASELINE 1b: Train-Only Majority")
    print(f"  {'─' * 60}")

    train_majority = compute_per_concept_majority(train_anchors, holdout_anchors, device)
    print(f"  Mean accuracy: {train_majority['mean_accuracy']:.1%} +/- {train_majority['std_accuracy']:.1%}")

    # --- Baseline 3: Random projections (no model needed) ---
    print(f"\n  {'─' * 60}")
    print(f"  BASELINE 3: Random Projections (R3 algebra)")
    print(f"  {'─' * 60}")

    random_proj = compute_random_projection_baseline(holdout_anchors, device, args.n_random)
    print(f"  Mean R3 accuracy: {random_proj['random_mean']:.1%} +/- {random_proj['random_std']:.1%}")
    print(f"  Range: [{random_proj['random_min']:.1%}, {random_proj['random_max']:.1%}]")

    # --- Load model for shuffled baseline ---
    ckpt_path = os.path.join(args.checkpoint, 'model_best.pt')
    if os.path.exists(ckpt_path):
        print(f"\n  Loading model for shuffled-label test...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt['config']

        tok_path = os.path.join(args.checkpoint, 'tokenizer.json')
        tokenizer = BPETokenizer.load(tok_path)

        config = TriadicGPTConfig(
            vocab_size=cfg['vocab_size'],
            block_size=cfg['block_size'],
            n_layer=cfg['n_layer'],
            n_embd=cfg['n_embd'],
            n_head=cfg['n_head'],
            n_triadic_bits=cfg['n_triadic_bits'],
        )
        model = DanzaTriadicGPT(config).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        # --- Baseline 2: Shuffled labels ---
        print(f"\n  {'─' * 60}")
        print(f"  BASELINE 2: Shuffled Gold Labels ({args.n_shuffles} permutations)")
        print(f"  {'─' * 60}")

        shuffled = compute_shuffled_baseline(
            model, tokenizer, train_anchors, holdout_anchors, device, args.n_shuffles
        )
        print(f"  Real R3 accuracy:     {shuffled['real_r3_mean']:.1%}")
        print(f"  Shuffled mean:        {shuffled['shuffled_mean']:.1%} +/- {shuffled['shuffled_std']:.1%}")
        print(f"  Shuffled range:       [{shuffled['shuffled_min']:.1%}, {shuffled['shuffled_max']:.1%}]")
        print(f"  p-value:              {shuffled['p_value']:.4f}")
        print(f"  Effect size (Cohen d):{shuffled['effect_size']:.2f}")

        if shuffled['p_value'] < 0.05:
            print(f"  RESULT: R3 accuracy is SIGNIFICANTLY above shuffled baseline (p < 0.05)")
        else:
            print(f"  RESULT: R3 accuracy is NOT significantly above shuffled baseline")
    else:
        print(f"\n  WARNING: No checkpoint found at {ckpt_path}")
        print(f"  Skipping shuffled-label baseline (requires model)")
        shuffled = None

    # --- Summary ---
    print(f"\n  {'=' * 60}")
    print(f"  SUMMARY — ALL BASELINES")
    print(f"  {'=' * 60}")
    print(f"  {'Baseline':35s} {'Accuracy':>10s}")
    print(f"  {'─' * 35} {'─' * 10}")
    print(f"  {'Majority-class (all concepts)':35s} {majority['mean_accuracy']:10.1%}")
    print(f"  {'Majority-class (train-only)':35s} {train_majority['mean_accuracy']:10.1%}")
    print(f"  {'Random projections + R3':35s} {random_proj['random_mean']:10.1%}")
    if shuffled:
        print(f"  {'Shuffled labels + R3':35s} {shuffled['shuffled_mean']:10.1%}")
        print(f"  {'─' * 35} {'─' * 10}")
        print(f"  {'D-A5 Real R3 algebraic':35s} {shuffled['real_r3_mean']:10.1%}")
    print(f"  {'D-A5 Reported (ensemble)':35s} {'90.7%':>10s}")

    # Save results
    out_path = os.path.join(args.checkpoint, 'negative_baselines.json')
    results = {
        'majority_global': majority['mean_accuracy'],
        'majority_train_only': train_majority['mean_accuracy'],
        'random_projection_mean': random_proj['random_mean'],
        'random_projection_std': random_proj['random_std'],
    }
    if shuffled:
        results.update({
            'shuffled_mean': shuffled['shuffled_mean'],
            'shuffled_std': shuffled['shuffled_std'],
            'shuffled_p_value': shuffled['p_value'],
            'real_r3_mean': shuffled['real_r3_mean'],
            'effect_size': shuffled['effect_size'],
        })

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == '__main__':
    main()
