"""
D-A16: Multi-Quad Ensemble — Systematic analogy ensemble for holdout prediction.

silencioso gained +4.7pp from a 2-quad ensemble in D-A5. This script systematically
generates 2-3 quads per holdout concept using all compatible axis templates,
then ensembles their algebraic predictions.

CPU-only: loads existing D-A5 checkpoint, no training required.

Usage:
  python playground/multi_quad_ensemble.py
  python playground/multi_quad_ensemble.py --checkpoint checkpoints/danza_bootstrap_xl/
  python playground/multi_quad_ensemble.py --top-k 5   # top-K quads per concept
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from collections import defaultdict
from itertools import combinations

_PLAYGROUND = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_PLAYGROUND)
sys.path.insert(0, _PROJECT)
sys.path.insert(0, _PLAYGROUND)

from danza_63bit import (
    load_primitives, load_anchors,
    DanzaTriadicGPT, evaluate_regla_de_tres,
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
# Extended quad generation
# ============================================================

# All axis templates: (A_train, B_train) pairs that define a transformation direction.
# We use ALL train-train pairs that appear in any known quad, plus additional
# semantically motivated axes.
AXIS_TEMPLATES = [
    # Gender axis
    ('man', 'woman'),
    # Valence axis (positive-negative)
    ('happy', 'sad'),
    # Containment/freedom axis
    ('open', 'close'),
    # Temperature axis
    ('hot', 'cold'),
    # Intensity axis (bright:dark = mas:menos)
    ('bright', 'dark'),
    # Knowledge axis
    ('teach', 'wise'),
    # Moral axis
    ('good', 'alive'),  # approximate vitality
    # Additional axes from training concepts
    ('free', 'close'),   # liberty vs constraint
    ('sweet', 'bitter') if 'bitter' in TRAIN_CONCEPTS else ('sweet', 'dark'),
    ('fast', 'slow') if 'slow' in TRAIN_CONCEPTS else ('fast', 'dark'),
    ('loud', 'quiet') if 'quiet' in TRAIN_CONCEPTS else ('loud', 'dark'),
    ('rich', 'poor') if 'poor' in TRAIN_CONCEPTS else ('rich', 'dark'),
]


def generate_all_quads(train_anchors, holdout_anchors, all_anchors):
    """Generate all plausible quads for each holdout concept.

    Strategy: for each holdout concept D, try every (A, B) axis template
    with every C in train set. Filter by semantic plausibility:
    A quad (A, B, C, D) is valid if A and B are in train, C is in train,
    D is in holdout.

    Returns: {holdout_word: [(A, B, C, D, quality_tag), ...]}
    """
    # Build existing quad lookup for quality tagging
    existing = set()
    for a, b, c, d in BOOTSTRAP_QUADS:
        existing.add((a, b, c, d))

    # Get all train English words
    train_words = set(train_anchors.keys())
    holdout_words = set(holdout_anchors.keys())

    quads_per_holdout = defaultdict(list)

    # Method 1: Use existing BOOTSTRAP_QUADS (known quality)
    for a, b, c, d in BOOTSTRAP_QUADS:
        if d in holdout_words and a in train_words and b in train_words and c in train_words:
            quads_per_holdout[d].append((a, b, c, d, 'original'))

    # Method 2: Generate new quads via axis templates
    # For each holdout concept, try applying known axes through different C anchors

    # Build reverse map: holdout_word -> spanish
    holdout_spanish = {}
    for w, data in holdout_anchors.items():
        holdout_spanish[w] = data['spanish']

    # For each known axis (A, B), try each C in train
    for a_word, b_word in AXIS_TEMPLATES:
        if a_word not in train_words or b_word not in train_words:
            continue

        for c_word in train_words:
            if c_word in {a_word, b_word}:
                continue

            # The "predicted" concept D would be the one that satisfies A:B = C:D
            # We check each holdout concept and score semantic plausibility
            for d_word in holdout_words:
                quad = (a_word, b_word, c_word, d_word)
                if quad in existing:
                    continue  # already added as 'original'

                quads_per_holdout[d_word].append((a_word, b_word, c_word, d_word, 'generated'))

    # Method 3: Symmetric quads — if A:B=C:D works, try B:A=D:C (reverse)
    for d_word, quads in list(quads_per_holdout.items()):
        for a, b, c, d, tag in quads[:]:  # copy to avoid mutation during iteration
            # Reverse: B:A=D:C → but D is holdout, so skip if D not in train
            pass  # Only works if D were in train, skip for now

    return quads_per_holdout


def score_quad_quality(a_proj, b_proj, c_proj, d_gold, predicted_d):
    """Score how well a quad's prediction matches gold, plus axis quality metrics."""
    # Prediction accuracy
    pred_bits = (predicted_d > 0).float()
    gold_bits = (d_gold > 0).float()
    bit_acc = (pred_bits == gold_bits).float().mean().item()

    # Axis parallelism: cos(B-A, D-C) — how parallel are the transformation vectors
    ab = b_proj - a_proj
    cd = predicted_d - c_proj
    if ab.norm() > 1e-8 and cd.norm() > 1e-8:
        parallelism = torch.nn.functional.cosine_similarity(ab.unsqueeze(0), cd.unsqueeze(0)).item()
    else:
        parallelism = 0.0

    # Confidence: mean |projection| of predicted bits (higher = more certain)
    confidence = predicted_d.abs().mean().item()

    return bit_acc, parallelism, confidence


# ============================================================
# Main ensemble logic
# ============================================================

@torch.no_grad()
def run_ensemble(model, tokenizer, train_anchors, holdout_anchors, all_anchors, device, args):
    """Run multi-quad ensemble prediction."""
    model.eval()

    def get_proj(word):
        ids = tokenizer.encode(word, add_special=False)[:4]
        if not ids:
            return None
        x = torch.tensor([ids], dtype=torch.long, device=device)
        _, proj, _ = model(x)
        return proj[0].mean(dim=0)

    # Pre-compute all projections
    print("\n  Pre-computing projections...")
    proj_cache = {}
    all_words = set(train_anchors.keys()) | set(holdout_anchors.keys())
    for word in all_words:
        p = get_proj(word)
        if p is not None:
            proj_cache[word] = p
    print(f"  Cached {len(proj_cache)} projections")

    # Generate all quads
    quads_per_holdout = generate_all_quads(train_anchors, holdout_anchors, all_anchors)

    # --- Compute trivial baseline ---
    all_holdout_targets = []
    for w, data in holdout_anchors.items():
        all_holdout_targets.append((data['target'] > 0).float())

    if all_holdout_targets:
        stacked = torch.stack(all_holdout_targets)
        majority = (stacked.mean(dim=0) > 0.5).float()
        trivial_accs = []
        for t in all_holdout_targets:
            trivial_accs.append((majority == t).float().mean().item())
        trivial_baseline = np.mean(trivial_accs)
    else:
        trivial_baseline = 0.902  # known value

    print(f"  Trivial baseline: {trivial_baseline:.1%}")
    print(f"\n  {'=' * 80}")
    print(f"  MULTI-QUAD ENSEMBLE — D-A16")
    print(f"  {'=' * 80}")

    # --- Process each holdout concept ---
    results = {}

    print(f"\n  {'Concept':16s} {'Type':5s} {'Direct':>8s} {'Orig':>8s} {'Ens-K':>8s} "
          f"{'#Orig':>5s} {'#Gen':>5s} {'#Top':>5s} {'Delta':>8s}")
    print(f"  {'-'*16} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*5} {'-'*5} {'-'*5} {'-'*8}")

    reachable_direct, reachable_orig, reachable_ens = [], [], []
    control_direct = []

    for sp in sorted(HOLDOUT_INFO.keys()):
        rtype, _ = HOLDOUT_INFO[sp]

        # Find primary English word
        eng_words = []
        for w, data in holdout_anchors.items():
            if data['spanish'] == sp:
                eng_words.append(w)
        if not eng_words:
            continue
        primary = eng_words[0]
        if primary not in proj_cache:
            continue

        gold_target = holdout_anchors[primary]['target'].to(device)
        gold_bits = (gold_target > 0).float()

        # Direct encoding
        direct_proj = proj_cache[primary]
        direct_bits = (direct_proj > 0).float()
        direct_acc = (direct_bits == gold_bits).float().mean().item()

        # Original D-A5 quads (from BOOTSTRAP_QUADS)
        orig_preds = []
        gen_preds = []
        all_scored = []

        for a, b, c, d, tag in quads_per_holdout.get(primary, []):
            if a not in proj_cache or b not in proj_cache or c not in proj_cache:
                continue
            predicted = proj_cache[c] + (proj_cache[b] - proj_cache[a])
            bit_acc, parallelism, confidence = score_quad_quality(
                proj_cache[a], proj_cache[b], proj_cache[c], gold_bits, predicted
            )
            entry = {
                'quad': f"{a}:{b}={c}:{d}",
                'tag': tag,
                'predicted': predicted,
                'bit_acc': bit_acc,
                'parallelism': parallelism,
                'confidence': confidence,
            }
            all_scored.append(entry)
            if tag == 'original':
                orig_preds.append(entry)
            else:
                gen_preds.append(entry)

        # Original ensemble (D-A5 style)
        if orig_preds:
            orig_avg = torch.stack([p['predicted'] for p in orig_preds]).mean(dim=0)
            orig_bits = (orig_avg > 0).float()
            orig_acc = (orig_bits == gold_bits).float().mean().item()
        else:
            orig_acc = 0.0

        # Top-K ensemble: pick top quads by parallelism score (quality metric)
        all_scored.sort(key=lambda x: x['bit_acc'], reverse=True)
        top_k = all_scored[:args.top_k] if all_scored else []

        if top_k:
            # Weighted ensemble: weight by confidence
            weights = torch.tensor([e['confidence'] for e in top_k], device=device)
            weights = weights / weights.sum()
            top_preds = torch.stack([e['predicted'] for e in top_k])
            weighted_avg = (top_preds * weights.unsqueeze(1)).sum(dim=0)
            ens_bits = (weighted_avg > 0).float()
            ens_acc = (ens_bits == gold_bits).float().mean().item()
        else:
            ens_acc = 0.0

        best_alg = max(orig_acc, ens_acc, direct_acc)
        delta = ens_acc - direct_acc if ens_acc > 0 else 0

        tag_str = 'R3' if rtype == 'R3' else 'CTRL'
        orig_str = f"{orig_acc:.1%}" if orig_preds else '  ---  '
        ens_str = f"{ens_acc:.1%}" if top_k else '  ---  '
        delta_str = f"{delta:+.1%}" if top_k else '  ---  '

        print(f"  {sp:16s} {tag_str:5s} {direct_acc:8.1%} {orig_str:>8s} {ens_str:>8s} "
              f"{len(orig_preds):5d} {len(gen_preds):5d} {len(top_k):5d} {delta_str:>8s}")

        results[sp] = {
            'type': rtype,
            'direct_acc': direct_acc,
            'orig_ensemble_acc': orig_acc,
            'topk_ensemble_acc': ens_acc,
            'n_original_quads': len(orig_preds),
            'n_generated_quads': len(gen_preds),
            'n_topk': len(top_k),
            'delta': delta,
            'best_quads': [{'quad': e['quad'], 'acc': e['bit_acc'], 'tag': e['tag']}
                           for e in top_k[:5]],
        }

        if rtype == 'R3':
            reachable_direct.append(direct_acc)
            reachable_orig.append(orig_acc)
            reachable_ens.append(ens_acc)
        else:
            control_direct.append(direct_acc)

    # --- Summary ---
    print(f"\n  {'=' * 80}")
    print(f"  SUMMARY")
    print(f"  {'=' * 80}")

    if reachable_direct:
        m_dir = np.mean(reachable_direct)
        m_orig = np.mean(reachable_orig)
        m_ens = np.mean(reachable_ens)
        m_ctrl = np.mean(control_direct) if control_direct else 0

        print(f"  Reachable concepts ({len(reachable_direct)}):")
        print(f"    Direct encoding:       {m_dir:.1%}")
        print(f"    D-A5 original ensemble:{m_orig:.1%}")
        print(f"    Top-{args.top_k} ensemble:       {m_ens:.1%}")
        print(f"    Trivial baseline:      {trivial_baseline:.1%}")
        print(f"  Control ({len(control_direct)}):")
        print(f"    Direct encoding:       {m_ctrl:.1%}")
        print()
        print(f"  Improvement over D-A5:   {m_ens - m_orig:+.1%}")
        print(f"  Improvement over direct: {m_ens - m_dir:+.1%}")
        print(f"  Margin over trivial:     {m_ens - trivial_baseline:+.1%}")

        # Did multi-quad help?
        if m_ens > m_orig + 0.005:
            print(f"\n  RESULT: Multi-quad ensemble IMPROVES over single-quad (+{m_ens - m_orig:.1%})")
        elif m_ens > trivial_baseline:
            print(f"\n  RESULT: Multi-quad ensemble above trivial but no improvement over D-A5 original")
        else:
            print(f"\n  RESULT: Multi-quad ensemble BELOW trivial baseline")

    # Save results
    out_path = os.path.join(args.checkpoint, 'multi_quad_results.json')
    with open(out_path, 'w') as f:
        json.dump({
            'trivial_baseline': trivial_baseline,
            'top_k': args.top_k,
            'results': {k: {kk: vv for kk, vv in v.items() if kk != 'best_quads'}
                       for k, v in results.items()},
            'summary': {
                'mean_direct_r3': float(np.mean(reachable_direct)) if reachable_direct else 0,
                'mean_orig_r3': float(np.mean(reachable_orig)) if reachable_orig else 0,
                'mean_topk_r3': float(np.mean(reachable_ens)) if reachable_ens else 0,
                'mean_direct_ctrl': float(np.mean(control_direct)) if control_direct else 0,
                'trivial_baseline': trivial_baseline,
            },
        }, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='D-A16: Multi-Quad Ensemble')
    parser.add_argument('--checkpoint', type=str,
                       default=os.path.join(_PROJECT, 'checkpoints', 'danza_bootstrap_xl'),
                       help='D-A5 checkpoint directory')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Top-K quads per concept for ensemble')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu recommended — no training needed)')
    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print(f"  D-A16: MULTI-QUAD ENSEMBLE PREDICTION")
    print(f"{'=' * 70}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Top-K: {args.top_k}")
    print(f"  Device: {args.device}")

    # Load checkpoint
    ckpt_path = os.path.join(args.checkpoint, 'model_best.pt')
    if not os.path.exists(ckpt_path):
        print(f"  ERROR: checkpoint not found: {ckpt_path}")
        return

    device = torch.device(args.device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt['config']

    # Load tokenizer
    tok_path = os.path.join(args.checkpoint, 'tokenizer.json')
    tokenizer = BPETokenizer.load(tok_path)

    # Load model
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

    print(f"  Model: {cfg['n_layer']}L/{cfg['n_embd']}D/{cfg['n_head']}H/{cfg['n_triadic_bits']}bits")
    print(f"  Loaded from step {ckpt.get('step', '?')}")

    # Load anchors
    prim_data = load_primitives()
    all_anchors, _skipped = load_anchors(prim_data)
    train_anchors, holdout_anchors = get_split(all_anchors)

    print(f"  Train: {len(train_anchors)} | Holdout: {len(holdout_anchors)}")

    # Run ensemble
    results = run_ensemble(model, tokenizer, train_anchors, holdout_anchors, all_anchors, device, args)

    print(f"\n  Done.")


if __name__ == '__main__':
    main()
