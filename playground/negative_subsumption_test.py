"""
D-A16: Negative Subsumption Test — False Positive Rate on Non-Subsumptive Pairs.

Tests whether the triadic projection erroneously reports subsumption (bit-subset)
on concept pairs that should NOT have a subsumption relationship.

Categories of negative pairs:
  - Siblings: dog/cat, red/blue, happy/sad (same level, neither subsumes)
  - Unrelated: fire/quiet, love/fast (no semantic link)
  - Reversed: dog does NOT subsume animal (direction matters)
  - Cross-domain: emotion/color, action/object

Also runs positive pairs for comparison (sensitivity / true positive rate).

Usage:
  python playground/negative_subsumption_test.py
  python playground/negative_subsumption_test.py --checkpoint checkpoints/danza_bootstrap_xl/
  python playground/negative_subsumption_test.py --checkpoint checkpoints/torch_run15_strongalign/
"""

import os
import sys
import argparse
import numpy as np
import torch

_PLAYGROUND = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_PLAYGROUND)
sys.path.insert(0, _PROJECT)
sys.path.insert(0, _PLAYGROUND)

from src.torch_transformer import TriadicGPT, TriadicGPTConfig
try:
    from src.fast_tokenizer import FastBPETokenizer as BPETokenizer
except ImportError:
    from src.tokenizer import BPETokenizer

try:
    from danza_63bit import DanzaTriadicGPT
except ImportError:
    DanzaTriadicGPT = None


# ============================================================
# Pair definitions
# ============================================================

# NEGATIVE pairs: A should NOT subsume B (bit-subset should fail)
NEGATIVE_PAIRS = [
    # --- Siblings (same hypernym, neither subsumes the other) ---
    ("dog", "cat"), ("cat", "bird"), ("dog", "fish"),
    ("horse", "cow"), ("pig", "sheep"), ("bird", "fish"),
    ("red", "blue"), ("blue", "green"), ("red", "green"),
    ("happy", "sad"), ("happy", "angry"), ("sad", "afraid"),
    ("love", "hate"), ("hope", "fear"), ("joy", "pain"),
    ("king", "queen"), ("prince", "princess"), ("brother", "sister"),
    ("mother", "father"), ("boy", "girl"), ("man", "woman"),
    ("morning", "night"), ("summer", "winter"),
    ("table", "chair"), ("bed", "lamp"),
    # --- Unrelated (no semantic link) ---
    ("fire", "quiet"), ("love", "fast"), ("water", "proud"),
    ("sun", "chair"), ("moon", "bread"), ("star", "angry"),
    ("rain", "king"), ("snow", "teach"), ("cloud", "red"),
    ("river", "happy"), ("mountain", "lamp"), ("ocean", "slow"),
    ("tree", "queen"), ("door", "fish"), ("window", "sad"),
    # --- Reversed true pairs (hyponym does NOT subsume hypernym) ---
    ("dog", "animal"), ("cat", "animal"), ("bird", "animal"),
    ("boy", "person"), ("girl", "person"), ("king", "person"),
    ("red", "color"), ("blue", "color"),
    ("happy", "feeling"), ("sad", "feeling"),
    ("run", "action"), ("walk", "action"),
    # --- Cross-domain ---
    ("happy", "red"), ("love", "table"), ("king", "river"),
    ("fast", "blue"), ("teach", "green"), ("brave", "chair"),
]

# POSITIVE pairs: A SHOULD subsume B (hypernym -> hyponym)
POSITIVE_PAIRS = [
    ("animal", "dog"), ("animal", "cat"), ("animal", "bird"),
    ("animal", "fish"), ("animal", "horse"), ("animal", "cow"),
    ("person", "boy"), ("person", "girl"), ("person", "man"),
    ("person", "woman"), ("person", "king"), ("person", "queen"),
    ("person", "mother"), ("person", "father"),
    ("feeling", "happy"), ("feeling", "sad"), ("feeling", "angry"),
    ("feeling", "love"), ("feeling", "hate"), ("feeling", "fear"),
    ("color", "red"), ("color", "blue"), ("color", "green"),
    ("place", "city"), ("place", "school"), ("place", "house"),
    ("action", "run"), ("action", "walk"), ("action", "swim"),
    ("nature", "fire"), ("nature", "water"), ("nature", "sun"),
]


# ============================================================
# Model loading (handles both checkpoint types)
# ============================================================

def load_checkpoint(ckpt_dir, device):
    """Load model + tokenizer from a checkpoint directory."""
    # Find best model file
    import glob as glob_mod
    step_ckpts = sorted(glob_mod.glob(os.path.join(ckpt_dir, 'model_step*.pt')))
    best_path = os.path.join(ckpt_dir, 'model_best.pt')
    named_ckpts = sorted(glob_mod.glob(os.path.join(ckpt_dir, 'model_*_best.pt')))

    if named_ckpts:
        ckpt_path = named_ckpts[-1]
    elif os.path.exists(best_path):
        ckpt_path = best_path
    elif step_ckpts:
        ckpt_path = step_ckpts[-1]
    else:
        raise FileNotFoundError(f"No model checkpoint found in {ckpt_dir}")

    print(f"  Checkpoint: {os.path.basename(ckpt_path)}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = ckpt['config']

    n_bits = cfg.get('n_triadic_bits', 63)
    config = TriadicGPTConfig(
        vocab_size=cfg['vocab_size'], block_size=cfg['block_size'],
        n_layer=cfg['n_layer'], n_embd=cfg['n_embd'],
        n_head=cfg['n_head'], n_triadic_bits=n_bits, dropout=0.0,
    )

    # Use DanzaTriadicGPT if available and checkpoint comes from danza pipeline
    is_danza = 'danza' in ckpt_dir.lower() and DanzaTriadicGPT is not None
    ModelClass = DanzaTriadicGPT if is_danza else TriadicGPT
    model = ModelClass(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Tokenizer
    tok_path = os.path.join(ckpt_dir, 'tokenizer.json')
    meta_path = tok_path + '.meta'
    if os.path.exists(meta_path):
        tokenizer = BPETokenizer.load(tok_path)
    else:
        tokenizer = BPETokenizer(vocab_size=cfg['vocab_size'])
        tokenizer.load(tok_path)

    print(f"  Model: {cfg['n_layer']}L/{cfg['n_embd']}D/{cfg['n_head']}H/{n_bits}bits")
    print(f"  Type:  {'DanzaTriadicGPT' if is_danza else 'TriadicGPT'}")
    return model, tokenizer, config


# ============================================================
# Projection + subsumption logic
# ============================================================

@torch.no_grad()
def get_projection(model, tokenizer, word, device):
    """Get the triadic projection for a single word. Returns (63,) tensor or None."""
    ids = tokenizer.encode(word, add_special=False)[:4]
    if not ids:
        return None
    x = torch.tensor([ids], dtype=torch.long, device=device)
    _, proj, _ = model(x)
    return proj[0].mean(dim=0)  # average over token positions


def check_subsumption(proj_a, proj_b):
    """Check if A's ON bits are a subset of B's ON bits (A subsumes B).

    Returns (is_subset, bit_inheritance_pct, n_on_a, n_on_b).
    """
    bits_a = (proj_a > 0)
    bits_b = (proj_b > 0)
    n_on_a = bits_a.sum().item()
    n_on_b = bits_b.sum().item()

    if n_on_a == 0:
        return True, 1.0, 0, n_on_b  # vacuously true

    # What fraction of A's ON bits are also ON in B?
    shared = (bits_a & bits_b).sum().item()
    inheritance = shared / n_on_a
    is_subset = (shared == n_on_a)

    return is_subset, inheritance, int(n_on_a), int(n_on_b)


# ============================================================
# Main evaluation
# ============================================================

def run_evaluation(model, tokenizer, device):
    """Run negative + positive subsumption tests. Returns results dict."""
    print("\n[1/2] Evaluating NEGATIVE pairs (should NOT show subsumption)...")
    neg_results = []
    for a, b in NEGATIVE_PAIRS:
        pa = get_projection(model, tokenizer, a, device)
        pb = get_projection(model, tokenizer, b, device)
        if pa is None or pb is None:
            continue
        is_sub, inherit, n_a, n_b = check_subsumption(pa, pb)
        neg_results.append({
            'a': a, 'b': b,
            'false_subsumption': is_sub,
            'bit_inheritance': inherit,
            'n_on_a': n_a, 'n_on_b': n_b,
        })

    print(f"\n[2/2] Evaluating POSITIVE pairs (should show subsumption)...")
    pos_results = []
    for a, b in POSITIVE_PAIRS:
        pa = get_projection(model, tokenizer, a, device)
        pb = get_projection(model, tokenizer, b, device)
        if pa is None or pb is None:
            continue
        is_sub, inherit, n_a, n_b = check_subsumption(pa, pb)
        pos_results.append({
            'a': a, 'b': b,
            'true_subsumption': is_sub,
            'bit_inheritance': inherit,
            'n_on_a': n_a, 'n_on_b': n_b,
        })

    return neg_results, pos_results


def print_report(neg_results, pos_results):
    """Print the full D-A16 report."""
    sep = "=" * 72

    # --- Negative detail table ---
    print(f"\n{sep}")
    print("  NEGATIVE PAIRS (should NOT subsume)")
    print(sep)
    print(f"  {'A':>12s} {'B':>12s}  {'Sub?':>5s}  {'Inherit':>8s}  {'ON_A':>5s}  {'ON_B':>5s}")
    print(f"  {'-'*12} {'-'*12}  {'-'*5}  {'-'*8}  {'-'*5}  {'-'*5}")
    for r in neg_results:
        mark = " FP!" if r['false_subsumption'] else "  ok"
        print(f"  {r['a']:>12s} {r['b']:>12s}  {mark}  {r['bit_inheritance']:8.1%}"
              f"  {r['n_on_a']:5d}  {r['n_on_b']:5d}")

    # --- Positive detail table ---
    print(f"\n{sep}")
    print("  POSITIVE PAIRS (should subsume)")
    print(sep)
    print(f"  {'A':>12s} {'B':>12s}  {'Sub?':>5s}  {'Inherit':>8s}  {'ON_A':>5s}  {'ON_B':>5s}")
    print(f"  {'-'*12} {'-'*12}  {'-'*5}  {'-'*8}  {'-'*5}  {'-'*5}")
    for r in pos_results:
        mark = "  TP" if r['true_subsumption'] else "MISS"
        print(f"  {r['a']:>12s} {r['b']:>12s}  {mark}  {r['bit_inheritance']:8.1%}"
              f"  {r['n_on_a']:5d}  {r['n_on_b']:5d}")

    # --- Summary statistics ---
    n_neg = len(neg_results)
    n_pos = len(pos_results)
    n_fp = sum(1 for r in neg_results if r['false_subsumption'])
    n_tp = sum(1 for r in pos_results if r['true_subsumption'])

    fpr = n_fp / n_neg if n_neg > 0 else 0.0
    tpr = n_tp / n_pos if n_pos > 0 else 0.0

    neg_inherit = [r['bit_inheritance'] for r in neg_results]
    pos_inherit = [r['bit_inheritance'] for r in pos_results]

    print(f"\n{sep}")
    print("  D-A16 SUMMARY")
    print(sep)
    print(f"  Negative pairs tested:    {n_neg}")
    print(f"  Positive pairs tested:    {n_pos}")
    print()
    print(f"  FALSE POSITIVE RATE:      {fpr:.1%}  ({n_fp}/{n_neg})")
    print(f"  TRUE POSITIVE RATE:       {tpr:.1%}  ({n_tp}/{n_pos})")
    print()
    print(f"  Neg bit inheritance:      mean={np.mean(neg_inherit):.1%}  "
          f"std={np.std(neg_inherit):.1%}  max={np.max(neg_inherit):.1%}")
    print(f"  Pos bit inheritance:      mean={np.mean(pos_inherit):.1%}  "
          f"std={np.std(pos_inherit):.1%}  min={np.min(pos_inherit):.1%}")
    print(f"  Inheritance gap:          {np.mean(pos_inherit) - np.mean(neg_inherit):+.1%}")
    print()
    print(f"  TARGETS:")
    print(f"    FPR < 10%:              {'PASS' if fpr < 0.10 else 'FAIL'}")
    print(f"    TPR > 50%:              {'PASS' if tpr > 0.50 else 'FAIL'}")
    print(f"    Inheritance gap > 15%:  {'PASS' if (np.mean(pos_inherit) - np.mean(neg_inherit)) > 0.15 else 'FAIL'}")
    print(sep)


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='D-A16: Negative Subsumption Test')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint directory (default: checkpoints/danza_bootstrap_xl/)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_dir = args.checkpoint or os.path.join(_PROJECT, 'checkpoints', 'danza_bootstrap_xl')
    if not os.path.isdir(ckpt_dir):
        alt = os.path.join(_PROJECT, 'checkpoints', 'torch_run15_strongalign')
        if os.path.isdir(alt):
            ckpt_dir = alt
        else:
            print(f"ERROR: checkpoint dir not found: {ckpt_dir}")
            sys.exit(1)

    print()
    print("=" * 72)
    print("  D-A16: NEGATIVE SUBSUMPTION TEST")
    print("=" * 72)
    print(f"  Device: {device}")

    model, tokenizer, config = load_checkpoint(ckpt_dir, device)
    neg_results, pos_results = run_evaluation(model, tokenizer, device)
    print_report(neg_results, pos_results)


if __name__ == '__main__':
    main()
