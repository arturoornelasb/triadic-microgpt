"""
R3 Formula Comparison: Discrete vs Continuous Rule of Three.

Loads D-A5 XL checkpoint, extracts continuous projections for all anchor
concepts, then compares 4 discrete formulas against the continuous R3
(D = C + (B - A)) that achieves 94.6%.

Formulas tested:
  A: C4 = C2 OR (C3 AND NOT C1)              — original boolean
  B: C4 = (C3 AND NOT removed) OR added       — transfer (add/remove delta)
  C: C4 = C3 XOR (C1 XOR C2)                  — symmetric XOR
  D: C4 = category-aware (dual flip + intra-category swap)

Each formula tested in:
  - Binary {0, 1} discretization (threshold > 0)
  - Ternary {-1, 0, +1} discretization (round to nearest)

CPU only. Uses existing D-A5 XL checkpoint.

Usage:
  python playground/r3_formula_comparison.py
  python playground/r3_formula_comparison.py --checkpoint checkpoints/danza_bootstrap_xl/
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

_PLAYGROUND = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_PLAYGROUND)
sys.path.insert(0, _PROJECT)
sys.path.insert(0, _PLAYGROUND)

from danza_63bit import (
    load_primitives, load_anchors, N_BITS, DanzaTriadicGPT,
)
from danza_bootstrap import (
    TRAIN_CONCEPTS, HOLDOUT_INFO, BOOTSTRAP_QUADS, get_split,
)
from src.torch_transformer import TriadicGPTConfig
try:
    from src.fast_tokenizer import FastBPETokenizer as BPETokenizer
except ImportError:
    from src.tokenizer import BPETokenizer


# ============================================================
# Load primitives metadata (for category-aware formula)
# ============================================================

def load_primitive_metadata():
    """Load layer and dual-axis info for each bit position."""
    prim_data = load_primitives()
    prims = prim_data['primitives']

    bit_to_layer = {}
    bit_to_name = {}
    for p in prims:
        bit_to_layer[p['bit']] = p['capa']
        bit_to_name[p['bit']] = p['nombre']

    # Dual axes: list of (bit_pos, bit_neg) pairs
    dual_axes = []
    ejes = prim_data.get('ejes_duales', [])
    name_to_bit = {p['nombre']: p['bit'] for p in prims}
    for pair in ejes:
        if len(pair) == 2:
            b0 = name_to_bit.get(pair[0])
            b1 = name_to_bit.get(pair[1])
            if b0 is not None and b1 is not None:
                dual_axes.append((b0, b1))

    return bit_to_layer, bit_to_name, dual_axes


# ============================================================
# Four discrete R3 formulas
# ============================================================

def formula_a_or_andnot(c1, c2, c3):
    """A: C4 = C2 OR (C3 AND NOT C1). Original boolean."""
    return c2 | (c3 & ~c1)


def formula_b_transfer(c1, c2, c3):
    """B: Transfer delta. Remove bits that C1 has but C2 doesn't,
    add bits that C2 has but C1 doesn't."""
    removed = c1 & ~c2   # bits in C1 not in C2
    added = c2 & ~c1     # bits in C2 not in C1
    return (c3 & ~removed) | added


def formula_c_xor(c1, c2, c3):
    """C: C4 = C3 XOR (C1 XOR C2). Fully symmetric."""
    return c3 ^ (c1 ^ c2)


def formula_d_category_aware(c1, c2, c3, dual_axes, bit_to_layer):
    """D: Category-aware with dual-axis flips + intra-layer swaps.

    Strategy:
    1. For each dual axis: if C1->C2 flips a pole, flip it in C3 too
    2. For non-dual bits in same layer: transfer the add/remove pattern
    3. For cross-layer additions: add to C3 as-is
    """
    result = c3.copy()
    n_bits = len(c1)

    # 1. Dual axis flips
    dual_bits = set()
    for pos, neg in dual_axes:
        if pos >= n_bits or neg >= n_bits:
            continue
        dual_bits.add(pos)
        dual_bits.add(neg)
        # Check if C1->C2 flips this axis
        c1_pos, c1_neg = c1[pos], c1[neg]
        c2_pos, c2_neg = c2[pos], c2[neg]
        if c1_pos != c2_pos or c1_neg != c2_neg:
            # Flip detected: apply same flip to C3->result
            if c3[pos] == c1_pos and c3[neg] == c1_neg:
                result[pos] = c2_pos
                result[neg] = c2_neg

    # 2. Intra-layer swaps (non-dual bits)
    layers = set(bit_to_layer.values())
    for layer in layers:
        layer_bits = [b for b in range(n_bits) if bit_to_layer.get(b) == layer and b not in dual_bits]
        for b in layer_bits:
            if c1[b] != c2[b]:
                # C1->C2 changed this bit: apply same change to C3
                if c3[b] == c1[b]:
                    result[b] = c2[b]

    return result


# ============================================================
# Ternary discretization helpers
# ============================================================

def to_binary(proj):
    """Continuous [-1,+1] -> binary {0, 1} via threshold > 0."""
    return (proj > 0).astype(np.int8)


def to_ternary(proj):
    """Continuous [-1,+1] -> ternary {-1, 0, +1} via rounding."""
    return np.clip(np.round(proj), -1, 1).astype(np.int8)


def hamming(a, b):
    """Hamming distance between two arrays."""
    return int(np.sum(a != b))


def bit_accuracy(pred, gold):
    """Bit accuracy between prediction and gold."""
    return float(np.mean(pred == gold))


# ============================================================
# Continuous R3 (what the neural model does)
# ============================================================

def continuous_r3(pa, pb, pc):
    """D = C + (B - A) in continuous space, clamped to [-1, +1]."""
    return np.clip(pc + (pb - pa), -1, 1)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='R3 Formula Comparison: Discrete vs Continuous')
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    ckpt_dir = args.checkpoint or os.path.join(_PROJECT, 'checkpoints', 'danza_bootstrap_xl')

    # Load checkpoint
    import glob as glob_mod
    step_ckpts = sorted(glob_mod.glob(os.path.join(ckpt_dir, 'model_step*.pt')))
    if step_ckpts:
        ckpt_path = step_ckpts[-1]
    else:
        ckpt_path = os.path.join(ckpt_dir, 'model_best.pt')

    print(f"{'=' * 80}")
    print(f"  R3 FORMULA COMPARISON — Discrete vs Continuous")
    print(f"{'=' * 80}")
    print(f"  Checkpoint: {os.path.basename(ckpt_path)}")

    device = torch.device('cpu')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = ckpt['config']

    config = TriadicGPTConfig(
        vocab_size=cfg['vocab_size'], block_size=cfg['block_size'],
        n_layer=cfg['n_layer'], n_embd=cfg['n_embd'],
        n_head=cfg['n_head'], n_triadic_bits=cfg['n_triadic_bits'],
    )
    model = DanzaTriadicGPT(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Load tokenizer
    tok_path = os.path.join(ckpt_dir, 'tokenizer.json')
    tokenizer = BPETokenizer.load(tok_path)

    # Load anchors and metadata
    prim_data = load_primitives()
    all_anchors, _ = load_anchors(prim_data)
    bit_to_layer, bit_to_name, dual_axes = load_primitive_metadata()

    print(f"  Anchors: {len(all_anchors)}")
    print(f"  Dual axes: {len(dual_axes)}")
    print(f"  Quads: {len(BOOTSTRAP_QUADS)}")

    # Extract continuous projections for all concepts
    @torch.no_grad()
    def get_proj(word):
        ids = tokenizer.encode(word, add_special=False)[:4]
        if not ids:
            return None
        x = torch.tensor([ids], dtype=torch.long, device=device)
        _, proj, _ = model(x)
        return proj[0].mean(dim=0).numpy()  # (63,)

    projections = {}
    for word in all_anchors:
        p = get_proj(word)
        if p is not None:
            projections[word] = p

    print(f"  Projections extracted: {len(projections)}")

    # Gold targets
    gold = {}
    for word, data in all_anchors.items():
        gold[word] = (data['target'] > 0).float().numpy().astype(np.int8)

    # ============================================================
    # Test each quad with all formulas
    # ============================================================

    formulas = ['Continuous', 'A (OR/ANDNOT)', 'B (Transfer)', 'C (XOR)', 'D (CatAware)']
    spaces = ['continuous', 'binary', 'ternary']

    # Results: formula -> space -> list of (quad_name, hamming, bit_acc)
    results = {f: {s: [] for s in spaces} for f in formulas}

    print(f"\n{'=' * 80}")
    print(f"  RESULTS PER QUAD")
    print(f"{'=' * 80}")

    header = f"  {'Quad':<35s} {'Gold':>3s}"
    for f in formulas:
        header += f" | {f[:12]:>12s}"
    print(f"\n  --- BINARY {{0,1}} ---")
    print(header)
    print(f"  {'-' * 35} {'---':>3s}" + (" | " + "-" * 12) * len(formulas))

    for a_w, b_w, c_w, d_w in BOOTSTRAP_QUADS:
        if any(w not in projections for w in [a_w, b_w, c_w]):
            continue
        if d_w not in projections or d_w not in gold:
            continue

        pa = projections[a_w]
        pb = projections[b_w]
        pc = projections[c_w]
        gold_d = gold[d_w]

        quad_name = f"{a_w}:{b_w}::{c_w}:{d_w}"

        # --- Continuous R3 ---
        cont_pred = continuous_r3(pa, pb, pc)
        cont_binary = to_binary(cont_pred)
        cont_ham = hamming(cont_binary, gold_d)
        results['Continuous']['continuous'].append((quad_name, cont_ham, bit_accuracy(cont_binary, gold_d)))
        results['Continuous']['binary'].append((quad_name, cont_ham, bit_accuracy(cont_binary, gold_d)))

        # --- Binary discretization ---
        ba = to_binary(pa)
        bb = to_binary(pb)
        bc = to_binary(pc)

        pred_a = formula_a_or_andnot(ba, bb, bc)
        pred_b = formula_b_transfer(ba, bb, bc)
        pred_c = formula_c_xor(ba, bb, bc)
        pred_d = formula_d_category_aware(ba, bb, bc, dual_axes, bit_to_layer)

        ham_a = hamming(pred_a, gold_d)
        ham_b = hamming(pred_b, gold_d)
        ham_c = hamming(pred_c, gold_d)
        ham_d = hamming(pred_d, gold_d)

        results['A (OR/ANDNOT)']['binary'].append((quad_name, ham_a, bit_accuracy(pred_a, gold_d)))
        results['B (Transfer)']['binary'].append((quad_name, ham_b, bit_accuracy(pred_b, gold_d)))
        results['C (XOR)']['binary'].append((quad_name, ham_c, bit_accuracy(pred_c, gold_d)))
        results['D (CatAware)']['binary'].append((quad_name, ham_d, bit_accuracy(pred_d, gold_d)))

        row = f"  {quad_name:<35s} {63:>3d}"
        row += f" | {cont_ham:>8d} bits"
        row += f" | {ham_a:>8d} bits"
        row += f" | {ham_b:>8d} bits"
        row += f" | {ham_c:>8d} bits"
        row += f" | {ham_d:>8d} bits"
        print(row)

    # --- Ternary discretization ---
    print(f"\n  --- TERNARY {{-1, 0, +1}} ---")
    print(header)
    print(f"  {'-' * 35} {'---':>3s}" + (" | " + "-" * 12) * len(formulas))

    for a_w, b_w, c_w, d_w in BOOTSTRAP_QUADS:
        if any(w not in projections for w in [a_w, b_w, c_w]):
            continue
        if d_w not in projections or d_w not in gold:
            continue

        pa = projections[a_w]
        pb = projections[b_w]
        pc = projections[c_w]
        gold_d = gold[d_w]

        quad_name = f"{a_w}:{b_w}::{c_w}:{d_w}"

        # Continuous (same as before)
        cont_pred = continuous_r3(pa, pb, pc)
        cont_ternary = to_ternary(cont_pred)
        cont_binary_from_tern = (cont_ternary > 0).astype(np.int8)
        cont_ham = hamming(cont_binary_from_tern, gold_d)
        results['Continuous']['ternary'].append((quad_name, cont_ham, bit_accuracy(cont_binary_from_tern, gold_d)))

        # Ternary discretization
        ta = to_ternary(pa)
        tb = to_ternary(pb)
        tc = to_ternary(pc)

        # For ternary, formulas operate on {-1,0,+1}
        # Continuous R3 in ternary space: D = clip(C + (B - A))
        tern_r3 = np.clip(tc + (tb - ta), -1, 1).astype(np.int8)
        tern_r3_bin = (tern_r3 > 0).astype(np.int8)
        ham_tern_r3 = hamming(tern_r3_bin, gold_d)

        # Formula B in ternary (transfer): works same way
        removed = ((ta == 1) & (tb != 1)).astype(np.int8)
        added = ((ta != 1) & (tb == 1)).astype(np.int8)
        neg_added = ((ta != -1) & (tb == -1)).astype(np.int8)
        pred_b_tern = tc.copy()
        pred_b_tern[removed == 1] = tb[removed == 1]
        pred_b_tern[added == 1] = 1
        pred_b_tern[neg_added == 1] = -1
        pred_b_tern_bin = (pred_b_tern > 0).astype(np.int8)
        ham_b_tern = hamming(pred_b_tern_bin, gold_d)

        # Formula D (category-aware) in ternary
        pred_d_tern = formula_d_category_aware(ta, tb, tc, dual_axes, bit_to_layer)
        pred_d_tern_bin = (pred_d_tern > 0).astype(np.int8)
        ham_d_tern = hamming(pred_d_tern_bin, gold_d)

        # For formulas A and C, convert to binary first (they need boolean ops)
        ba = (ta > 0).astype(np.int8)
        bb = (tb > 0).astype(np.int8)
        bc = (tc > 0).astype(np.int8)
        ham_a = hamming(formula_a_or_andnot(ba, bb, bc), gold_d)
        ham_c = hamming(formula_c_xor(ba, bb, bc), gold_d)

        results['A (OR/ANDNOT)']['ternary'].append((quad_name, ham_a, bit_accuracy(formula_a_or_andnot(ba, bb, bc), gold_d)))
        results['B (Transfer)']['ternary'].append((quad_name, ham_b_tern, bit_accuracy(pred_b_tern_bin, gold_d)))
        results['C (XOR)']['ternary'].append((quad_name, ham_c, bit_accuracy(formula_c_xor(ba, bb, bc), gold_d)))
        results['D (CatAware)']['ternary'].append((quad_name, ham_d_tern, bit_accuracy(pred_d_tern_bin, gold_d)))

        row = f"  {quad_name:<35s} {63:>3d}"
        row += f" | {cont_ham:>8d} bits"
        row += f" | {ham_a:>8d} bits"
        row += f" | {ham_b_tern:>8d} bits"
        row += f" | {ham_c:>8d} bits"
        row += f" | {ham_d_tern:>8d} bits"
        print(row)

    # ============================================================
    # Summary table
    # ============================================================

    print(f"\n{'=' * 80}")
    print(f"  SUMMARY — Mean Hamming Distance (lower = better)")
    print(f"{'=' * 80}")
    print(f"  {'Formula':<20s} {'Binary H':>10s} {'Binary Acc':>10s} {'Ternary H':>10s} {'Tern Acc':>10s}")
    print(f"  {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")

    for f in formulas:
        bin_data = results[f]['binary']
        tern_data = results[f]['ternary']
        if bin_data:
            bin_h = np.mean([h for _, h, _ in bin_data])
            bin_a = np.mean([a for _, _, a in bin_data])
        else:
            bin_h = bin_a = float('nan')
        if tern_data:
            tern_h = np.mean([h for _, h, _ in tern_data])
            tern_a = np.mean([a for _, _, a in tern_data])
        else:
            tern_h = tern_a = float('nan')
        print(f"  {f:<20s} {bin_h:>10.1f} {bin_a:>9.1%} {tern_h:>10.1f} {tern_a:>9.1%}")

    # Ternary arithmetic R3 (D = clip(C + B - A) in ternary)
    tern_arith = []
    for a_w, b_w, c_w, d_w in BOOTSTRAP_QUADS:
        if any(w not in projections for w in [a_w, b_w, c_w, d_w]):
            continue
        if d_w not in gold:
            continue
        ta = to_ternary(projections[a_w])
        tb = to_ternary(projections[b_w])
        tc = to_ternary(projections[c_w])
        pred = np.clip(tc + (tb - ta), -1, 1).astype(np.int8)
        pred_bin = (pred > 0).astype(np.int8)
        h = hamming(pred_bin, gold[d_w])
        a = bit_accuracy(pred_bin, gold[d_w])
        tern_arith.append((f"{a_w}:{b_w}::{c_w}:{d_w}", h, a))

    if tern_arith:
        h_mean = np.mean([h for _, h, _ in tern_arith])
        a_mean = np.mean([a for _, _, a in tern_arith])
        print(f"  {'Tern Arith D=C+B-A':<20s} {'—':>10s} {'—':>10s} {h_mean:>10.1f} {a_mean:>9.1%}")

    # ============================================================
    # Key findings
    # ============================================================

    print(f"\n{'=' * 80}")
    print(f"  KEY FINDINGS")
    print(f"{'=' * 80}")

    # Find best formula per space
    for space in ['binary', 'ternary']:
        best_f = None
        best_h = float('inf')
        for f in formulas:
            data = results[f][space]
            if data:
                h = np.mean([h for _, h, _ in data])
                if h < best_h:
                    best_h = h
                    best_f = f
        print(f"  Best in {space}: {best_f} (mean Hamming = {best_h:.1f})")

    if tern_arith:
        print(f"  Ternary arithmetic: mean Hamming = {np.mean([h for _, h, _ in tern_arith]):.1f}")

    # Compare continuous vs best discrete
    cont_bin = results['Continuous']['binary']
    if cont_bin:
        cont_h = np.mean([h for _, h, _ in cont_bin])
        print(f"\n  Continuous R3 (D=C+B-A, then threshold): Hamming = {cont_h:.1f}")
        print(f"  -> This is what the neural model does (94.6% ensemble)")

    # Analyze where formulas disagree most
    print(f"\n  WORST QUADS (highest Hamming) per formula (binary):")
    for f in ['Continuous', 'B (Transfer)', 'D (CatAware)']:
        data = sorted(results[f]['binary'], key=lambda x: -x[1])
        if data:
            worst = data[0]
            print(f"    {f}: {worst[0]} -> H={worst[1]}, acc={worst[2]:.1%}")

    # Save results
    results_path = os.path.join(ckpt_dir, 'r3_formula_comparison.json')
    serializable = {}
    for f in formulas:
        serializable[f] = {}
        for s in spaces:
            serializable[f][s] = [
                {'quad': q, 'hamming': int(h), 'bit_accuracy': round(a, 4)}
                for q, h, a in results[f][s]
            ]
    if tern_arith:
        serializable['Tern_Arithmetic'] = {
            'ternary': [
                {'quad': q, 'hamming': int(h), 'bit_accuracy': round(a, 4)}
                for q, h, a in tern_arith
            ]
        }

    with open(results_path, 'w') as fp:
        json.dump(serializable, fp, indent=2)
    print(f"\n  Results saved: {results_path}")


if __name__ == '__main__':
    main()
