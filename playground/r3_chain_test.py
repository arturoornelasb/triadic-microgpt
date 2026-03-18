"""
R3 Chain & Fork Composition Test.

Tests whether the Rule of Three COMPOSES across multiple steps.
Three test types:

1. ROUND-TRIP: A:B::C:D_pred -> B:A::D_pred:C_pred. Compare C_pred vs C_gold.
   If R3 is perfectly invertible, round-trip accuracy = single-step accuracy.
   If errors compound multiplicatively: round-trip ~ single^2 (~81%).
   If errors are sub-linear (coherent structure): round-trip > single^2.

2. FORK: Same relationship A:B applied to multiple C->D pairs.
   Measures consistency of the transformation vector across targets.
   Groups: bright:dark (5 quads), happy:sad (3), hot:cold (3), man:woman (2), open:close (2).

3. TRANSITIVE CHAIN: Step1 with relationship R1, Step2 with relationship R2.
   A1:B1::C:D_pred -> A2:B2::D_pred:E_pred. Compare E_pred vs E_gold.
   Tests if predicted vectors carry enough info for a second transformation.

CPU only. Uses D-A5 XL checkpoint.

Usage:
  python playground/r3_chain_test.py
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
# R3 operations
# ============================================================

def continuous_r3(pa, pb, pc):
    """D = C + (B - A) in continuous space, clamped."""
    return np.clip(pc + (pb - pa), -1, 1)


def to_ternary(proj):
    """Continuous -> ternary {-1, 0, +1}."""
    return np.clip(np.round(proj), -1, 1).astype(np.int8)


def to_binary(proj):
    """Continuous -> binary {0, 1}."""
    return (proj > 0).astype(np.int8)


def hamming(a, b):
    return int(np.sum(a != b))


def bit_accuracy(a, b):
    return float(np.mean(a == b))


def cosine_sim(a, b):
    """Cosine similarity in continuous space."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='R3 Chain & Fork Composition Test')
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    ckpt_dir = args.checkpoint or os.path.join(_PROJECT, 'checkpoints', 'danza_bootstrap_xl')

    # Load checkpoint
    import glob as glob_mod
    step_ckpts = sorted(glob_mod.glob(os.path.join(ckpt_dir, 'model_step*.pt')))
    ckpt_path = step_ckpts[-1] if step_ckpts else os.path.join(ckpt_dir, 'model_best.pt')

    print(f"{'=' * 80}")
    print(f"  R3 CHAIN & FORK COMPOSITION TEST")
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

    tok_path = os.path.join(ckpt_dir, 'tokenizer.json')
    tokenizer = BPETokenizer.load(tok_path)

    prim_data = load_primitives()
    all_anchors, _ = load_anchors(prim_data)

    # Extract projections
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

    gold_binary = {}
    for word, data in all_anchors.items():
        gold_binary[word] = (data['target'] > 0).float().numpy().astype(np.int8)

    n_bits = N_BITS
    print(f"  Anchors: {len(projections)} | Bits: {n_bits}")
    print(f"  Quads: {len(BOOTSTRAP_QUADS)}")

    # ============================================================
    # TEST 1: ROUND-TRIP (Forward + Reverse)
    # ============================================================

    print(f"\n{'=' * 80}")
    print(f"  TEST 1: ROUND-TRIP (A:B::C:D_pred -> B:A::D_pred:C_pred)")
    print(f"{'=' * 80}")
    print(f"  If multiplicative degradation: round-trip ~ single^2")
    print(f"  If sub-linear (coherent): round-trip > single^2\n")

    header = f"  {'Quad':<35s} {'1-step H':>8s} {'1-step Acc':>10s} {'RT H':>6s} {'RT Acc':>8s} {'Predict':>8s}"
    print(header)
    print(f"  {'-' * 35} {'-' * 8} {'-' * 10} {'-' * 6} {'-' * 8} {'-' * 8}")

    single_accs_cont = []
    rt_accs_cont = []
    single_accs_tern = []
    rt_accs_tern = []

    for a_w, b_w, c_w, d_w in BOOTSTRAP_QUADS:
        if any(w not in projections for w in [a_w, b_w, c_w, d_w]):
            continue

        pa, pb, pc, pd = projections[a_w], projections[b_w], projections[c_w], projections[d_w]
        gold_d = gold_binary[d_w]
        gold_c = gold_binary[c_w]

        # --- Continuous round-trip ---
        # Step 1: A:B::C:D_pred
        d_pred_cont = continuous_r3(pa, pb, pc)
        d_pred_bin = to_binary(d_pred_cont)
        step1_h = hamming(d_pred_bin, gold_d)
        step1_acc = bit_accuracy(d_pred_bin, gold_d)

        # Step 2: B:A::D_pred:C_pred (reverse direction)
        c_pred_cont = continuous_r3(pb, pa, d_pred_cont)
        c_pred_bin = to_binary(c_pred_cont)
        rt_h = hamming(c_pred_bin, gold_c)
        rt_acc = bit_accuracy(c_pred_bin, gold_c)

        # Prediction: if multiplicative, expected RT acc = step1_acc^2
        predicted = step1_acc ** 2

        single_accs_cont.append(step1_acc)
        rt_accs_cont.append(rt_acc)

        qname = f"{a_w}:{b_w}::{c_w}:{d_w}"
        print(f"  {qname:<35s} {step1_h:>8d} {step1_acc:>10.1%} {rt_h:>6d} {rt_acc:>8.1%} {predicted:>8.1%}")

    mean_single_cont = np.mean(single_accs_cont)
    mean_rt_cont = np.mean(rt_accs_cont)
    predicted_rt = mean_single_cont ** 2

    print(f"\n  --- Continuous R3 Summary ---")
    print(f"  Mean 1-step accuracy:     {mean_single_cont:.1%}")
    print(f"  Mean round-trip accuracy: {mean_rt_cont:.1%}")
    print(f"  Predicted (multiplicative): {predicted_rt:.1%}")
    print(f"  Actual vs predicted:      {mean_rt_cont - predicted_rt:+.1%}")
    if mean_rt_cont > predicted_rt:
        print(f"  --> SUB-LINEAR: errors are coherent, structure preserved")
    else:
        print(f"  --> SUPER-LINEAR: errors compound, structure degrades")

    # --- Ternary round-trip ---
    print(f"\n  --- Ternary Round-Trip ---")
    header = f"  {'Quad':<35s} {'1-step H':>8s} {'1-step Acc':>10s} {'RT H':>6s} {'RT Acc':>8s} {'Predict':>8s}"
    print(header)
    print(f"  {'-' * 35} {'-' * 8} {'-' * 10} {'-' * 6} {'-' * 8} {'-' * 8}")

    for a_w, b_w, c_w, d_w in BOOTSTRAP_QUADS:
        if any(w not in projections for w in [a_w, b_w, c_w, d_w]):
            continue

        pa, pb, pc, pd = projections[a_w], projections[b_w], projections[c_w], projections[d_w]
        gold_d_t = to_ternary(np.array([(2 * g - 1) for g in gold_binary[d_w]], dtype=np.float32))
        gold_c_t = to_ternary(np.array([(2 * g - 1) for g in gold_binary[c_w]], dtype=np.float32))

        # Step 1: ternary R3
        d_pred_cont = continuous_r3(pa, pb, pc)
        d_pred_t = to_ternary(d_pred_cont)
        step1_h = hamming(d_pred_t, gold_d_t)
        step1_acc = bit_accuracy(d_pred_t, gold_d_t)

        # Step 2: reverse using ternary prediction as continuous input
        # Convert ternary back to "continuous" (it's already in [-1, +1])
        d_pred_as_cont = d_pred_t.astype(np.float32)
        c_pred_cont = continuous_r3(pb, pa, d_pred_as_cont)
        c_pred_t = to_ternary(c_pred_cont)
        rt_h = hamming(c_pred_t, gold_c_t)
        rt_acc = bit_accuracy(c_pred_t, gold_c_t)

        predicted = step1_acc ** 2
        single_accs_tern.append(step1_acc)
        rt_accs_tern.append(rt_acc)

        qname = f"{a_w}:{b_w}::{c_w}:{d_w}"
        print(f"  {qname:<35s} {step1_h:>8d} {step1_acc:>10.1%} {rt_h:>6d} {rt_acc:>8.1%} {predicted:>8.1%}")

    mean_single_tern = np.mean(single_accs_tern)
    mean_rt_tern = np.mean(rt_accs_tern)
    predicted_rt_tern = mean_single_tern ** 2

    print(f"\n  --- Ternary Summary ---")
    print(f"  Mean 1-step accuracy:     {mean_single_tern:.1%}")
    print(f"  Mean round-trip accuracy: {mean_rt_tern:.1%}")
    print(f"  Predicted (multiplicative): {predicted_rt_tern:.1%}")
    print(f"  Actual vs predicted:      {mean_rt_tern - predicted_rt_tern:+.1%}")
    if mean_rt_tern > predicted_rt_tern:
        print(f"  --> SUB-LINEAR: errors are coherent, structure preserved")
    else:
        print(f"  --> SUPER-LINEAR: errors compound, structure degrades")

    # ============================================================
    # TEST 2: FORK CONSISTENCY
    # ============================================================

    print(f"\n{'=' * 80}")
    print(f"  TEST 2: FORK CONSISTENCY (same A:B, multiple C->D)")
    print(f"{'=' * 80}")
    print(f"  Same relationship applied to N targets.")
    print(f"  Measures: is the transformation vector consistent?\n")

    # Group quads by (A, B) relationship
    rel_groups = defaultdict(list)
    for a_w, b_w, c_w, d_w in BOOTSTRAP_QUADS:
        rel_groups[(a_w, b_w)].append((c_w, d_w))

    fork_results = []

    for (a_w, b_w), targets in sorted(rel_groups.items(), key=lambda x: -len(x[1])):
        if len(targets) < 2:
            continue
        if a_w not in projections or b_w not in projections:
            continue

        pa, pb = projections[a_w], projections[b_w]

        # The "transformation vector" in continuous space
        transform = pb - pa

        print(f"  Relationship: {a_w} -> {b_w} ({len(targets)} targets)")
        print(f"  Transform vector norm: {np.linalg.norm(transform):.4f}")

        # Apply to each target and collect the "effective transforms"
        effective_transforms = []
        accs = []
        for c_w, d_w in targets:
            if c_w not in projections or d_w not in projections:
                continue
            pc, pd = projections[c_w], projections[d_w]

            # Predicted D
            d_pred = continuous_r3(pa, pb, pc)
            d_pred_bin = to_binary(d_pred)
            gold_d = gold_binary[d_w]
            acc = bit_accuracy(d_pred_bin, gold_d)
            accs.append(acc)

            # The "effective transform" for this pair: D_gold - C
            eff_t = pd - pc
            effective_transforms.append(eff_t)

            print(f"    {c_w:>12s} -> {d_w:<12s}  acc={acc:.1%}  eff_norm={np.linalg.norm(eff_t):.4f}")

        # Measure consistency: pairwise cosine between effective transforms
        if len(effective_transforms) >= 2:
            cosines = []
            for i in range(len(effective_transforms)):
                for j in range(i + 1, len(effective_transforms)):
                    cs = cosine_sim(effective_transforms[i], effective_transforms[j])
                    cosines.append(cs)

            # Also: cosine between the "canonical" transform (B-A) and each effective
            canon_cosines = []
            for eff in effective_transforms:
                canon_cosines.append(cosine_sim(transform, eff))

            mean_pairwise = np.mean(cosines)
            mean_canon = np.mean(canon_cosines)
            print(f"    Pairwise cosine (effective transforms): {mean_pairwise:.4f}")
            print(f"    Canonical cosine (B-A vs effective):    {mean_canon:.4f}")
            print(f"    Mean accuracy: {np.mean(accs):.1%}")
            fork_results.append({
                'relationship': f"{a_w}->{b_w}",
                'n_targets': len(targets),
                'pairwise_cosine': mean_pairwise,
                'canonical_cosine': mean_canon,
                'mean_acc': float(np.mean(accs)),
            })
        print()

    # ============================================================
    # TEST 3: TRANSITIVE 2-STEP CHAINS
    # ============================================================

    print(f"{'=' * 80}")
    print(f"  TEST 3: TRANSITIVE 2-STEP CHAINS")
    print(f"{'=' * 80}")
    print(f"  Step1: R1 applied to C -> D_pred")
    print(f"  Step2: R2 applied to D_pred -> E_pred")
    print(f"  Compare E_pred vs E_gold\n")

    # Build chains: find quads where D of one is C of another
    # Since our quads don't share C/D concepts, we construct synthetic chains
    # using concepts that appear in multiple quads in different roles.
    #
    # Strategy: if concept X appears as D in quad1 and as C or A or B in quad2,
    # we can chain. But our quads mostly don't overlap this way.
    #
    # Alternative: construct 2-step chains by applying TWO different relationships
    # to the same starting concept. E.g.:
    #   Step 1: happy:sad :: good:bad_pred    (valence flip on good)
    #   Step 2: bright:dark :: bad_pred:?     (perception flip on bad_pred)
    #   Compare ? vs the gold for bright:dark applied to bad

    # Collect all relationship pairs and their transforms
    rel_transforms = {}
    for a_w, b_w, c_w, d_w in BOOTSTRAP_QUADS:
        key = (a_w, b_w)
        if key not in rel_transforms and a_w in projections and b_w in projections:
            rel_transforms[key] = projections[b_w] - projections[a_w]

    # Build 2-step chains: apply R1 to get intermediate, then R2 to get final
    # We need: a concept that appears as D in some quad AND as C in another
    # If not available, we construct synthetic chains using any anchor as intermediary.
    #
    # Synthetic chain: pick concept X, apply R1 to get X', apply R2 to get X''
    # Gold for X'': apply R2 to actual X_gold (using actual quad data)

    chains = []

    # Find concepts that are D (output) in one quad and also have known gold projections
    # Then apply a second relationship to them
    for i, (a1, b1, c1, d1) in enumerate(BOOTSTRAP_QUADS):
        for j, (a2, b2, c2, d2) in enumerate(BOOTSTRAP_QUADS):
            if i == j:
                continue
            # Chain: if d1 == c2, we can chain quad_i -> quad_j
            if d1 == c2:
                chains.append(('natural', i, j, a1, b1, c1, d1, a2, b2, d2))
            # Also: if d1 appears as a concept we can use as C in another rel
            # Synthetic: use d1 as new C for relationship (a2, b2)
            # Gold would be: R3(a2, b2, d1_gold) where d1_gold is the actual projection
            elif d1 in projections and d2 in projections:
                # Only add if d1 != c2 (otherwise it's the natural case above)
                # and d1 is not already part of quad j
                if d1 not in (a2, b2, c2, d2) and (a1, b1) != (a2, b2):
                    chains.append(('synthetic', i, j, a1, b1, c1, d1, a2, b2, None))

    # Limit synthetic chains to avoid explosion
    natural_chains = [c for c in chains if c[0] == 'natural']
    synthetic_chains = [c for c in chains if c[0] == 'synthetic']

    # Sample synthetic chains if too many
    if len(synthetic_chains) > 30:
        rng = np.random.RandomState(42)
        synthetic_chains = [synthetic_chains[i] for i in rng.choice(len(synthetic_chains), 30, replace=False)]

    all_chains = natural_chains + synthetic_chains

    print(f"  Natural chains (D1=C2): {len(natural_chains)}")
    print(f"  Synthetic chains (sample): {len(synthetic_chains)}")
    print()

    chain_1step_accs = []
    chain_2step_accs = []
    chain_gold_2step_accs = []  # 2-step with gold intermediate (upper bound)

    header = f"  {'Chain':<55s} {'1-step':>7s} {'2-step':>7s} {'Gold2s':>7s} {'Type':>10s}"
    print(header)
    print(f"  {'-' * 55} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 10}")

    for chain_info in all_chains:
        ctype = chain_info[0]
        _, qi, qj, a1, b1, c1, d1, a2, b2, d2_or_none = chain_info

        if any(w not in projections for w in [a1, b1, c1, d1, a2, b2]):
            continue

        pa1, pb1, pc1 = projections[a1], projections[b1], projections[c1]
        pd1_gold = projections[d1]
        pa2, pb2 = projections[a2], projections[b2]

        # Step 1: A1:B1::C1:D1_pred
        d1_pred_cont = continuous_r3(pa1, pb1, pc1)
        d1_pred_bin = to_binary(d1_pred_cont)
        gold_d1 = gold_binary[d1]
        step1_acc = bit_accuracy(d1_pred_bin, gold_d1)

        # Step 2: A2:B2::D1_pred:E_pred
        e_pred_cont = continuous_r3(pa2, pb2, d1_pred_cont)
        e_pred_bin = to_binary(e_pred_cont)

        # Step 2 with gold intermediate: A2:B2::D1_gold:E_gold_pred
        e_gold_pred_cont = continuous_r3(pa2, pb2, pd1_gold)
        e_gold_pred_bin = to_binary(e_gold_pred_cont)

        # What's the gold for E?
        if ctype == 'natural' and d2_or_none in gold_binary:
            gold_e = gold_binary[d2_or_none]
            e_name = d2_or_none
        else:
            # Synthetic: gold E is R3(a2, b2, d1_gold) discretized from actual projection
            # We use the gold binary of d1 to construct what the "ideal" output would be
            # Actually, for synthetic we don't have a "gold E" concept name.
            # Instead, compare predicted vs gold-intermediate-predicted:
            # This measures degradation from using predicted vs gold D1
            gold_e = e_gold_pred_bin  # upper bound = what you'd get with perfect step 1
            e_name = f"R2({d1})"

        step2_acc = bit_accuracy(e_pred_bin, gold_e)
        gold2_acc = bit_accuracy(e_gold_pred_bin, gold_e)

        chain_1step_accs.append(step1_acc)
        chain_2step_accs.append(step2_acc)
        chain_gold_2step_accs.append(gold2_acc)

        chain_desc = f"{a1}:{b1}::{c1}->{d1} | {a2}:{b2}::{d1}->{e_name}"
        if len(chain_desc) > 55:
            chain_desc = chain_desc[:52] + "..."
        print(f"  {chain_desc:<55s} {step1_acc:>7.1%} {step2_acc:>7.1%} {gold2_acc:>7.1%} {ctype:>10s}")

    if chain_1step_accs:
        mean_1s = np.mean(chain_1step_accs)
        mean_2s = np.mean(chain_2step_accs)
        mean_g2 = np.mean(chain_gold_2step_accs)
        predicted_2s = mean_1s ** 2

        print(f"\n  --- 2-Step Chain Summary ---")
        print(f"  Mean step-1 accuracy:         {mean_1s:.1%}")
        print(f"  Mean 2-step accuracy:          {mean_2s:.1%}")
        print(f"  Mean 2-step (gold intermed.):  {mean_g2:.1%}")
        print(f"  Predicted (multiplicative):    {predicted_2s:.1%}")
        print(f"  Actual vs predicted:           {mean_2s - predicted_2s:+.1%}")
        if mean_2s > predicted_2s:
            print(f"  --> SUB-LINEAR: chained predictions preserve structure")
        else:
            print(f"  --> SUPER-LINEAR: errors compound across steps")

    # ============================================================
    # GRAND SUMMARY
    # ============================================================

    print(f"\n{'=' * 80}")
    print(f"  GRAND SUMMARY")
    print(f"{'=' * 80}")

    summary = {
        'round_trip_continuous': {
            'single_step_acc': float(mean_single_cont),
            'round_trip_acc': float(mean_rt_cont),
            'predicted_multiplicative': float(predicted_rt),
            'delta': float(mean_rt_cont - predicted_rt),
            'sub_linear': bool(mean_rt_cont > predicted_rt),
        },
        'round_trip_ternary': {
            'single_step_acc': float(mean_single_tern),
            'round_trip_acc': float(mean_rt_tern),
            'predicted_multiplicative': float(predicted_rt_tern),
            'delta': float(mean_rt_tern - predicted_rt_tern),
            'sub_linear': bool(mean_rt_tern > predicted_rt_tern),
        },
        'fork_consistency': fork_results,
    }

    if chain_1step_accs:
        summary['transitive_chains'] = {
            'n_chains': len(chain_1step_accs),
            'mean_step1_acc': float(mean_1s),
            'mean_step2_acc': float(mean_2s),
            'mean_gold_step2_acc': float(mean_g2),
            'predicted_multiplicative': float(predicted_2s),
            'delta': float(mean_2s - predicted_2s),
            'sub_linear': bool(mean_2s > predicted_2s),
        }

    rows = [
        ("Continuous R3 (1-step)", f"{mean_single_cont:.1%}", ""),
        ("Continuous R3 (round-trip)", f"{mean_rt_cont:.1%}", f"pred={predicted_rt:.1%}, delta={mean_rt_cont - predicted_rt:+.1%}"),
        ("Ternary R3 (1-step)", f"{mean_single_tern:.1%}", ""),
        ("Ternary R3 (round-trip)", f"{mean_rt_tern:.1%}", f"pred={predicted_rt_tern:.1%}, delta={mean_rt_tern - predicted_rt_tern:+.1%}"),
    ]

    if chain_1step_accs:
        rows.append(("2-step chain", f"{mean_2s:.1%}", f"pred={predicted_2s:.1%}, delta={mean_2s - predicted_2s:+.1%}"))

    if fork_results:
        mean_fork_cos = np.mean([f['pairwise_cosine'] for f in fork_results])
        mean_canon_cos = np.mean([f['canonical_cosine'] for f in fork_results])
        rows.append(("Fork pairwise cosine", f"{mean_fork_cos:.4f}", "1.0 = perfectly consistent"))
        rows.append(("Fork canonical cosine", f"{mean_canon_cos:.4f}", "T_eff vs T_canonical"))

    print(f"\n  {'Test':<30s} {'Result':>10s}  {'Notes':<40s}")
    print(f"  {'-' * 30} {'-' * 10}  {'-' * 40}")
    for label, val, notes in rows:
        print(f"  {label:<30s} {val:>10s}  {notes:<40s}")

    # Verdict
    rt_sub = mean_rt_cont > predicted_rt
    tern_sub = mean_rt_tern > predicted_rt_tern
    chain_sub = mean_2s > predicted_2s if chain_1step_accs else None

    print(f"\n  VERDICT:")
    if rt_sub and tern_sub:
        print(f"  The triadic bit space supports COMPOSITIONAL operations.")
        print(f"  Errors are coherent (same bits fail), not random.")
        print(f"  This is evidence of COMPUTATIONAL SUBSTRATE, not just encoding.")
    elif rt_sub or tern_sub:
        print(f"  Mixed results: sub-linear in {'continuous' if rt_sub else 'ternary'},")
        print(f"  super-linear in {'ternary' if rt_sub else 'continuous'}.")
    else:
        print(f"  Errors compound multiplicatively or worse.")
        print(f"  The space encodes but does not support computation.")

    # Save
    out_path = os.path.join(ckpt_dir, 'r3_chain_test.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved: {out_path}")


if __name__ == '__main__':
    main()
