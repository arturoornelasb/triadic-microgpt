"""
Post-training analysis of Bootstrap D-A5.

Run after the XL training finishes:
  python playground/analyze_bootstrap.py --checkpoint checkpoints/danza_bootstrap_xl/

Generates:
  1. Training curves (loss, bit_acc, dead_bits)
  2. Per-bit accuracy heatmap (which bits it learns and which it doesn't)
  3. Exact vs approximate quads comparison
  4. Success criteria evaluation
  5. Comparison vs base scale and vs D2 full-supervision baseline
"""

import os
import sys
import csv
import json
import argparse
import numpy as np

_PLAYGROUND = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_PLAYGROUND)
sys.path.insert(0, _PROJECT)
sys.path.insert(0, _PLAYGROUND)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

import torch
import torch.nn.functional as F

from danza_63bit import (
    load_primitives, load_anchors, N_BITS,
    DanzaTriadicGPT, ANCHOR_TRANSLATIONS, SKIP_ANCHORS,
)
from danza_bootstrap import (
    TRAIN_CONCEPTS, HOLDOUT_INFO, BOOTSTRAP_QUADS,
    get_split,
)


def load_training_log(ckpt_dir):
    path = os.path.join(ckpt_dir, 'training_log.csv')
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def load_bootstrap_results(ckpt_dir):
    path = os.path.join(ckpt_dir, 'bootstrap_results.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_training_curves(log, ckpt_dir):
    if not HAS_PLT:
        print("  [SKIP] matplotlib not available")
        return

    steps = [r['step'] for r in log]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss curves
    ax = axes[0, 0]
    ax.plot(steps, [r['loss'] for r in log], 'b-', label='total', linewidth=2)
    ax.plot(steps, [r['lang_loss'] for r in log], 'g--', label='lang', alpha=0.7)
    tri = [r['tri_loss'] for r in log]
    if any(t > 0 for t in tri):
        ax.plot(steps, tri, 'r-', label='triadic', linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Supervision losses
    ax = axes[0, 1]
    sup = [r['sup_loss'] for r in log]
    sub = [r['sub_loss'] for r in log]
    if any(s > 0 for s in sup):
        ax.plot(steps, sup, 'orange', label='sup_loss', linewidth=1.5)
    if any(s > 0 for s in sub):
        ax.plot(steps, sub, 'purple', label='sub_loss', linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Supervision Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bit accuracy
    ax = axes[1, 0]
    ax.plot(steps, [r['bit_acc_train'] for r in log], 'b-', label='train', linewidth=2)
    ax.plot(steps, [r['bit_acc_holdout'] for r in log], 'r--', label='holdout', linewidth=2)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='chance')
    ax.set_xlabel('Step')
    ax.set_ylabel('Bit Accuracy')
    ax.set_title('Bit Accuracy (Train vs Holdout)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.3, 1.0)

    # Dead bits
    ax = axes[1, 1]
    ax.plot(steps, [r['dead_bits'] for r in log], 'k-', linewidth=2)
    ax.axhline(y=61, color='red', linestyle=':', alpha=0.5, label='base (61 dead)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Dead Bits (all-same)')
    ax.set_title('Dead Bits Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 63)

    plt.tight_layout()
    out_path = os.path.join(ckpt_dir, 'training_curves.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def per_bit_analysis(model, tokenizer, all_anchors, prim_data, device):
    """Compute per-bit accuracy across all anchor concepts."""
    model.eval()
    bit_to_name = prim_data['bit_to_name']

    def get_proj(word):
        ids = tokenizer.encode(word, add_special=False)[:4]
        if not ids:
            return None
        x = torch.tensor([ids], dtype=torch.long, device=device)
        _, proj, _ = model(x)
        return proj[0].mean(dim=0)

    train_a, holdout_a = get_split(all_anchors)

    results = {'train': {}, 'holdout': {}}

    for split_name, anchors in [('train', train_a), ('holdout', holdout_a)]:
        bit_correct = np.zeros(N_BITS)
        bit_total = np.zeros(N_BITS)
        bit_pred_sum = np.zeros(N_BITS)

        for word, data in anchors.items():
            proj = get_proj(word)
            if proj is None:
                continue
            pred = (proj > 0).float().cpu().numpy()
            gold = (data['target'] > 0).float().cpu().numpy()
            correct = (pred == gold).astype(float)
            bit_correct += correct
            bit_total += 1
            bit_pred_sum += pred

        mask = bit_total > 0
        bit_acc = np.where(mask, bit_correct / bit_total, 0)
        bit_activation = np.where(mask, bit_pred_sum / bit_total, 0)
        n_concepts = int(bit_total.max()) if mask.any() else 0

        # Dead bits = always 0 or always 1
        dead = ((bit_activation < 0.05) | (bit_activation > 0.95)) & mask
        n_dead = int(dead.sum())

        results[split_name] = {
            'bit_acc': bit_acc,
            'bit_activation': bit_activation,
            'dead_mask': dead,
            'n_dead': n_dead,
            'n_concepts': n_concepts,
        }

    # Print per-bit table
    print(f"\n{'=' * 90}")
    print(f"  PER-BIT ACCURACY ANALYSIS")
    print(f"{'=' * 90}")
    print(f"  {'Bit':>3s} {'Primitive':20s} {'Capa':>4s} {'Train%':>7s} {'Hold%':>7s} {'Act%':>6s} {'Status':>8s}")
    print(f"  {'-'*3} {'-'*20} {'-'*4} {'-'*7} {'-'*7} {'-'*6} {'-'*8}")

    prims = prim_data['primitives']
    bit_info = {p['bit']: p for p in prims}

    problem_bits = []
    strong_bits = []

    for b in range(N_BITS):
        p = bit_info.get(b, {})
        name = p.get('nombre', '???')
        capa = p.get('capa', '?')
        t_acc = results['train']['bit_acc'][b]
        h_acc = results['holdout']['bit_acc'][b]
        act = results['holdout']['bit_activation'][b]

        dead_t = results['train']['dead_mask'][b]
        dead_h = results['holdout']['dead_mask'][b]

        if dead_h:
            status = 'DEAD'
            problem_bits.append((b, name, capa, t_acc, h_acc))
        elif h_acc < 0.6:
            status = 'WEAK'
            problem_bits.append((b, name, capa, t_acc, h_acc))
        elif h_acc > 0.85:
            status = 'STRONG'
            strong_bits.append((b, name, capa, h_acc))
        else:
            status = 'ok'

        print(f"  {b:3d} {name:20s} {capa:>4} {t_acc:7.1%} {h_acc:7.1%} {act:6.1%}   {status}")

    print(f"\n  Train: {results['train']['n_dead']}/{N_BITS} dead bits, "
          f"{results['train']['n_concepts']} concepts")
    print(f"  Holdout: {results['holdout']['n_dead']}/{N_BITS} dead bits, "
          f"{results['holdout']['n_concepts']} concepts")

    print(f"\n  PROBLEM BITS ({len(problem_bits)}):")
    for b, name, capa, ta, ha in sorted(problem_bits, key=lambda x: x[4]):
        print(f"    bit {b:2d} ({name:20s}, capa {capa}): train={ta:.1%} holdout={ha:.1%}")

    print(f"\n  STRONG BITS ({len(strong_bits)}):")
    for b, name, capa, ha in sorted(strong_bits, key=lambda x: -x[3]):
        print(f"    bit {b:2d} ({name:20s}, capa {capa}): holdout={ha:.1%}")

    return results


def quad_type_analysis(results_json):
    """Compare exact-axis quads vs approximate quads."""
    if results_json is None:
        print("  [SKIP] No bootstrap_results.json found")
        return

    # Classify quads
    exact_quads = {
        'queen': 'man:woman=king:queen',
        'hate': 'happy:sad=love:hate',
        'prisoner': 'open:close=free:prisoner',
        'liquid': 'man:woman=solid:liquid',
        'logical': 'hot:cold=creative:logical',
    }
    partial_quads = {'quiet', 'silent'}  # hot:cold=loud:quiet, bright:dark=loud:quiet
    approx_template_quads = {'slow', 'poor', 'bitter', 'humble'}  # bright:dark as template
    approx_valence_quads = {'bad', 'evil', 'dead'}  # happy:sad as template
    approx_action_quads = {'learn'}  # open:close=teach:learn
    approx_knowledge_quads = {'ignorant'}  # hot:cold=wise:ignorant

    print(f"\n{'=' * 80}")
    print(f"  QUAD TYPE ANALYSIS: Exact vs Approximate")
    print(f"{'=' * 80}")

    categories = {
        'Exact axis': [],
        'Partial axis': [],
        'Approx (bright:dark template)': [],
        'Approx (happy:sad template)': [],
        'Approx (action direction)': [],
        'Approx (knowledge)': [],
        'Control (no path)': [],
    }

    for concept, data in results_json.items():
        eng_words = data.get('english', [concept])
        rtype = data.get('type', 'CTRL')
        d_acc = data.get('direct_acc', 0)
        r3_acc = data.get('best_r3_acc', 0)
        delta = data.get('algebraic_improvement', 0)

        primary = eng_words[0] if eng_words else concept

        if rtype == 'CTRL':
            categories['Control (no path)'].append((concept, d_acc, r3_acc, delta))
        elif primary in exact_quads:
            categories['Exact axis'].append((concept, d_acc, r3_acc, delta))
        elif primary in partial_quads:
            categories['Partial axis'].append((concept, d_acc, r3_acc, delta))
        elif primary in approx_template_quads:
            categories['Approx (bright:dark template)'].append((concept, d_acc, r3_acc, delta))
        elif primary in approx_valence_quads:
            categories['Approx (happy:sad template)'].append((concept, d_acc, r3_acc, delta))
        elif primary in approx_action_quads:
            categories['Approx (action direction)'].append((concept, d_acc, r3_acc, delta))
        elif primary in approx_knowledge_quads:
            categories['Approx (knowledge)'].append((concept, d_acc, r3_acc, delta))
        else:
            categories['Exact axis'].append((concept, d_acc, r3_acc, delta))

    for cat, items in categories.items():
        if not items:
            continue
        mean_d = np.mean([x[1] for x in items])
        mean_r3 = np.mean([x[2] for x in items]) if any(x[2] > 0 for x in items) else 0
        mean_delta = np.mean([x[3] for x in items])
        print(f"\n  {cat} ({len(items)} concepts):")
        print(f"    Mean direct: {mean_d:.1%}  |  Mean R3: {mean_r3:.1%}  |  Mean delta: {mean_delta:+.1%}")
        for concept, da, r3a, delta in sorted(items, key=lambda x: -x[3]):
            r3_str = f"{r3a:.1%}" if r3a > 0 else " ---  "
            delta_str = f"{delta:+.1%}" if r3a > 0 else "  --- "
            print(f"    {concept:20s}  direct={da:.1%}  R3={r3_str}  delta={delta_str}")


def compare_scales(base_dir, xl_dir):
    """Compare base vs XL results."""
    base = load_bootstrap_results(base_dir)
    xl = load_bootstrap_results(xl_dir)

    if base is None or xl is None:
        print("  [SKIP] Need both base and XL results for comparison")
        return

    print(f"\n{'=' * 80}")
    print(f"  SCALE COMPARISON: Base (5M) vs XL (40M)")
    print(f"{'=' * 80}")
    print(f"  {'Concept':20s} {'Type':5s} {'Base_D':>7s} {'XL_D':>7s} {'Base_R3':>8s} {'XL_R3':>8s} {'d_D':>7s} {'d_R3':>8s}")
    print(f"  {'-'*20} {'-'*5} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*7} {'-'*8}")

    for concept in sorted(base.keys()):
        if concept not in xl:
            continue
        b, x = base[concept], xl[concept]
        rtype = b.get('type', '?')
        bd = b.get('direct_acc', 0)
        xd = x.get('direct_acc', 0)
        br3 = b.get('best_r3_acc', 0)
        xr3 = x.get('best_r3_acc', 0)

        dd = xd - bd
        dr3 = xr3 - br3 if (br3 > 0 and xr3 > 0) else 0

        br3_s = f"{br3:.1%}" if br3 > 0 else "  --- "
        xr3_s = f"{xr3:.1%}" if xr3 > 0 else "  --- "
        dr3_s = f"{dr3:+.1%}" if dr3 != 0 else "  --- "

        print(f"  {concept:20s} {rtype:5s} {bd:7.1%} {xd:7.1%} {br3_s:>8s} {xr3_s:>8s} {dd:+7.1%} {dr3_s:>8s}")


def success_criteria(results_json):
    """Evaluate D-A5 success criteria from results JSON."""
    if results_json is None:
        return

    print(f"\n{'=' * 80}")
    print(f"  D-A5 SUCCESS CRITERIA")
    print(f"{'=' * 80}")

    r3_direct, r3_alg, ctrl_direct = [], [], []

    for concept, data in results_json.items():
        rtype = data.get('type', 'CTRL')
        d_acc = data.get('direct_acc', 0)
        r3_acc = data.get('best_r3_acc', 0)
        ens_acc = data.get('ensemble_acc', 0)
        alg = max(r3_acc, ens_acc) if r3_acc > 0 else 0

        if rtype == 'R3':
            r3_direct.append(d_acc)
            r3_alg.append(alg)
        else:
            ctrl_direct.append(d_acc)

    md = np.mean(r3_direct) if r3_direct else 0
    ma = np.mean(r3_alg) if r3_alg else 0
    mc = np.mean(ctrl_direct) if ctrl_direct else 0

    criteria = [
        ("Holdout direct > 75%", md > 0.75, f"{md:.1%}"),
        ("Algebraic > 80%", ma > 0.80, f"{ma:.1%}"),
        ("Algebraic > direct + 5%", (ma - md) > 0.05, f"delta={ma-md:+.1%}"),
        ("Reachable > control + 10%", (ma - mc) > 0.10, f"delta={ma-mc:+.1%}"),
    ]

    for name, passed, value in criteria:
        icon = "PASS" if passed else "FAIL"
        print(f"  [{icon}] {name:35s} -> {value}")

    # D2 baseline comparison
    print(f"\n  Reference baselines:")
    print(f"    D2 full supervision: 89.5% bit accuracy, 90% subsumption")
    print(f"    Base scale (5M, 100 steps triadic): ~90% (but 61/63 dead bits)")
    print(f"\n  This run:")
    print(f"    R3 direct mean:  {md:.1%}")
    print(f"    R3 algebraic:    {ma:.1%}")
    print(f"    CTRL direct:     {mc:.1%}")
    print(f"    Algebraic lift:  {ma - md:+.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/danza_bootstrap_xl/')
    parser.add_argument('--base', type=str,
                        default='checkpoints/danza_bootstrap_base/')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no-model', action='store_true',
                        help='Skip model-based analysis (just parse logs/results)')
    args = parser.parse_args()

    ckpt_dir = os.path.join(_PROJECT, args.checkpoint)
    base_dir = os.path.join(_PROJECT, args.base)

    print(f"\n{'#' * 80}")
    print(f"  D-A5 BOOTSTRAP ANALYSIS")
    print(f"  Checkpoint: {ckpt_dir}")
    print(f"{'#' * 80}")

    # 1. Training curves
    print(f"\n[1/5] Training Curves")
    log = load_training_log(ckpt_dir)
    last = log[-1]
    print(f"  Steps: {int(last['step'])} | Loss: {last['loss']:.4f} | "
          f"Bit acc train: {last['bit_acc_train']:.1%} | "
          f"Bit acc holdout: {last['bit_acc_holdout']:.1%} | "
          f"Dead bits: {int(last['dead_bits'])}")

    # Check if triadic losses have kicked in
    triadic_start = None
    for r in log:
        if r['tri_loss'] > 0 and triadic_start is None:
            triadic_start = int(r['step'])
    if triadic_start:
        triadic_steps = int(last['step']) - triadic_start
        print(f"  Triadic loss active since step {triadic_start} ({triadic_steps} triadic steps)")

    plot_training_curves(log, ckpt_dir)

    # 2. Bootstrap results
    print(f"\n[2/5] Bootstrap Results")
    results = load_bootstrap_results(ckpt_dir)
    if results is None:
        print("  No bootstrap_results.json yet.")
        print("  Run prediction phase:")
        print(f"    python playground/danza_bootstrap.py --phase predict --checkpoint {args.checkpoint}")
    else:
        success_criteria(results)

    # 3. Quad type analysis
    print(f"\n[3/5] Quad Type Analysis")
    quad_type_analysis(results)

    # 4. Scale comparison
    print(f"\n[4/5] Scale Comparison")
    compare_scales(base_dir, ckpt_dir)

    # 5. Per-bit analysis (requires loading model)
    print(f"\n[5/5] Per-Bit Analysis")
    if args.no_model:
        print("  [SKIP] --no-model flag set")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        print(f"  Loading model on {device}...")

        prim_data = load_primitives()
        all_anchors, _skipped = load_anchors(prim_data)

        # Load model
        from src.fast_tokenizer import FastBPETokenizer
        tok = FastBPETokenizer.load(os.path.join(ckpt_dir, 'tokenizer.json'))

        from src.torch_transformer import TriadicGPTConfig
        config = TriadicGPTConfig(
            vocab_size=tok.vocab_size,
            n_triadic_bits=N_BITS,
            n_layer=12, n_head=8, n_embd=512,
        )
        model = DanzaTriadicGPT(config).to(device)
        ckpt = torch.load(os.path.join(ckpt_dir, 'model_best.pt'),
                          map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Loaded model from step {ckpt.get('step', '?')}")

        per_bit_analysis(model, tokenizer=tok, all_anchors=all_anchors,
                         prim_data=prim_data, device=device)

    print(f"\n{'#' * 80}")
    print(f"  ANALYSIS COMPLETE")
    print(f"{'#' * 80}\n")


if __name__ == '__main__':
    main()
