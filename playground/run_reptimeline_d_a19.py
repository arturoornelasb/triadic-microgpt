"""
Run reptimeline BitDiscovery on D-A19 (GPT-2 Medium 355M, fixed algebra).

Extracts bit codes for 158 v2 anchors, runs BitDiscovery, saves results.
Compares against D-A14 and D-A18.

Usage:
    conda run -n triadic-microgpt python playground/run_reptimeline_d_a19.py
"""

import os, sys, json, numpy as np, torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'playground'))

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from playground.gpt2_355m_sparsity import GPT2MediumSparsity
from playground.danza_63bit import load_primitives, load_all_anchors, N_BITS
from src.triadic import BitwiseMapper
from reptimeline.core import ConceptSnapshot
from reptimeline.discovery import BitDiscovery

CKPT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints', 'danza_gpt2_355m_sparsity_v2')
CKPT = os.path.join(CKPT_DIR, 'model_best.pt')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'reptimeline', 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("  REPTIMELINE: D-A19 GPT-2 Medium 355M BitDiscovery")
    print("=" * 60)

    # Load model
    print("\n  [1/3] Loading D-A19 model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state = torch.load(CKPT, map_location=device, weights_only=False)
    qmode = state.get('quantize_mode', 'fsq')
    step = state.get('step', '?')

    gpt2 = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2MediumSparsity(gpt2, n_triadic_bits=N_BITS, quantize_mode=qmode)
    model.load_state_dict(state['model_state_dict'])
    model.to(device)
    model.eval()
    mapper = BitwiseMapper(N_BITS)

    print(f"    Loaded: {N_BITS} bits, quantize={qmode}, step={step}")
    print(f"    Device: {device}")

    # Extract bit codes for all anchors
    print("\n  [2/3] Extracting bit codes for anchors...")
    prim_data = load_primitives()
    anchors, _ = load_all_anchors(prim_data)

    codes = {}
    for word in anchors.keys():
        ids = tokenizer.encode(word, add_special_tokens=False)[:8]
        if not ids:
            continue
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(x)
        proj = out[1][0].mean(dim=0).cpu().numpy()
        bits = mapper.get_bits(proj)
        codes[word] = bits

    print(f"    Extracted {len(codes)} / {len(anchors)} concepts")

    # Create snapshot and run discovery
    print("\n  [3/3] Running BitDiscovery...")
    snapshot = ConceptSnapshot(
        step=step if isinstance(step, int) else 50000,
        codes=codes,
        metadata={
            'model': 'D-A19 GPT2MediumSparsity (355M, fixed algebra)',
            'checkpoint': CKPT,
            'n_bits': N_BITS,
            'quantize_mode': qmode,
            'backbone': 'gpt2-medium',
        },
    )

    discovery = BitDiscovery(
        dead_threshold=0.02,
        dual_threshold=-0.3,
        dep_confidence=0.9,
        triadic_threshold=0.7,
        triadic_min_interaction=0.2,
    )
    report = discovery.discover(snapshot, top_k=15)
    discovery.print_report(report)

    # Save results
    results = {
        'model': 'D-A19 GPT2MediumSparsity (355M, fixed algebra)',
        'checkpoint': CKPT,
        'step': step,
        'n_concepts': len(codes),
        'n_bits': N_BITS,
        'quantize_mode': qmode,
        'n_active_bits': report.n_active_bits,
        'n_dead_bits': report.n_dead_bits,
        'n_duals': len(report.discovered_duals),
        'n_dependencies': len(report.discovered_deps),
        'n_triadic_interactions': len(report.discovered_triadic_deps),
        'bit_semantics': [
            {
                'bit': bs.bit_index,
                'activation_rate': round(bs.activation_rate, 3),
                'top_concepts': bs.top_concepts[:10],
                'anti_concepts': bs.anti_concepts[:5],
            }
            for bs in report.bit_semantics
        ],
        'discovered_duals': [
            {
                'bit_a': d.bit_a,
                'bit_b': d.bit_b,
                'anti_correlation': round(d.anti_correlation, 3),
            }
            for d in report.discovered_duals[:20]
        ],
        'discovered_triadic': [
            {
                'bit_i': t.bit_i,
                'bit_j': t.bit_j,
                'bit_r': t.bit_r,
                'p_r_ij': round(t.p_r_given_ij, 3),
                'p_r_i': round(t.p_r_given_i, 3),
                'p_r_j': round(t.p_r_given_j, 3),
                'strength': round(t.interaction_strength, 3),
                'support': t.support,
            }
            for t in report.discovered_triadic_deps[:30]
        ],
    }

    out_path = os.path.join(OUTPUT_DIR, 'd_a19_discovery.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out_path}")

    # Compare with D-A14 and D-A18
    for label, filename in [('D-A14 v2', 'd_a14_v2_autolabel_report.json'),
                            ('D-A18', 'd_a18_discovery.json')]:
        prev_path = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(prev_path):
            with open(prev_path) as f:
                prev = json.load(f)
            print(f"\n  COMPARISON: D-A19 vs {label}")
            print("  " + "-" * 50)
            print(f"  {'Metric':<25} {'D-A19':>10} {label:>10}")
            print("  " + "-" * 50)
            print(f"  {'Active bits':<25} {report.n_active_bits:>10} {prev.get('n_active_bits', '?'):>10}")
            print(f"  {'Dead bits':<25} {report.n_dead_bits:>10} {prev.get('n_dead_bits', '?'):>10}")
            print(f"  {'Duals':<25} {len(report.discovered_duals):>10} {prev.get('n_duals', '?'):>10}")
            print(f"  {'Dependencies':<25} {len(report.discovered_deps):>10} {prev.get('n_dependencies', '?'):>10}")
            print(f"  {'Triadic interactions':<25} {len(report.discovered_triadic_deps):>10} {prev.get('n_triadic_interactions', '?'):>10}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
