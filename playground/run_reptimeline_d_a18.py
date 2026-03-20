"""
Run reptimeline BitDiscovery on D-A18 (UnifiedTriadicGPT).

Standalone script because TriadicExtractor hardcodes TriadicGPT loading.
Extracts bit codes for 158 v2 anchors, runs BitDiscovery, saves results.

Usage:
    conda run -n triadic-microgpt python playground/run_reptimeline_d_a18.py
"""

import os, sys, json, numpy as np, torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'playground'))

from playground.unified_final import (
    UnifiedTriadicGPT, load_primitives, load_all_anchors, N_BITS,
)
from src.torch_transformer import TriadicGPTConfig
from src.triadic import BitwiseMapper
from src.fast_tokenizer import FastBPETokenizer
from reptimeline.core import ConceptSnapshot
from reptimeline.discovery import BitDiscovery

CKPT = os.path.join(PROJECT_ROOT, 'checkpoints', 'danza_unified_xl', 'model_best.pt')
TOK = os.path.join(os.path.dirname(CKPT), 'tokenizer.json')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'reptimeline', 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("  REPTIMELINE: D-A18 UnifiedTriadicGPT BitDiscovery")
    print("=" * 60)

    # Load model
    print("\n  [1/3] Loading D-A18 model (CPU)...")
    state = torch.load(CKPT, map_location='cpu', weights_only=False)
    config = state.get('config', state.get('model_config'))
    if isinstance(config, dict):
        config = TriadicGPTConfig(**config)
    model = UnifiedTriadicGPT(config)
    model.load_state_dict(state['model_state_dict'], strict=False)
    model.eval()
    tokenizer = FastBPETokenizer.load(TOK)
    mapper = BitwiseMapper(N_BITS)

    n_sup = getattr(config, 'n_supervised_bits', 30)
    n_free = config.n_triadic_bits - n_sup
    print(f"    Loaded: {config.n_triadic_bits} bits ({n_sup} sup + {n_free} free)")

    # Extract bit codes for all anchors
    print("\n  [2/3] Extracting bit codes for 158 anchors...")
    prim_data = load_primitives()
    anchors, _ = load_all_anchors(prim_data)

    codes = {}
    for word in anchors.keys():
        ids = tokenizer.encode(word, add_special=False)[:4]
        if not ids:
            continue
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            out = model(x)
        proj = out[1][0].mean(dim=0).cpu().numpy()
        bits = mapper.get_bits(proj)
        codes[word] = bits

    print(f"    Extracted {len(codes)} / {len(anchors)} concepts")

    # Create snapshot and run discovery
    print("\n  [3/3] Running BitDiscovery...")
    snapshot = ConceptSnapshot(
        step=50000,
        codes=codes,
        metadata={
            'model': 'D-A18 UnifiedTriadicGPT',
            'checkpoint': CKPT,
            'n_bits': N_BITS,
            'n_supervised': n_sup,
            'n_free': n_free,
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
        'model': 'D-A18 UnifiedTriadicGPT (iFSQ + hybrid 30+33)',
        'checkpoint': CKPT,
        'n_concepts': len(codes),
        'n_bits': N_BITS,
        'n_supervised': n_sup,
        'n_free': n_free,
        'n_active_bits': report.n_active_bits,
        'n_dead_bits': report.n_dead_bits,
        'n_duals': len(report.discovered_duals),
        'n_dependencies': len(report.discovered_deps),
        'n_triadic_interactions': len(report.discovered_triadic_deps),
        'bit_semantics': [
            {
                'bit': bs.bit_index,
                'activation_rate': round(bs.activation_rate, 3),
                'is_supervised': bs.bit_index < n_sup,
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
                'a_is_supervised': d.bit_a < n_sup,
                'b_is_supervised': d.bit_b < n_sup,
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

    out_path = os.path.join(OUTPUT_DIR, 'd_a18_discovery.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out_path}")

    # Compare with D-A14
    d14_path = os.path.join(OUTPUT_DIR, 'd_a14_v2_autolabel_report.json')
    if os.path.exists(d14_path):
        with open(d14_path) as f:
            d14 = json.load(f)
        print("\n" + "=" * 60)
        print("  COMPARISON: D-A18 vs D-A14 v2")
        print("=" * 60)
        print(f"  {'Metric':<25} {'D-A18':>10} {'D-A14':>10}")
        print("  " + "-" * 47)
        print(f"  {'Active bits':<25} {report.n_active_bits:>10} {d14.get('n_active_bits', '?'):>10}")
        print(f"  {'Dead bits':<25} {report.n_dead_bits:>10} {d14.get('n_dead_bits', '?'):>10}")
        print(f"  {'Duals':<25} {len(report.discovered_duals):>10} {d14.get('n_duals', '?'):>10}")
        print(f"  {'Dependencies':<25} {len(report.discovered_deps):>10} {d14.get('n_dependencies', '?'):>10}")
        print(f"  {'Triadic interactions':<25} {len(report.discovered_triadic_deps):>10} {d14.get('n_triadic_interactions', '?'):>10}")
    print("=" * 60)


if __name__ == '__main__':
    main()
