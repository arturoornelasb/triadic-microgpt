"""
Analyze danza_v2 (158 anchors) model with reptimeline discovery.

Compares with bootstrap and hybrid models.
"""

import os
import sys
import json
import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PLAYGROUND = os.path.dirname(_THIS_DIR)
_PROJECT = os.path.dirname(_PLAYGROUND)
sys.path.insert(0, _PROJECT)
sys.path.insert(0, _PLAYGROUND)

from common import (
    load_primitives, N_BITS, print_header, print_section, save_results,
)
from danza_63bit import load_anchors, load_all_anchors, DanzaTriadicGPT
from src.torch_transformer import TriadicGPTConfig
from src.triadic import BitwiseValidator

try:
    from src.fast_tokenizer import FastBPETokenizer as BPETokenizer
except ImportError:
    from src.tokenizer import BPETokenizer

from reptimeline.core import ConceptSnapshot
from reptimeline.discovery import BitDiscovery
from reptimeline.reconcile import Reconciler
from reptimeline.overlays.primitive_overlay import PrimitiveOverlay


CKPT_DIR = os.path.join(_PROJECT, 'checkpoints', 'danza_63bit_xl_v2')
CKPT = os.path.join(CKPT_DIR, 'model_best.pt')
TOK = os.path.join(CKPT_DIR, 'tokenizer.json')


def load_v2(device='cpu'):
    """Load danza v2 model."""
    ckpt = torch.load(CKPT, map_location=device, weights_only=True)
    cfg = ckpt['config']

    config = TriadicGPTConfig(
        vocab_size=cfg['vocab_size'], block_size=cfg['block_size'],
        n_layer=cfg['n_layer'], n_embd=cfg['n_embd'],
        n_head=cfg['n_head'], n_triadic_bits=cfg['n_triadic_bits'],
    )
    model = DanzaTriadicGPT(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    tokenizer = BPETokenizer.load(TOK)
    return model, tokenizer


@torch.no_grad()
def extract_projections(model, tokenizer, words, device='cpu'):
    """Extract projections for a list of words."""
    results = {}
    for word in words:
        ids = tokenizer.encode(word, add_special=False)[:4]
        if not ids:
            continue
        x = torch.tensor([ids], dtype=torch.long, device=device)
        triadic_proj = model(x)[1]  # (1, T, 63)
        proj = triadic_proj[0].mean(dim=0).float().cpu().numpy()
        bits = (proj > 0).astype(np.int8)
        results[word] = bits
    return results


def bits_to_mask(bits):
    """Convert numpy bit array to bitmask integer."""
    mask = 0
    for i, b in enumerate(bits):
        if b:
            mask |= (1 << i)
    return mask


def test_subsumption_bitwise(codes, anchors):
    """Test subsumption using BitwiseValidator."""
    bv = BitwiseValidator()
    total = 0
    correct = 0

    for word, bits in codes.items():
        if word not in anchors:
            continue
        anchor = anchors[word]
        # anchors have 'target' tensor with -1/+1 values
        target = anchor['target'] if isinstance(anchor, dict) else anchor
        if hasattr(target, 'numpy'):
            target = target.numpy()
        gold_bits = (np.array(target) > 0).astype(np.int8)
        mask_model = bits_to_mask(bits)
        mask_gold = bits_to_mask(gold_bits)
        if bv.subsumes(mask_model, mask_gold):
            correct += 1
        total += 1

    return correct, total


def test_analogies_bitwise(codes):
    """Test analogies using BitwiseValidator."""
    bv = BitwiseValidator()
    analogies = [
        ('man', 'woman', 'king', 'queen'),
        ('cold', 'hot', 'quiet', 'loud'),
        ('happy', 'sad', 'love', 'hate'),
        ('good', 'evil', 'light', 'dark'),
        ('teacher', 'student', 'doctor', 'patient'),
        ('big', 'small', 'fast', 'slow'),
    ]

    results = []
    for a_w, b_w, c_w, d_expected in analogies:
        if not all(w in codes for w in [a_w, b_w, c_w, d_expected]):
            continue
        a = bits_to_mask(codes[a_w])
        b = bits_to_mask(codes[b_w])
        c = bits_to_mask(codes[c_w])
        d_gold = bits_to_mask(codes[d_expected])

        d_pred = bv.analogy(a, b, c)
        sim = bv.similarity(d_pred, d_gold)
        results.append({
            'analogy': f"{a_w}:{b_w}::{c_w}:?",
            'expected': d_expected,
            'similarity': sim,
            'exact_match': d_pred == d_gold,
        })

    return results


def main():
    print_header("REPTIMELINE ANALYSIS -- DANZA V2 (158 ANCHORS)")
    print(f"  Checkpoint: {CKPT}")
    print(f"  Architecture: 63 supervised bits, 158 anchor concepts")

    device = 'cpu'
    print(f"\n  Loading model on {device}...")
    model, tokenizer = load_v2(device)

    # Load concepts
    prim_data = load_primitives()
    all_anchors, _ = load_all_anchors(prim_data)
    test_words = [
        'king', 'queen', 'fire', 'water', 'love', 'hate', 'truth', 'lie',
        'god', 'devil', 'light', 'dark', 'life', 'death', 'war', 'peace',
        'freedom', 'slavery', 'beauty', 'ugliness', 'courage', 'fear',
        'wisdom', 'ignorance', 'order', 'chaos', 'creation', 'destruction',
        'hope', 'despair', 'joy', 'sadness', 'anger', 'calm', 'fast', 'slow',
        'hot', 'cold', 'big', 'small', 'old', 'young', 'rich', 'poor',
        'strong', 'weak', 'happy', 'sad', 'good', 'evil', 'friend', 'enemy',
        'doctor', 'patient', 'teacher', 'student', 'loud', 'quiet',
    ]
    all_words = list(set(list(all_anchors.keys()) + test_words))

    print(f"  Extracting projections for {len(all_words)} concepts...")
    codes = extract_projections(model, tokenizer, all_words, device)
    print(f"  Got {len(codes)} concepts")

    # ============================================================
    # Run discovery
    # ============================================================
    print_section("BIT DISCOVERY")

    discovery = BitDiscovery(
        dead_threshold=0.02,
        dual_threshold=-0.3,
        dep_confidence=0.9,
        triadic_threshold=0.7,
        triadic_min_interaction=0.2,
    )
    snapshot = ConceptSnapshot(
        step=50000,
        codes={w: bits.tolist() for w, bits in codes.items()},
    )
    report = discovery.discover(snapshot, top_k=10)
    discovery.print_report(report)

    # ============================================================
    # Bitwise operations test
    # ============================================================
    print_section("BITWISE VALIDATOR TESTS")

    # Subsumption
    correct, total = test_subsumption_bitwise(codes, all_anchors)
    print(f"  Subsumption (bitwise): {correct}/{total} = {100*correct/total:.1f}%")

    # Analogies
    analogy_results = test_analogies_bitwise(codes)
    if analogy_results:
        print(f"\n  Analogies (bitwise):")
        for r in analogy_results:
            status = "EXACT" if r['exact_match'] else f"sim={r['similarity']:.3f}"
            print(f"    {r['analogy']:30s} -> {r['expected']:10s}  {status}")
        mean_sim = np.mean([r['similarity'] for r in analogy_results])
        exact = sum(1 for r in analogy_results if r['exact_match'])
        print(f"  Mean similarity: {mean_sim:.3f}, Exact matches: {exact}/{len(analogy_results)}")

    # ============================================================
    # Gap analysis for key pairs
    # ============================================================
    print_section("GAP ANALYSIS (BITWISE)")

    bv = BitwiseValidator()
    pairs = [
        ('love', 'hate'), ('life', 'death'), ('light', 'dark'),
        ('good', 'evil'), ('man', 'woman'), ('king', 'queen'),
    ]
    for w1, w2 in pairs:
        if w1 not in codes or w2 not in codes:
            continue
        m1 = bits_to_mask(codes[w1])
        m2 = bits_to_mask(codes[w2])
        gap = bv.explain_gap(m1, m2)
        shared_count = bin(gap['shared']).count('1')
        only1 = bin(gap['only_in_a']).count('1')
        only2 = bin(gap['only_in_b']).count('1')
        sim = bv.similarity(m1, m2)
        print(f"  {w1:>10s} vs {w2:<10s}  shared={shared_count:>2d}  "
              f"only_{w1[:4]}={only1:>2d}  only_{w2[:4]}={only2:>2d}  "
              f"sim={sim:.3f}")

    # ============================================================
    # Reconcile with manual primitives
    # ============================================================
    print_section("RECONCILIATION")

    primitivos_path = os.path.join(_PROJECT, 'data', 'primitivos.json')
    if os.path.exists(primitivos_path):
        overlay = PrimitiveOverlay(primitivos_path)
        reconciler = Reconciler(overlay)
        codes_matrix = {w: bits.tolist() for w, bits in codes.items()}
        recon = reconciler.reconcile(report, codes_matrix)
        reconciler.print_report(recon)
    else:
        print("  primitivos.json not found -- skipping reconciliation")
        recon = None

    # ============================================================
    # Comparison with prior models
    # ============================================================
    print_section("COMPARISON WITH PRIOR MODELS")

    print("  | Model               | Test Acc | Dead | Subsumption | Triadic 3-way |")
    print("  |---------------------|----------|------|-------------|---------------|")
    print(f"  | danza_v2 (158 anc)  | 93.0%    | {report.n_dead_bits}/63  | {100*correct/total:.1f}%        | {len(report.discovered_triadic_deps):>13d} |")
    print(f"  | bootstrap (54 anc)  | 79.4%    | 26/63 | --          | --            |")
    print(f"  | hybrid_adv          | 69.3%    |  6/63 | 80.0%       | 17            |")
    print(f"  | gradient_decoupling | 49.6%    | 21/63 | --          | --            |")

    # ============================================================
    # Save
    # ============================================================
    results = {
        'model': 'DanzaTriadicGPT v2 (158 anchors)',
        'checkpoint': str(CKPT),
        'n_concepts': len(codes),
        'n_active_bits': report.n_active_bits,
        'n_dead_bits': report.n_dead_bits,
        'n_duals': len(report.discovered_duals),
        'n_deps': len(report.discovered_deps),
        'n_triadic_deps': len(report.discovered_triadic_deps),
        'subsumption_bitwise': {'correct': correct, 'total': total,
                                 'accuracy': correct / total if total else 0},
        'analogies_bitwise': analogy_results,
        'bit_semantics': [{
            'bit': bs.bit_index,
            'activation_rate': bs.activation_rate,
            'top_words': bs.top_concepts[:5] if hasattr(bs, 'top_concepts') else [],
        } for bs in report.bit_semantics[:20]],
    }
    save_results(results, 'v2_reptimeline_analysis.json')
    print(f"\n  Results saved to playground/audit_tests/results/v2_reptimeline_analysis.json")


if __name__ == '__main__':
    main()
