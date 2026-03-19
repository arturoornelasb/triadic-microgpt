"""
Analyze hybrid adversarial model with reptimeline discovery.

Discovers what the 33 FREE bits learned without supervision,
compares with the 30 supervised bits, and finds triadic interactions.
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
from src.triadic import PrimeMapper

try:
    from src.fast_tokenizer import FastBPETokenizer as BPETokenizer
except ImportError:
    from src.tokenizer import BPETokenizer

from reptimeline.core import ConceptSnapshot
from reptimeline.discovery import BitDiscovery
from reptimeline.reconcile import Reconciler
from reptimeline.overlays.primitive_overlay import PrimitiveOverlay

# Import hybrid model class
sys.path.insert(0, _PLAYGROUND)
from hybrid_adversarial import HybridTriadicGPT, N_SUPERVISED, N_FREE


CKPT_DIR = os.path.join(_PROJECT, 'checkpoints', 'danza_hybrid_adv_xl')
CKPT = os.path.join(CKPT_DIR, 'model_best.pt')
TOK = os.path.join(CKPT_DIR, 'tokenizer.json')


def load_hybrid(device='cpu'):
    """Load hybrid adversarial model."""
    ckpt = torch.load(CKPT, map_location=device, weights_only=True)
    cfg = ckpt['config']

    config = TriadicGPTConfig(
        vocab_size=cfg['vocab_size'], block_size=cfg['block_size'],
        n_layer=cfg['n_layer'], n_embd=cfg['n_embd'],
        n_head=cfg['n_head'], n_triadic_bits=cfg['n_triadic_bits'],
    )
    model = HybridTriadicGPT(config).to(device)
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
        proj = triadic_proj[0].mean(dim=0).cpu().numpy()
        bits = (proj > 0).astype(np.int8)
        results[word] = bits
    return results


def main():
    print_header("REPTIMELINE ANALYSIS — HYBRID ADVERSARIAL MODEL")
    print(f"  Checkpoint: {CKPT}")
    print(f"  Architecture: 30 supervised + 33 free bits")

    device = 'cpu'
    print(f"\n  Loading model on {device}...")
    model, tokenizer = load_hybrid(device)

    # Load concepts — all anchors + extra test words
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
    ]
    all_words = list(set(list(all_anchors.keys()) + test_words))

    print(f"  Extracting projections for {len(all_words)} concepts...")
    codes = extract_projections(model, tokenizer, all_words, device)
    print(f"  Got {len(codes)} concepts")

    # Build snapshot
    snapshot = ConceptSnapshot(
        step=50000,
        codes={w: bits.tolist() for w, bits in codes.items()},
    )

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
    report = discovery.discover(snapshot, top_k=10)
    discovery.print_report(report)

    # ============================================================
    # Analyze supervised vs free bits separately
    # ============================================================
    print_section("SUPERVISED vs FREE BITS ANALYSIS")

    sup_active = 0
    free_active = 0
    sup_dead = 0
    free_dead = 0

    for bs in report.bit_semantics:
        is_sup = bs.bit_index < N_SUPERVISED
        if bs.activation_rate > 0.02:
            if is_sup:
                sup_active += 1
            else:
                free_active += 1
        else:
            if is_sup:
                sup_dead += 1
            else:
                free_dead += 1

    print(f"  Supervised bits (0-29): {sup_active} active, {sup_dead} dead")
    print(f"  Free bits (30-62):      {free_active} active, {free_dead} dead")

    # Triadic deps involving free bits
    free_triadic = [t for t in report.discovered_triadic_deps
                    if t.bit_r >= N_SUPERVISED or t.bit_i >= N_SUPERVISED or t.bit_j >= N_SUPERVISED]
    print(f"\n  Triadic interactions involving free bits: {len(free_triadic)}")
    for td in free_triadic[:10]:
        sup_i = "S" if td.bit_i < N_SUPERVISED else "F"
        sup_j = "S" if td.bit_j < N_SUPERVISED else "F"
        sup_r = "S" if td.bit_r < N_SUPERVISED else "F"
        print(f"    [{sup_i}]bit {td.bit_i:>2d} + [{sup_j}]bit {td.bit_j:>2d} -> [{sup_r}]bit {td.bit_r:>2d}"
              f"  P(r|i,j)={td.p_r_given_ij:.2f}"
              f"  strength={td.interaction_strength:.2f}"
              f"  n={td.support}")

    # Cross-domain triadic: supervised + supervised -> free
    cross_triadic = [t for t in report.discovered_triadic_deps
                     if t.bit_i < N_SUPERVISED and t.bit_j < N_SUPERVISED and t.bit_r >= N_SUPERVISED]
    print(f"\n  Cross-domain (sup+sup -> free): {len(cross_triadic)}")
    for td in cross_triadic[:10]:
        print(f"    bit {td.bit_i:>2d} + bit {td.bit_j:>2d} -> FREE bit {td.bit_r:>2d}"
              f"  P(r|i,j)={td.p_r_given_ij:.2f}"
              f"  strength={td.interaction_strength:.2f}")

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
        print("  primitivos.json not found — skipping reconciliation")
        recon = None

    # ============================================================
    # Save
    # ============================================================
    results = {
        'model': 'HybridTriadicGPT (D-A9)',
        'checkpoint': str(CKPT),
        'n_concepts': len(codes),
        'n_active_bits': report.n_active_bits,
        'n_dead_bits': report.n_dead_bits,
        'supervised': {'active': sup_active, 'dead': sup_dead},
        'free': {'active': free_active, 'dead': free_dead},
        'n_duals': len(report.discovered_duals),
        'n_deps': len(report.discovered_deps),
        'n_triadic_deps': len(report.discovered_triadic_deps),
        'n_free_triadic': len(free_triadic),
        'n_cross_triadic': len(cross_triadic),
        'reconciliation': {
            'agreement': recon.agreement_rate if recon and hasattr(recon, 'agreement_rate') else None,
            'n_mismatches': len(recon.bit_mismatches) if recon and hasattr(recon, 'bit_mismatches') else None,
        },
    }
    save_results(results, 'hybrid_reptimeline_analysis.json')


if __name__ == '__main__':
    main()
