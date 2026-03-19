"""
Test Reconciler — run discovery on real checkpoint and compare with theory.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from reptimeline.discovery import BitDiscovery
from reptimeline.overlays.primitive_overlay import PrimitiveOverlay
from reptimeline.reconcile import Reconciler
from reptimeline.extractors.triadic import TriadicExtractor
from reptimeline.tracker import TimelineTracker


def main():
    ckpt_dir = os.path.join(
        os.path.dirname(__file__), '..', '..', 'checkpoints', 'danza_63bit_xl'
    )
    if not os.path.exists(ckpt_dir):
        print(f"[SKIP] Checkpoint dir not found: {ckpt_dir}")
        return

    # Use the last checkpoint only for discovery
    extractor = TriadicExtractor(n_bits=63, max_tokens=4)
    checkpoints = extractor.discover_checkpoints(ckpt_dir)
    if not checkpoints:
        print("[SKIP] No checkpoints found")
        return

    # Extract last checkpoint with many concepts
    concepts = [
        # Anchor concepts (English)
        'fire', 'water', 'earth', 'stone', 'wind', 'light', 'dark',
        'love', 'hate', 'truth', 'lie', 'life', 'death',
        'good', 'evil', 'order', 'chaos', 'king', 'queen',
        'freedom', 'control', 'joy', 'pain', 'fear', 'hope',
        'war', 'peace', 'man', 'woman', 'child', 'animal',
        'tree', 'river', 'mountain', 'sky', 'sun', 'moon',
        'gold', 'iron', 'sword', 'shield', 'song', 'silence',
        'dream', 'memory', 'time', 'space', 'nothing', 'everything',
        'beginning', 'end', 'door', 'wall', 'road', 'bridge',
    ]

    print(f"Extracting from last checkpoint with {len(concepts)} concepts...")
    last_step, last_path = checkpoints[-1]
    snapshot = extractor.extract(last_path, concepts, device='cpu')
    print(f"  Got codes for {len(snapshot.codes)} concepts at step {last_step}")

    # Also extract a few checkpoints for timeline
    print("Extracting timeline (4 checkpoints)...")
    snapshots = extractor.extract_sequence(
        ckpt_dir, concepts, device='cpu', max_checkpoints=4
    )
    tracker = TimelineTracker(extractor)
    timeline = tracker.analyze(snapshots)

    # Discovery (no primitives knowledge)
    print("\nRunning BitDiscovery (no theory used)...")
    discovery = BitDiscovery(dead_threshold=0.02, dual_threshold=-0.3,
                             dep_confidence=0.9)
    report = discovery.discover(snapshot, timeline=timeline, top_k=8)
    discovery.print_report(report)

    # Reconcile with theory
    print("\nReconciling with primitivos.json...")
    overlay = PrimitiveOverlay()
    reconciler = Reconciler(overlay)
    recon = reconciler.reconcile(report, snapshot.codes)
    reconciler.print_report(recon)


if __name__ == '__main__':
    main()
