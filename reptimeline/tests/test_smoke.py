"""
Smoke test — verify reptimeline core works without model checkpoints.

Creates synthetic snapshots and runs the full pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from reptimeline.core import ConceptSnapshot, Timeline
from reptimeline.extractors.base import RepresentationExtractor
from reptimeline.tracker import TimelineTracker
from reptimeline.overlays.primitive_overlay import PrimitiveOverlay


class SyntheticExtractor(RepresentationExtractor):
    """Dummy extractor for testing — no model needed."""

    def extract(self, checkpoint_path, concepts, device='cpu'):
        raise NotImplementedError("Use synthetic snapshots directly")

    def similarity(self, code_a, code_b):
        active_a = set(i for i, v in enumerate(code_a) if v == 1)
        active_b = set(i for i, v in enumerate(code_b) if v == 1)
        union = active_a | active_b
        if not union:
            return 1.0
        return len(active_a & active_b) / len(union)

    def shared_features(self, code_a, code_b):
        return [i for i in range(min(len(code_a), len(code_b)))
                if code_a[i] == 1 and code_b[i] == 1]


def make_code(n_bits, active_indices):
    """Create a binary code with specific bits active."""
    code = [0] * n_bits
    for i in active_indices:
        code[i] = 1
    return code


def test_basic_pipeline():
    """Test tracker with synthetic evolving codes."""
    n_bits = 63

    # Simulate 5 training steps where concepts gradually activate bits
    snapshots = [
        # Step 0: minimal activation
        ConceptSnapshot(
            step=0,
            codes={
                'king': make_code(n_bits, [0, 1]),           # vacío, información
                'queen': make_code(n_bits, [0, 1]),          # same
                'fire': make_code(n_bits, [2]),              # fuerza
            },
        ),
        # Step 1000: more bits activate, king/queen diverge
        ConceptSnapshot(
            step=1000,
            codes={
                'king': make_code(n_bits, [0, 1, 2, 44]),   # +fuerza, +uno
                'queen': make_code(n_bits, [0, 1, 26]),      # +unión
                'fire': make_code(n_bits, [2, 3, 24, 25]),   # +fuego, +creación, +destrucción
            },
        ),
        # Step 2000: connections form
        ConceptSnapshot(
            step=2000,
            codes={
                'king': make_code(n_bits, [0, 1, 2, 44, 31]),  # +control
                'queen': make_code(n_bits, [0, 1, 26, 30]),    # +libertad
                'fire': make_code(n_bits, [2, 3, 24, 25, 4]),  # +tierra
            },
        ),
        # Step 3000: stabilization
        ConceptSnapshot(
            step=3000,
            codes={
                'king': make_code(n_bits, [0, 1, 2, 44, 31]),  # same as 2000
                'queen': make_code(n_bits, [0, 1, 26, 30]),    # same
                'fire': make_code(n_bits, [2, 3, 24, 25, 4]),  # same
            },
        ),
        # Step 4000: a death occurs — king loses bit 0 (vacío)
        ConceptSnapshot(
            step=4000,
            codes={
                'king': make_code(n_bits, [1, 2, 44, 31]),     # lost vacío
                'queen': make_code(n_bits, [0, 1, 26, 30]),    # stable
                'fire': make_code(n_bits, [2, 3, 24, 25, 4]),  # stable
            },
        ),
    ]

    extractor = SyntheticExtractor()
    tracker = TimelineTracker(extractor, stability_window=2)
    timeline = tracker.analyze(snapshots)

    # Verify basic timeline properties
    assert len(timeline.steps) == 5
    assert timeline.steps == [0, 1000, 2000, 3000, 4000]

    # Should have births (bits activating for first time)
    assert len(timeline.births) > 0
    print(f"  Births: {len(timeline.births)}")

    # Should have at least 1 death (king's bit 0)
    assert len(timeline.deaths) >= 1
    king_deaths = [d for d in timeline.deaths if d.concept == 'king']
    assert any(d.code_index == 0 for d in king_deaths), "Expected king to lose bit 0"
    print(f"  Deaths: {len(timeline.deaths)}")

    # Should detect connections (king & queen share bits 0, 1)
    assert len(timeline.connections) > 0
    print(f"  Connections: {len(timeline.connections)}")

    # Curves should have correct length
    assert len(timeline.curves['entropy']) == 5
    assert len(timeline.curves['churn_rate']) == 5
    assert len(timeline.curves['utilization']) == 5
    print(f"  Entropy range: {timeline.curves['entropy'][0]:.4f} -> {timeline.curves['entropy'][-1]:.4f}")
    print(f"  Churn range: {timeline.curves['churn_rate'][0]:.3f} -> {timeline.curves['churn_rate'][-1]:.3f}")

    # Stability should be computed
    assert len(timeline.stability) > 0
    print(f"  Stability entries: {len(timeline.stability)}")

    timeline.print_summary()
    print("\n  [PASS] Basic pipeline")
    return timeline


def test_primitive_overlay(timeline):
    """Test the PrimitiveOverlay on synthetic data."""
    prim_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'playground', 'danza_data', 'primitivos.json'
    )
    if not os.path.exists(prim_path):
        print("  [SKIP] primitivos.json not found")
        return

    overlay = PrimitiveOverlay(prim_path)
    assert len(overlay.primitives) == 63

    report = overlay.analyze(timeline, concepts=['king', 'queen', 'fire'])

    assert len(report.activations) > 0
    print(f"  Activations: {len(report.activations)}")
    print(f"  Layer emergence entries: {len(report.layer_emergence)}")
    print(f"  Dual coherence pairs: {len(report.dual_coherence)}")
    print(f"  Deps completions: {len(report.deps_completions)}")

    overlay.print_report(report)
    print("\n  [PASS] Primitive overlay")


if __name__ == '__main__':
    print("reptimeline smoke test\n" + "=" * 40)
    tl = test_basic_pipeline()
    test_primitive_overlay(tl)
    print("\nAll tests passed.")
