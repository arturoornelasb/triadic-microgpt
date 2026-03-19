"""
Test BitDiscovery against real checkpoints — discover what the model
learned WITHOUT using primitivos.json.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from reptimeline.core import ConceptSnapshot
from reptimeline.extractors.base import RepresentationExtractor
from reptimeline.tracker import TimelineTracker
from reptimeline.discovery import BitDiscovery


class SyntheticExtractor(RepresentationExtractor):
    def extract(self, checkpoint_path, concepts, device='cpu'):
        raise NotImplementedError
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
    code = [0] * n_bits
    for i in active_indices:
        code[i] = 1
    return code


def test_discovery_synthetic():
    """Test discovery with synthetic data that has clear patterns."""
    n_bits = 16

    # Design codes with intentional structure:
    # - Bits 0,1 = "basic" (active in almost everything)
    # - Bits 2,3 = dual pair (never both active)
    # - Bit 4 depends on bit 0 (only active when 0 is active)
    # - Bits 14,15 = dead (never active)
    # - Triadic: bit 13 activates ONLY when bits 2 AND 5 are both active,
    #   but NOT when only 2 or only 5 is active alone
    snapshot = ConceptSnapshot(
        step=5000,
        codes={
            # "Physical" concepts: bit 2 active, bit 3 off
            'fire':    make_code(n_bits, [0, 1, 2, 5, 6, 13]),   # 2+5 -> 13 ON
            'water':   make_code(n_bits, [0, 1, 2, 7]),           # 2 only -> 13 OFF
            'earth':   make_code(n_bits, [0, 1, 2, 8]),           # 2 only -> 13 OFF
            'stone':   make_code(n_bits, [0, 1, 2, 8, 9]),        # 2 only -> 13 OFF
            'metal':   make_code(n_bits, [0, 1, 2, 8, 9, 10]),    # 2 only -> 13 OFF
            'wind':    make_code(n_bits, [0, 1, 2, 5, 13]),       # 2+5 -> 13 ON
            'storm':   make_code(n_bits, [0, 1, 2, 5, 6, 13]),   # 2+5 -> 13 ON
            # "Abstract" concepts: bit 3 active, bit 2 off
            'love':    make_code(n_bits, [0, 1, 3, 4, 11]),
            'hate':    make_code(n_bits, [0, 1, 3, 4, 12]),
            'truth':   make_code(n_bits, [0, 1, 3, 11]),
            'lie':     make_code(n_bits, [0, 1, 3, 12]),
            'justice': make_code(n_bits, [0, 1, 3, 4, 11]),
            'freedom': make_code(n_bits, [0, 1, 3, 4]),
            # Concepts with bit 5 but NOT bit 2 -> 13 OFF
            'spirit':  make_code(n_bits, [0, 1, 3, 5]),
            'soul':    make_code(n_bits, [0, 1, 3, 5, 11]),
            'dream':   make_code(n_bits, [0, 1, 3, 5, 12]),
            # Outlier: neither 2 nor 3
            'void':    make_code(n_bits, [0]),
            'nothing': make_code(n_bits, []),
        },
    )

    discovery = BitDiscovery(dead_threshold=0.05, dual_threshold=-0.3,
                             dep_confidence=0.85)
    report = discovery.discover(snapshot, top_k=5)

    # Verify: bits 0,1 should be most active
    rates = {bs.bit_index: bs.activation_rate for bs in report.bit_semantics}
    assert rates[0] > 0.8, f"Bit 0 should be very active, got {rates[0]}"
    assert rates[1] > 0.8, f"Bit 1 should be very active, got {rates[1]}"

    # Verify: bits 14,15 should be dead
    assert rates[14] < 0.05, f"Bit 14 should be dead, got {rates[14]}"
    assert rates[15] < 0.05, f"Bit 15 should be dead, got {rates[15]}"
    assert report.n_dead_bits >= 2

    # Verify: bits 2,3 should be discovered as duals
    dual_pairs = {(d.bit_a, d.bit_b) for d in report.discovered_duals}
    assert (2, 3) in dual_pairs, f"Bits 2,3 should be duals. Found: {dual_pairs}"
    print(f"  Discovered {len(report.discovered_duals)} dual pairs")

    # Verify: bit 4 depends on bit 0 or bit 1 (or bit 3)
    dep_edges = {(d.bit_parent, d.bit_child) for d in report.discovered_deps}
    has_dep = any(d.bit_child == 4 for d in report.discovered_deps)
    assert has_dep, f"Bit 4 should have a dependency. Found: {dep_edges}"
    print(f"  Discovered {len(report.discovered_deps)} dependencies")

    # Verify: triadic dep — bit 2 + bit 5 -> bit 13
    triadic_triples = [(t.bit_i, t.bit_j, t.bit_r) for t in report.discovered_triadic_deps]
    has_triadic = any(t.bit_r == 13 and set([t.bit_i, t.bit_j]) == {2, 5}
                      for t in report.discovered_triadic_deps)
    assert has_triadic, f"Should find triadic 2+5->13. Found: {triadic_triples}"
    print(f"  Discovered {len(report.discovered_triadic_deps)} triadic interactions")

    discovery.print_report(report)
    print("\n  [PASS] Discovery synthetic")
    return report


def test_discovery_with_timeline():
    """Test hierarchy discovery with synthetic timeline."""
    n_bits = 8
    extractor = SyntheticExtractor()

    # Bits 0,1 stabilize early; bits 4,5 stabilize late
    snapshots = [
        ConceptSnapshot(step=0, codes={
            'a': make_code(n_bits, [0, 1]),
            'b': make_code(n_bits, [0]),
        }),
        ConceptSnapshot(step=1000, codes={
            'a': make_code(n_bits, [0, 1, 2]),
            'b': make_code(n_bits, [0, 3]),
        }),
        ConceptSnapshot(step=2000, codes={
            'a': make_code(n_bits, [0, 1, 2]),    # bits 0,1,2 stable
            'b': make_code(n_bits, [0, 3]),        # bits 0,3 stable
        }),
        ConceptSnapshot(step=3000, codes={
            'a': make_code(n_bits, [0, 1, 2]),    # still stable
            'b': make_code(n_bits, [0, 3]),
        }),
        ConceptSnapshot(step=4000, codes={
            'a': make_code(n_bits, [0, 1, 2, 4]),  # bit 4 appears late
            'b': make_code(n_bits, [0, 3, 5]),      # bit 5 appears late
        }),
        ConceptSnapshot(step=5000, codes={
            'a': make_code(n_bits, [0, 1, 2, 4]),  # bit 4 stable now
            'b': make_code(n_bits, [0, 3, 5]),      # bit 5 stable now
        }),
        ConceptSnapshot(step=6000, codes={
            'a': make_code(n_bits, [0, 1, 2, 4]),
            'b': make_code(n_bits, [0, 3, 5]),
        }),
    ]

    tracker = TimelineTracker(extractor)
    timeline = tracker.analyze(snapshots)

    discovery = BitDiscovery()
    report = discovery.discover(snapshots[-1], timeline=timeline, top_k=3)

    # Bits 0,1 should be in earlier layers than bits 4,5
    hierarchy = {h.bit_index: h for h in report.discovered_hierarchy}
    if hierarchy.get(0) and hierarchy.get(4):
        if hierarchy[0].first_stable_step and hierarchy[4].first_stable_step:
            assert hierarchy[0].first_stable_step <= hierarchy[4].first_stable_step, \
                "Bit 0 should stabilize before bit 4"

    discovery.print_report(report)
    print("\n  [PASS] Discovery with timeline")


if __name__ == '__main__':
    print("reptimeline discovery test\n" + "=" * 40)
    test_discovery_synthetic()
    test_discovery_with_timeline()
    print("\nAll discovery tests passed.")
