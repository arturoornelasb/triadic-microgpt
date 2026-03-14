"""
Triadic Bridge: connects the Conceptual Tokenizer to the existing
TriadicValidator from src/triadic.py.

Provides high-level semantic operations on ConceptTokens using
prime algebra: subsumption, composition, gap analysis, analogy,
and similarity — with human-readable explanations using primitive names.
"""

from __future__ import annotations

import sys
import os
import math

# Add src/ to path so we can import the original triadic module
_src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
if _src_path not in sys.path:
    sys.path.insert(0, os.path.abspath(_src_path))

from triadic import TriadicValidator

from .config import PRIME_TO_PRIMITIVE, PRIMITIVE_TO_PRIME
from .primitives import ConceptToken, decompose_composite


class ConceptBridge:
    """
    Bridges ConceptTokens to algebraic verification via TriadicValidator.
    All operations work on active_composite (the [+] primes).
    """

    def __init__(self):
        self.validator = TriadicValidator()

    def subsumes(self, a: ConceptToken, b: ConceptToken) -> bool:
        """Does concept A contain all semantic features of concept B?"""
        return self.validator.subsumes(a.active_composite, b.active_composite)

    def compose(self, *tokens: ConceptToken) -> int:
        """Combine all features from multiple concepts. Returns composite."""
        composites = [t.active_composite for t in tokens]
        return self.validator.compose(*composites)

    def similarity(self, a: ConceptToken, b: ConceptToken) -> float:
        """Semantic similarity [0.0, 1.0] based on shared prime factors."""
        return self.validator.similarity(a.active_composite, b.active_composite)

    def explain_gap(self, a: ConceptToken, b: ConceptToken) -> dict:
        """
        Explain exactly WHY two concepts differ, with primitive names.

        Returns dict with:
            shared: list of primitive names both concepts have
            only_in_a: primitives only in A
            only_in_b: primitives only in B
            similarity: float [0, 1]
        """
        raw = self.validator.explain_gap(a.active_composite, b.active_composite)
        return {
            "shared": decompose_composite(raw["shared"]),
            "only_in_a": decompose_composite(raw["only_in_a"]),
            "only_in_b": decompose_composite(raw["only_in_b"]),
            "a_contains_b": raw["a_contains_b"],
            "b_contains_a": raw["b_contains_a"],
            "similarity": self.similarity(a, b),
            "a_word": a.word,
            "b_word": b.word,
        }

    def analogy(self, a: ConceptToken, b: ConceptToken, c: ConceptToken) -> dict:
        """
        A is to B as C is to ?

        Returns the target composite and its decomposition.
        Example: king:queen::man:? → woman
        """
        target = self.validator.analogy(
            a.active_composite, b.active_composite, c.active_composite
        )
        return {
            "a": a.word,
            "b": b.word,
            "c": c.word,
            "target_composite": target,
            "target_primitives": decompose_composite(target),
            "transformation": {
                "removed": decompose_composite(
                    a.active_composite // math.gcd(a.active_composite, b.active_composite)
                ),
                "added": decompose_composite(
                    b.active_composite // math.gcd(a.active_composite, b.active_composite)
                ),
            }
        }

    def verify_emergence(self, emergent: ConceptToken, components: list[ConceptToken]) -> dict:
        """
        Verify that an emergent concept contains all its component primitives.

        Example: verify that "music" contains Aire + Oído + Play + Orden
        """
        composed = self.compose(*components)
        contains_all = self.validator.subsumes(emergent.active_composite, composed)
        missing = []
        extra = []

        for comp in components:
            for act in comp.active_primitives:
                if emergent.active_composite % act.prime != 0:
                    missing.append(act.name)

        for act in emergent.active_primitives:
            found = False
            for comp in components:
                if comp.active_composite % act.prime == 0:
                    found = True
                    break
            if not found:
                extra.append(act.name)

        return {
            "emergent": emergent.word,
            "components": [c.word for c in components],
            "contains_all": contains_all,
            "missing": missing,
            "extra": extra,
        }

    def report(self, a: ConceptToken, b: ConceptToken) -> str:
        """Human-readable report of the relationship between two concepts."""
        gap_info = self.explain_gap(a, b)
        lines = [
            f"{'═' * 50}",
            f"  {a.word} ↔ {b.word}",
            f"{'═' * 50}",
            f"  Similarity: {gap_info['similarity']:.1%}",
            f"  Shared:     {', '.join(gap_info['shared']) or '(nothing)'}",
            f"  Only {a.word}: {', '.join(gap_info['only_in_a']) or '(nothing)'}",
            f"  Only {b.word}: {', '.join(gap_info['only_in_b']) or '(nothing)'}",
            f"  {a.word} ⊇ {b.word}: {gap_info['a_contains_b']}",
            f"  {b.word} ⊇ {a.word}: {gap_info['b_contains_a']}",
        ]
        return "\n".join(lines)
