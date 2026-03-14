"""
Primitive concepts and data structures for the Conceptual Tokenizer.

Defines ConceptToken (single token) and ConceptSequence (tokenized text),
plus utilities for working with the 49 primitives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .config import (
    CATEGORY_NAMES,
    CATEGORY_TO_LAYER,
    DUAL_INDICES,
    DUAL_POLES,
    N_PRIMITIVES,
    PRIMITIVE_NAMES,
    PRIMITIVE_PRIMES,
    PRIMITIVE_TO_CATEGORY,
    PRIMITIVE_TO_PRIME,
)


class State(Enum):
    """The three states of existence for any primitive."""
    ACTIVE = "+"    # Present and operating
    ZERO = "0"      # Actively absent (silence, darkness, void)
    NA = "∅"        # Not applicable (irrelevant to this context)


class Intensity(Enum):
    """Intensity level of an active or zero primitive."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class PrimitiveActivation:
    """A single primitive's activation state."""
    name: str
    prime: int
    category: str
    layer: str
    state: State
    intensity: Intensity
    raw_value: float        # Original projection value [-1, 1]
    polarity: Optional[str] = None  # Only for dual principles: "Bien"/"Mal" etc.

    @property
    def is_active(self) -> bool:
        return self.state == State.ACTIVE

    @property
    def is_zero(self) -> bool:
        return self.state == State.ZERO

    @property
    def is_na(self) -> bool:
        return self.state == State.NA


@dataclass
class ConceptToken:
    """A single concept-token: a word mapped to its 49 primitive activations."""

    # Source text
    word: str
    span: tuple[int, int] = (0, 0)  # Character offsets in original text

    # All 49 activations
    activations: list[PrimitiveActivation] = field(default_factory=list)

    # Raw projection vector (49 values in [-1, 1])
    projections: Optional[np.ndarray] = None

    # Prime signatures (computed from activations)
    active_composite: int = 1   # Product of primes for [+] primitives
    zero_composite: int = 1     # Product of primes for [0] primitives

    @property
    def active_primitives(self) -> list[PrimitiveActivation]:
        return [a for a in self.activations if a.is_active]

    @property
    def zero_primitives(self) -> list[PrimitiveActivation]:
        return [a for a in self.activations if a.is_zero]

    @property
    def na_primitives(self) -> list[PrimitiveActivation]:
        return [a for a in self.activations if a.is_na]

    @property
    def active_categories(self) -> set[str]:
        return {a.category for a in self.activations if not a.is_na}

    @property
    def depth(self) -> int:
        """Emergence level: how many categories have at least one non-N/A primitive."""
        return len(self.active_categories)

    @property
    def polarities(self) -> dict[str, str]:
        """Polarities of dual principles that are active."""
        return {
            a.name: a.polarity
            for a in self.activations
            if a.polarity is not None and not a.is_na
        }

    def summary(self) -> str:
        """Human-readable summary."""
        parts = []
        for a in self.activations:
            if a.is_na:
                continue
            state_str = f"[{a.state.value}]"
            intensity_str = f"({a.intensity.value})"
            if a.polarity:
                parts.append(f"{a.polarity}{state_str}{intensity_str}")
            else:
                parts.append(f"{a.name}{state_str}{intensity_str}")
        return f"{self.word}: {' + '.join(parts)} [depth={self.depth}]"


@dataclass
class ConceptSequence:
    """The full output of the tokenizer for a text passage."""
    tokens: list[ConceptToken]
    text: str
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx) -> ConceptToken:
        return self.tokens[idx]

    def summary(self) -> str:
        lines = [f"ConceptSequence({len(self.tokens)} tokens):"]
        for t in self.tokens:
            lines.append(f"  {t.summary()}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────

def prime_index(name: str) -> int:
    """Get the index (0-48) of a primitive by name."""
    return PRIMITIVE_NAMES.index(name)


def compute_composite(names: list[str]) -> int:
    """Compute the composite prime product for a list of primitive names."""
    result = 1
    for name in names:
        result *= PRIMITIVE_TO_PRIME[name]
    return result


def decompose_composite(composite: int) -> list[str]:
    """Decompose a composite prime back into primitive names."""
    if composite <= 1:
        return []
    names = []
    for name, prime in PRIMITIVE_TO_PRIME.items():
        if composite % prime == 0:
            names.append(name)
    return names


def subsumes(a_composite: int, b_composite: int) -> bool:
    """Does concept A contain all features of concept B? (A % B == 0)"""
    if b_composite == 0 or b_composite == 1:
        return True
    return a_composite % b_composite == 0


def compose(*composites: int) -> int:
    """Combine concepts via LCM (lowest common multiple)."""
    result = composites[0]
    for c in composites[1:]:
        result = result * c // math.gcd(result, c)
    return result


def gap(a_composite: int, b_composite: int) -> tuple[int, int, int]:
    """
    Analyze the gap between two concepts.
    Returns: (shared, only_in_a, only_in_b)
    """
    shared = math.gcd(a_composite, b_composite)
    only_a = a_composite // shared if shared else a_composite
    only_b = b_composite // shared if shared else b_composite
    return shared, only_a, only_b
