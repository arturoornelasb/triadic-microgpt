"""
State resolution: converts continuous projections to discrete states.

Takes a 49-dim projection vector (each value in [-1, 1]) and resolves
each position into Active[+], Zero[0], or N/A[∅], with intensity and
polarity for dual principles.
"""

from __future__ import annotations

import numpy as np

from .config import (
    CATEGORY_TO_LAYER,
    DUAL_INDICES,
    DUAL_POLES,
    N_PRIMITIVES,
    PRIMITIVE_NAMES,
    PRIMITIVE_PRIMES,
    PRIMITIVE_TO_CATEGORY,
    PRIMITIVE_TO_PRIME,
    StateConfig,
)
from .primitives import (
    ConceptToken,
    Intensity,
    PrimitiveActivation,
    State,
)


class StateResolver:
    """Resolves continuous 49-dim projections into discrete concept tokens."""

    def __init__(self, config: StateConfig | None = None):
        self.config = config or StateConfig()

    def resolve(self, word: str, projections: np.ndarray, span: tuple[int, int] = (0, 0)) -> ConceptToken:
        """
        Convert a 49-dim projection vector into a ConceptToken.

        Args:
            word: The source word
            projections: Array of shape (49,), values in [-1, 1]
            span: Character offsets in original text

        Returns:
            ConceptToken with resolved states, intensities, polarities, and prime signatures
        """
        assert len(projections) == N_PRIMITIVES, f"Expected {N_PRIMITIVES} projections, got {len(projections)}"

        activations = []
        active_primes = []
        zero_primes = []

        for i, (name, value) in enumerate(zip(PRIMITIVE_NAMES, projections)):
            prime = PRIMITIVE_PRIMES[i]
            category = PRIMITIVE_TO_CATEGORY[name]
            layer = CATEGORY_TO_LAYER[category]
            abs_value = abs(float(value))

            # Resolve state
            if abs_value < self.config.na_threshold:
                state = State.NA
            elif float(value) > 0:
                state = State.ACTIVE
            else:
                state = State.ZERO

            # Resolve intensity
            if abs_value < self.config.low_threshold:
                intensity = Intensity.LOW
            elif abs_value < self.config.high_threshold:
                intensity = Intensity.MEDIUM
            else:
                intensity = Intensity.HIGH

            # Resolve polarity for dual principles
            polarity = None
            if name in DUAL_POLES:
                pos_pole, neg_pole = DUAL_POLES[name]
                if state == State.ACTIVE:
                    polarity = pos_pole
                elif state == State.ZERO:
                    polarity = neg_pole

            activation = PrimitiveActivation(
                name=name,
                prime=prime,
                category=category,
                layer=layer,
                state=state,
                intensity=intensity,
                raw_value=float(value),
                polarity=polarity,
            )
            activations.append(activation)

            # Accumulate primes for composites
            if state == State.ACTIVE:
                active_primes.append(prime)
            elif state == State.ZERO:
                zero_primes.append(prime)

        # Compute composite signatures
        active_composite = 1
        for p in active_primes:
            active_composite *= p

        zero_composite = 1
        for p in zero_primes:
            zero_composite *= p

        return ConceptToken(
            word=word,
            span=span,
            activations=activations,
            projections=projections,
            active_composite=active_composite,
            zero_composite=zero_composite,
        )

    def from_lexicon_entry(self, word: str, entry: dict[str, tuple[str, float]]) -> ConceptToken:
        """
        Create a ConceptToken from a seed lexicon entry.

        Args:
            word: The word
            entry: Dict of {primitive_name: (state_char, intensity_value)}
                   state_char: "+" for active, "0" for zero, "-" for negative pole

        Returns:
            ConceptToken
        """
        projections = np.zeros(N_PRIMITIVES)

        for prim_name, (state_char, intensity) in entry.items():
            idx = PRIMITIVE_NAMES.index(prim_name)
            if state_char == "+":
                projections[idx] = intensity
            elif state_char == "0":
                projections[idx] = -intensity  # Negative = zero state
            elif state_char == "-":
                projections[idx] = -intensity  # Negative pole of dual principle

        return self.resolve(word, projections)
