"""
Prime Encoder: converts resolved concept states to composite prime signatures.

Provides both the encoding (states → primes) and the bridge to the
existing TriadicValidator for algebraic verification.
"""

from __future__ import annotations

import math

from .config import PRIMITIVE_TO_PRIME, PRIME_TO_PRIMITIVE, PRIMITIVE_PRIMES
from .primitives import ConceptToken, decompose_composite


class PrimeEncoder:
    """Encodes concept activations into composite prime signatures."""

    def __init__(self):
        self.primes = PRIMITIVE_PRIMES
        self.prime_to_name = PRIME_TO_PRIMITIVE
        self.name_to_prime = PRIMITIVE_TO_PRIME

    def encode(self, token: ConceptToken) -> ConceptToken:
        """
        Compute prime signatures from a token's activations.
        Mutates token in-place and returns it.
        """
        active = 1
        zero = 1
        for act in token.activations:
            if act.is_active:
                active *= act.prime
            elif act.is_zero:
                zero *= act.prime
        token.active_composite = active
        token.zero_composite = zero
        return token

    def decode_composite(self, composite: int) -> list[str]:
        """Decompose a composite back into primitive names."""
        return decompose_composite(composite)

    def full_signature(self, token: ConceptToken) -> int:
        """LCM of active and zero composites — everything that's relevant."""
        a, z = token.active_composite, token.zero_composite
        if a == 1:
            return z
        if z == 1:
            return a
        return a * z // math.gcd(a, z)
