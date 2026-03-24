"""
Prime-factor algebra for neurosymbolic semantic operations.

Provides exact algebraic operations (subsumption, composition, gap analysis)
on prime-factor signatures — things impossible with cosine similarity.

Zero external dependencies.
"""

import math
from typing import Dict, List


# ============================================================
# Prime Number Utilities
# ============================================================

def sieve_primes(limit: int) -> List[int]:
    """Return all primes up to `limit` via Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]


_PRIMES_CACHE = sieve_primes(1200)  # covers n_bits up to ~200


def nth_prime(n: int) -> int:
    """Return the nth prime (1-indexed: nth_prime(1) = 2)."""
    if n <= 0:
        raise ValueError("n must be >= 1")
    while len(_PRIMES_CACHE) < n:
        candidate = _PRIMES_CACHE[-1] + 2
        while True:
            if all(candidate % p != 0 for p in _PRIMES_CACHE if p * p <= candidate):
                _PRIMES_CACHE.append(candidate)
                break
            candidate += 2
    return _PRIMES_CACHE[n - 1]


def prime_factors(n: int) -> List[int]:
    """Return sorted list of unique prime factors of n."""
    if n <= 1:
        return []
    factors = []
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            factors.append(d)
            while temp % d == 0:
                temp //= d
        d += 1
    if temp > 1:
        factors.append(temp)
    return factors


# ============================================================
# PrimeMapper
# ============================================================

class PrimeMapper:
    """
    Maps continuous projections (tanh outputs in [-1, 1]) to composite prime integers.

    Each bit position is assigned a unique prime. If projection[i] > 0,
    prime[i] is included in the composite product.

    Example:
        projections = [0.5, -0.2, 0.8, 0.1]  (4 bits)
        primes      = [2,    3,    5,   7  ]
        composite   = 2 * 5 * 7 = 70  (bits 0, 2, 3 are positive)
    """

    def __init__(self, n_bits: int):
        self.n_bits = n_bits
        self.primes = [nth_prime(i + 1) for i in range(n_bits)]

    def encode(self, projections) -> int:
        """
        Convert projection values to a composite prime integer.

        Args:
            projections: sequence of floats (tanh outputs in [-1, 1])
                         or a 1D tensor.

        Returns:
            int — composite prime product.
        """
        composite = 1
        for proj, prime in zip(projections, self.primes):
            val = float(proj)
            if val > 0:
                composite *= prime
        # Return 1 (identity element) when all projections negative,
        # rather than prime 2 which has specific semantic meaning.
        return composite

    # Alias for compatibility with src/triadic.py which uses map()
    map = encode

    def get_bits(self, projections) -> List[int]:
        """Return binary bit pattern from projections."""
        return [1 if float(p) > 0 else 0 for p in projections]

    def explain(self, composite: int) -> Dict:
        """Decompose a composite into its constituent prime factors."""
        factors = []
        bit_indices = []
        for i, prime in enumerate(self.primes):
            if composite % prime == 0:
                factors.append(prime)
                bit_indices.append(i)
        return {
            'composite': composite,
            'factors': factors,
            'active_bits': bit_indices,
            'n_active': len(factors),
            'n_total': self.n_bits,
        }

    def similarity(self, a: int, b: int) -> float:
        """Jaccard similarity over prime factor sets."""
        return TriadicValidator.similarity(a, b)


# ============================================================
# TriadicValidator — Algebraic semantic operations
# ============================================================

class TriadicValidator:
    """
    Verifies semantic relationships using prime-factor algebra.

    Three operations IMPOSSIBLE with cosine similarity:
      1. Subsumption — Does concept A contain all features of B?
      2. Composition — Create a concept with features of A and B.
      3. Gap Analysis — Exactly WHICH features differ between A and B?
    """

    @staticmethod
    def subsumes(a: int, b: int) -> bool:
        """Does concept A contain ALL semantic features of B?  (A % B == 0)"""
        if b == 0:
            return False
        return a % b == 0

    @staticmethod
    def compose(*concepts: int) -> int:
        """Combine features from multiple concepts (LCM)."""
        result = concepts[0]
        for c in concepts[1:]:
            result = (result * c) // math.gcd(result, c)
        return result

    @staticmethod
    def explain_gap(a: int, b: int) -> Dict:
        """Explain exactly WHY two concepts differ."""
        shared = math.gcd(a, b)
        only_a = a // shared
        only_b = b // shared
        return {
            'shared': shared,
            'shared_factors': prime_factors(shared),
            'only_in_a': only_a,
            'only_in_a_factors': prime_factors(only_a),
            'only_in_b': only_b,
            'only_in_b_factors': prime_factors(only_b),
            'a_contains_b': (a % b == 0) if b > 0 else False,
            'b_contains_a': (b % a == 0) if a > 0 else False,
        }

    @staticmethod
    def similarity(a: int, b: int) -> float:
        """Jaccard similarity over prime factor sets. Range: [0.0, 1.0]."""
        fa = set(prime_factors(a))
        fb = set(prime_factors(b))
        if not fa and not fb:
            return 1.0
        total = fa | fb
        return len(fa & fb) / len(total) if total else 0.0

    @staticmethod
    def analogy(a: int, b: int, c: int) -> int:
        """
        Analogy: A is to B as C is to ?

        Computes the transformation from A->B (factors removed/added),
        then applies it to C to find D.

        D = (C / GCD(C, only_in_a)) * only_in_b
        where only_in_a = A/GCD(A,B) and only_in_b = B/GCD(A,B)

        Two steps: (1) remove A-specific factors from C, (2) add B-specific.
        """
        shared_ab = math.gcd(a, b)
        only_in_a = a // shared_ab  # factors to remove
        only_in_b = b // shared_ab  # factors to add

        # Remove A-specific factors from C (where they overlap)
        c_reduced = c // math.gcd(c, only_in_a)
        # Add B-specific factors (avoiding duplicates via GCD)
        return (c_reduced * only_in_b) // math.gcd(c_reduced, only_in_b)
