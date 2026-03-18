"""
Triadic Algebra — Pure-Python neurosymbolic bridge.

Maps continuous hidden states from the transformer into discrete
prime-factor integers. Provides algebraic operations (subsumption,
composition, gap analysis) for transparent semantic verification.

Zero external dependencies — primes are computed via a simple sieve.
"""

import math


# ============================================================
# Prime Number Utilities
# ============================================================

def sieve_primes(limit):
    """Return all primes up to `limit` using the Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]


# Pre-compute first 100 primes (more than enough for n_bits <= 64)
_PRIMES_CACHE = sieve_primes(600)  # first 100+ primes are all < 600


def nth_prime(n):
    """Return the nth prime (1-indexed: nth_prime(1) = 2, nth_prime(2) = 3, ...)."""
    if n <= 0:
        raise ValueError("n must be >= 1")
    if n <= len(_PRIMES_CACHE):
        return _PRIMES_CACHE[n - 1]
    # Extend cache if needed
    while len(_PRIMES_CACHE) < n:
        candidate = _PRIMES_CACHE[-1] + 2
        while True:
            if all(candidate % p != 0 for p in _PRIMES_CACHE if p * p <= candidate):
                _PRIMES_CACHE.append(candidate)
                break
            candidate += 2
    return _PRIMES_CACHE[n - 1]


def prime_factors(n):
    """Return the sorted list of unique prime factors of n."""
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
# Prime Mapper — Maps projections to composite primes
# ============================================================

class PrimeMapper:
    """
    Maps continuous projections (from tanh, in [-1, 1]) to composite prime integers.

    Each bit position is assigned a unique prime. If projection[i] > 0,
    then prime[i] is included in the composite product.

    Example:
        projections = [0.5, -0.2, 0.8, 0.1]  (4 bits)
        primes      = [2,    3,    5,   7  ]
        composite   = 2 * 5 * 7 = 70  (bits 0, 2, 3 are positive)
    """

    def __init__(self, n_bits):
        self.n_bits = n_bits
        self.primes = [nth_prime(i + 1) for i in range(n_bits)]

    def map(self, projections):
        """
        Convert a list of projection values to a composite prime integer.

        Args:
            projections: list of floats or Value objects (tanh outputs in [-1, 1])

        Returns:
            int — composite prime product
        """
        composite = 1
        for proj, prime in zip(projections, self.primes):
            val = proj.data if hasattr(proj, 'grad') else proj
            if val > 0:
                composite *= prime
        # Degenerate case: all projections negative = no active primitives.
        # Return 1 (identity element for multiplication) rather than
        # assigning prime 2 ("vacío") which has specific semantic meaning.
        # Callers should check for composite == 1 to handle this case.
        return composite

    def get_bits(self, projections):
        """Return the binary bit pattern from projections."""
        return [1 if (p.data if hasattr(p, 'grad') else p) > 0 else 0
                for p in projections]

    def explain(self, composite):
        """
        Decompose a composite prime into its constituent factors with labels.

        Returns:
            dict with 'factors' (list of primes), 'bit_indices' (which bits are active),
            and 'composite' (the integer).
        """
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


# ============================================================
# Triadic Validator — Algebraic semantic operations
# ============================================================

class TriadicValidator:
    """
    Verifies semantic relationships using prime-factor algebra.

    Three operations IMPOSSIBLE with cosine similarity:
      1. Subsumption — Does concept A contain all features of B?
      2. Composition — Create a concept with all features of A and B
      3. Gap Analysis — Exactly WHICH features differ between A and B?
    """

    @staticmethod
    def subsumes(a, b):
        """
        Logical Subsumption: Does concept A contain ALL semantic features of B?

        A subsumes B iff B divides A (A % B == 0).
        Example: King(2*3*5) subsumes Male(3) → True
        """
        if b == 0:
            return False
        return a % b == 0

    @staticmethod
    def compose(*concepts):
        """
        Algebraic Composition: Combine features from multiple concepts.

        Returns LCM of all concepts — the smallest integer containing
        all prime factors from all inputs.
        Example: compose(6, 10) = compose(2*3, 2*5) = 30 = 2*3*5
        """
        result = concepts[0]
        for c in concepts[1:]:
            result = (result * c) // math.gcd(result, c)
        return result

    @staticmethod
    def explain_gap(a, b):
        """
        Abductive Discovery: Explains exactly WHY two concepts differ.

        Returns:
            shared   — GCD(a, b): the common semantic backbone
            only_in_a — factors present in A but missing from B
            only_in_b — factors present in B but missing from A

        Example: explain_gap(King=30, Queen=70)
            shared=10, only_in_king=3, only_in_queen=7
        """
        shared = math.gcd(a, b)
        only_in_a = a // shared
        only_in_b = b // shared
        return {
            'shared': shared,
            'shared_factors': prime_factors(shared),
            'only_in_a': only_in_a,
            'only_in_a_factors': prime_factors(only_in_a),
            'only_in_b': only_in_b,
            'only_in_b_factors': prime_factors(only_in_b),
            'a_contains_b': (a % b == 0),
            'b_contains_a': (b % a == 0),
        }

    @staticmethod
    def analogy(a, b, c):
        """
        Algebraic Analogy: A is to B as C is to ?

        Computes the transformation from A→B (factors removed/added),
        then applies it to C to find D.

        D = (C / GCD(C, only_in_a)) * only_in_b
        where only_in_a = A/GCD(A,B) and only_in_b = B/GCD(A,B)

        Example: king:queen::man:?
            shared(king,queen) = GCD → only_in_king=male, only_in_queen=female
            target = (man / male_factors) * female_factors = woman
        """
        shared_ab = math.gcd(a, b)
        only_in_a = a // shared_ab  # factors to remove
        only_in_b = b // shared_ab  # factors to add

        # Remove A-specific factors from C (where they overlap)
        c_reduced = c // math.gcd(c, only_in_a)
        # Add B-specific factors
        target = (c_reduced * only_in_b) // math.gcd(c_reduced, only_in_b)
        return target

    @staticmethod
    def similarity(a, b):
        """
        Semantic similarity based on shared prime factors.

        Returns the ratio of shared factors to total unique factors.
        Range: [0.0, 1.0] where 1.0 means identical prime signatures.
        """
        factors_a = set(prime_factors(a))
        factors_b = set(prime_factors(b))
        if not factors_a and not factors_b:
            return 1.0
        shared = factors_a & factors_b
        total = factors_a | factors_b
        return len(shared) / len(total) if total else 0.0


# ============================================================
# Triadic Loss — Differentiable loss for prime alignment
# ============================================================

def triadic_loss(projections_a, projections_b, should_share=True):
    """
    Differentiable loss that encourages/discourages shared prime factors
    between two sets of projections.

    For concepts that SHOULD share factors (same document/group):
      Push projections toward the same sign → low loss when both positive or both negative

    For concepts that SHOULD NOT share factors (different documents):
      Push projections toward different signs → low loss when signs differ

    Uses a smooth binary cross-entropy-like formulation over tanh outputs.

    Args:
        projections_a: list of Value nodes (tanh outputs) from concept A
        projections_b: list of Value nodes (tanh outputs) from concept B
        should_share: if True, encourage agreement; if False, encourage disagreement

    Returns:
        Value — scalar loss
    """
    loss = projections_a[0] * 0  # start at zero, keeping the graph

    for pa, pb in zip(projections_a, projections_b):
        # Agreement score: pa * pb is high when they have the same sign
        agreement = pa * pb

        if should_share:
            # We want agreement → maximize pa * pb → minimize -(pa * pb)
            loss = loss + (1 - agreement)
        else:
            # We want disagreement → minimize pa * pb → minimize (pa * pb)
            loss = loss + (1 + agreement)

    # Normalize by number of bits
    loss = loss * (1.0 / len(projections_a))

    return loss
