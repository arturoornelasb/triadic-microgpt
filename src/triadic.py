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

    # Alias for compatibility with triadic-head PyPI package which uses encode()
    encode = map

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

    @staticmethod
    def intersect(a, b):
        """
        Semantic Intersection: Features SHARED between A and B.

        Returns GCD(A, B) — the common semantic backbone.
        Example: intersect(30, 70) = 10 (shared: {2, 5})
        """
        return math.gcd(a, b)

    @staticmethod
    def difference(a, b):
        """
        Semantic Difference: Features in A but NOT in B.

        Returns A / GCD(A, B).
        Example: difference(30, 70) = 3 (only in A: {3})
        """
        return a // math.gcd(a, b)

    @staticmethod
    def symmetric_difference(a, b):
        """
        Symmetric Difference: Features in EXACTLY ONE of A or B.

        Returns (A \\ B) * (B \\ A) — all distinguishing features.
        Example: symmetric_difference(30, 70) = 21 = 3*7
        """
        g = math.gcd(a, b)
        return (a // g) * (b // g)

    @staticmethod
    def negate(a, n_bits=64):
        """
        Semantic Negation: Complement of A within the universe of n_bits primes.

        Returns Omega / A where Omega = product of first n_bits primes.
        Caveat: Algebraically exact, semantically approximate — use dual axes
        (bien/mal, vida/muerte) for domain-appropriate negation.
        """
        omega = 1
        for i in range(n_bits):
            omega *= nth_prime(i + 1)
        return omega // a

    @staticmethod
    def project(a, category_primes):
        """
        Category Projection: Extract only features from a specific category.

        Args:
            a: composite integer
            category_primes: list of primes defining the category

        Returns: composite containing only A's factors that are in category_primes.
        Example: project(210, [2, 3, 5]) = 30 (keeps {2,3,5} from {2,3,5,7})
        """
        result = 1
        for p in category_primes:
            if a % p == 0:
                result *= p
        return result


# ============================================================
# Bitwise Mapper — O(1) alternative to PrimeMapper
# ============================================================

class BitwiseMapper:
    """Maps projections to bitmask integers instead of prime composites.

    Isomorphic to PrimeMapper but uses O(1) bitwise operations instead
    of BigInt arithmetic. Scales to any number of bits.

    Example:
        projections = [0.5, -0.2, 0.8, 0.1]  (4 bits)
        bitmask     = 0b1101 = 13  (bits 0, 2, 3 are positive)
    """

    def __init__(self, n_bits):
        self.n_bits = n_bits

    def map(self, projections):
        """Convert projections to a bitmask integer."""
        bitmask = 0
        for i, proj in enumerate(projections[:self.n_bits]):
            val = proj.data if hasattr(proj, 'grad') else proj
            if val > 0:
                bitmask |= (1 << i)
        return bitmask

    encode = map

    def get_bits(self, projections):
        """Return the binary bit pattern from projections."""
        return [1 if (p.data if hasattr(p, 'grad') else p) > 0 else 0
                for p in projections[:self.n_bits]]

    def to_prime(self, bitmask, prime_mapper):
        """Convert bitmask to equivalent prime composite."""
        composite = 1
        for i in range(self.n_bits):
            if bitmask & (1 << i):
                composite *= prime_mapper.primes[i]
        return composite

    def from_prime(self, composite, prime_mapper):
        """Convert prime composite to equivalent bitmask."""
        bitmask = 0
        for i, p in enumerate(prime_mapper.primes):
            if composite % p == 0:
                bitmask |= (1 << i)
        return bitmask

    def explain(self, bitmask):
        """Decompose a bitmask into active bit indices."""
        active = []
        for i in range(self.n_bits):
            if bitmask & (1 << i):
                active.append(i)
        return {
            'bitmask': bitmask,
            'active_bits': active,
            'n_active': len(active),
            'n_total': self.n_bits,
        }


class BitwiseValidator:
    """Algebraic semantic operations using O(1) bitwise ops.

    Mathematically equivalent to TriadicValidator but operates on
    bitmask integers instead of prime composites. Every operation
    that TriadicValidator does with GCD/LCM/division, this class
    does with AND/OR/XOR.

    Equivalences:
        GCD(A, B)         <->  A & B          (shared features)
        LCM(A, B)         <->  A | B          (union of features)
        A / GCD(A, B)     <->  A & ~B         (features only in A)
        A % B == 0        <->  (A & B) == B   (subsumption)
    """

    @staticmethod
    def subsumes(a, b):
        """Does concept A contain ALL features of B?  O(1)."""
        return (a & b) == b

    @staticmethod
    def compose(*concepts):
        """Combine features from multiple concepts.  O(n)."""
        result = 0
        for c in concepts:
            result |= c
        return result

    @staticmethod
    def explain_gap(a, b):
        """Exactly WHICH features differ between A and B.  O(1)."""
        shared = a & b
        only_in_a = a & ~b
        only_in_b = b & ~a
        return {
            'shared': shared,
            'only_in_a': only_in_a,
            'only_in_b': only_in_b,
            'a_contains_b': (a & b) == b,
            'b_contains_a': (a & b) == a,
        }

    @staticmethod
    def analogy(a, b, c):
        """A is to B as C is to ?  O(1).

        Same transformation as TriadicValidator.analogy but in bit space:
        1. Find what A has that B doesn't (only_a)
        2. Find what B has that A doesn't (only_b)
        3. Remove only_a from C, add only_b
        """
        only_a = a & ~b  # bits to remove
        only_b = b & ~a  # bits to add
        return (c & ~only_a) | only_b

    @staticmethod
    def similarity(a, b):
        """Jaccard similarity on active bits.  O(1)."""
        shared = bin(a & b).count('1')
        total = bin(a | b).count('1')
        if total == 0:
            return 1.0
        return shared / total

    @staticmethod
    def intersect(a, b):
        """Shared features.  O(1)."""
        return a & b

    @staticmethod
    def difference(a, b):
        """Features in A but NOT in B.  O(1)."""
        return a & ~b

    @staticmethod
    def symmetric_difference(a, b):
        """Features in EXACTLY ONE of A or B.  O(1)."""
        return a ^ b

    @staticmethod
    def negate(a, n_bits=64):
        """Complement within universe of n_bits.  O(1)."""
        mask = (1 << n_bits) - 1
        return a ^ mask

    @staticmethod
    def project(a, category_bits):
        """Extract only features from specific bit positions.  O(1).

        Args:
            a: bitmask integer
            category_bits: list of bit indices defining the category
        """
        mask = 0
        for b in category_bits:
            mask |= (1 << b)
        return a & mask


# ============================================================
# Default aliases — BitwiseMapper/BitwiseValidator are O(1)
# isomorphic to PrimeMapper/TriadicValidator. Use as default.
# ============================================================

DefaultMapper = BitwiseMapper
DefaultValidator = BitwiseValidator


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
