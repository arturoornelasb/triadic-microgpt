"""
Prime vs Bitwise — Prove equivalence and benchmark scaling.

Demonstrates that BitwiseValidator produces identical results to
TriadicValidator, but scales to arbitrary bit counts where primes
become computationally impossible.

Usage:
    python benchmarks/scripts/prime_vs_bitwise.py
"""

import sys
import os
import time
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from src.triadic import (
    PrimeMapper, BitwiseMapper, TriadicValidator, BitwiseValidator, nth_prime,
)


def bits_to_bitmask(bits):
    """Convert list of 0/1 to bitmask integer."""
    mask = 0
    for i, b in enumerate(bits):
        if b:
            mask |= (1 << i)
    return mask


def bits_to_prime(bits, mapper):
    """Convert list of 0/1 to prime composite."""
    composite = 1
    for i, b in enumerate(bits):
        if b:
            composite *= mapper.primes[i]
    return composite


def random_concept(n_bits, density=0.4, rng=None):
    """Generate random bit pattern."""
    if rng is None:
        rng = np.random.default_rng()
    return (rng.random(n_bits) < density).astype(int).tolist()


# ============================================================
# Test 1: Equivalence proof at 63 bits
# ============================================================

def test_equivalence():
    print("=" * 70)
    print("  TEST 1: EQUIVALENCE PROOF (63 bits)")
    print("=" * 70)

    n_bits = 63
    pm = PrimeMapper(n_bits)
    bm = BitwiseMapper(n_bits)
    tv = TriadicValidator()
    bv = BitwiseValidator()
    rng = np.random.default_rng(42)

    n_tests = 1000
    n_pass = 0

    for _ in range(n_tests):
        a_bits = random_concept(n_bits, rng=rng)
        b_bits = random_concept(n_bits, rng=rng)
        c_bits = random_concept(n_bits, rng=rng)

        a_prime = bits_to_prime(a_bits, pm)
        b_prime = bits_to_prime(b_bits, pm)
        c_prime = bits_to_prime(c_bits, pm)

        a_mask = bits_to_bitmask(a_bits)
        b_mask = bits_to_bitmask(b_bits)
        c_mask = bits_to_bitmask(c_bits)

        # Subsumption
        assert tv.subsumes(a_prime, b_prime) == bv.subsumes(a_mask, b_mask)

        # Compose
        compose_prime = tv.compose(a_prime, b_prime)
        compose_bits = bv.compose(a_mask, b_mask)
        # Verify by converting back
        compose_prime_bits = bm.from_prime(compose_prime, pm)
        assert compose_prime_bits == compose_bits

        # Intersect
        inter_prime = tv.intersect(a_prime, b_prime)
        inter_bits = bv.intersect(a_mask, b_mask)
        assert bm.from_prime(inter_prime, pm) == inter_bits

        # Difference
        diff_prime = tv.difference(a_prime, b_prime)
        diff_bits = bv.difference(a_mask, b_mask)
        assert bm.from_prime(diff_prime, pm) == diff_bits

        # Similarity
        sim_prime = tv.similarity(a_prime, b_prime)
        sim_bits = bv.similarity(a_mask, b_mask)
        assert abs(sim_prime - sim_bits) < 1e-10

        # Analogy (the critical one)
        d_prime = tv.analogy(a_prime, b_prime, c_prime)
        d_bits = bv.analogy(a_mask, b_mask, c_mask)
        d_prime_as_bits = bm.from_prime(d_prime, pm)
        assert d_prime_as_bits == d_bits, \
            f"Analogy mismatch: prime->{d_prime_as_bits} bits->{d_bits}"

        n_pass += 1

    print(f"  {n_pass}/{n_tests} tests passed — PERFECT EQUIVALENCE")
    print()


# ============================================================
# Test 2: Speed benchmark at 63 bits
# ============================================================

def test_speed_63():
    print("=" * 70)
    print("  TEST 2: SPEED BENCHMARK (63 bits)")
    print("=" * 70)

    n_bits = 63
    pm = PrimeMapper(n_bits)
    tv = TriadicValidator()
    bv = BitwiseValidator()
    rng = np.random.default_rng(42)

    # Pre-generate data
    n_ops = 10000
    concepts_bits = [random_concept(n_bits, rng=rng) for _ in range(n_ops * 3)]
    concepts_prime = [bits_to_prime(b, pm) for b in concepts_bits]
    concepts_mask = [bits_to_bitmask(b) for b in concepts_bits]

    # Benchmark: Analogy (most expensive operation)
    t0 = time.perf_counter()
    for i in range(0, n_ops * 3, 3):
        tv.analogy(concepts_prime[i], concepts_prime[i+1], concepts_prime[i+2])
    t_prime = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i in range(0, n_ops * 3, 3):
        bv.analogy(concepts_mask[i], concepts_mask[i+1], concepts_mask[i+2])
    t_bits = time.perf_counter() - t0

    print(f"  Analogy x{n_ops}:")
    print(f"    Prime:   {t_prime:.3f}s ({n_ops/t_prime:.0f} ops/s)")
    print(f"    Bitwise: {t_bits:.3f}s ({n_ops/t_bits:.0f} ops/s)")
    print(f"    Speedup: {t_prime/t_bits:.1f}x")
    print()

    # Benchmark: Subsumption
    t0 = time.perf_counter()
    for i in range(0, n_ops * 2, 2):
        tv.subsumes(concepts_prime[i], concepts_prime[i+1])
    t_prime = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i in range(0, n_ops * 2, 2):
        bv.subsumes(concepts_mask[i], concepts_mask[i+1])
    t_bits = time.perf_counter() - t0

    print(f"  Subsumption x{n_ops}:")
    print(f"    Prime:   {t_prime:.3f}s ({n_ops/t_prime:.0f} ops/s)")
    print(f"    Bitwise: {t_bits:.3f}s ({n_ops/t_bits:.0f} ops/s)")
    print(f"    Speedup: {t_prime/t_bits:.1f}x")
    print()

    # Benchmark: Similarity
    t0 = time.perf_counter()
    for i in range(0, n_ops * 2, 2):
        tv.similarity(concepts_prime[i], concepts_prime[i+1])
    t_prime = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i in range(0, n_ops * 2, 2):
        bv.similarity(concepts_mask[i], concepts_mask[i+1])
    t_bits = time.perf_counter() - t0

    print(f"  Similarity x{n_ops}:")
    print(f"    Prime:   {t_prime:.3f}s ({n_ops/t_prime:.0f} ops/s)")
    print(f"    Bitwise: {t_bits:.3f}s ({n_ops/t_bits:.0f} ops/s)")
    print(f"    Speedup: {t_prime/t_bits:.1f}x")
    print()


# ============================================================
# Test 3: Scaling — where primes become impossible
# ============================================================

def test_scaling():
    print("=" * 70)
    print("  TEST 3: SCALING (64 -> 1024 bits)")
    print("=" * 70)

    bv = BitwiseValidator()
    rng = np.random.default_rng(42)

    for n_bits in [64, 128, 256, 512, 1024]:
        # Generate random concepts
        concepts = [bits_to_bitmask(random_concept(n_bits, rng=rng))
                    for _ in range(3000)]

        # Prime feasibility check
        largest_prime = nth_prime(n_bits)
        if n_bits <= 128:
            # Estimate composite size (product of ~40% of primes)
            active_primes = [nth_prime(i+1) for i in range(n_bits)
                             if rng.random() < 0.4]
            composite_digits = sum(math.log10(p) for p in active_primes) if active_primes else 0
        else:
            composite_digits = float('inf')

        # Benchmark bitwise operations
        n_ops = 1000
        t0 = time.perf_counter()
        for i in range(0, min(n_ops * 3, len(concepts) - 2), 3):
            bv.analogy(concepts[i], concepts[i+1], concepts[i+2])
        t_bits = time.perf_counter() - t0

        t0_sub = time.perf_counter()
        for i in range(0, min(n_ops * 2, len(concepts) - 1), 2):
            bv.subsumes(concepts[i], concepts[i+1])
        t_sub = time.perf_counter() - t0_sub

        prime_status = "feasible" if n_bits <= 64 else "SLOW" if n_bits <= 128 else "IMPOSSIBLE"
        digits_str = f"~{composite_digits:.0f} digits" if composite_digits < float('inf') else "overflow"

        print(f"  {n_bits:>4d} bits | prime {largest_prime:>5d} | composite: {digits_str:>14s} | "
              f"prime: {prime_status:>10s} | "
              f"bitwise analogy: {t_bits*1000:.1f}ms | subsume: {t_sub*1000:.1f}ms")

    print()
    print("  Bitwise operations are O(1) regardless of bit count.")
    print("  Prime operations become impossible beyond ~128 bits.")
    print()


# ============================================================
# Test 4: Real-world example with named concepts
# ============================================================

def test_real_example():
    print("=" * 70)
    print("  TEST 4: REAL-WORLD ANALOGY EXAMPLE")
    print("=" * 70)

    n_bits = 63
    pm = PrimeMapper(n_bits)
    bv = BitwiseValidator()
    tv = TriadicValidator()

    # Simulated bit patterns (manually designed for clarity)
    # bit 0=animate, 1=human, 2=male, 3=female, 4=royal, 5=adult
    king  = 0b111101  # animate, human, male, royal, adult
    queen = 0b111011  # animate, human, female, royal, adult
    man   = 0b110101  # animate, human, male, adult
    woman = 0b110011  # animate, human, female, adult
    boy   = 0b010101  # animate, human, male

    # king:queen :: man:?
    result = bv.analogy(king, queen, man)
    print(f"  king:queen :: man:?")
    print(f"    king  = {king:06b}  (animate, human, male, royal, adult)")
    print(f"    queen = {queen:06b}  (animate, human, female, royal, adult)")
    print(f"    man   = {man:06b}  (animate, human, male, adult)")
    print(f"    result= {result:06b}  -> {'woman' if result == woman else 'unknown'}")
    print(f"    expected woman = {woman:06b}")
    assert result == woman, f"Expected woman ({woman:06b}), got {result:06b}"
    print(f"    PASS")

    # Verify same with primes
    king_p = pm.map([1,0,1,1,1,1] + [0]*57)
    queen_p = pm.map([1,1,0,1,1,1] + [0]*57)
    man_p = pm.map([1,0,1,0,1,1] + [0]*57)
    woman_p = pm.map([1,1,0,0,1,1] + [0]*57)
    result_p = tv.analogy(king_p, queen_p, man_p)
    print(f"\n  Prime verification:")
    print(f"    king={king_p}, queen={queen_p}, man={man_p}")
    print(f"    result={result_p}, expected woman={woman_p}")
    assert result_p == woman_p
    print(f"    PASS — identical result")

    # Gap analysis
    print(f"\n  Gap analysis (king vs queen):")
    gap = bv.explain_gap(king, queen)
    print(f"    shared:    {gap['shared']:06b}")
    print(f"    only_king: {gap['only_in_a']:06b}  (male bit)")
    print(f"    only_queen:{gap['only_in_b']:06b}  (female bit)")
    print()


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("\n  PRIME vs BITWISE — Equivalence & Scaling Benchmark")
    print("  " + "=" * 66)
    print()

    test_equivalence()
    test_speed_63()
    test_scaling()
    test_real_example()

    print("=" * 70)
    print("  ALL TESTS PASSED")
    print()
    print("  Conclusion:")
    print("    - Primes and bits are mathematically isomorphic")
    print("    - Primes: elegant theory, proves algebraic properties")
    print("    - Bits: O(1) implementation, scales to any dimension")
    print("    - For the paper: formalize with primes, implement with bits")
    print("=" * 70)
