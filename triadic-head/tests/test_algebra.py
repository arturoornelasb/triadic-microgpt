"""Tests for prime-factor algebra."""

import pytest
from triadic_head.algebra import (
    PrimeMapper, TriadicValidator, prime_factors, nth_prime, sieve_primes
)


class TestPrimes:
    def test_sieve(self):
        p = sieve_primes(30)
        assert p == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

    def test_nth_prime(self):
        assert nth_prime(1) == 2
        assert nth_prime(4) == 7
        assert nth_prime(10) == 29

    def test_prime_factors(self):
        assert prime_factors(1) == []
        assert prime_factors(12) == [2, 3]
        assert prime_factors(30) == [2, 3, 5]
        assert prime_factors(7) == [7]


class TestPrimeMapper:
    def test_encode_basic(self):
        m = PrimeMapper(4)  # primes: 2, 3, 5, 7
        # All positive -> 2*3*5*7 = 210
        assert m.encode([0.5, 0.5, 0.5, 0.5]) == 210
        # Only first two -> 2*3 = 6
        assert m.encode([0.5, 0.5, -0.5, -0.5]) == 6

    def test_encode_all_negative(self):
        m = PrimeMapper(4)
        # All negative -> composite = 1 (identity element, no active primitives)
        assert m.encode([-0.5, -0.5, -0.5, -0.5]) == 1

    def test_get_bits(self):
        m = PrimeMapper(4)
        assert m.get_bits([0.5, -0.5, 0.5, -0.5]) == [1, 0, 1, 0]

    def test_explain(self):
        m = PrimeMapper(4)
        info = m.explain(30)  # 2*3*5
        assert info['factors'] == [2, 3, 5]
        assert info['active_bits'] == [0, 1, 2]
        assert info['n_active'] == 3

    def test_64_bits(self):
        m = PrimeMapper(64)
        assert len(m.primes) == 64
        proj = [0.1] * 32 + [-0.1] * 32
        c = m.encode(proj)
        assert c > 1
        bits = m.get_bits(proj)
        assert sum(bits) == 32


class TestTriadicValidator:
    def test_subsumes(self):
        # 30 = 2*3*5; 6 = 2*3
        assert TriadicValidator.subsumes(30, 6) is True
        assert TriadicValidator.subsumes(6, 30) is False

    def test_compose(self):
        # LCM(6, 10) = LCM(2*3, 2*5) = 30
        assert TriadicValidator.compose(6, 10) == 30
        assert TriadicValidator.compose(6, 10, 21) == 210  # LCM(30, 21=3*7) = 210

    def test_explain_gap(self):
        gap = TriadicValidator.explain_gap(30, 70)  # 30=2*3*5, 70=2*5*7
        assert gap['shared_factors'] == [2, 5]
        assert gap['only_in_a_factors'] == [3]
        assert gap['only_in_b_factors'] == [7]

    def test_similarity(self):
        # 30=2*3*5, 70=2*5*7 -> shared={2,5}, total={2,3,5,7} -> 2/4 = 0.5
        assert TriadicValidator.similarity(30, 70) == 0.5
        assert TriadicValidator.similarity(30, 30) == 1.0

    def test_analogy(self):
        # king(30=2*3*5) : queen(70=2*5*7) :: man(6=2*3) : woman(?)
        # only_in_a=3 (male factor), only_in_b=7 (female factor)
        # c_reduced = 6/gcd(6,3) = 2 (remove male), then 2*7 = 14 (add female)
        result = TriadicValidator.analogy(30, 70, 6)
        assert result == 14  # woman = 2*7


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
