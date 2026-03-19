"""
Test Suite — Automated tests for Triadic MicroGPT.

Tests cover:
  1. Autograd engine: forward/backward correctness
  2. Transformer: forward pass, softmax, rmsnorm
  3. Triadic algebra: primes, subsumption, composition, gap analysis
  4. Integration: end-to-end training smoke test

Run with:  python tests/test_all.py
Or with:   python -m pytest tests/test_all.py -v
"""

import sys
import os
import math
import unittest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.autograd import Value
from src.transformer import GPT, GPTConfig, softmax, rmsnorm, linear
from src.triadic import (
    nth_prime, prime_factors, sieve_primes,
    PrimeMapper, TriadicValidator, triadic_loss,
)


# ============================================================
# Autograd Tests
# ============================================================

class TestAutograd(unittest.TestCase):
    """Test the Value autograd engine."""

    def test_addition(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        self.assertAlmostEqual(c.data, 5.0)

    def test_multiplication(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        self.assertAlmostEqual(c.data, 6.0)

    def test_power(self):
        a = Value(3.0)
        c = a ** 2
        self.assertAlmostEqual(c.data, 9.0)

    def test_backward_simple(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        c.backward()
        self.assertAlmostEqual(a.grad, 3.0)  # dc/da = b
        self.assertAlmostEqual(b.grad, 2.0)  # dc/db = a

    def test_backward_chain(self):
        x = Value(2.0)
        y = x * x + x * 3  # y = x^2 + 3x, dy/dx = 2x + 3 = 7
        y.backward()
        self.assertAlmostEqual(x.grad, 7.0)

    def test_relu(self):
        a = Value(5.0)
        b = Value(-3.0)
        self.assertAlmostEqual(a.relu().data, 5.0)
        self.assertAlmostEqual(b.relu().data, 0.0)

    def test_relu_backward(self):
        a = Value(5.0)
        c = a.relu()
        c.backward()
        self.assertAlmostEqual(a.grad, 1.0)

        b = Value(-3.0)
        d = b.relu()
        d.backward()
        self.assertAlmostEqual(b.grad, 0.0)

    def test_tanh(self):
        a = Value(0.0)
        self.assertAlmostEqual(a.tanh().data, 0.0, places=5)

        b = Value(1.0)
        self.assertAlmostEqual(b.tanh().data, math.tanh(1.0), places=5)

    def test_tanh_backward(self):
        a = Value(0.5)
        c = a.tanh()
        c.backward()
        expected_grad = 1 - math.tanh(0.5) ** 2
        self.assertAlmostEqual(a.grad, expected_grad, places=5)

    def test_exp(self):
        a = Value(1.0)
        c = a.exp()
        self.assertAlmostEqual(c.data, math.e, places=5)

    def test_log(self):
        a = Value(math.e)
        c = a.log()
        self.assertAlmostEqual(c.data, 1.0, places=5)

    def test_division(self):
        a = Value(6.0)
        b = Value(3.0)
        c = a / b
        self.assertAlmostEqual(c.data, 2.0)

    def test_subtraction(self):
        a = Value(5.0)
        b = Value(3.0)
        c = a - b
        self.assertAlmostEqual(c.data, 2.0)

    def test_negation(self):
        a = Value(5.0)
        c = -a
        self.assertAlmostEqual(c.data, -5.0)

    def test_scalar_operations(self):
        a = Value(3.0)
        c = a + 2
        self.assertAlmostEqual(c.data, 5.0)
        d = 2 + a
        self.assertAlmostEqual(d.data, 5.0)
        e = a * 4
        self.assertAlmostEqual(e.data, 12.0)
        f = 4 * a
        self.assertAlmostEqual(f.data, 12.0)


# ============================================================
# Transformer Tests
# ============================================================

class TestTransformer(unittest.TestCase):
    """Test the GPT transformer model."""

    def setUp(self):
        self.config = GPTConfig(
            vocab_size=10,
            block_size=8,
            n_layer=1,
            n_embd=8,
            n_head=2,
            n_triadic_bits=4,
        )
        self.model = GPT(self.config)

    def test_forward_shape(self):
        keys = [[] for _ in range(self.config.n_layer)]
        values = [[] for _ in range(self.config.n_layer)]
        logits, hidden = self.model.forward(0, 0, keys, values)
        self.assertEqual(len(logits), self.config.vocab_size)
        self.assertEqual(len(hidden), self.config.n_embd)

    def test_forward_value_type(self):
        keys = [[] for _ in range(self.config.n_layer)]
        values = [[] for _ in range(self.config.n_layer)]
        logits, hidden = self.model.forward(0, 0, keys, values)
        self.assertIsInstance(logits[0], Value)
        self.assertIsInstance(hidden[0], Value)

    def test_softmax_sums_to_one(self):
        logits = [Value(1.0), Value(2.0), Value(3.0)]
        probs = softmax(logits)
        total = sum(p.data for p in probs)
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_softmax_order(self):
        logits = [Value(1.0), Value(2.0), Value(3.0)]
        probs = softmax(logits)
        self.assertLess(probs[0].data, probs[1].data)
        self.assertLess(probs[1].data, probs[2].data)

    def test_rmsnorm_scale(self):
        x = [Value(2.0), Value(3.0), Value(4.0), Value(5.0)]
        normed = rmsnorm(x)
        # RMSNorm should produce values close to unit scale
        ms = sum(v.data ** 2 for v in normed) / len(normed)
        self.assertAlmostEqual(ms, 1.0, places=1)

    def test_triadic_projection(self):
        keys = [[] for _ in range(self.config.n_layer)]
        values = [[] for _ in range(self.config.n_layer)]
        _, hidden = self.model.forward(0, 0, keys, values)
        projections = self.model.project_to_triadic(hidden)
        self.assertEqual(len(projections), self.config.n_triadic_bits)
        # tanh outputs should be in [-1, 1]
        for p in projections:
            self.assertGreaterEqual(p.data, -1.0)
            self.assertLessEqual(p.data, 1.0)

    def test_params_count(self):
        params = self.model.params()
        # Should have some params
        self.assertGreater(len(params), 0)
        # All should be Value objects
        for p in params:
            self.assertIsInstance(p, Value)

    def test_linear(self):
        x = [Value(1.0), Value(2.0)]
        w = [[Value(1.0), Value(0.0)], [Value(0.0), Value(1.0)]]
        y = linear(x, w)
        self.assertAlmostEqual(y[0].data, 1.0)
        self.assertAlmostEqual(y[1].data, 2.0)


# ============================================================
# Triadic Tests
# ============================================================

class TestTriadic(unittest.TestCase):
    """Test the triadic algebra module."""

    def test_sieve_primes(self):
        primes = sieve_primes(30)
        self.assertEqual(primes, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29])

    def test_nth_prime(self):
        self.assertEqual(nth_prime(1), 2)
        self.assertEqual(nth_prime(2), 3)
        self.assertEqual(nth_prime(3), 5)
        self.assertEqual(nth_prime(4), 7)
        self.assertEqual(nth_prime(10), 29)

    def test_prime_factors(self):
        self.assertEqual(prime_factors(30), [2, 3, 5])
        self.assertEqual(prime_factors(70), [2, 5, 7])
        self.assertEqual(prime_factors(7), [7])
        self.assertEqual(prime_factors(1), [])

    def test_prime_mapper(self):
        mapper = PrimeMapper(4)
        # Primes: [2, 3, 5, 7]
        # projections: [+, -, +, +] → 2 * 5 * 7 = 70
        prime = mapper.map([0.5, -0.2, 0.8, 0.1])
        self.assertEqual(prime, 70)

    def test_prime_mapper_all_negative(self):
        mapper = PrimeMapper(4)
        # All negative → identity element 1 (no active primitives)
        # Bug #3 fix: returns 1 instead of 2 to avoid conflating with vacío
        prime = mapper.map([-0.1, -0.2, -0.3, -0.4])
        self.assertEqual(prime, 1)

    def test_prime_mapper_all_positive(self):
        mapper = PrimeMapper(4)
        # All positive → 2 * 3 * 5 * 7 = 210
        prime = mapper.map([0.1, 0.2, 0.3, 0.4])
        self.assertEqual(prime, 210)

    def test_subsumption(self):
        v = TriadicValidator()
        # 30 = 2*3*5, 6 = 2*3 → 30 % 6 == 0, so 30 subsumes 6
        self.assertTrue(v.subsumes(30, 6))
        # 6 does not subsume 30
        self.assertFalse(v.subsumes(6, 30))
        # Everything subsumes itself
        self.assertTrue(v.subsumes(30, 30))

    def test_composition(self):
        v = TriadicValidator()
        # compose(6, 10) = lcm(2*3, 2*5) = 30
        self.assertEqual(v.compose(6, 10), 30)
        # compose(6, 6) = lcm(6, 6) = 6
        self.assertEqual(v.compose(6, 6), 6)

    def test_explain_gap(self):
        v = TriadicValidator()
        # 30 = 2*3*5, 70 = 2*5*7
        # shared = gcd(30, 70) = 10 = 2*5
        # only_in_a = 30/10 = 3
        # only_in_b = 70/10 = 7
        gap = v.explain_gap(30, 70)
        self.assertEqual(gap['shared'], 10)
        self.assertEqual(gap['only_in_a'], 3)
        self.assertEqual(gap['only_in_b'], 7)
        self.assertEqual(gap['shared_factors'], [2, 5])
        self.assertEqual(gap['only_in_a_factors'], [3])
        self.assertEqual(gap['only_in_b_factors'], [7])

    def test_similarity(self):
        v = TriadicValidator()
        # Same prime → similarity = 1.0
        self.assertAlmostEqual(v.similarity(30, 30), 1.0)
        # 30 = {2,3,5}, 70 = {2,5,7} → shared={2,5}, total={2,3,5,7} → 2/4 = 0.5
        self.assertAlmostEqual(v.similarity(30, 70), 0.5)
        # Coprime → similarity = 0.0
        self.assertAlmostEqual(v.similarity(3, 7), 0.0)

    def test_triadic_loss_agreement(self):
        # Two identical projection sets should have low loss when should_share=True
        proj_a = [Value(0.5), Value(-0.3), Value(0.8)]
        proj_b = [Value(0.5), Value(-0.3), Value(0.8)]
        loss = triadic_loss(proj_a, proj_b, should_share=True)
        self.assertGreater(loss.data, 0)

        # Opposite projections should have higher loss when should_share=True
        proj_c = [Value(-0.5), Value(0.3), Value(-0.8)]
        loss2 = triadic_loss(proj_a, proj_c, should_share=True)
        self.assertGreater(loss2.data, loss.data)

    def test_mapper_explain(self):
        mapper = PrimeMapper(4)
        info = mapper.explain(70)  # 70 = 2 * 5 * 7
        self.assertEqual(info['composite'], 70)
        self.assertIn(2, info['factors'])
        self.assertIn(5, info['factors'])
        self.assertIn(7, info['factors'])
        self.assertEqual(info['n_active'], 3)


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration(unittest.TestCase):
    """End-to-end integration tests."""

    def test_training_smoke(self):
        """Run a few training steps and verify loss decreases."""
        import random
        random.seed(42)

        # Tiny model for speed
        config = GPTConfig(
            vocab_size=10,
            block_size=8,
            n_layer=1,
            n_embd=8,
            n_head=2,
            n_triadic_bits=4,
        )
        model = GPT(config)
        mapper = PrimeMapper(config.n_triadic_bits)

        # Tiny corpus
        docs = ["abcdef", "ghijkl"]
        uchars = sorted(set(''.join(docs)))
        char_to_id = {ch: i for i, ch in enumerate(uchars)}
        BOS = len(uchars)

        # Override vocab_size to match
        config.vocab_size = len(uchars) + 1
        model = GPT(config)
        params = model.params()

        # Adam buffers
        lr = 0.01
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        m_buf = [0.0] * len(params)
        v_buf = [0.0] * len(params)

        first_loss = None
        last_loss = None

        for step in range(20):
            doc = docs[step % len(docs)]
            tokens = [BOS] + [char_to_id[ch] for ch in doc] + [BOS]
            n = min(config.block_size, len(tokens) - 1)

            keys = [[] for _ in range(config.n_layer)]
            values = [[] for _ in range(config.n_layer)]
            losses = []

            for pos_id in range(n):
                token_id = tokens[pos_id]
                target_id = tokens[pos_id + 1]
                logits, _ = model.forward(token_id, pos_id, keys, values)
                probs = softmax(logits)
                loss_t = -probs[target_id].log()
                losses.append(loss_t)

            loss = (1 / n) * sum(losses)

            if first_loss is None:
                first_loss = loss.data
            last_loss = loss.data

            loss.backward()

            for i, p in enumerate(params):
                m_buf[i] = beta1 * m_buf[i] + (1 - beta1) * p.grad
                v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * p.grad ** 2
                m_hat = m_buf[i] / (1 - beta1 ** (step + 1))
                v_hat = v_buf[i] / (1 - beta2 ** (step + 1))
                p.data -= lr * m_hat / (v_hat ** 0.5 + eps)
                p.grad = 0

        # Loss should decrease
        self.assertLess(last_loss, first_loss)

    def test_triadic_projection_consistency(self):
        """Same input should produce the same triadic projection."""
        config = GPTConfig(
            vocab_size=10,
            block_size=8,
            n_layer=1,
            n_embd=8,
            n_head=2,
            n_triadic_bits=4,
        )
        model = GPT(config)
        mapper = PrimeMapper(config.n_triadic_bits)

        # First pass
        keys1 = [[] for _ in range(config.n_layer)]
        values1 = [[] for _ in range(config.n_layer)]
        _, hidden1 = model.forward(0, 0, keys1, values1)
        proj1 = model.project_to_triadic(hidden1)
        prime1 = mapper.map(proj1)

        # Second pass with same input
        keys2 = [[] for _ in range(config.n_layer)]
        values2 = [[] for _ in range(config.n_layer)]
        _, hidden2 = model.forward(0, 0, keys2, values2)
        proj2 = model.project_to_triadic(hidden2)
        prime2 = mapper.map(proj2)

        self.assertEqual(prime1, prime2)


# ============================================================
# Entry point
# ============================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
