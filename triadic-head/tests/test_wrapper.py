"""Tests for TriadicWrapper — requires torch but NOT a GPU."""

import pytest
import torch
from triadic_head import TriadicWrapper, TriadicHead


class TestTriadicHead:
    def test_output_shape(self):
        head = TriadicHead(n_embd=128, n_bits=32)
        x = torch.randn(2, 10, 128)
        out = head(x)
        assert out.shape == (2, 10, 32)

    def test_output_range(self):
        head = TriadicHead(n_embd=128, n_bits=32)
        x = torch.randn(2, 10, 128) * 10  # large input
        out = head(x)
        assert out.min() >= -1.0
        assert out.max() <= 1.0

    def test_params(self):
        head = TriadicHead(n_embd=768, n_bits=64)
        assert sum(p.numel() for p in head.parameters()) == 768 * 64


class TestTriadicWrapper:
    """Integration tests using a tiny GPT-2 config (no downloads needed)."""

    @pytest.fixture
    def tiny_model(self):
        """Create a minimal GPT-2-style model for testing."""
        from transformers import GPT2LMHeadModel, GPT2Config
        config = GPT2Config(
            vocab_size=100,
            n_positions=64,
            n_embd=32,
            n_layer=2,
            n_head=2,
        )
        return GPT2LMHeadModel(config)

    def test_wrap(self, tiny_model):
        wrapper = TriadicWrapper(tiny_model, n_bits=16)
        assert wrapper.n_embd == 32
        assert wrapper.n_bits == 16
        assert wrapper.triadic_params() == 32 * 16

    def test_forward(self, tiny_model):
        wrapper = TriadicWrapper(tiny_model, n_bits=16)
        ids = torch.randint(0, 100, (2, 10))
        logits, proj, loss = wrapper(ids)
        assert logits.shape == (2, 10, 100)
        assert proj.shape == (2, 10, 16)
        assert loss is None

    def test_forward_with_labels(self, tiny_model):
        wrapper = TriadicWrapper(tiny_model, n_bits=16)
        ids = torch.randint(0, 100, (2, 10))
        logits, proj, loss = wrapper(ids, labels=ids)
        assert loss is not None
        assert loss.ndim == 0  # scalar

    def test_triadic_loss(self, tiny_model):
        wrapper = TriadicWrapper(tiny_model, n_bits=16, align_mode='mse')
        ids = torch.randint(0, 100, (2, 10))
        _, proj, _ = wrapper(ids)
        tri_loss = wrapper.triadic_loss(proj, input_ids=ids)
        assert tri_loss.ndim == 0
        assert tri_loss.item() >= 0

    def test_triadic_loss_all_modes(self, tiny_model):
        wrapper = TriadicWrapper(tiny_model, n_bits=16)
        ids = torch.randint(0, 100, (2, 20))
        _, proj, _ = wrapper(ids)
        for mode in ('mse', 'rank', 'infonce'):
            loss = wrapper.triadic_loss(proj, input_ids=ids, align_mode=mode)
            assert loss.ndim == 0, f"Mode {mode} failed"

    def test_freeze_unfreeze(self, tiny_model):
        wrapper = TriadicWrapper(tiny_model, n_bits=16)
        wrapper.freeze_backbone()
        assert wrapper.num_params(trainable_only=True) == 32 * 16

        wrapper.unfreeze_last_n(1)
        trainable = wrapper.num_params(trainable_only=True)
        assert trainable > 32 * 16

    def test_repr(self, tiny_model):
        wrapper = TriadicWrapper(tiny_model, n_bits=16)
        r = repr(wrapper)
        assert 'GPT2LMHeadModel' in r
        assert 'n_bits=16' in r

    def test_validate(self, tiny_model):
        """validate() should return results even on untrained model."""
        from transformers import AutoTokenizer
        wrapper = TriadicWrapper(tiny_model, n_bits=16)
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        report = wrapper.validate(tokenizer=tokenizer, verbose=False)
        assert 'checks' in report
        assert 'overall_pass' in report
        assert 'signatures' in report
        assert 'diversity' in report['checks']
        assert 'active_bits' in report['checks']
        assert 'semantic_ordering' in report['checks']
        # Each check has 'pass' and 'detail'
        for name, check in report['checks'].items():
            assert 'pass' in check
            assert 'detail' in check

    def test_validate_custom_groups(self, tiny_model):
        """validate() accepts custom word groups."""
        from transformers import AutoTokenizer
        wrapper = TriadicWrapper(tiny_model, n_bits=16)
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        report = wrapper.validate(
            tokenizer=tokenizer,
            word_groups={'a': ['cat', 'dog'], 'b': ['red', 'blue']},
            verbose=False,
        )
        assert len(report['signatures']) == 4

    def test_explore(self, tiny_model):
        """explore() should return matrix, ranked pairs, and per-pair details."""
        from transformers import AutoTokenizer
        wrapper = TriadicWrapper(tiny_model, n_bits=16)
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        result = wrapper.explore(['cat', 'dog', 'red'], tokenizer=tokenizer, verbose=False)
        assert result['words'] == ['cat', 'dog', 'red']
        assert len(result['matrix']) == 3
        assert len(result['matrix'][0]) == 3
        assert len(result['pairs_ranked']) == 3  # 3 choose 2 = 3 pairs
        # Diagonal should be 1.0
        for i in range(3):
            assert result['matrix'][i][i] == 1.0
        # Pairs are dicts with full details
        for p in result['pairs_ranked']:
            assert 'similarity' in p
            assert 'word_a' in p
            assert 'word_b' in p
            assert 'shared_factors' in p
            assert 'n_shared' in p
        # Pairs should be sorted descending
        sims = [p['similarity'] for p in result['pairs_ranked']]
        assert sims == sorted(sims, reverse=True)

    def test_explore_threshold(self, tiny_model):
        """explore() with threshold flags high-similarity pairs."""
        from transformers import AutoTokenizer
        wrapper = TriadicWrapper(tiny_model, n_bits=16)
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        result = wrapper.explore(
            ['cat', 'dog', 'red'], tokenizer=tokenizer,
            threshold=0.0, verbose=False,
        )
        # threshold=0.0 should flag all pairs
        assert len(result['flagged']) == 3

    def test_config(self, tiny_model):
        """config() returns current settings and allows changes."""
        wrapper = TriadicWrapper(tiny_model, n_bits=16, align_mode='mse')
        cfg = wrapper.config()
        assert cfg['align_mode'] == 'mse'
        assert cfg['n_bits'] == 16
        assert cfg['n_embd'] == 32

        # Change align_mode
        wrapper.config(align_mode='rank')
        assert wrapper.align_mode == 'rank'
        assert wrapper.config()['align_mode'] == 'rank'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
