"""
Shared utilities for audit tests.

Correct model loading, projection extraction, and prime mapping
following the exact patterns from danza_63bit.py and r3_chain_test.py.

CRITICAL:
  - Run 15 (40M):  max_tokens=4, custom BPETokenizer, DanzaTriadicGPT
  - D-A13 (355M):  max_tokens=8, GPT2Tokenizer, GPT2MediumTernary
"""

import os
import sys
import json
import math
import numpy as np
import torch
from datetime import datetime

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PLAYGROUND = os.path.dirname(_THIS_DIR)
_PROJECT = os.path.dirname(_PLAYGROUND)
sys.path.insert(0, _PROJECT)
sys.path.insert(0, _PLAYGROUND)

from src.torch_transformer import TriadicGPTConfig
from src.triadic import PrimeMapper, TriadicValidator, prime_factors

try:
    from src.fast_tokenizer import FastBPETokenizer as BPETokenizer
except ImportError:
    from src.tokenizer import BPETokenizer

from danza_63bit import (
    load_primitives, load_anchors, load_anchors_v2, load_all_anchors,
    N_BITS, DanzaTriadicGPT,
)

# Default paths
RUN15_CKPT_DIR = os.path.join(_PROJECT, 'checkpoints', 'danza_bootstrap_xl')
RUN15_CKPT = os.path.join(RUN15_CKPT_DIR, 'model_best.pt')
RUN15_TOK = os.path.join(RUN15_CKPT_DIR, 'tokenizer.json')

DA13_CKPT_DIR = os.path.join(_PROJECT, 'checkpoints', 'danza_gpt2medium_ternary')
DA13_CKPT = os.path.join(DA13_CKPT_DIR, 'model_best.pt')

RESULTS_DIR = os.path.join(_THIS_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Run 15 (40M) model loading
# ============================================================

def load_run15(device='cpu'):
    """Load Run 15 (40M, DanzaTriadicGPT) with custom BPE tokenizer."""
    ckpt = torch.load(RUN15_CKPT, map_location=device, weights_only=True)
    cfg = ckpt['config']

    config = TriadicGPTConfig(
        vocab_size=cfg['vocab_size'], block_size=cfg['block_size'],
        n_layer=cfg['n_layer'], n_embd=cfg['n_embd'],
        n_head=cfg['n_head'], n_triadic_bits=cfg['n_triadic_bits'],
    )
    model = DanzaTriadicGPT(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    tokenizer = BPETokenizer.load(RUN15_TOK)
    return model, tokenizer


def load_da13(device='cpu'):
    """Load D-A13 (355M, GPT2MediumTernary) with GPT2Tokenizer."""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer as HFTokenizer
    from gpt2_medium_ternary import GPT2MediumTernary

    gpt2 = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    model = GPT2MediumTernary(gpt2, n_triadic_bits=N_BITS, quantize_mode='fsq')

    ckpt = torch.load(DA13_CKPT, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    tokenizer = HFTokenizer.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# ============================================================
# Projection extraction
# ============================================================

@torch.no_grad()
def get_projection(model, tokenizer, word, device='cpu', max_tokens=4):
    """Extract mean-pooled triadic projection for a word.

    Args:
        max_tokens: 4 for Run 15 (custom BPE), 8 for D-A13 (GPT-2 tokenizer)
    Returns:
        numpy array of shape (n_bits,) or None if word not in vocab
    """
    if hasattr(tokenizer, 'encode'):
        # Check if it's HuggingFace tokenizer (has add_special_tokens kwarg)
        try:
            ids = tokenizer.encode(word, add_special_tokens=False)[:max_tokens]
        except TypeError:
            ids = tokenizer.encode(word, add_special=False)[:max_tokens]
    else:
        ids = tokenizer.encode(word, add_special=False)[:max_tokens]

    if not ids:
        return None

    x = torch.tensor([ids], dtype=torch.long, device=device)
    _, proj, _ = model(x)
    return proj[0].mean(dim=0).cpu().numpy()


def get_projections_batch(model, tokenizer, words, device='cpu', max_tokens=4):
    """Extract projections for a list of words."""
    results = {}
    for word in words:
        p = get_projection(model, tokenizer, word, device, max_tokens)
        if p is not None:
            results[word] = p
    return results


# ============================================================
# Metrics
# ============================================================

def to_binary(proj):
    """Continuous projection -> binary {0, 1}."""
    return (proj > 0).astype(np.int8)


def to_ternary(proj):
    """Continuous projection -> ternary {-1, 0, +1}."""
    return np.clip(np.round(proj), -1, 1).astype(np.int8)


def hamming(a, b):
    """Hamming distance between two bit vectors."""
    return int(np.sum(to_binary(a) != to_binary(b)))


def cosine_sim(a, b):
    """Cosine similarity in continuous space."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def bits_shared(a, b):
    """Fraction of active bits shared between two projections."""
    ba, bb = to_binary(a), to_binary(b)
    active_a = set(np.where(ba == 1)[0])
    active_b = set(np.where(bb == 1)[0])
    if not active_a and not active_b:
        return 1.0
    union = active_a | active_b
    if not union:
        return 0.0
    return len(active_a & active_b) / len(union)


def proj_to_prime(proj, mapper):
    """Convert projection to composite prime integer."""
    return mapper.map(proj.tolist())


# ============================================================
# Gold target access (for dual-level evaluation)
# ============================================================

_anchors_cache = None

def get_gold_target(word, include_v2=True):
    """Get the gold target bits for an anchor word (or None if not an anchor).

    Args:
        include_v2: If True, also search anclas_v2.json (default: True).

    Returns numpy int8 array of shape (N_BITS,) with values {0, 1}.
    """
    global _anchors_cache
    if _anchors_cache is None:
        prim_data = load_primitives()
        if include_v2:
            _anchors_cache, _ = load_all_anchors(prim_data)
        else:
            _anchors_cache, _ = load_anchors(prim_data)
    if word not in _anchors_cache:
        return None
    return (_anchors_cache[word]['target'] > 0).float().numpy().astype(np.int8)


def get_best_bits(word, model, tokenizer, device='cpu', max_tokens=4):
    """Get the best available bits for a word: gold target if anchor, else model prediction.

    Returns (bits, source) where source is 'gold' or 'model'.
    """
    gold = get_gold_target(word)
    if gold is not None:
        return gold, 'gold'
    proj = get_projection(model, tokenizer, word, device, max_tokens)
    if proj is not None:
        return to_binary(proj), 'model'
    return None, None


# ============================================================
# Output helpers
# ============================================================

def save_results(data, filename):
    """Save results as JSON to the results directory."""
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Results saved to: {path}")
    return path


def print_header(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}")


def print_section(title):
    print(f"\n{'-' * 70}")
    print(f"  {title}")
    print(f"{'-' * 70}")
