"""
triadic-head — Drop-in triadic projection head for any HuggingFace transformer.

Adds interpretable prime-factor semantic signatures to language models
at zero language cost. A single linear layer (n_embd -> n_bits) produces
discrete prime composites that enable exact algebraic operations:
subsumption, composition, analogy, and gap analysis.

Quick start:
    from triadic_head import TriadicWrapper

    model = TriadicWrapper("gpt2", n_bits=64)
    sigs = model.encode(["king", "queen", "dog"])
    result = model.compare("king", "queen")
"""

# Algebra is pure Python — always available
from .algebra import PrimeMapper, TriadicValidator, prime_factors, nth_prime

__version__ = "0.1.0"


def __getattr__(name):
    """Lazy import for torch-dependent classes."""
    if name in ('TriadicWrapper', 'TriadicHead'):
        from .wrapper import TriadicWrapper, TriadicHead
        return {'TriadicWrapper': TriadicWrapper, 'TriadicHead': TriadicHead}[name]
    raise AttributeError(f"module 'triadic_head' has no attribute {name!r}")


__all__ = [
    "TriadicWrapper",
    "TriadicHead",
    "PrimeMapper",
    "TriadicValidator",
    "prime_factors",
    "nth_prime",
]
