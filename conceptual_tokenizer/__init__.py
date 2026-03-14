"""
Conceptual Tokenizer based on Sistema 7×7

Tokenizes text by MEANING (49 conceptual primitives) instead of
by statistical frequency (BPE). Each token is a combination of
primitives with prime number assignments, enabling algebraic
verification via the Triadic Engine.

49 primitives · 3 states · 6 operations · 7 rules
"""

__version__ = "0.1.0"
