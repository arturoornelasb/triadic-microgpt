"""
Inference Engine — Generation and triadic explanation for Triadic MicroGPT.

Provides:
  - Text generation with temperature sampling
  - Triadic analysis: prime signatures, subsumption, gap analysis
  - Interactive CLI for experimentation
"""

import os
import sys
import random

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.autograd import Value
from src.transformer import GPT, GPTConfig, softmax
from src.triadic import PrimeMapper, TriadicValidator, prime_factors


# ============================================================
# Generation
# ============================================================

def generate(model, uchars, BOS, mapper, max_tokens=32, temperature=0.5, return_primes=False):
    """
    Generate text from the model with optional triadic analysis.

    Args:
        model: trained GPT model
        uchars: list of characters in vocabulary
        BOS: BOS token id
        mapper: PrimeMapper instance
        max_tokens: maximum tokens to generate
        temperature: sampling temperature (lower = more deterministic)
        return_primes: if True, also return prime signatures per token

    Returns:
        generated text (str), and optionally a list of (char, prime) tuples
    """
    config = model.config
    n_layers = config.n_layer

    keys = [[] for _ in range(n_layers)]
    values = [[] for _ in range(n_layers)]
    token_id = BOS
    sample = []
    primes_log = []

    for pos_id in range(min(max_tokens, config.block_size)):
        logits, hidden = model.forward(token_id, pos_id, keys, values)

        # Temperature scaling
        if temperature != 1.0:
            scaled_logits = [Value(l.data / temperature) for l in logits]
        else:
            scaled_logits = logits

        probs = softmax(scaled_logits)
        weights = [p.data for p in probs]

        # Sample next token
        token_id = random.choices(range(config.vocab_size), weights=weights)[0]

        if token_id == BOS:
            break

        char = uchars[token_id]
        sample.append(char)

        # Compute triadic projection
        if return_primes:
            projections = model.project_to_triadic(hidden)
            prime = mapper.map(projections)
            primes_log.append((char, prime))

    text = ''.join(sample)

    if return_primes:
        return text, primes_log
    return text


def analyze_text(model, text, uchars, char_to_id, BOS, mapper):
    """
    Run text through the model and return triadic projections for each position.

    Returns:
        list of (char, prime_integer, projection_bits) tuples
    """
    config = model.config
    n_layers = config.n_layer
    tokens = [BOS] + [char_to_id[ch] for ch in text if ch in char_to_id]

    keys = [[] for _ in range(n_layers)]
    values = [[] for _ in range(n_layers)]
    results = []

    for pos_id in range(min(len(tokens) - 1, config.block_size)):
        token_id = tokens[pos_id]
        logits, hidden = model.forward(token_id, pos_id, keys, values)
        projections = model.project_to_triadic(hidden)
        prime = mapper.map(projections)
        bits = mapper.get_bits(projections)

        if pos_id < len(text):
            char = text[pos_id] if pos_id < len(text) else '?'
        else:
            char = '?'

        results.append({
            'position': pos_id,
            'char': char,
            'prime': prime,
            'bits': bits,
            'factors': prime_factors(prime),
        })

    return results


def compare_texts(model, text_a, text_b, uchars, char_to_id, BOS, mapper, validator):
    """
    Compare two text strings using triadic algebra.

    Returns a composite prime for each text (average of token primes via LCM),
    and explains their relationship.
    """
    results_a = analyze_text(model, text_a, uchars, char_to_id, BOS, mapper)
    results_b = analyze_text(model, text_b, uchars, char_to_id, BOS, mapper)

    # Compose all token primes within each text
    if results_a:
        composite_a = results_a[0]['prime']
        for r in results_a[1:]:
            composite_a = validator.compose(composite_a, r['prime'])
    else:
        composite_a = 2

    if results_b:
        composite_b = results_b[0]['prime']
        for r in results_b[1:]:
            composite_b = validator.compose(composite_b, r['prime'])
    else:
        composite_b = 2

    gap = validator.explain_gap(composite_a, composite_b)
    similarity = validator.similarity(composite_a, composite_b)

    return {
        'text_a': text_a,
        'text_b': text_b,
        'prime_a': composite_a,
        'prime_b': composite_b,
        'similarity': similarity,
        'gap': gap,
        'subsumes_a_b': validator.subsumes(composite_a, composite_b),
        'subsumes_b_a': validator.subsumes(composite_b, composite_a),
    }


# ============================================================
# Interactive CLI
# ============================================================

HELP_TEXT = """
╔══════════════════════════════════════════════════════════════╗
║  TRIADIC MICROGPT — Interactive CLI                         ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Commands:                                                   ║
║    /generate [prompt]    Generate text (optionally from a    ║
║                          seed character)                     ║
║    /compare <a> | <b>    Compare two texts algebraically     ║
║    /factors <text>       Show prime factorization per token   ║
║    /sample [n]           Generate n random samples            ║
║    /help                 Show this message                    ║
║    /quit                 Exit                                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""


def interactive_cli(model, mapper, validator, config, uchars, char_to_id, vocab_size, BOS):
    """Launch an interactive REPL for experimenting with the model."""
    print(HELP_TEXT)

    while True:
        try:
            user_input = input("triadic> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.startswith('/quit') or user_input.startswith('/exit'):
            print("Goodbye!")
            break

        elif user_input.startswith('/help'):
            print(HELP_TEXT)

        elif user_input.startswith('/generate'):
            text = generate(model, uchars, BOS, mapper, temperature=0.5)
            print(f"  Generated: {text}")

            # Also show triadic info
            text_gen, primes_log = generate(model, uchars, BOS, mapper,
                                            temperature=0.5, return_primes=True)
            if primes_log:
                print(f"  Text:      {text_gen}")
                print(f"  Prime signatures:")
                for ch, prime in primes_log[:10]:
                    factors = prime_factors(prime)
                    print(f"    '{ch}' → Φ={prime} = {' × '.join(map(str, factors)) if factors else '1'}")

        elif user_input.startswith('/compare'):
            parts = user_input[len('/compare'):].strip()
            if '|' not in parts:
                print("  Usage: /compare text A | text B")
                continue
            text_a, text_b = parts.split('|', 1)
            text_a, text_b = text_a.strip(), text_b.strip()

            if not text_a or not text_b:
                print("  Both texts must be non-empty.")
                continue

            result = compare_texts(model, text_a, text_b, uchars, char_to_id, BOS, mapper, validator)

            print()
            print(f"  ╔═══ Triadic Comparison ═══╗")
            print(f"  ║ A: \"{text_a}\"")
            print(f"  ║    Φ(A) = {result['prime_a']}")
            print(f"  ║    Factors: {prime_factors(result['prime_a'])}")
            print(f"  ║")
            print(f"  ║ B: \"{text_b}\"")
            print(f"  ║    Φ(B) = {result['prime_b']}")
            print(f"  ║    Factors: {prime_factors(result['prime_b'])}")
            print(f"  ║")
            print(f"  ║ Similarity:  {result['similarity']:.2%}")
            print(f"  ║ Shared:      {result['gap']['shared_factors']}")
            print(f"  ║ Only in A:   {result['gap']['only_in_a_factors']}")
            print(f"  ║ Only in B:   {result['gap']['only_in_b_factors']}")
            print(f"  ║ A ⊇ B:       {result['subsumes_a_b']}")
            print(f"  ║ B ⊇ A:       {result['subsumes_b_a']}")
            print(f"  ╚{'═' * 28}╝")
            print()

        elif user_input.startswith('/factors'):
            text = user_input[len('/factors'):].strip()
            if not text:
                print("  Usage: /factors <text>")
                continue

            results = analyze_text(model, text, uchars, char_to_id, BOS, mapper)
            print()
            print(f"  Prime Factorization of \"{text}\":")
            print(f"  {'Pos':>3}  {'Char':>5}  {'Prime':>12}  {'Bits':>12}  Factors")
            print(f"  {'---':>3}  {'----':>5}  {'-----':>12}  {'----':>12}  -------")
            for r in results:
                bits_str = ''.join(map(str, r['bits']))
                factors_str = ' × '.join(map(str, r['factors'])) if r['factors'] else '1'
                print(f"  {r['position']:3d}  {repr(r['char']):>5}  {r['prime']:12d}  {bits_str:>12}  {factors_str}")
            print()

        elif user_input.startswith('/sample'):
            parts = user_input.split()
            n = int(parts[1]) if len(parts) > 1 else 5
            print(f"\n  Generating {n} samples:")
            for i in range(n):
                text = generate(model, uchars, BOS, mapper, temperature=0.6)
                print(f"    {i+1:2d}. {text}")
            print()

        else:
            print(f"  Unknown command. Type /help for available commands.")


# ============================================================
# Entry point
# ============================================================

def load_and_run(checkpoint_path=None):
    """Load a trained model and launch the interactive CLI."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if checkpoint_path is None:
        checkpoint_path = os.path.join(project_root, 'model.ckpt')

    vocab_path = checkpoint_path + '.vocab'

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        print("Train the model first: python src/train.py")
        sys.exit(1)

    if not os.path.exists(vocab_path):
        print(f"ERROR: Vocabulary not found: {vocab_path}")
        sys.exit(1)

    # Load vocabulary
    with open(vocab_path, 'r') as f:
        uchars = list(f.read())
    char_to_id = {ch: i for i, ch in enumerate(uchars)}
    BOS = len(uchars)
    vocab_size = len(uchars) + 1

    print(f"  Loaded vocabulary: {vocab_size} tokens")

    # Build model with same config
    config = GPTConfig(vocab_size=vocab_size)
    model = GPT(config)

    # Load checkpoint
    model.load_checkpoint(checkpoint_path)
    print(f"  Loaded checkpoint: {checkpoint_path}")

    # Triadic components
    mapper = PrimeMapper(config.n_triadic_bits)
    validator = TriadicValidator()

    # Launch CLI
    interactive_cli(model, mapper, validator, config, uchars, char_to_id, vocab_size, BOS)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Triadic MicroGPT Inference')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    args = parser.parse_args()

    load_and_run(checkpoint_path=args.checkpoint)
