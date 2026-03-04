"""
Training Loop — Dual-objective training for the Triadic MicroGPT.

Two losses are combined:
  1. Language Loss — standard cross-entropy on next-token prediction
  2. Triadic Loss — encourages related concepts to share prime factors

The triadic loss activates after a warmup period, allowing the model
to first learn basic language patterns before adding the algebraic constraint.
"""

import os
import sys
import math
import random
import time

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.autograd import Value
from src.transformer import GPT, GPTConfig, softmax
from src.triadic import PrimeMapper, TriadicValidator, triadic_loss


# ============================================================
# Data Loading & Tokenization
# ============================================================

def load_corpus(filepath):
    """Load the concept corpus. Each line is a group of related concepts."""
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def build_tokenizer(docs):
    """
    Build a character-level tokenizer from the corpus.

    Returns:
        uchars: sorted list of unique characters
        char_to_id: dict mapping char → int
        vocab_size: total number of tokens (chars + BOS)
        BOS: the BOS token id
    """
    uchars = sorted(set(''.join(docs)))
    char_to_id = {ch: i for i, ch in enumerate(uchars)}
    BOS = len(uchars)
    vocab_size = len(uchars) + 1
    return uchars, char_to_id, vocab_size, BOS


def tokenize(doc, char_to_id, BOS):
    """Convert a string to a list of token ids, wrapped with BOS."""
    return [BOS] + [char_to_id[ch] for ch in doc] + [BOS]


# ============================================================
# Training
# ============================================================

def train(
    data_path=None,
    num_steps=200,
    learning_rate=0.01,
    triadic_warmup=50,
    triadic_alpha=0.1,
    checkpoint_path=None,
    n_layer=2,
    n_embd=32,
    n_head=4,
    n_triadic_bits=8,
    block_size=32,
    print_every=10,
):
    """
    Train the Triadic MicroGPT.

    Args:
        data_path: path to concepts.txt
        num_steps: total training steps
        learning_rate: initial learning rate
        triadic_warmup: steps before activating triadic loss
        triadic_alpha: weight of triadic loss relative to language loss
        checkpoint_path: path to save the model checkpoint
        n_layer, n_embd, n_head, n_triadic_bits, block_size: model config
        print_every: how often to print training status
    """
    # --- Resolve data path ---
    if data_path is None:
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'concepts.txt'
        )

    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found: {data_path}")
        sys.exit(1)

    # --- Load data ---
    print("=" * 60)
    print("  TRIADIC MICROGPT — Training")
    print("=" * 60)
    print()

    docs = load_corpus(data_path)
    random.seed(42)
    random.shuffle(docs)
    print(f"  Corpus: {len(docs)} concept groups")

    uchars, char_to_id, vocab_size, BOS = build_tokenizer(docs)
    print(f"  Vocab:  {vocab_size} tokens ({len(uchars)} chars + BOS)")

    # --- Initialize model ---
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_embd=n_embd,
        n_head=n_head,
        n_triadic_bits=n_triadic_bits,
    )
    model = GPT(config)
    params = model.params()
    print(f"  Model:  {len(params)} parameters")
    print(f"  Config: {n_layer} layers, {n_embd} dim, {n_head} heads, {n_triadic_bits} triadic bits")
    print()

    # --- Triadic components ---
    mapper = PrimeMapper(n_triadic_bits)
    validator = TriadicValidator()

    # --- Adam optimizer buffers ---
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    m_buf = [0.0] * len(params)
    v_buf = [0.0] * len(params)

    # --- Training loop ---
    print(f"  Training for {num_steps} steps (triadic warmup: {triadic_warmup} steps)")
    print("-" * 60)

    start_time = time.time()
    concept_primes = {}  # Track prime signatures for concepts

    for step in range(num_steps):
        doc = docs[step % len(docs)]
        tokens = tokenize(doc, char_to_id, BOS)
        n = min(block_size, len(tokens) - 1)

        # --- Forward pass ---
        n_layers = config.n_layer
        keys = [[] for _ in range(n_layers)]
        values = [[] for _ in range(n_layers)]
        losses = []
        hidden_states = []

        for pos_id in range(n):
            token_id = tokens[pos_id]
            target_id = tokens[pos_id + 1]

            logits, hidden = model.forward(token_id, pos_id, keys, values)
            probs = softmax(logits)
            loss_t = -probs[target_id].log()
            losses.append(loss_t)
            hidden_states.append(hidden)

        # --- Language loss ---
        lang_loss = (1 / n) * sum(losses)

        # --- Triadic loss (after warmup) ---
        total_loss = lang_loss
        tri_loss_val = 0.0

        if step >= triadic_warmup and len(hidden_states) >= 2:
            # Project hidden states to triadic space
            projections = [model.project_to_triadic(h) for h in hidden_states]

            # Encourage adjacent tokens in the same doc to share prime factors
            tri_losses = []
            for i in range(len(projections) - 1):
                tri_l = triadic_loss(projections[i], projections[i + 1], should_share=True)
                tri_losses.append(tri_l)

            if tri_losses:
                avg_tri_loss = (1 / len(tri_losses)) * sum(tri_losses)
                total_loss = lang_loss + triadic_alpha * avg_tri_loss
                tri_loss_val = avg_tri_loss.data

            # Track prime signatures for some words
            if step % print_every == 0:
                words = doc.split()
                for i, word in enumerate(words[:min(3, len(hidden_states))]):
                    if i < len(projections):
                        prime = mapper.map(projections[i])
                        concept_primes[word] = prime

        # --- Backward pass ---
        total_loss.backward()

        # --- Adam optimizer ---
        lr_t = learning_rate * (1 - step / num_steps)
        for i, p in enumerate(params):
            m_buf[i] = beta1 * m_buf[i] + (1 - beta1) * p.grad
            v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * p.grad ** 2
            m_hat = m_buf[i] / (1 - beta1 ** (step + 1))
            v_hat = v_buf[i] / (1 - beta2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
            p.grad = 0

        # --- Logging ---
        if step % print_every == 0 or step == num_steps - 1:
            elapsed = time.time() - start_time
            status = f"  step {step+1:4d}/{num_steps:4d}"
            status += f" | lang_loss {lang_loss.data:.4f}"
            if step >= triadic_warmup:
                status += f" | tri_loss {tri_loss_val:.4f}"
            status += f" | lr {lr_t:.5f}"
            status += f" | {elapsed:.1f}s"
            print(status)

    # --- Final report ---
    elapsed = time.time() - start_time
    print()
    print("-" * 60)
    print(f"  Training complete in {elapsed:.1f}s")
    print()

    # Show some concept primes
    if concept_primes:
        print("  Learned Prime Signatures:")
        print("  " + "-" * 40)
        for concept, prime in sorted(concept_primes.items()):
            factors = mapper.explain(prime)
            print(f"    {concept:15s} → Φ = {prime:10d}  ({factors['n_active']}/{factors['n_total']} bits active)")

        # Show some triadic relationships
        concepts_list = list(concept_primes.items())
        if len(concepts_list) >= 2:
            print()
            print("  Triadic Relationships:")
            print("  " + "-" * 40)
            for i in range(min(5, len(concepts_list))):
                for j in range(i + 1, min(i + 3, len(concepts_list))):
                    a_name, a_prime = concepts_list[i]
                    b_name, b_prime = concepts_list[j]
                    sim = validator.similarity(a_prime, b_prime)
                    gap = validator.explain_gap(a_prime, b_prime)
                    sub_ab = validator.subsumes(a_prime, b_prime)
                    print(f"    {a_name} vs {b_name}: "
                          f"sim={sim:.2f}, "
                          f"shared={gap['shared_factors']}, "
                          f"subsumes={sub_ab}")

    # --- Save checkpoint ---
    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'model.ckpt'
        )
    model.save_checkpoint(checkpoint_path)
    print(f"\n  Checkpoint saved: {checkpoint_path}")

    # Save tokenizer info
    tokenizer_path = checkpoint_path + '.vocab'
    with open(tokenizer_path, 'w') as f:
        f.write(''.join(uchars))
    print(f"  Vocabulary saved: {tokenizer_path}")

    print()
    print("=" * 60)

    return model, mapper, validator, config, uchars, char_to_id, vocab_size, BOS


# ============================================================
# Entry point
# ============================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train Triadic MicroGPT')
    parser.add_argument('--steps', type=int, default=200, help='Number of training steps')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--warmup', type=int, default=50, help='Triadic warmup steps')
    parser.add_argument('--alpha', type=float, default=0.1, help='Triadic loss weight')
    parser.add_argument('--layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--dim', type=int, default=32, help='Embedding dimension')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--bits', type=int, default=8, help='Number of triadic bits')
    parser.add_argument('--data', type=str, default=None, help='Path to concepts.txt')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint save path')
    args = parser.parse_args()

    train(
        data_path=args.data,
        num_steps=args.steps,
        learning_rate=args.lr,
        triadic_warmup=args.warmup,
        triadic_alpha=args.alpha,
        n_layer=args.layers,
        n_embd=args.dim,
        n_head=args.heads,
        n_triadic_bits=args.bits,
        checkpoint_path=args.checkpoint,
    )
