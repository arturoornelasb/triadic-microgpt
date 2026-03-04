"""
Transformer Model — A minimal GPT built on the scalar autograd engine.

Architecture follows GPT-2 with minor simplifications:
  - RMSNorm instead of LayerNorm
  - No biases
  - ReLU instead of GeLU

Enhanced with a triadic projection head that maps hidden states
to a discrete prime-factor space for algebraic verification.
"""

import random
from dataclasses import dataclass
from .autograd import Value


# ============================================================
# Model Configuration
# ============================================================

@dataclass
class GPTConfig:
    """Configuration for the GPT model."""
    vocab_size: int = 28          # character-level vocab + BOS
    block_size: int = 32          # maximum context length
    n_layer: int = 2              # number of transformer layers
    n_embd: int = 32              # embedding dimension
    n_head: int = 4               # number of attention heads
    n_triadic_bits: int = 8       # number of triadic projection bits (= number of primes)
    init_std: float = 0.08        # weight initialization std


# ============================================================
# Neural Network Primitives
# ============================================================

def linear(x, w):
    """Matrix-vector multiply: y = W @ x. w is a list of rows."""
    return [sum(wi * xi for wi, xi in zip(row, x)) for row in w]


def softmax(logits):
    """Numerically stable softmax over a list of Value nodes."""
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x):
    """Root Mean Square Normalization."""
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


# ============================================================
# GPT Model
# ============================================================

class GPT:
    """
    A minimal GPT language model with a triadic projection head.

    The forward pass returns both:
      1. Logits for next-token prediction (standard LM)
      2. Hidden state that can be projected to triadic prime space

    The projection head maps the hidden state through a learned linear
    layer + tanh to produce values in [-1, 1], which are then discretized
    into prime factors.
    """

    def __init__(self, config: GPTConfig):
        self.config = config
        self.rng = random.Random(42)

        # --- Weight initialization helper ---
        def matrix(nout, nin, std=None):
            std = std or config.init_std
            return [
                [Value(self.rng.gauss(0, std)) for _ in range(nin)]
                for _ in range(nout)
            ]

        # --- Token and position embeddings ---
        self.wte = matrix(config.vocab_size, config.n_embd)
        self.wpe = matrix(config.block_size, config.n_embd)
        self.lm_head = matrix(config.vocab_size, config.n_embd)

        # --- Transformer layers ---
        self.layers = []
        for i in range(config.n_layer):
            layer = {
                'attn_wq': matrix(config.n_embd, config.n_embd),
                'attn_wk': matrix(config.n_embd, config.n_embd),
                'attn_wv': matrix(config.n_embd, config.n_embd),
                'attn_wo': matrix(config.n_embd, config.n_embd),
                'mlp_fc1': matrix(4 * config.n_embd, config.n_embd),
                'mlp_fc2': matrix(config.n_embd, 4 * config.n_embd),
            }
            self.layers.append(layer)

        # --- Triadic Projection Head ---
        # Maps n_embd → n_triadic_bits
        self.triadic_head = matrix(config.n_triadic_bits, config.n_embd)

    def params(self):
        """Return a flat list of all trainable Value parameters."""
        all_params = []

        # Embeddings
        for mat in [self.wte, self.wpe, self.lm_head]:
            for row in mat:
                all_params.extend(row)

        # Transformer layers
        for layer in self.layers:
            for mat in layer.values():
                for row in mat:
                    all_params.extend(row)

        # Triadic projection head
        for row in self.triadic_head:
            all_params.extend(row)

        return all_params

    def forward(self, token_id, pos_id, keys, values):
        """
        Forward pass for a single token.

        Args:
            token_id: integer token index
            pos_id: integer position index
            keys: list of key caches per layer (mutated in-place)
            values: list of value caches per layer (mutated in-place)

        Returns:
            (logits, hidden_state) — logits for next-token prediction,
            and the hidden state before the LM head.
        """
        config = self.config
        head_dim = config.n_embd // config.n_head

        # Token + position embedding
        tok_emb = self.wte[token_id]
        pos_emb = self.wpe[pos_id]
        x = [t + p for t, p in zip(tok_emb, pos_emb)]
        x = rmsnorm(x)

        # Transformer layers
        for li, layer in enumerate(self.layers):
            # 1) Multi-head Self-Attention
            x_residual = x
            x = rmsnorm(x)

            q = linear(x, layer['attn_wq'])
            k = linear(x, layer['attn_wk'])
            v = linear(x, layer['attn_wv'])

            keys[li].append(k)
            values[li].append(v)

            x_attn = []
            for h in range(config.n_head):
                hs = h * head_dim
                q_h = q[hs:hs + head_dim]
                k_h = [ki[hs:hs + head_dim] for ki in keys[li]]
                v_h = [vi[hs:hs + head_dim] for vi in values[li]]

                # Scaled dot-product attention
                attn_logits = [
                    sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim ** 0.5
                    for t in range(len(k_h))
                ]
                attn_weights = softmax(attn_logits)

                head_out = [
                    sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                    for j in range(head_dim)
                ]
                x_attn.extend(head_out)

            x = linear(x_attn, layer['attn_wo'])
            x = [a + b for a, b in zip(x, x_residual)]

            # 2) Feed-Forward MLP
            x_residual = x
            x = rmsnorm(x)
            x = linear(x, layer['mlp_fc1'])
            x = [xi.relu() for xi in x]
            x = linear(x, layer['mlp_fc2'])
            x = [a + b for a, b in zip(x, x_residual)]

        # hidden_state is the final representation before the LM head
        hidden_state = x

        # LM head: project to vocabulary logits
        logits = linear(x, self.lm_head)

        return logits, hidden_state

    def project_to_triadic(self, hidden_state):
        """
        Map the hidden state through the triadic projection head.

        Returns a list of n_triadic_bits Value nodes, each in [-1, 1] (via tanh).
        These are later discretized into prime factors:
          - projection[i] > 0 → include prime[i]
          - projection[i] <= 0 → exclude prime[i]
        """
        projections = linear(hidden_state, self.triadic_head)
        return [p.tanh() for p in projections]

    def save_checkpoint(self, filepath):
        """Save model weights as a simple text format."""
        params = self.params()
        with open(filepath, 'w') as f:
            f.write(f"{len(params)}\n")
            for p in params:
                f.write(f"{p.data}\n")

    def load_checkpoint(self, filepath):
        """Load model weights from a text file."""
        params = self.params()
        with open(filepath, 'r') as f:
            n = int(f.readline().strip())
            assert n == len(params), f"Checkpoint has {n} params, model has {len(params)}"
            for p in params:
                p.data = float(f.readline().strip())
