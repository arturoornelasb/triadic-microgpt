"""
Fast Transformer — NumPy-based GPT model for scalable training.

Uses tensor_ops for bulk array operations (100× faster than scalar autograd).
Keeps the triadic projection head using the original Value autograd for
its small computational graph.
"""

import numpy as np
from dataclasses import dataclass
from .tensor_ops import (
    Parameter, init_matrix,
    linear_forward, linear_backward,
    rmsnorm_forward, rmsnorm_backward,
    relu_forward, relu_backward,
    softmax_forward, cross_entropy_loss,
    attention_forward, attention_backward,
    tanh_forward, tanh_backward,
)
from .autograd import Value


# ============================================================
# Configuration
# ============================================================

@dataclass
class FastGPTConfig:
    """Configuration for the fast GPT model."""
    vocab_size: int = 4096
    block_size: int = 128
    n_layer: int = 4
    n_embd: int = 128
    n_head: int = 4
    n_triadic_bits: int = 16
    init_std: float = 0.02
    dropout: float = 0.0  # reserved for future use


# ============================================================
# Fast GPT Model
# ============================================================

class FastGPT:
    """
    NumPy-based GPT language model with triadic projection head.

    Forward pass processes an entire sequence at once (not token-by-token),
    giving massive speedup via array operations.

    The triadic projection head still uses Value autograd since it's small
    and needs to integrate with the triadic loss.
    """

    def __init__(self, config: FastGPTConfig):
        self.config = config
        np.random.seed(42)
        std = config.init_std

        # --- Embeddings ---
        self.wte = Parameter(np.random.randn(config.vocab_size, config.n_embd) * std, name="wte")
        self.wpe = Parameter(np.random.randn(config.block_size, config.n_embd) * std, name="wpe")

        # --- Transformer layers ---
        self.layers = []
        for i in range(config.n_layer):
            layer = {
                'attn_wq': init_matrix(config.n_embd, config.n_embd, std, f"L{i}.attn_wq"),
                'attn_wk': init_matrix(config.n_embd, config.n_embd, std, f"L{i}.attn_wk"),
                'attn_wv': init_matrix(config.n_embd, config.n_embd, std, f"L{i}.attn_wv"),
                'attn_wo': init_matrix(config.n_embd, config.n_embd, std, f"L{i}.attn_wo"),
                'mlp_fc1': init_matrix(4 * config.n_embd, config.n_embd, std, f"L{i}.mlp_fc1"),
                'mlp_fc2': init_matrix(config.n_embd, 4 * config.n_embd, std, f"L{i}.mlp_fc2"),
            }
            self.layers.append(layer)

        # --- LM Head ---
        self.lm_head = init_matrix(config.vocab_size, config.n_embd, std, "lm_head")

        # --- Triadic Projection Head ---
        self.triadic_head = init_matrix(config.n_triadic_bits, config.n_embd, std, "triadic_head")

    def params(self):
        """Return all trainable parameters as a flat list."""
        all_params = [self.wte, self.wpe, self.lm_head, self.triadic_head]
        for layer in self.layers:
            all_params.extend(layer.values())
        return all_params

    def num_params(self):
        """Count total number of scalar parameters."""
        return sum(p.data.size for p in self.params())

    def forward(self, token_ids):
        """
        Forward pass for an entire sequence.

        Args:
            token_ids: list or array of integer token ids, length T

        Returns:
            logits: (T, vocab_size) — pre-softmax predictions
            hidden: (T, n_embd) — final hidden states (for triadic projection)
            caches: list of caches needed for backward pass
        """
        config = self.config
        T = len(token_ids)
        head_dim = config.n_embd // config.n_head
        caches = []

        # --- Embedding lookup ---
        tok_emb = self.wte.data[token_ids]  # (T, n_embd)
        pos_ids = np.arange(T)
        pos_emb = self.wpe.data[pos_ids]    # (T, n_embd)
        x = tok_emb + pos_emb               # (T, n_embd)

        # Initial RMSNorm
        x, rn_cache0 = rmsnorm_forward(x)
        caches.append(('rmsnorm0', rn_cache0, tok_emb, pos_emb, token_ids, pos_ids))

        # --- Transformer layers ---
        for li, layer in enumerate(self.layers):
            # 1) Multi-head Self-Attention
            x_residual = x.copy()
            x, rn_cache = rmsnorm_forward(x)
            caches.append(('rn_attn', li, rn_cache))

            q, q_cache = linear_forward(x, layer['attn_wq'])
            k, k_cache = linear_forward(x, layer['attn_wk'])
            v, v_cache = linear_forward(x, layer['attn_wv'])
            caches.append(('qkv', li, q_cache, k_cache, v_cache))

            attn_out, attn_cache = attention_forward(q, k, v, config.n_head, head_dim)
            caches.append(('attn', li, attn_cache))

            x_proj, wo_cache = linear_forward(attn_out, layer['attn_wo'])
            caches.append(('attn_wo', li, wo_cache))

            x = x_proj + x_residual
            caches.append(('res_attn', li, x_residual))

            # 2) MLP
            x_residual = x.copy()
            x, rn_cache2 = rmsnorm_forward(x)
            caches.append(('rn_mlp', li, rn_cache2))

            x_fc1, fc1_cache = linear_forward(x, layer['mlp_fc1'])
            caches.append(('fc1', li, fc1_cache))

            x_act, relu_cache = relu_forward(x_fc1)
            caches.append(('relu', li, relu_cache))

            x_fc2, fc2_cache = linear_forward(x_act, layer['mlp_fc2'])
            caches.append(('fc2', li, fc2_cache))

            x = x_fc2 + x_residual
            caches.append(('res_mlp', li, x_residual))

        # --- Final hidden state ---
        hidden = x.copy()

        # --- LM Head ---
        logits, lm_cache = linear_forward(x, self.lm_head)
        caches.append(('lm_head', lm_cache))

        return logits, hidden, caches

    def backward(self, grad_logits, grad_hidden, caches):
        """
        Backward pass through the entire model.

        Args:
            grad_logits: (T, vocab_size) — gradient from language loss
            grad_hidden: (T, n_embd) or None — gradient from triadic loss
            caches: list of caches from forward pass
        """
        config = self.config

        # LM Head backward
        _, lm_cache = caches.pop()
        grad_x = linear_backward(grad_logits, lm_cache)

        # Add triadic gradient if available
        if grad_hidden is not None:
            grad_x = grad_x + grad_hidden

        # Transformer layers (reverse order)
        for li in range(config.n_layer - 1, -1, -1):
            layer = self.layers[li]

            # MLP residual backward
            _, _, x_residual = caches.pop()
            grad_residual = grad_x.copy()

            # FC2 backward
            _, _, fc2_cache = caches.pop()
            grad_act = linear_backward(grad_x, fc2_cache)

            # ReLU backward
            _, _, relu_cache = caches.pop()
            grad_fc1 = relu_backward(grad_act, relu_cache)

            # FC1 backward
            _, _, fc1_cache = caches.pop()
            grad_x = linear_backward(grad_fc1, fc1_cache)

            # RMSNorm (MLP) backward
            _, _, rn_cache2 = caches.pop()
            grad_x = rmsnorm_backward(grad_x, rn_cache2)

            # Add residual gradient
            grad_x = grad_x + grad_residual

            # Attention residual backward
            _, _, x_residual = caches.pop()
            grad_residual = grad_x.copy()

            # Attention output projection backward
            _, _, wo_cache = caches.pop()
            grad_attn_out = linear_backward(grad_x, wo_cache)

            # Attention backward
            _, _, attn_cache = caches.pop()
            grad_q, grad_k, grad_v = attention_backward(grad_attn_out, attn_cache)

            # QKV backward
            _, _, q_cache, k_cache, v_cache = caches.pop()
            grad_x_q = linear_backward(grad_q, q_cache)
            grad_x_k = linear_backward(grad_k, k_cache)
            grad_x_v = linear_backward(grad_v, v_cache)
            grad_x = grad_x_q + grad_x_k + grad_x_v

            # RMSNorm (attention) backward
            _, _, rn_cache = caches.pop()
            grad_x = rmsnorm_backward(grad_x, rn_cache)

            # Add residual gradient
            grad_x = grad_x + grad_residual

        # Initial RMSNorm backward
        _, rn_cache0, tok_emb, pos_emb, token_ids, pos_ids = caches.pop()
        grad_x = rmsnorm_backward(grad_x, rn_cache0)

        # Embedding gradients
        for t in range(len(token_ids)):
            self.wte.grad[token_ids[t]] += grad_x[t]
            self.wpe.grad[pos_ids[t]] += grad_x[t]

    def project_to_triadic_np(self, hidden):
        """
        Project hidden state to triadic space using NumPy.

        Args:
            hidden: (T, n_embd) or (n_embd,)

        Returns:
            projections: (T, n_triadic_bits) or (n_triadic_bits,) — tanh outputs
        """
        raw = hidden @ self.triadic_head.data.T
        return np.tanh(raw)

    def project_to_triadic_autograd(self, hidden_np):
        """
        Project a single hidden state vector using Value autograd.
        Used when we need gradients for the triadic loss.

        Args:
            hidden_np: (n_embd,) numpy array

        Returns:
            projections: list of Value objects (tanh outputs)
        """
        n_bits = self.config.n_triadic_bits
        n_embd = self.config.n_embd
        projections = []
        for b in range(n_bits):
            dot = Value(0.0)
            for e in range(n_embd):
                dot = dot + Value(hidden_np[e]) * Value(self.triadic_head.data[b, e])
            projections.append(dot.tanh())
        return projections

    def save_checkpoint(self, filepath):
        """Save all parameters to a numpy .npz file."""
        save_dict = {}
        for p in self.params():
            save_dict[p.name] = p.data
        np.savez(filepath, **save_dict)

    def load_checkpoint(self, filepath):
        """Load parameters from a numpy .npz file."""
        if not filepath.endswith('.npz'):
            filepath = filepath + '.npz'
        data = np.load(filepath)
        for p in self.params():
            if p.name in data:
                p.data = data[p.name].astype(np.float32)
            else:
                print(f"  Warning: parameter '{p.name}' not found in checkpoint")
