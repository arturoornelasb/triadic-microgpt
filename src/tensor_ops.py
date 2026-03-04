"""
Tensor Operations — NumPy-based forward and backward passes.

Replaces the scalar Value autograd for the main transformer weights.
Each operation computes both forward pass and caches values needed
for the backward pass. Gradients are computed manually using the
chain rule on array operations.

This gives us ~100× speedup over scalar autograd while keeping
the entire implementation transparent and educational.
"""

import numpy as np


# ============================================================
# Parameter Container
# ============================================================

class Parameter:
    """
    A named tensor with data and gradient, analogous to torch.nn.Parameter.
    """
    __slots__ = ('data', 'grad', 'name')

    def __init__(self, data, name=""):
        self.data = data.astype(np.float32) if isinstance(data, np.ndarray) else np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data, dtype=np.float32)
        self.name = name

    def zero_grad(self):
        self.grad[:] = 0.0

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Parameter({self.name}, shape={self.shape})"


def init_matrix(nout, nin, std=0.02, name=""):
    """Initialize a weight matrix with Gaussian noise."""
    return Parameter(np.random.randn(nout, nin).astype(np.float32) * std, name=name)


# ============================================================
# Forward + Backward Primitives
# ============================================================

def linear_forward(x, W):
    """
    Linear layer: y = W @ x (for 1D x) or y = x @ W^T (for batch).

    Args:
        x: input array, shape (d_in,) or (batch, d_in)
        W: Parameter, shape (d_out, d_in)

    Returns:
        y: output, shape (d_out,) or (batch, d_out)
        cache: (x, W) for backward
    """
    y = x @ W.data.T
    return y, (x, W)


def linear_backward(grad_y, cache):
    """
    Backward pass for linear layer.

    Returns:
        grad_x: gradient w.r.t. input
    """
    x, W = cache
    if x.ndim == 1:
        grad_x = grad_y @ W.data
        W.grad += np.outer(grad_y, x)
    else:
        grad_x = grad_y @ W.data
        W.grad += grad_y.T @ x
    return grad_x


def rmsnorm_forward(x, eps=1e-5):
    """
    Root Mean Square Normalization.

    Args:
        x: input, shape (d,) or (batch, d)

    Returns:
        y: normalized output, same shape
        cache: values needed for backward
    """
    ms = np.mean(x * x, axis=-1, keepdims=True)
    scale = 1.0 / np.sqrt(ms + eps)
    y = x * scale
    return y, (x, scale, ms, eps)


def rmsnorm_backward(grad_y, cache):
    """Backward pass for RMSNorm."""
    x, scale, ms, eps = cache
    d = x.shape[-1]
    # Simplified backward: d(x * scale)/dx
    grad_x = grad_y * scale - x * np.sum(grad_y * x, axis=-1, keepdims=True) * (scale ** 3) / d
    return grad_x


def relu_forward(x):
    """ReLU activation."""
    mask = (x > 0).astype(np.float32)
    return x * mask, mask


def relu_backward(grad_y, cache):
    """Backward pass for ReLU."""
    mask = cache
    return grad_y * mask


def softmax_forward(logits):
    """
    Numerically stable softmax.

    Args:
        logits: shape (vocab,) or (batch, vocab)

    Returns:
        probs: same shape, sums to 1 along last axis
    """
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    probs = exps / np.sum(exps, axis=-1, keepdims=True)
    return probs


def cross_entropy_loss(probs, target_ids):
    """
    Cross-entropy loss for next-token prediction.

    Args:
        probs: shape (seq_len, vocab_size) — softmax probabilities
        target_ids: shape (seq_len,) — integer target token ids

    Returns:
        loss: scalar
        grad_logits: shape (seq_len, vocab_size) — gradient w.r.t. pre-softmax logits
    """
    seq_len = probs.shape[0]
    # Clip for numerical stability
    probs_clipped = np.clip(probs, 1e-8, 1.0)

    # Loss = -mean(log(p[target]))
    log_probs = np.log(probs_clipped)
    loss = 0.0
    for t in range(seq_len):
        loss -= log_probs[t, target_ids[t]]
    loss /= seq_len

    # Gradient of cross-entropy w.r.t. logits (softmax + CE combined)
    grad_logits = probs.copy()
    for t in range(seq_len):
        grad_logits[t, target_ids[t]] -= 1.0
    grad_logits /= seq_len

    return loss, grad_logits


def tanh_forward(x):
    """Tanh activation."""
    y = np.tanh(x)
    return y, y


def tanh_backward(grad_y, cache):
    """Backward pass for tanh."""
    y = cache
    return grad_y * (1 - y * y)


# ============================================================
# Attention
# ============================================================

def attention_forward(q, k, v, n_head, head_dim):
    """
    Multi-head causal self-attention (single sequence, no batch dim).

    Args:
        q: (seq_len, n_embd) — queries
        k: (seq_len, n_embd) — keys
        v: (seq_len, n_embd) — values
        n_head: number of attention heads
        head_dim: dimension per head

    Returns:
        out: (seq_len, n_embd) — attention output
        cache: values needed for backward
    """
    seq_len = q.shape[0]
    scale = 1.0 / np.sqrt(head_dim)

    # Reshape to heads: (seq_len, n_head, head_dim)
    q_h = q.reshape(seq_len, n_head, head_dim)
    k_h = k.reshape(seq_len, n_head, head_dim)
    v_h = v.reshape(seq_len, n_head, head_dim)

    # Compute attention scores: (n_head, seq_len, seq_len)
    # attn[h, i, j] = dot(q_h[i, h, :], k_h[j, h, :]) * scale
    attn_logits = np.einsum('ihe,jhe->hij', q_h, k_h) * scale

    # Causal mask: position i can only attend to positions <= i
    causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=np.float32) * (-1e9), k=1)
    attn_logits = attn_logits + causal_mask[np.newaxis, :, :]

    # Softmax per head per query position
    attn_weights = np.zeros_like(attn_logits)
    for h in range(n_head):
        attn_weights[h] = softmax_forward(attn_logits[h])

    # Weighted sum of values: out[i, h, :] = sum_j attn_weights[h, i, j] * v_h[j, h, :]
    out_h = np.einsum('hij,jhe->ihe', attn_weights, v_h)

    # Reshape back: (seq_len, n_embd)
    out = out_h.reshape(seq_len, -1)

    cache = (q_h, k_h, v_h, attn_weights, scale, n_head, head_dim, seq_len)
    return out, cache


def attention_backward(grad_out, cache):
    """
    Backward pass for multi-head causal self-attention.

    Returns:
        grad_q, grad_k, grad_v: each (seq_len, n_embd)
    """
    q_h, k_h, v_h, attn_weights, scale, n_head, head_dim, seq_len = cache

    # Reshape grad_out to heads
    grad_out_h = grad_out.reshape(seq_len, n_head, head_dim)

    # Gradient w.r.t. attn_weights and v_h
    # out_h = attn_weights @ v_h (per head)
    grad_attn_weights = np.einsum('ihe,jhe->hij', grad_out_h, v_h)
    grad_v_h = np.einsum('hij,ihe->jhe', attn_weights, grad_out_h)

    # Gradient through softmax (per head, per query position)
    grad_attn_logits = np.zeros_like(attn_weights)
    for h in range(n_head):
        for i in range(seq_len):
            s = attn_weights[h, i]  # (seq_len,)
            g = grad_attn_weights[h, i]  # (seq_len,)
            dot = np.sum(g * s)
            grad_attn_logits[h, i] = s * (g - dot)

    # Gradient w.r.t. q_h and k_h from attention logits
    grad_q_h = np.einsum('hij,jhe->ihe', grad_attn_logits, k_h) * scale
    grad_k_h = np.einsum('hij,ihe->jhe', grad_attn_logits, q_h) * scale

    # Reshape back to (seq_len, n_embd)
    grad_q = grad_q_h.reshape(seq_len, -1)
    grad_k = grad_k_h.reshape(seq_len, -1)
    grad_v = grad_v_h.reshape(seq_len, -1)

    return grad_q, grad_k, grad_v


# ============================================================
# Adam Optimizer
# ============================================================

class AdamOptimizer:
    """
    Adam optimizer operating on a list of Parameter objects.
    """

    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8,
                 weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0

        # Moment buffers
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]

    def step(self, lr_override=None):
        """Perform one Adam update step."""
        self.step_count += 1
        lr = lr_override if lr_override is not None else self.lr

        for i, p in enumerate(self.params):
            grad = p.grad
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * p.data

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2

            m_hat = self.m[i] / (1 - self.beta1 ** self.step_count)
            v_hat = self.v[i] / (1 - self.beta2 ** self.step_count)

            p.data -= lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """Zero all parameter gradients."""
        for p in self.params:
            p.zero_grad()
