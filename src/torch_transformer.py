"""
PyTorch Transformer — GPU-accelerated GPT with Triadic Projection Head.

Drop-in replacement for fast_transformer.py using PyTorch nn.Module.
Keeps the same triadic projection head architecture but with full CUDA support,
batching, and automatic differentiation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TriadicGPTConfig:
    """Configuration for the PyTorch GPT model."""
    def __init__(
        self,
        vocab_size=4096,
        block_size=256,
        n_layer=6,
        n_embd=256,
        n_head=8,
        n_triadic_bits=32,
        dropout=0.1,
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_triadic_bits = n_triadic_bits
        self.dropout = dropout
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()
        # QKV projection
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape to heads: (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask (uses Flash Attention when available)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.attn_dropout.p if self.training else 0.0
        )

        # Reassemble heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.dropout(self.fc2(x))
        return x


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN → Attn → residual → LN → MLP → residual."""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TriadicGPT(nn.Module):
    """
    GPT Language Model with Triadic Projection Head.

    Architecture:
        Token + Position Embedding → N × TransformerBlock → LayerNorm
            → LM Head (next-token prediction)
            → Triadic Head (prime-factor semantic encoding)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)

        # LM Head (language modeling)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Triadic Projection Head
        self.triadic_head = nn.Linear(config.n_embd, config.n_triadic_bits, bias=False)

        # Weight tying: share embedding and LM head weights
        self.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def initialize_triadic_pca(self, hidden_states):
        """
        Initialize triadic_head weights using PCA of the provided hidden states.
        This follows the 'pca' projection mode of the Triadic Engine.
        
        Args:
            hidden_states: (N, n_embd) tensor of hidden representations.
        """
        # Center the data
        X = hidden_states.float()
        n_bits = self.config.n_triadic_bits
        
        print(f"Computing PCA for {X.shape[0]} hidden states to extract {n_bits} semantic bits...")
        
        # torch.pca_lowrank is efficient for extracting top-k components
        # it returns (U, S, V) where X approx U @ diag(S) @ V.T
        _, _, V = torch.pca_lowrank(X, q=n_bits)
        
        # The principal components are the columns of V
        # triadic_head.weight expects (n_bits, n_embd)
        pca_weights = V.T  # (n_bits, n_embd)
        
        # Ensure it fits
        if pca_weights.shape[0] < n_bits:
            # Pad with random if we have fewer components than bits (rare if N is large)
            num_extra = n_bits - pca_weights.shape[0]
            extra = torch.randn(num_extra, self.config.n_embd, device=X.device) * 0.02
            pca_weights = torch.cat([pca_weights, extra], dim=0)
        elif pca_weights.shape[0] > n_bits:
            pca_weights = pca_weights[:n_bits]
            
        # Update weights
        self.triadic_head.weight.copy_(pca_weights)
        print("Triadic head initialized with PCA components.")

    def forward(self, input_ids, targets=None):
        """
        Forward pass.

        Args:
            input_ids: (B, T) integer token IDs
            targets: (B, T) integer target IDs (optional, for loss computation)

        Returns:
            logits: (B, T, vocab_size)
            triadic_proj: (B, T, n_triadic_bits) — tanh projections
            loss: scalar loss if targets provided, else None
        """
        B, T = input_ids.shape
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block_size {self.config.block_size}"

        # Token + position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        tok_emb = self.wte(input_ids)       # (B, T, n_embd)
        pos_emb = self.wpe(pos)             # (T, n_embd)
        x = self.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)                    # (B, T, n_embd)

        # LM Head
        logits = self.lm_head(x)            # (B, T, vocab_size)

        # Triadic Head
        triadic_proj = torch.tanh(self.triadic_head(x))  # (B, T, n_triadic_bits)

        # Compute loss if targets are given
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, triadic_proj, loss

    def triadic_loss(self, triadic_proj):
        """
        Compute triadic loss with three objectives:
          1. Coherence: adjacent tokens in same sequence should agree
          2. Diversity: bits should be diverse across the batch (not all same)
          3. Contrastive: different sequences should have different projections

        Args:
            triadic_proj: (B, T, n_bits)

        Returns:
            loss: scalar
        """
        if triadic_proj.size(1) < 2:
            return torch.tensor(0.0, device=triadic_proj.device)

        # 1. Coherence: adjacent tokens agree (original loss)
        pa = triadic_proj[:, :-1, :]  # (B, T-1, n_bits)
        pb = triadic_proj[:, 1:, :]   # (B, T-1, n_bits)
        agreement = (pa * pb).mean(dim=-1)  # (B, T-1)
        coherence_loss = (1.0 - agreement).mean()

        # 2. Diversity: each bit should be active ~50% of the time across the batch
        #    This prevents the trivial solution of all bits being identical
        bit_means = triadic_proj.mean(dim=(0, 1))  # (n_bits,) — mean activation per bit
        # Ideal is 0.0 (half positive, half negative). Penalize deviation.
        diversity_loss = (bit_means ** 2).mean()

        # 3. Contrastive: different batch items should differ
        #    Take mean projection per sequence, then push apart pairs
        seq_proj = triadic_proj.mean(dim=1)  # (B, n_bits)
        B = seq_proj.size(0)
        if B > 1:
            # Cosine similarity matrix
            seq_norm = F.normalize(seq_proj, dim=-1)
            sim_matrix = seq_norm @ seq_norm.T  # (B, B)
            # We want diagonal=1, off-diagonal=low
            # Mask out diagonal and penalize high off-diagonal similarity
            mask = ~torch.eye(B, device=sim_matrix.device, dtype=torch.bool)
            off_diag = sim_matrix[mask]
            contrastive_loss = off_diag.pow(2).mean()
        else:
            contrastive_loss = torch.tensor(0.0, device=triadic_proj.device)

        # Combined: coherence + diversity + contrastive
        loss = coherence_loss + 0.5 * diversity_loss + 0.3 * contrastive_loss
        return loss

    def distillation_loss(self, triadic_proj, targets_proj, mask):
        """
        Compute Binary Cross Entropy loss to align triadic projections with gold standard bits.

        Args:
            triadic_proj: (B, T, n_bits) — current model projections in [-1, 1] range via tanh
            targets_proj: (B, T, n_bits) — target bits [0, 1] 
            mask:         (B, T) — boolean mask, True for tokens that exist in the gold dictionary

        Returns:
            loss: scalar representing the BCE across masked tokens
        """
        if not mask.any():
            return torch.tensor(0.0, device=triadic_proj.device)
            
        # Map tanh outputs from [-1, 1] to [0, 1] probability range
        probs = (triadic_proj + 1.0) / 2.0
        
        # Apply Binary Cross Entropy
        dist_loss = F.binary_cross_entropy(probs[mask], targets_proj[mask])
        
        return dist_loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=0.7, top_k=50):
        """
        Autoregressive text generation.

        Args:
            input_ids: (1, T) starting tokens
            max_new_tokens: how many tokens to generate
            temperature: sampling temperature
            top_k: keep only top-k logits for sampling
        """
        for _ in range(max_new_tokens):
            # Crop to block size
            idx_cond = input_ids[:, -self.config.block_size:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)

        return input_ids

    def num_params(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
