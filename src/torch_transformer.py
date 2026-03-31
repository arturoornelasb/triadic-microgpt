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

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing to trade compute for VRAM."""
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self._gradient_checkpointing = False

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
            if getattr(self, '_gradient_checkpointing', False) and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
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

    def triadic_loss(self, triadic_proj, entropy_weight=0.0, input_ids=None,
                     align_weight=0.0, align_mode='mse',
                     sparsity_weight=0.0, target_active_bits=8,
                     subsumption_weight=0.0, subsumption_pairs=None):
        """
        Compute triadic loss with up to six objectives:
          1. Diversity: bits should be diverse across the batch (not all same)
          2. Contrastive: different sequences should have different projections
          3. Entropy: each bit should have high entropy across the batch (prevent dead bits)
          4. Embedding alignment (configurable mode):
             - 'mse': MSE on absolute similarity values (original)
             - 'rank': Margin ranking loss — preserve similarity ordering
             - 'infonce': InfoNCE contrastive with embedding-mined positives
          5. Sparsity: penalize deviation from target number of active bits.
             Critical for subsumption — with k=30 active bits, P(divisibility)
             is near zero. Target 6-10 active bits for functional subsumption.
          6. Subsumption: for known hypernym pairs, penalize when hypernym bits
             are not a subset of hyponym bits. Requires subsumption_pairs dict
             mapping token_id -> list of token_ids that should subsume it.

        NOTE: Coherence loss (adjacent tokens agree) was REMOVED in Run 13.
        It caused triadic collapse by pushing all tokens toward the same projection.

        Args:
            triadic_proj: (B, T, n_bits)
            entropy_weight: weight for entropy regularization term (0 = disabled)
            input_ids: (B, T) token IDs — needed for embedding alignment
            align_weight: weight for embedding alignment loss (0 = disabled)
            align_mode: 'mse' | 'rank' | 'infonce'
            sparsity_weight: weight for sparsity loss (0 = disabled)
            target_active_bits: target number of active bits per token (default 8)
            subsumption_weight: weight for subsumption loss (0 = disabled)
            subsumption_pairs: dict {hyper_token_id: [hypo_token_id, ...]}

        Returns:
            loss: scalar
        """
        if triadic_proj.size(1) < 2:
            return torch.tensor(0.0, device=triadic_proj.device)

        B, T, n_bits = triadic_proj.shape

        # 1. Diversity: each bit should be diverse across the batch
        # NOTE: When sparsity loss is active, diversity loss is disabled
        # because they are contradictory — diversity pushes toward 50%
        # activation while sparsity targets ~12.5%.
        bit_means = triadic_proj.mean(dim=(0, 1))  # (n_bits,)
        if sparsity_weight > 0:
            diversity_loss = torch.tensor(0.0, device=triadic_proj.device)
        else:
            diversity_loss = (bit_means ** 2).mean()

        # 2. Contrastive: different batch items should differ
        seq_proj = triadic_proj.mean(dim=1)  # (B, n_bits)
        if B > 1:
            seq_norm = F.normalize(seq_proj, dim=-1)
            sim_matrix = seq_norm @ seq_norm.T  # (B, B)
            mask = ~torch.eye(B, device=sim_matrix.device, dtype=torch.bool)
            off_diag = sim_matrix[mask]
            contrastive_loss = off_diag.pow(2).mean()
        else:
            contrastive_loss = torch.tensor(0.0, device=triadic_proj.device)

        # 3. Entropy regularization (disabled when sparsity is active —
        #    entropy pushes bits toward 50% activation, opposing sparsity)
        entropy_loss = torch.tensor(0.0, device=triadic_proj.device)
        if entropy_weight > 0 and sparsity_weight == 0:
            flat_proj = triadic_proj.reshape(-1, n_bits)  # (B*T, n_bits)
            probs = (flat_proj.mean(dim=0) + 1.0) / 2.0  # (n_bits,)
            eps = 1e-7
            probs = probs.clamp(eps, 1.0 - eps)
            bit_entropy = -(probs * probs.log() + (1 - probs) * (1 - probs).log())
            entropy_loss = (1.0 - bit_entropy / math.log(2)).mean()

        # 4. Embedding alignment
        alignment_loss = torch.tensor(0.0, device=triadic_proj.device)
        if align_weight > 0 and input_ids is not None:
            with torch.no_grad():
                embeds = self.wte(input_ids).detach()  # (B, T, n_embd)

            if align_mode == 'mse':
                alignment_loss = self._align_mse(triadic_proj, embeds, B, T, n_bits)
            elif align_mode == 'rank':
                alignment_loss = self._align_rank(triadic_proj, embeds, B, T, n_bits)
            elif align_mode == 'infonce':
                alignment_loss = self._align_infonce(triadic_proj, embeds, B, T, n_bits)

        # 5. Sparsity: push activation rate toward target
        sparsity_loss = torch.tensor(0.0, device=triadic_proj.device)
        if sparsity_weight > 0:
            # activation_rate = fraction of bits > 0 (tanh output)
            activation_rate = (triadic_proj > 0).float().mean(dim=-1)  # (B, T)
            target_rate = target_active_bits / n_bits
            sparsity_loss = ((activation_rate - target_rate) ** 2).mean()

        # 6. Subsumption: hypernym bits must be subset of hyponym bits
        subsumption_loss = torch.tensor(0.0, device=triadic_proj.device)
        if subsumption_weight > 0 and subsumption_pairs and input_ids is not None:
            subsumption_loss = self._subsumption_loss(
                triadic_proj, input_ids, subsumption_pairs)

        # Combined loss
        loss = diversity_loss + contrastive_loss
        if entropy_weight > 0:
            loss = loss + entropy_weight * entropy_loss
        if align_weight > 0:
            loss = loss + align_weight * alignment_loss
        if sparsity_weight > 0:
            loss = loss + sparsity_weight * sparsity_loss
        if subsumption_weight > 0:
            loss = loss + subsumption_weight * subsumption_loss
        return loss

    def _subsumption_loss(self, triadic_proj, input_ids, subsumption_pairs):
        """Penalize when hypernym bits are not a subset of hyponym bits.

        For each (hyper, hypo) pair found in the batch:
          For each bit where hyper is active (>0), hypo should also be active.
          Loss = mean of ReLU(hyper_activation - hypo_activation) across those bits.
        """
        B, T, n_bits = triadic_proj.shape
        device = triadic_proj.device
        losses = []

        # Build token_id -> mean projection lookup from this batch
        token_projs = {}
        for b in range(B):
            for t in range(T):
                tid = input_ids[b, t].item()
                if tid not in token_projs:
                    token_projs[tid] = []
                token_projs[tid].append(triadic_proj[b, t])

        # Average projections per token
        token_mean_projs = {}
        for tid, projs in token_projs.items():
            token_mean_projs[tid] = torch.stack(projs).mean(dim=0)

        # Evaluate subsumption pairs present in this batch
        for hyper_tid, hypo_tids in subsumption_pairs.items():
            if hyper_tid not in token_mean_projs:
                continue
            hyper_proj = token_mean_projs[hyper_tid]
            for hypo_tid in hypo_tids:
                if hypo_tid not in token_mean_projs:
                    continue
                hypo_proj = token_mean_projs[hypo_tid]
                # Where hyper is active, hypo must also be active
                # penalty = ReLU(hyper - hypo) — hyper active but hypo not
                penalty = F.relu(hyper_proj - hypo_proj)
                losses.append(penalty.mean())

        if losses:
            return torch.stack(losses).mean()
        return torch.tensor(0.0, device=device)

    def _align_mse(self, triadic_proj, embeds, B, T, n_bits):
        """Original MSE alignment: match absolute similarity values."""
        n_pairs = 64
        idx = torch.randint(0, T, (B, n_pairs, 2), device=triadic_proj.device)
        idx_i, idx_j = idx[:, :, 0], idx[:, :, 1]

        idx_i_e = idx_i.unsqueeze(-1).expand(-1, -1, embeds.size(-1))
        idx_j_e = idx_j.unsqueeze(-1).expand(-1, -1, embeds.size(-1))
        e_i = torch.gather(embeds, 1, idx_i_e)
        e_j = torch.gather(embeds, 1, idx_j_e)
        embed_sim = F.cosine_similarity(e_i, e_j, dim=-1)

        idx_i_p = idx_i.unsqueeze(-1).expand(-1, -1, n_bits)
        idx_j_p = idx_j.unsqueeze(-1).expand(-1, -1, n_bits)
        p_i = torch.gather(triadic_proj, 1, idx_i_p)
        p_j = torch.gather(triadic_proj, 1, idx_j_p)
        triadic_sim = F.cosine_similarity(p_i, p_j, dim=-1)

        return F.mse_loss(triadic_sim, embed_sim)

    def _align_rank(self, triadic_proj, embeds, B, T, n_bits, margin=0.1):
        """Margin ranking loss: preserve similarity ORDERING, ignore absolute values."""
        n_anchors = 32
        n_candidates = 16

        anchor_idx = torch.randint(0, T, (B, n_anchors), device=triadic_proj.device)
        cand_idx = torch.randint(0, T, (B, n_anchors, n_candidates), device=triadic_proj.device)

        anchor_e = torch.gather(embeds, 1,
                                anchor_idx.unsqueeze(-1).expand(-1, -1, embeds.size(-1)))
        cand_e = torch.gather(embeds, 1,
                              cand_idx.reshape(B, -1).unsqueeze(-1).expand(-1, -1, embeds.size(-1))
                              ).reshape(B, n_anchors, n_candidates, embeds.size(-1))

        embed_sim = F.cosine_similarity(anchor_e.unsqueeze(2), cand_e, dim=-1)

        pos_idx_local = embed_sim.argmax(dim=-1)
        neg_idx_local = embed_sim.argmin(dim=-1)

        pos_idx = torch.gather(cand_idx, 2, pos_idx_local.unsqueeze(-1)).squeeze(-1)
        neg_idx = torch.gather(cand_idx, 2, neg_idx_local.unsqueeze(-1)).squeeze(-1)

        anchor_p = torch.gather(triadic_proj, 1,
                                anchor_idx.unsqueeze(-1).expand(-1, -1, n_bits))
        pos_p = torch.gather(triadic_proj, 1,
                             pos_idx.unsqueeze(-1).expand(-1, -1, n_bits))
        neg_p = torch.gather(triadic_proj, 1,
                             neg_idx.unsqueeze(-1).expand(-1, -1, n_bits))

        pos_sim = F.cosine_similarity(anchor_p, pos_p, dim=-1)
        neg_sim = F.cosine_similarity(anchor_p, neg_p, dim=-1)

        return F.relu(margin - (pos_sim - neg_sim)).mean()

    def _align_infonce(self, triadic_proj, embeds, B, T, n_bits, temperature=0.1):
        """InfoNCE contrastive alignment with embedding-mined positives."""
        n_anchors = 32

        anchor_idx = torch.randint(0, T, (B, n_anchors), device=triadic_proj.device)
        pool_idx = torch.randint(0, T, (B, n_anchors), device=triadic_proj.device)
        # Avoid trivial self-reference where anchor == pool at same position
        collisions = anchor_idx == pool_idx
        if collisions.any():
            pool_idx = pool_idx.clone()
            pool_idx[collisions] = (pool_idx[collisions] + 1) % T

        anchor_e = torch.gather(embeds, 1,
                                anchor_idx.unsqueeze(-1).expand(-1, -1, embeds.size(-1)))
        pool_e = torch.gather(embeds, 1,
                              pool_idx.unsqueeze(-1).expand(-1, -1, embeds.size(-1)))
        anchor_e_norm = F.normalize(anchor_e, dim=-1)
        pool_e_norm = F.normalize(pool_e, dim=-1)

        embed_sim_matrix = torch.bmm(anchor_e_norm, pool_e_norm.transpose(1, 2))
        pos_labels = embed_sim_matrix.argmax(dim=-1)

        anchor_p = torch.gather(triadic_proj, 1,
                                anchor_idx.unsqueeze(-1).expand(-1, -1, n_bits))
        pool_p = torch.gather(triadic_proj, 1,
                              pool_idx.unsqueeze(-1).expand(-1, -1, n_bits))
        anchor_p_norm = F.normalize(anchor_p, dim=-1)
        pool_p_norm = F.normalize(pool_p, dim=-1)

        triadic_sim_matrix = torch.bmm(anchor_p_norm, pool_p_norm.transpose(1, 2))

        logits = triadic_sim_matrix / temperature
        return F.cross_entropy(logits.reshape(-1, n_anchors), pos_labels.reshape(-1))

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
            
        # Note: Autocast requires BCEWithLogitsLoss for mixed precision stability
        # triadic_proj is currently tanh[-1, 1]. We need raw logits for BCEWithLogits.
        # Since we applied tanh in the forward pass, we need the raw projection here,
        # but to minimize refactoring we can invert tanh (artanh), or better, apply 
        # MSE on the tanh outputs directly which is mathematically similar and stable.
        
        dist_loss = F.mse_loss(triadic_proj[mask], (targets_proj[mask] * 2.0) - 1.0)
        
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
