"""
GPT-2 + Triadic Projection Head -- Transfer Experiment (Experiment 10).

Wraps a pre-trained HuggingFace GPT-2 with our triadic projection head
to test whether richer pre-trained embeddings produce better triadic structure.

Key hypothesis: GPT-2's embeddings (trained on 8M web pages) should provide
a much richer semantic teacher for the alignment loss than our from-scratch
model trained on TinyStories (50K children's stories).

Two-phase training strategy:
  Phase 1: Backbone frozen, train only triadic head (~49K params)
  Phase 2: Unfreeze last N layers, joint optimization (~14M params)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GPT2TriadicModel(nn.Module):
    """
    Pre-trained GPT-2 + Triadic Projection Head.

    Architecture:
        GPT-2 Backbone (frozen or partially unfrozen)
            -> LM Head (GPT-2's original, weight-tied with wte)
            -> Triadic Head (new linear layer, trained from scratch)
    """

    def __init__(self, gpt2_model, n_triadic_bits=64):
        super().__init__()
        self.gpt2 = gpt2_model
        self.n_embd = gpt2_model.config.n_embd
        self.n_triadic_bits = n_triadic_bits
        self.block_size = gpt2_model.config.n_positions  # 1024 for GPT-2

        # Triadic projection head (new, random init)
        self.triadic_head = nn.Linear(self.n_embd, n_triadic_bits, bias=False)
        nn.init.normal_(self.triadic_head.weight, mean=0.0, std=0.02)

    def freeze_backbone(self):
        """Freeze all GPT-2 parameters. Only triadic head is trainable."""
        for param in self.gpt2.parameters():
            param.requires_grad = False
        for param in self.triadic_head.parameters():
            param.requires_grad = True

    def unfreeze_last_n(self, n=2):
        """Unfreeze the last N transformer blocks + final layer norm."""
        for param in self.gpt2.transformer.ln_f.parameters():
            param.requires_grad = True
        total = len(self.gpt2.transformer.h)
        for i in range(max(0, total - n), total):
            for param in self.gpt2.transformer.h[i].parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass.

        Args:
            input_ids: (B, T) token IDs (GPT-2 tokenizer)
            attention_mask: (B, T) optional
            labels: (B, T) for language loss (shifted internally)

        Returns:
            logits: (B, T, vocab_size)
            triadic_proj: (B, T, n_triadic_bits)
            loss: scalar language loss or None
        """
        outputs = self.gpt2.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state  # (B, T, n_embd)

        # LM Head (GPT-2's original)
        logits = self.gpt2.lm_head(hidden_states)  # (B, T, vocab_size)

        # Triadic Head (new)
        triadic_proj = torch.tanh(self.triadic_head(hidden_states))

        # Language loss
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, triadic_proj, loss

    def triadic_loss(self, triadic_proj, entropy_weight=0.0,
                     input_ids=None, align_weight=0.0,
                     align_mode='mse'):
        """
        Compute triadic loss.

        Components:
          1. Diversity: bits should fire ~50% of the time
          2. Contrastive: different sequences should differ
          3. Entropy: prevent dead bits
          4. Embedding alignment (configurable mode):
             - 'mse': MSE on absolute similarity values (original)
             - 'rank': Margin ranking loss — preserve similarity ordering
             - 'infonce': InfoNCE contrastive with embedding-mined positives

        Args:
            align_mode: 'mse' | 'rank' | 'infonce'
        """
        if triadic_proj.size(1) < 2:
            return torch.tensor(0.0, device=triadic_proj.device)

        B, T, n_bits = triadic_proj.shape

        # 1. Diversity
        bit_means = triadic_proj.mean(dim=(0, 1))
        diversity_loss = (bit_means ** 2).mean()

        # 2. Contrastive
        seq_proj = triadic_proj.mean(dim=1)
        if B > 1:
            seq_norm = F.normalize(seq_proj, dim=-1)
            sim_matrix = seq_norm @ seq_norm.T
            mask = ~torch.eye(B, device=sim_matrix.device, dtype=torch.bool)
            contrastive_loss = sim_matrix[mask].pow(2).mean()
        else:
            contrastive_loss = torch.tensor(0.0, device=triadic_proj.device)

        # 3. Entropy regularization
        entropy_loss = torch.tensor(0.0, device=triadic_proj.device)
        if entropy_weight > 0:
            flat_proj = triadic_proj.reshape(-1, n_bits)
            probs = (flat_proj.mean(dim=0) + 1.0) / 2.0
            eps = 1e-7
            probs = probs.clamp(eps, 1.0 - eps)
            bit_entropy = -(probs * probs.log() + (1 - probs) * (1 - probs).log())
            entropy_loss = (1.0 - bit_entropy / math.log(2)).mean()

        # 4. Embedding alignment
        alignment_loss = torch.tensor(0.0, device=triadic_proj.device)
        if align_weight > 0 and input_ids is not None:
            with torch.no_grad():
                embeds = self.gpt2.transformer.wte(input_ids).detach()

            if align_mode == 'mse':
                alignment_loss = self._align_mse(triadic_proj, embeds, B, T, n_bits)
            elif align_mode == 'rank':
                alignment_loss = self._align_rank(triadic_proj, embeds, B, T, n_bits)
            elif align_mode == 'infonce':
                alignment_loss = self._align_infonce(triadic_proj, embeds, B, T, n_bits)

        # Combined
        loss = diversity_loss + contrastive_loss
        if entropy_weight > 0:
            loss = loss + entropy_weight * entropy_loss
        if align_weight > 0:
            loss = loss + align_weight * alignment_loss
        return loss

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
        """Margin ranking loss: preserve similarity ORDERING, ignore absolute values.

        For each batch item, sample triplets (anchor, pos, neg) where:
          embed_sim(anchor, pos) > embed_sim(anchor, neg)
        Then enforce:
          triadic_sim(anchor, pos) > triadic_sim(anchor, neg) + margin

        This is more robust than MSE because:
        - Doesn't try to match absolute similarity values across different spaces
        - Only penalizes ordering violations
        - Naturally handles scale mismatch between 768D and 64-bit spaces
        """
        n_anchors = 32
        n_candidates = 16

        # Sample anchor indices and candidate pair indices
        anchor_idx = torch.randint(0, T, (B, n_anchors), device=triadic_proj.device)
        cand_idx = torch.randint(0, T, (B, n_anchors, n_candidates), device=triadic_proj.device)

        # Get anchor embeddings: (B, n_anchors, n_embd)
        anchor_e = torch.gather(embeds, 1,
                                anchor_idx.unsqueeze(-1).expand(-1, -1, embeds.size(-1)))
        # Get candidate embeddings: (B, n_anchors, n_candidates, n_embd)
        cand_e = torch.gather(embeds, 1,
                              cand_idx.reshape(B, -1).unsqueeze(-1).expand(-1, -1, embeds.size(-1))
                              ).reshape(B, n_anchors, n_candidates, embeds.size(-1))

        # Embedding similarities: (B, n_anchors, n_candidates)
        embed_sim = F.cosine_similarity(anchor_e.unsqueeze(2), cand_e, dim=-1)

        # Find most similar (pos) and least similar (neg) candidates per anchor
        pos_idx_local = embed_sim.argmax(dim=-1)  # (B, n_anchors)
        neg_idx_local = embed_sim.argmin(dim=-1)  # (B, n_anchors)

        # Map back to token indices
        pos_idx = torch.gather(cand_idx, 2, pos_idx_local.unsqueeze(-1)).squeeze(-1)
        neg_idx = torch.gather(cand_idx, 2, neg_idx_local.unsqueeze(-1)).squeeze(-1)

        # Get triadic projections for anchor, pos, neg
        anchor_p = torch.gather(triadic_proj, 1,
                                anchor_idx.unsqueeze(-1).expand(-1, -1, n_bits))
        pos_p = torch.gather(triadic_proj, 1,
                             pos_idx.unsqueeze(-1).expand(-1, -1, n_bits))
        neg_p = torch.gather(triadic_proj, 1,
                             neg_idx.unsqueeze(-1).expand(-1, -1, n_bits))

        # Triadic similarities
        pos_sim = F.cosine_similarity(anchor_p, pos_p, dim=-1)  # (B, n_anchors)
        neg_sim = F.cosine_similarity(anchor_p, neg_p, dim=-1)  # (B, n_anchors)

        # Margin ranking loss: pos_sim should exceed neg_sim by margin
        rank_loss = F.relu(margin - (pos_sim - neg_sim)).mean()

        return rank_loss

    def _align_infonce(self, triadic_proj, embeds, B, T, n_bits, temperature=0.5):
        """InfoNCE contrastive alignment with embedding-mined positives.

        For each anchor token, find the most similar token in embedding space
        (positive), then use all other sampled tokens as negatives. Maximize
        triadic similarity to positive vs negatives.

        Advantages over MSE:
        - Structured positive/negative mining from embedding space
        - Temperature-controlled softmax creates sharper gradients
        - Natural information-theoretic objective

        Bug #7 fix: temperature 0.1->0.5 to prevent logit overflow in bfloat16,
        added eps to F.normalize, clamped logits before cross_entropy.
        """
        n_anchors = 32

        # Sample anchor and candidate pool
        anchor_idx = torch.randint(0, T, (B, n_anchors), device=triadic_proj.device)
        pool_idx = torch.randint(0, T, (B, n_anchors), device=triadic_proj.device)

        # Embedding similarities between all anchor-pool pairs
        anchor_e = torch.gather(embeds, 1,
                                anchor_idx.unsqueeze(-1).expand(-1, -1, embeds.size(-1)))
        pool_e = torch.gather(embeds, 1,
                              pool_idx.unsqueeze(-1).expand(-1, -1, embeds.size(-1)))
        anchor_e_norm = F.normalize(anchor_e, dim=-1, eps=1e-6)
        pool_e_norm = F.normalize(pool_e, dim=-1, eps=1e-6)

        # (B, n_anchors, n_anchors) — cross-similarity matrix in embedding space
        embed_sim_matrix = torch.bmm(anchor_e_norm, pool_e_norm.transpose(1, 2))

        # For each anchor, the positive is the most similar pool token
        pos_labels = embed_sim_matrix.argmax(dim=-1)  # (B, n_anchors)

        # Triadic similarities between all anchor-pool pairs
        anchor_p = torch.gather(triadic_proj, 1,
                                anchor_idx.unsqueeze(-1).expand(-1, -1, n_bits))
        pool_p = torch.gather(triadic_proj, 1,
                              pool_idx.unsqueeze(-1).expand(-1, -1, n_bits))
        anchor_p_norm = F.normalize(anchor_p, dim=-1, eps=1e-6)
        pool_p_norm = F.normalize(pool_p, dim=-1, eps=1e-6)

        # (B, n_anchors, n_anchors) — cross-similarity in triadic space
        triadic_sim_matrix = torch.bmm(anchor_p_norm, pool_p_norm.transpose(1, 2))

        # InfoNCE: maximize triadic sim to embedding-selected positive
        # Clamp logits to prevent overflow in bfloat16 softmax
        logits = torch.clamp(triadic_sim_matrix / temperature, -30, 30)
        loss = F.cross_entropy(logits.reshape(-1, n_anchors), pos_labels.reshape(-1))

        return loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=0.7, top_k=50):
        """Autoregressive text generation."""
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.block_size:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids

    def num_params(self, trainable_only=False):
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
