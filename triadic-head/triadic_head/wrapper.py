"""
TriadicWrapper — Drop-in triadic projection head for any HuggingFace transformer.

Usage:
    from triadic_head import TriadicWrapper

    model = TriadicWrapper("gpt2", n_bits=64)
    model.freeze_backbone()
    # ... train phase 1 (triadic head only) ...
    model.unfreeze_last_n(2)
    # ... train phase 2 (joint optimization) ...

    sigs = model.encode(["king", "queen", "dog"])
    model.compare("king", "queen")  # -> 0.89
"""

import math
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Architecture detection for HuggingFace models
# ============================================================

def _detect_architecture(model) -> Dict:
    """
    Auto-detect backbone, embedding layer, LM head, and layer list
    from a HuggingFace causal LM model.

    Supports: GPT-2, GPT-Neo, GPT-J, LLaMA, Mistral, Phi, Qwen, OPT, Falcon.
    """
    info = {}

    # Find the transformer backbone
    for attr in ('transformer', 'model', 'gpt_neox'):
        if hasattr(model, attr):
            info['backbone'] = getattr(model, attr)
            info['backbone_attr'] = attr
            break
    else:
        raise ValueError(
            f"Cannot detect backbone for {type(model).__name__}. "
            "Pass `embedding_attr` and `layers_attr` manually."
        )

    backbone = info['backbone']

    # Find embedding layer
    for attr in ('wte', 'embed_tokens', 'word_embeddings', 'embed_in'):
        if hasattr(backbone, attr):
            info['embed_layer'] = getattr(backbone, attr)
            info['embed_attr'] = attr
            break
    else:
        raise ValueError(f"Cannot find embedding layer in {type(backbone).__name__}.")

    info['n_embd'] = info['embed_layer'].weight.shape[1]

    # Find transformer layers
    for attr in ('h', 'layers', 'block'):
        if hasattr(backbone, attr):
            info['layers'] = getattr(backbone, attr)
            info['layers_attr'] = attr
            break
    else:
        raise ValueError(f"Cannot find layer list in {type(backbone).__name__}.")

    # Find final layer norm
    for attr in ('ln_f', 'norm', 'final_layer_norm', 'ln_final'):
        if hasattr(backbone, attr):
            info['ln_f'] = getattr(backbone, attr)
            info['ln_f_attr'] = attr
            break

    # Find LM head
    for attr in ('lm_head',):
        if hasattr(model, attr):
            info['lm_head'] = getattr(model, attr)
            break

    # Find block size / max position
    config = getattr(model, 'config', None)
    if config:
        for attr in ('n_positions', 'max_position_embeddings', 'max_seq_len'):
            if hasattr(config, attr):
                info['block_size'] = getattr(config, attr)
                break

    return info


# ============================================================
# TriadicHead — The projection layer itself
# ============================================================

class TriadicHead(nn.Module):
    """
    Single linear projection: hidden_states -> tanh -> n_bits continuous values.

    This is the only new parameter added to the backbone model.
    For GPT-2 (768D, 64 bits): 768 * 64 = 49,152 parameters.
    """

    def __init__(self, n_embd: int, n_bits: int):
        super().__init__()
        self.proj = nn.Linear(n_embd, n_bits, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """(B, T, n_embd) -> (B, T, n_bits) in [-1, 1]."""
        return torch.tanh(self.proj(hidden_states))


# ============================================================
# TriadicWrapper — Main public class
# ============================================================

class TriadicWrapper(nn.Module):
    """
    Wraps any HuggingFace causal LM with a triadic projection head.

    The wrapper adds a single linear layer (n_embd -> n_bits) that produces
    discrete prime-factor signatures alongside normal language model outputs.
    Training uses a multi-component triadic loss that transfers semantic
    structure from the model's own embeddings to the triadic head.

    Args:
        model: HuggingFace model instance or model name string (e.g. "gpt2").
        n_bits: Number of triadic bits (default 64). Each bit maps to a prime.
        align_mode: Alignment loss type — "mse", "rank", or "infonce".
                    Use "infonce" for pre-trained models, "mse" for from-scratch.
        device: Device to place model on (auto-detected if None).
    """

    def __init__(
        self,
        model: Union[str, nn.Module],
        n_bits: int = 64,
        align_mode: str = 'infonce',
        device: Optional[str] = None,
    ):
        super().__init__()

        # Load model from string if needed
        if isinstance(model, str):
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(model)

        self._arch = _detect_architecture(model)
        self.backbone_model = model
        self.n_bits = n_bits
        self.n_embd = self._arch['n_embd']
        self.align_mode = align_mode
        self.block_size = self._arch.get('block_size', 1024)

        # The one new thing: triadic projection head
        self.triadic_head = TriadicHead(self.n_embd, n_bits)

        if device:
            self.to(device)

    # ----------------------------------------------------------
    # Configuration — view and change settings
    # ----------------------------------------------------------

    def config(self, **kwargs) -> Dict:
        """
        View or update model configuration.

        Call with no arguments to view current config.
        Pass keyword arguments to update:

            model.config(align_mode='rank', n_bits=32)

        Changeable settings:
            align_mode: 'mse' | 'rank' | 'infonce'

        Read-only (shown but not changeable):
            n_bits, n_embd, backbone, block_size, triadic_params, total_params
        """
        VALID_ALIGN = ('mse', 'rank', 'infonce')

        if 'align_mode' in kwargs:
            mode = kwargs['align_mode']
            if mode not in VALID_ALIGN:
                raise ValueError(f"align_mode must be one of {VALID_ALIGN}, got {mode!r}")
            self.align_mode = mode

        return {
            'align_mode': self.align_mode,
            'n_bits': self.n_bits,
            'n_embd': self.n_embd,
            'backbone': type(self.backbone_model).__name__,
            'block_size': self.block_size,
            'triadic_params': self.triadic_params(),
            'total_params': self.num_params(),
        }

    # ----------------------------------------------------------
    # Freeze / unfreeze
    # ----------------------------------------------------------

    def freeze_backbone(self):
        """Freeze entire backbone. Only triadic head receives gradients."""
        for param in self.backbone_model.parameters():
            param.requires_grad = False
        for param in self.triadic_head.parameters():
            param.requires_grad = True

    def unfreeze_last_n(self, n: int = 2):
        """Unfreeze the last N transformer layers + final layer norm."""
        if 'ln_f' in self._arch:
            for param in self._arch['ln_f'].parameters():
                param.requires_grad = True
        layers = self._arch['layers']
        total = len(layers)
        for i in range(max(0, total - n), total):
            for param in layers[i].parameters():
                param.requires_grad = True

    def unfreeze_all(self):
        """Unfreeze everything."""
        for param in self.parameters():
            param.requires_grad = True

    # ----------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass through backbone + triadic head.

        Returns:
            logits:       (B, T, vocab_size)
            triadic_proj: (B, T, n_bits) in [-1, 1]
            lang_loss:    scalar or None
        """
        backbone = self._arch['backbone']
        outputs = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Handle different return types
        if hasattr(outputs, 'last_hidden_state'):
            hidden = outputs.last_hidden_state
        elif isinstance(outputs, tuple):
            hidden = outputs[0]
        else:
            hidden = outputs

        # LM head
        lm_head = self._arch.get('lm_head')
        if lm_head is not None:
            logits = lm_head(hidden)
        else:
            logits = self.backbone_model.lm_head(hidden)

        # Triadic head
        triadic_proj = self.triadic_head(hidden)

        # Language loss
        lang_loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lang_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, triadic_proj, lang_loss

    # ----------------------------------------------------------
    # Triadic loss
    # ----------------------------------------------------------

    def triadic_loss(
        self,
        triadic_proj: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        alpha: float = 0.05,
        entropy_weight: float = 1.0,
        align_weight: float = 5.0,
        align_mode: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Compute multi-component triadic loss.

        Components:
          1. Diversity — each bit should fire ~50% of the time.
          2. Contrastive — different sequences should produce different signatures.
          3. Entropy — prevent dead bits (bits stuck at +1 or -1).
          4. Embedding alignment — transfer semantic structure from embeddings.

        Args:
            triadic_proj: (B, T, n_bits) from forward().
            input_ids: (B, T) needed for alignment loss.
            alpha: Weight of total triadic loss (default 0.05, DO NOT exceed 0.10).
            entropy_weight: Weight for entropy term (default 1.0).
            align_weight: Weight for alignment term (default 5.0).
            align_mode: Override self.align_mode for this call.

        Returns:
            Weighted triadic loss scalar (already multiplied by alpha).
        """
        mode = align_mode or self.align_mode

        if triadic_proj.size(1) < 2:
            return torch.tensor(0.0, device=triadic_proj.device)

        # Force float32 for numerical stability (critical under AMP/mixed precision)
        triadic_proj = triadic_proj.float()

        B, T, n_bits = triadic_proj.shape

        # 1. Diversity: push per-bit mean toward 0
        bit_means = triadic_proj.mean(dim=(0, 1))
        diversity = (bit_means ** 2).mean()

        # 2. Contrastive: push sequence-level projections apart
        if B > 1:
            seq_proj = F.normalize(triadic_proj.mean(dim=1), dim=-1)
            sim = seq_proj @ seq_proj.T
            mask = ~torch.eye(B, device=sim.device, dtype=torch.bool)
            contrastive = sim[mask].pow(2).mean()
        else:
            contrastive = torch.tensor(0.0, device=triadic_proj.device)

        # 3. Entropy: maximize per-bit entropy across the batch
        entropy = torch.tensor(0.0, device=triadic_proj.device)
        if entropy_weight > 0:
            flat = triadic_proj.reshape(-1, n_bits)
            probs = ((flat.mean(dim=0) + 1.0) / 2.0).clamp(1e-7, 1 - 1e-7)
            H = -(probs * probs.log() + (1 - probs) * (1 - probs).log())
            entropy = (1.0 - H / math.log(2)).mean()

        # 4. Embedding alignment
        alignment = torch.tensor(0.0, device=triadic_proj.device)
        if align_weight > 0 and input_ids is not None:
            with torch.no_grad():
                embeds = self._arch['embed_layer'](input_ids).detach().float()

            if mode == 'mse':
                alignment = self._align_mse(triadic_proj, embeds)
            elif mode == 'rank':
                alignment = self._align_rank(triadic_proj, embeds)
            elif mode == 'infonce':
                alignment = self._align_infonce(triadic_proj, embeds)

        raw = diversity + contrastive
        if entropy_weight > 0:
            raw = raw + entropy_weight * entropy
        if align_weight > 0:
            raw = raw + align_weight * alignment

        return alpha * raw

    # ----------------------------------------------------------
    # Alignment loss implementations
    # ----------------------------------------------------------

    def _align_mse(self, proj, embeds, n_pairs=64):
        """MSE on absolute cosine similarity values. Best for from-scratch."""
        B, T, n_bits = proj.shape
        idx = torch.randint(0, T, (B, n_pairs, 2), device=proj.device)
        i, j = idx[:, :, 0], idx[:, :, 1]

        e_i = torch.gather(embeds, 1, i.unsqueeze(-1).expand(-1, -1, embeds.size(-1)))
        e_j = torch.gather(embeds, 1, j.unsqueeze(-1).expand(-1, -1, embeds.size(-1)))
        embed_sim = F.cosine_similarity(e_i, e_j, dim=-1)

        p_i = torch.gather(proj, 1, i.unsqueeze(-1).expand(-1, -1, n_bits))
        p_j = torch.gather(proj, 1, j.unsqueeze(-1).expand(-1, -1, n_bits))
        tri_sim = F.cosine_similarity(p_i, p_j, dim=-1)

        return F.mse_loss(tri_sim, embed_sim)

    def _align_rank(self, proj, embeds, n_anchors=32, n_cands=16, margin=0.1):
        """Margin ranking: preserve similarity ORDERING. Best for analogies."""
        B, T, n_bits = proj.shape
        d = embeds.size(-1)

        a_idx = torch.randint(0, T, (B, n_anchors), device=proj.device)
        c_idx = torch.randint(0, T, (B, n_anchors, n_cands), device=proj.device)

        a_e = torch.gather(embeds, 1, a_idx.unsqueeze(-1).expand(-1, -1, d))
        c_e = torch.gather(
            embeds, 1, c_idx.reshape(B, -1).unsqueeze(-1).expand(-1, -1, d)
        ).reshape(B, n_anchors, n_cands, d)

        e_sim = F.cosine_similarity(a_e.unsqueeze(2), c_e, dim=-1)
        pos_local = e_sim.argmax(dim=-1)
        neg_local = e_sim.argmin(dim=-1)

        pos_idx = torch.gather(c_idx, 2, pos_local.unsqueeze(-1)).squeeze(-1)
        neg_idx = torch.gather(c_idx, 2, neg_local.unsqueeze(-1)).squeeze(-1)

        a_p = torch.gather(proj, 1, a_idx.unsqueeze(-1).expand(-1, -1, n_bits))
        pos_p = torch.gather(proj, 1, pos_idx.unsqueeze(-1).expand(-1, -1, n_bits))
        neg_p = torch.gather(proj, 1, neg_idx.unsqueeze(-1).expand(-1, -1, n_bits))

        pos_sim = F.cosine_similarity(a_p, pos_p, dim=-1)
        neg_sim = F.cosine_similarity(a_p, neg_p, dim=-1)

        return F.relu(margin - (pos_sim - neg_sim)).mean()

    def _align_infonce(self, proj, embeds, n_anchors=32, temperature=0.1):
        """InfoNCE with embedding-mined positives. Best for pre-trained models."""
        B, T, n_bits = proj.shape
        d = embeds.size(-1)

        a_idx = torch.randint(0, T, (B, n_anchors), device=proj.device)
        p_idx = torch.randint(0, T, (B, n_anchors), device=proj.device)

        a_e = F.normalize(
            torch.gather(embeds, 1, a_idx.unsqueeze(-1).expand(-1, -1, d)), dim=-1)
        p_e = F.normalize(
            torch.gather(embeds, 1, p_idx.unsqueeze(-1).expand(-1, -1, d)), dim=-1)

        e_sim = torch.bmm(a_e, p_e.transpose(1, 2))
        pos_labels = e_sim.argmax(dim=-1)

        a_p = F.normalize(
            torch.gather(proj, 1, a_idx.unsqueeze(-1).expand(-1, -1, n_bits)), dim=-1)
        p_p = F.normalize(
            torch.gather(proj, 1, p_idx.unsqueeze(-1).expand(-1, -1, n_bits)), dim=-1)

        logits = torch.bmm(a_p, p_p.transpose(1, 2)) / temperature
        return F.cross_entropy(logits.reshape(-1, n_anchors), pos_labels.reshape(-1))

    # ----------------------------------------------------------
    # Encode: text -> prime signatures
    # ----------------------------------------------------------

    @torch.no_grad()
    def encode(
        self,
        texts: Union[str, List[str]],
        tokenizer=None,
    ) -> Dict[str, dict]:
        """
        Encode text(s) to prime-factor signatures.

        Args:
            texts: Single string or list of strings.
            tokenizer: HuggingFace tokenizer. If None, uses AutoTokenizer
                       matching the backbone model.

        Returns:
            Dict mapping each text to:
              - 'composite': int (prime product)
              - 'bits': List[int] (binary pattern)
              - 'projection': List[float] (raw tanh values)
              - 'n_active': int (number of active bits)
        """
        from .algebra import PrimeMapper

        if isinstance(texts, str):
            texts = [texts]

        if tokenizer is None:
            from transformers import AutoTokenizer
            config = getattr(self.backbone_model, 'config', None)
            model_name = getattr(config, '_name_or_path', 'gpt2')
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        device = next(self.parameters()).device
        mapper = PrimeMapper(self.n_bits)
        results = {}

        for text in texts:
            tokens = tokenizer(text, return_tensors='pt', truncation=True,
                               max_length=self.block_size).to(device)
            _, triadic_proj, _ = self(tokens['input_ids'],
                                     attention_mask=tokens.get('attention_mask'))
            # Average across all token positions
            avg_proj = triadic_proj.mean(dim=1).squeeze(0)  # (n_bits,)
            proj_list = avg_proj.cpu().tolist()

            composite = mapper.encode(proj_list)
            bits = mapper.get_bits(proj_list)

            results[text] = {
                'composite': composite,
                'bits': bits,
                'projection': proj_list,
                'n_active': sum(bits),
                'factors': mapper.explain(composite)['factors'],
            }

        return results

    @torch.no_grad()
    def compare(
        self,
        text_a: str,
        text_b: str,
        tokenizer=None,
    ) -> Dict:
        """
        Compare two texts using prime-factor algebra.

        Returns:
            Dict with 'similarity', 'shared_factors', 'only_in_a', 'only_in_b'.
        """
        from .algebra import TriadicValidator

        sigs = self.encode([text_a, text_b], tokenizer=tokenizer)
        a = sigs[text_a]['composite']
        b = sigs[text_b]['composite']

        gap = TriadicValidator.explain_gap(a, b)
        gap['similarity'] = TriadicValidator.similarity(a, b)
        gap['a_text'] = text_a
        gap['b_text'] = text_b
        return gap

    # ----------------------------------------------------------
    # Validation — Is training working?
    # ----------------------------------------------------------

    @torch.no_grad()
    def validate(
        self,
        tokenizer=None,
        word_groups: Optional[Dict[str, List[str]]] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Run diagnostic checks to verify triadic head training quality.

        Checks:
          1. Diversity — are signatures unique? (>75% = PASS)
          2. Active bits — are enough bits firing? (15-85% = PASS)
          3. Semantic ordering — related words more similar than unrelated? (gap > 0 = PASS)

        Also provides per-group breakdown so you can see which semantic
        categories the model handles well and which need more training.

        Args:
            tokenizer: HuggingFace tokenizer (auto-loaded if None).
            word_groups: Custom word groups to test, e.g.
                         {"animals": ["dog", "cat"], "colors": ["red", "blue"]}.
                         Uses built-in groups if None.
            verbose: Print results to console.

        Returns:
            Dict with check results, per-group breakdown, overall PASS/FAIL,
            config snapshot, and all signatures.
        """
        from .algebra import TriadicValidator as TV

        if word_groups is None:
            word_groups = {
                'royalty': ['king', 'queen', 'prince', 'throne'],
                'animals': ['dog', 'cat', 'fish', 'bird'],
                'emotions': ['happy', 'sad', 'angry', 'love'],
                'food': ['bread', 'water', 'rice', 'fruit'],
            }

        all_words = [w for group in word_groups.values() for w in group]
        sigs = self.encode(all_words, tokenizer=tokenizer)

        # --- Check 1: Diversity ---
        composites = [sigs[w]['composite'] for w in all_words]
        unique_ratio = len(set(composites)) / len(composites)

        # --- Check 2: Active bits ---
        active_counts = [sigs[w]['n_active'] for w in all_words]
        avg_active = sum(active_counts) / len(active_counts)
        active_frac = avg_active / self.n_bits

        # --- Check 3: Semantic ordering (global + per-group) ---
        group_names = list(word_groups.keys())
        groups = list(word_groups.values())
        intra_sims, inter_sims = [], []
        group_details = {}

        for gname, group in word_groups.items():
            g_intra = []
            for i, w1 in enumerate(group):
                for w2 in group[i + 1:]:
                    sim = TV.similarity(sigs[w1]['composite'], sigs[w2]['composite'])
                    g_intra.append(sim)
                    intra_sims.append(sim)
            # Inter: this group vs all other groups
            g_inter = []
            for other_name, other_group in word_groups.items():
                if other_name == gname:
                    continue
                for w1 in group:
                    for w2 in other_group:
                        sim = TV.similarity(sigs[w1]['composite'], sigs[w2]['composite'])
                        g_inter.append(sim)
                        inter_sims.append(sim)

            avg_gi = sum(g_intra) / len(g_intra) if g_intra else 0
            avg_ge = sum(g_inter) / len(g_inter) if g_inter else 0
            group_details[gname] = {
                'words': group,
                'intra_sim': avg_gi,
                'inter_sim': avg_ge,
                'gap': avg_gi - avg_ge,
                'pass': (avg_gi - avg_ge) > 0,
            }

        # Deduplicate inter_sims (each pair counted from both group perspectives)
        # Recalculate cleanly
        inter_sims_clean = []
        for gi, g1 in enumerate(groups):
            for g2 in groups[gi + 1:]:
                for w1 in g1:
                    for w2 in g2:
                        inter_sims_clean.append(
                            TV.similarity(sigs[w1]['composite'], sigs[w2]['composite'])
                        )

        avg_intra = sum(intra_sims) / len(intra_sims) if intra_sims else 0
        avg_inter = sum(inter_sims_clean) / len(inter_sims_clean) if inter_sims_clean else 0
        semantic_gap = avg_intra - avg_inter

        # --- Build results ---
        checks = {
            'diversity': {
                'value': unique_ratio,
                'pass': unique_ratio > 0.75,
                'detail': f"{len(set(composites))}/{len(composites)} unique signatures ({unique_ratio:.0%})",
            },
            'active_bits': {
                'value': active_frac,
                'pass': 0.15 < active_frac < 0.85,
                'detail': f"{avg_active:.1f}/{self.n_bits} bits active on avg ({active_frac:.0%})",
            },
            'semantic_ordering': {
                'intra': avg_intra,
                'inter': avg_inter,
                'gap': semantic_gap,
                'pass': semantic_gap > 0,
                'detail': f"within-group {avg_intra:.1%} vs between-group {avg_inter:.1%} (gap {semantic_gap:+.1%})",
            },
        }

        overall = all(c['pass'] for c in checks.values())

        if verbose:
            print(f"\n{'=' * 60}")
            print("  TRIADIC HEAD — VALIDATION REPORT")
            print(f"{'=' * 60}")
            print(f"  Config: {self.n_bits} bits | align_mode={self.align_mode} | "
                  f"{self.triadic_params():,} triadic params")
            print(f"{'-' * 60}")

            for name, check in checks.items():
                status = "PASS" if check['pass'] else "FAIL"
                print(f"  [{status}] {name}: {check['detail']}")

            # Per-group breakdown
            print(f"\n  PER-GROUP BREAKDOWN:")
            for gname, gd in group_details.items():
                status = "PASS" if gd['pass'] else "FAIL"
                print(f"    [{status}] {gname}: intra {gd['intra_sim']:.0%} vs inter {gd['inter_sim']:.0%} "
                      f"(gap {gd['gap']:+.1%}) — {gd['words']}")

            # Per-word active bits
            print(f"\n  PER-WORD ACTIVE BITS:")
            for w in all_words:
                print(f"    {w:>12}: {sigs[w]['n_active']:2d}/{self.n_bits} bits active")

            print(f"{'-' * 60}")
            if overall:
                print("  RESULT: PASS — Triadic head is producing meaningful signatures.")
            else:
                failed = [n for n, c in checks.items() if not c['pass']]
                print(f"  RESULT: FAIL — Issues detected: {', '.join(failed)}")
                if 'diversity' in failed:
                    print("    -> Signatures are too similar. Train longer or increase entropy_weight.")
                if 'active_bits' in failed:
                    if active_frac <= 0.15:
                        print("    -> Too few bits active. Increase entropy_weight to activate dead bits.")
                    else:
                        print("    -> Too many bits active. Decrease alpha or increase training steps.")
                if 'semantic_ordering' in failed:
                    print("    -> Related words aren't more similar than unrelated.")
                    print("       Try: increase align_weight, or use align_mode='infonce' for pre-trained models.")
                failing_groups = [g for g, d in group_details.items() if not d['pass']]
                if failing_groups:
                    print(f"    -> Weak groups: {', '.join(failing_groups)} — consider adding more training data for these domains.")
            print(f"{'=' * 60}")

        return {
            'checks': checks,
            'overall_pass': overall,
            'group_details': group_details,
            'config': self.config(),
            'signatures': sigs,
        }

    # ----------------------------------------------------------
    # Explore — Discover relationships between words
    # ----------------------------------------------------------

    @torch.no_grad()
    def explore(
        self,
        words: List[str],
        tokenizer=None,
        top_k: int = 0,
        threshold: Optional[float] = None,
        show_factors: bool = False,
        verbose: bool = True,
    ) -> Dict:
        """
        Full audit of relationships between words with similarity matrix.

        Args:
            words: List of words/phrases to compare.
            tokenizer: HuggingFace tokenizer (auto-loaded if None).
            top_k: Show only top-K and bottom-K pairs (0 = show ALL pairs).
            threshold: If set, flag pairs above this similarity for review.
            show_factors: Show shared/unique prime factors for every pair.
            verbose: Print results to console.

        Returns:
            Dict with similarity matrix, ranked pairs, signatures, and flagged pairs.
        """
        from .algebra import TriadicValidator as TV

        sigs = self.encode(words, tokenizer=tokenizer)
        n = len(words)

        # Build similarity matrix
        matrix = [[0.0] * n for _ in range(n)]
        pairs = []

        for i in range(n):
            matrix[i][i] = 1.0
            for j in range(i + 1, n):
                sim = TV.similarity(
                    sigs[words[i]]['composite'],
                    sigs[words[j]]['composite'],
                )
                gap_info = TV.explain_gap(
                    sigs[words[i]]['composite'],
                    sigs[words[j]]['composite'],
                )
                matrix[i][j] = sim
                matrix[j][i] = sim
                pairs.append({
                    'similarity': sim,
                    'word_a': words[i],
                    'word_b': words[j],
                    'shared_factors': gap_info['shared_factors'],
                    'only_a_factors': gap_info['only_in_a_factors'],
                    'only_b_factors': gap_info['only_in_b_factors'],
                    'n_shared': len(gap_info['shared_factors']),
                    'n_only_a': len(gap_info['only_in_a_factors']),
                    'n_only_b': len(gap_info['only_in_b_factors']),
                })

        pairs.sort(key=lambda p: p['similarity'], reverse=True)

        # Flag pairs above threshold
        flagged = []
        if threshold is not None:
            flagged = [p for p in pairs if p['similarity'] >= threshold]

        if verbose:
            col_w = min(max(max(len(w) for w in words), 6), 10)

            print(f"\n{'=' * 60}")
            print("  SIMILARITY MATRIX")
            print(f"{'=' * 60}")

            # Header
            header = " " * (col_w + 2) + "".join(f"{w[:col_w]:>{col_w + 1}}" for w in words)
            print(header)

            for i, w in enumerate(words):
                row_vals = []
                for j in range(n):
                    if i == j:
                        row_vals.append(f"{'---':>{col_w + 1}}")
                    else:
                        row_vals.append(f"{matrix[i][j]:>{col_w}.0%} ")
                print(f"{w[:col_w]:>{col_w}}  {''.join(row_vals)}")

            # Pairs listing
            show_pairs = pairs if top_k == 0 else (pairs[:top_k] + pairs[-top_k:])
            if top_k > 0 and len(pairs) > top_k * 2:
                print(f"\n  TOP {top_k} most similar:")
                for p in pairs[:top_k]:
                    line = f"    {p['word_a']} <-> {p['word_b']}: {p['similarity']:.0%} ({p['n_shared']} shared)"
                    if show_factors:
                        line += f" shared={p['shared_factors']}"
                    print(line)
                print(f"\n  TOP {top_k} least similar:")
                for p in pairs[-top_k:]:
                    line = f"    {p['word_a']} <-> {p['word_b']}: {p['similarity']:.0%}"
                    print(line)
            else:
                print(f"\n  ALL PAIRS (ranked by similarity):")
                for p in pairs:
                    line = f"    {p['word_a']:>12} <-> {p['word_b']:<12}: {p['similarity']:.0%} ({p['n_shared']} shared, {p['n_only_a']} only-a, {p['n_only_b']} only-b)"
                    if show_factors:
                        line += f"\n{'':>42}shared={p['shared_factors']}"
                        if p['only_a_factors']:
                            line += f"\n{'':>42}only-{p['word_a']}={p['only_a_factors']}"
                        if p['only_b_factors']:
                            line += f"\n{'':>42}only-{p['word_b']}={p['only_b_factors']}"
                    print(line)

            # Flagged pairs
            if threshold is not None:
                print(f"\n  FLAGGED (similarity >= {threshold:.0%}): {len(flagged)} pairs")
                if flagged:
                    for p in flagged:
                        print(f"    {p['word_a']} <-> {p['word_b']}: {p['similarity']:.0%}")
                else:
                    print("    (none)")

            # Per-word signature card
            print(f"\n  PER-WORD SIGNATURES:")
            for i, w in enumerate(words):
                sims_for_w = [matrix[i][j] for j in range(n) if i != j]
                avg = sum(sims_for_w) / len(sims_for_w)
                most_sim_j = max((j for j in range(n) if j != i), key=lambda j: matrix[i][j])
                least_sim_j = min((j for j in range(n) if j != i), key=lambda j: matrix[i][j])
                primes = sigs[w]['factors']
                print(f"    {w:>12}: {sigs[w]['n_active']} bits | avg sim {avg:.0%} "
                      f"| closest: {words[most_sim_j]} ({matrix[i][most_sim_j]:.0%}) "
                      f"| furthest: {words[least_sim_j]} ({matrix[i][least_sim_j]:.0%})")
                if show_factors:
                    print(f"{'':>16}primes={primes}")

            # Factor index: which words share each prime?
            if show_factors:
                from .algebra import nth_prime as _nth_prime
                factor_index = {}  # prime -> list of words
                for w in words:
                    for p in sigs[w]['factors']:
                        factor_index.setdefault(p, []).append(w)

                # Sort by number of words sharing the factor (most shared first)
                sorted_factors = sorted(factor_index.items(),
                                        key=lambda x: len(x[1]), reverse=True)

                print(f"\n  FACTOR INDEX (what each prime means):")
                print(f"    {'prime':>8}  {'bit':>3}  {'words':>5}  shared by")
                print(f"    {'-'*8}  {'-'*3}  {'-'*5}  {'-'*30}")
                for prime, sharing_words in sorted_factors:
                    # Find bit index for this prime
                    bit_idx = next(
                        (i for i in range(self.n_bits) if _nth_prime(i + 1) == prime),
                        -1
                    )
                    count = len(sharing_words)
                    label = ', '.join(sharing_words)
                    if count == len(words):
                        label += '  [UNIVERSAL]'
                    elif count == 1:
                        label += '  [UNIQUE]'
                    print(f"    {prime:>8}  {bit_idx:>3}  {count:>5}  {label}")

            print(f"{'=' * 60}")

        return {
            'matrix': matrix,
            'words': words,
            'pairs_ranked': pairs,
            'signatures': sigs,
            'flagged': flagged,
        }

    # ----------------------------------------------------------
    # Generation (pass-through to backbone)
    # ----------------------------------------------------------

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=0.7, top_k=50):
        """Autoregressive text generation."""
        for _ in range(max_new_tokens):
            idx = input_ids[:, -self.block_size:]
            logits, _, _ = self(idx)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids

    # ----------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------

    def num_params(self, trainable_only=False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def triadic_params(self) -> int:
        """Number of parameters in the triadic head only."""
        return sum(p.numel() for p in self.triadic_head.parameters())

    def __repr__(self):
        return (
            f"TriadicWrapper(\n"
            f"  backbone={type(self.backbone_model).__name__},\n"
            f"  n_embd={self.n_embd}, n_bits={self.n_bits},\n"
            f"  align_mode='{self.align_mode}',\n"
            f"  triadic_params={self.triadic_params():,},\n"
            f"  total_params={self.num_params():,}\n"
            f")"
        )
