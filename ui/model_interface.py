"""
ModelInterface — Unified abstraction for TriadicGPT and HuggingFace models.

Provides consistent encode/compare/explore/validate/chat methods regardless
of the underlying model backend.
"""
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "triadic-head"))

from src.triadic import PrimeMapper, TriadicValidator, prime_factors


# ---------------------------------------------------------------------------
# Encoding helper
# ---------------------------------------------------------------------------

def _encode_with_native(model, tokenizer, text: str, device, mapper: PrimeMapper, block_size: int) -> dict:
    """Encode text using native TriadicGPT model."""
    ids = tokenizer.encode(text, add_special=False)
    if not ids:
        ids = [tokenizer.special_tokens.get('<UNK>', 3)]
    ids = ids[:block_size]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        _, triadic_proj, _ = model(x)
    proj = triadic_proj[0].mean(dim=0).cpu().numpy()
    bits = mapper.get_bits(proj)
    composite = mapper.map(proj)
    return {
        'composite': composite,
        'bits': bits,
        'projection': proj.tolist(),
        'n_active': int(sum(bits)),
        'factors': prime_factors(composite),
        'text': text,
    }


def _encode_with_hf(wrapper, text: str) -> dict:
    """Encode text using HuggingFace TriadicWrapper."""
    results = wrapper.encode(text)
    enc = results[text] if isinstance(results, dict) else results
    return {
        'composite': enc['composite'],
        'bits': enc['bits'],
        'projection': enc['projection'],
        'n_active': enc['n_active'],
        'factors': prime_factors(enc['composite']),
        'text': text,
    }


# ---------------------------------------------------------------------------
# ModelInterface
# ---------------------------------------------------------------------------

class ModelInterface:
    """
    Unified interface for both native TriadicGPT and HF TriadicWrapper.
    Created by ModelPanel after loading; shared across all tabs.
    """

    BACKENDS = ('native', 'hf')

    def __init__(self, backend: str, model, tokenizer, mapper: PrimeMapper,
                 validator: TriadicValidator, device, config=None, hf_wrapper=None):
        self.backend = backend          # 'native' | 'hf'
        self.model = model
        self.tokenizer = tokenizer
        self.mapper = mapper
        self.validator = validator
        self.device = device
        self.config = config            # TriadicGPTConfig | None
        self.hf_wrapper = hf_wrapper    # TriadicWrapper | None
        self._session_vocab: dict[str, dict] = {}  # cache of encoded words

    # ------------------------------------------------------------------
    # Info properties
    # ------------------------------------------------------------------

    @property
    def n_bits(self) -> int:
        if self.config:
            return self.config.n_triadic_bits
        if self.hf_wrapper:
            return self.hf_wrapper.n_bits
        return 64

    @property
    def info_str(self) -> str:
        if self.config:
            cfg = self.config
            return f"{cfg.n_layer}L/{cfg.n_embd}D/{cfg.n_head}H/{cfg.n_triadic_bits}bits"
        if self.hf_wrapper:
            return f"HF/{self.n_bits}bits"
        return "Unknown config"

    @property
    def device_str(self) -> str:
        return str(self.device).upper()

    @property
    def param_count(self) -> int:
        if hasattr(self.model, 'num_params'):
            return self.model.num_params()
        if self.model is not None:
            return sum(p.numel() for p in self.model.parameters())
        return 0

    # ------------------------------------------------------------------
    # Core encoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> dict:
        """Encode text to prime signature dict."""
        if text in self._session_vocab:
            return self._session_vocab[text]

        if self.backend == 'native':
            block_size = self.config.block_size if self.config else 256
            result = _encode_with_native(
                self.model, self.tokenizer, text,
                self.device, self.mapper, block_size
            )
        else:
            result = _encode_with_hf(self.hf_wrapper, text)

        self._session_vocab[text] = result
        return result

    # ------------------------------------------------------------------
    # Algebraic operations
    # ------------------------------------------------------------------

    def compare(self, text_a: str, text_b: str) -> dict:
        """Compare two concepts, return similarity + factor analysis."""
        enc_a = self.encode(text_a)
        enc_b = self.encode(text_b)
        sim = self.validator.similarity(enc_a['composite'], enc_b['composite'])
        gap = self.validator.explain_gap(enc_a['composite'], enc_b['composite'])
        return {
            'similarity': sim,
            'shared_factors': gap['shared_factors'],
            'only_a_factors': gap['only_in_a_factors'],
            'only_b_factors': gap['only_in_b_factors'],
            'a_subsumes_b': self.validator.subsumes(enc_a['composite'], enc_b['composite']),
            'b_subsumes_a': self.validator.subsumes(enc_b['composite'], enc_a['composite']),
            'composition': self.validator.compose(enc_a['composite'], enc_b['composite']),
            'enc_a': enc_a,
            'enc_b': enc_b,
        }

    def explore(self, words: list[str]) -> dict:
        """Compute pairwise similarity matrix + ranked pairs for N words."""
        signatures = {w: self.encode(w) for w in words}
        n = len(words)
        matrix = [[0.0] * n for _ in range(n)]
        pairs = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    sim = self.validator.similarity(
                        signatures[words[i]]['composite'],
                        signatures[words[j]]['composite']
                    )
                    matrix[i][j] = sim
                    if i < j:
                        gap = self.validator.explain_gap(
                            signatures[words[i]]['composite'],
                            signatures[words[j]]['composite']
                        )
                        pairs.append({
                            'word_a': words[i],
                            'word_b': words[j],
                            'similarity': sim,
                            'shared_factors': gap['shared_factors'],
                            'only_a_factors': gap['only_in_a_factors'],
                            'only_b_factors': gap['only_in_b_factors'],
                            'n_shared': len(gap['shared_factors']),
                        })
        pairs.sort(key=lambda x: x['similarity'], reverse=True)
        return {'matrix': matrix, 'words': words, 'pairs': pairs, 'signatures': signatures}

    def analogy(self, word_a: str, word_b: str, word_c: str,
                vocab_pool: Optional[list[str]] = None) -> dict:
        """
        Compute A:B::C:? and rank vocab_pool by similarity to target prime.
        Returns transform, target_prime, and ranked matches.
        """
        enc_a = self.encode(word_a)
        enc_b = self.encode(word_b)
        enc_c = self.encode(word_c)

        target = self.validator.analogy(
            enc_a['composite'], enc_b['composite'], enc_c['composite']
        )
        gap_ab = self.validator.explain_gap(enc_a['composite'], enc_b['composite'])

        # Build pool to search
        pool = vocab_pool or self._get_default_vocab()
        pool = [w for w in pool if w not in (word_a, word_b, word_c)]

        matches = []
        pool_sims = []
        for word in pool:
            enc = self.encode(word)
            sim = self.validator.similarity(target, enc['composite'])
            matches.append({'word': word, 'similarity': sim, 'enc': enc})
            pool_sims.append(sim)

        matches.sort(key=lambda x: x['similarity'], reverse=True)
        median_sim = float(np.median(pool_sims)) if pool_sims else 0.5

        return {
            'target_prime': target,
            'target_factors': prime_factors(target),
            'transform_added': gap_ab['only_in_b_factors'],    # B has but A doesn't
            'transform_removed': gap_ab['only_in_a_factors'],  # A has but B doesn't
            'matches': matches[:10],
            'median_sim': median_sim,
            'enc_a': enc_a,
            'enc_b': enc_b,
            'enc_c': enc_c,
        }

    def validate(self, word_groups: Optional[dict[str, list[str]]] = None) -> dict:
        """Run validation checks: diversity, active bits, semantic ordering."""
        if word_groups is None:
            word_groups = {
                'royalty': ['king', 'queen', 'prince', 'throne'],
                'animals': ['dog', 'cat', 'fish', 'bird'],
                'emotions': ['happy', 'sad', 'angry', 'love'],
                'nature': ['sun', 'moon', 'fire', 'water'],
            }

        all_words = list({w for ws in word_groups.values() for w in ws})
        signatures = {w: self.encode(w) for w in all_words}
        primes = [signatures[w]['composite'] for w in all_words]

        # Check 1: Signature diversity
        unique_sigs = len(set(primes))
        diversity_ratio = unique_sigs / max(len(primes), 1)
        diversity_pass = diversity_ratio >= 0.75

        # Check 2: Active bits (15–85%)
        avg_active = sum(signatures[w]['n_active'] for w in all_words) / max(len(all_words) * self.n_bits, 1)
        active_pass = 0.15 <= avg_active <= 0.85

        # Check 3: Semantic ordering per group
        group_results = {}
        all_gaps = []
        for group_name, words in word_groups.items():
            valid_words = [w for w in words if w in signatures]
            if len(valid_words) < 2:
                continue
            group_primes = [signatures[w]['composite'] for w in valid_words]
            other_words = [w for g, ws in word_groups.items() for w in ws
                           if g != group_name and w in signatures]

            intra_sims = [
                self.validator.similarity(group_primes[i], group_primes[j])
                for i in range(len(group_primes))
                for j in range(i + 1, len(group_primes))
            ]
            inter_sims = [
                self.validator.similarity(signatures[wi]['composite'], signatures[wo]['composite'])
                for wi in valid_words for wo in other_words
            ]
            intra = float(np.mean(intra_sims)) if intra_sims else 0.0
            inter = float(np.mean(inter_sims)) if inter_sims else 0.0
            gap = intra - inter
            all_gaps.append(gap)
            group_results[group_name] = {
                'words': valid_words,
                'intra_sim': intra,
                'inter_sim': inter,
                'gap': gap,
                'pass': gap > 0,
            }

        mean_gap = float(np.mean(all_gaps)) if all_gaps else 0.0
        ordering_pass = mean_gap > 0

        return {
            'checks': {
                'diversity': {
                    'value': diversity_ratio,
                    'detail': f'{unique_sigs}/{len(primes)} unique signatures ({diversity_ratio:.0%})',
                    'pass': diversity_pass,
                },
                'active_bits': {
                    'value': avg_active,
                    'detail': f'{avg_active * self.n_bits:.1f}/{self.n_bits} bits active on avg ({avg_active:.0%})',
                    'pass': active_pass,
                },
                'semantic_ordering': {
                    'gap': mean_gap,
                    'detail': f'mean gap {mean_gap:+.3f} across {len(all_gaps)} groups',
                    'pass': ordering_pass,
                },
            },
            'overall_pass': diversity_pass and active_pass and ordering_pass,
            'group_details': group_results,
            'n_concepts': len(all_words),
            'unique_signatures': unique_sigs,
        }

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> dict:
        """
        Generate response + triadic analysis.
        Returns response text, prompt/response prime signatures, similarity.
        """
        if self.backend != 'native':
            raise NotImplementedError("Chat is only supported for native TriadicGPT backend.")

        from src.chat import generate_response
        response_text, h_prompt, h_resp = generate_response(
            self.model, self.tokenizer, prompt, self.device,
            max_tokens=max_tokens, temperature=temperature
        )

        with torch.no_grad():
            p_proj = torch.tanh(self.model.triadic_head(h_prompt)).squeeze().cpu().numpy()
            r_proj = torch.tanh(self.model.triadic_head(h_resp)).squeeze().cpu().numpy()

        prime_p = self.mapper.map(p_proj)
        prime_r = self.mapper.map(r_proj)
        bits_p = self.mapper.get_bits(p_proj)
        bits_r = self.mapper.get_bits(r_proj)

        gap = self.validator.explain_gap(prime_r, prime_p)
        sim = self.validator.similarity(prime_r, prime_p)

        return {
            'response': response_text,
            'prompt_prime': prime_p,
            'prompt_bits': bits_p,
            'prompt_projection': p_proj.tolist(),
            'response_prime': prime_r,
            'response_bits': bits_r,
            'response_projection': r_proj.tolist(),
            'similarity': sim,
            'resp_subsumes_prompt': self.validator.subsumes(prime_r, prime_p),
            'shared_factors': gap['shared_factors'],
            'resp_extra_factors': gap['only_in_a_factors'],   # response has but prompt doesn't
            'prompt_extra_factors': gap['only_in_b_factors'],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_default_vocab(self) -> list[str]:
        """Return a default vocabulary pool for analogy search."""
        vocab_path = PROJECT_ROOT / 'data' / 'core_concepts.txt'
        if vocab_path.exists():
            with open(vocab_path) as f:
                return [line.strip() for line in f if line.strip()]
        # Fallback built-in
        return [
            'king', 'queen', 'man', 'woman', 'prince', 'princess', 'boy', 'girl',
            'father', 'mother', 'brother', 'sister', 'son', 'daughter',
            'dog', 'cat', 'fish', 'bird', 'horse', 'cow', 'sheep', 'lion',
            'happy', 'sad', 'angry', 'love', 'hate', 'hope', 'fear', 'peace', 'war',
            'fire', 'water', 'sun', 'moon', 'earth', 'wind', 'rain', 'snow',
            'big', 'small', 'fast', 'slow', 'old', 'young', 'hot', 'cold',
            'doctor', 'teacher', 'hospital', 'school', 'castle', 'church', 'garden',
            'run', 'walk', 'swim', 'fly', 'jump', 'fall', 'sleep', 'eat',
            'red', 'blue', 'green', 'white', 'black', 'gold', 'silver',
            'friend', 'enemy', 'morning', 'night', 'summer', 'winter',
            'book', 'tree', 'flower', 'cake', 'bread', 'milk', 'apple',
            'table', 'chair', 'door', 'window', 'car', 'boat', 'bridge',
            'music', 'dance', 'song', 'game', 'dream', 'cloud', 'star',
        ]

    def add_to_vocab(self, word: str) -> None:
        """Pre-encode and cache a word in the session vocabulary."""
        if word and word not in self._session_vocab:
            self.encode(word)
