# triadic-head

Drop-in triadic projection head for any HuggingFace transformer. Adds interpretable prime-factor semantic signatures at zero language cost.

## What it does

Adds a single linear layer (49K params for GPT-2) that maps hidden states to discrete **prime-factor signatures**. Each concept becomes a composite integer like `Φ(king) = 2 × 3 × 5 × 7` where each prime represents an active semantic feature.

This enables **exact algebraic operations** impossible with cosine similarity:

| Operation | Cosine Similarity | Prime Algebra |
|-----------|:-:|:-:|
| "Does A contain all features of B?" | Approximate | `Φ(A) % Φ(B) == 0` |
| "What features do A and B share?" | Not possible | `GCD(Φ(A), Φ(B))` |
| "Combine features of A and B" | Not possible | `LCM(Φ(A), Φ(B))` |
| "A is to B as C is to ?" | Approximate | Exact factor transfer |

## Install

```bash
pip install triadic-head
```

## Quick start

```python
from triadic_head import TriadicWrapper

# Wrap any HuggingFace model
model = TriadicWrapper("gpt2", n_bits=64, align_mode="infonce")

# Train (see examples/train_gpt2.py for full loop)
model.freeze_backbone()
# ... phase 1: train triadic head only ...
model.unfreeze_last_n(2)
# ... phase 2: joint optimization ...

# Encode concepts to prime signatures
sigs = model.encode(["king", "queen", "dog"])

# Compare
result = model.compare("king", "queen")
# {'similarity': 0.89, 'shared_factors': [2, 3, 5, ...], ...}
```

## Training API

```python
# Forward pass returns (logits, triadic_proj, lang_loss)
logits, triadic_proj, lang_loss = model(input_ids, labels=input_ids)

# Triadic loss (4 components: diversity + contrastive + entropy + alignment)
tri_loss = model.triadic_loss(
    triadic_proj,
    input_ids=input_ids,
    alpha=0.05,           # triadic weight (DO NOT exceed 0.10)
    entropy_weight=1.0,   # prevent dead bits
    align_weight=5.0,     # transfer semantic structure from embeddings
    align_mode="infonce", # "mse" | "rank" | "infonce"
)

total_loss = lang_loss + tri_loss
total_loss.backward()
```

### Alignment modes

| Mode | Best for | Why |
|------|----------|-----|
| `infonce` | Pre-trained models (GPT-2, LLaMA, ...) | Mines positive/negative pairs from rich embeddings |
| `mse` | From-scratch training | Dense local gradients work with weak embeddings |
| `rank` | Best analogy accuracy | Preserves similarity ordering, not absolute values |

## Training guide — How long to train

The number of training steps directly determines result quality. Short runs are useful for smoke-testing the pipeline, but **will NOT produce reliable semantic signatures**. The triadic head needs enough steps to learn real word relationships beyond statistical noise.

| Level | Steps | Time (GPT-2, 1 GPU) | What to expect |
|-------|------:|-----:|----------------|
| Smoke test | 5,000 | ~5 min | Pipeline works, results are mostly noise |
| Minimum viable | 20,000 | ~20 min | Basic semantic ordering emerges |
| Good quality | 50,000 | ~50 min | Reliable word relationships, gap well above random |
| Production | 100,000+ | ~2 hours | Publish-ready signatures |

**Important**: Larger models (LLaMA, Mistral, etc.) need proportionally more steps. The `validate()` method includes a **random baseline** — it generates random bit patterns and measures what gap you'd get by pure chance. If your model's gap is close to the random baseline, you need more training.

```bash
# Quick smoke test (verify the pipeline works)
python examples/train_gpt2.py --data corpus.txt --phase1-steps 1000 --phase2-steps 4000

# Good quality training
python examples/train_gpt2.py --data corpus.txt --phase1-steps 10000 --phase2-steps 40000

# Production quality
python examples/train_gpt2.py --data corpus.txt --phase1-steps 20000 --phase2-steps 80000
```

## Validation — Did training work?

```python
# Automatic diagnostic: runs standard word groups and checks
# diversity, active bits, and semantic ordering
report = model.validate()

# Output:
# ============================================================
#   TRIADIC HEAD — VALIDATION REPORT
# ============================================================
#   [PASS] diversity: 16/16 unique signatures (100%)
#   [PASS] active_bits: 35.2/64 bits active on avg (55%)
#   [PASS] semantic_ordering: within-group 72% vs between-group 58% (gap +14%)
#   [PASS] random_baseline: model gap +14% vs random baseline +0.3% (signal +13.7%)
# ------------------------------------------------------------
#   RESULT: PASS — Triadic head is producing meaningful signatures.
#   Signal above random: +13.7%
# ============================================================

# Use your own domain-specific word groups:
report = model.validate(word_groups={
    "medical": ["heart", "lung", "brain", "kidney"],
    "legal": ["court", "judge", "law", "trial"],
})
```

## Explore — Discover relationships

```python
# See how any set of words relate to each other
model.explore(["king", "queen", "prince", "dog", "cat", "happy", "sad"])

# Output:
#   SIMILARITY MATRIX
#            king  queen prince    dog    cat  happy    sad
#   king      ---   78%   72%   45%   43%   38%   35%
#   queen    78%    ---   69%   41%   44%   42%   37%
#   ...
#
#   TOP 3 most similar:
#     king <-> queen: 78% (12 shared factors)
#     king <-> prince: 72% (10 shared factors)
#     dog <-> cat: 68% (9 shared factors)
```

## Algebra API

```python
from triadic_head import PrimeMapper, TriadicValidator

mapper = PrimeMapper(n_bits=64)
sig = mapper.encode(projection_values)  # -> composite integer

# Subsumption: does A contain all features of B?
TriadicValidator.subsumes(sig_a, sig_b)  # -> bool

# Composition: combine features
TriadicValidator.compose(sig_a, sig_b)  # -> LCM

# Gap analysis: exactly which features differ?
TriadicValidator.explain_gap(sig_a, sig_b)
# {'shared_factors': [2, 5], 'only_in_a_factors': [3], 'only_in_b_factors': [7]}

# Similarity: Jaccard over prime factor sets
TriadicValidator.similarity(sig_a, sig_b)  # -> float [0, 1]

# Analogy: A is to B as C is to ?
TriadicValidator.analogy(sig_a, sig_b, sig_c)  # -> target composite
```

## Supported models

Works with any HuggingFace `AutoModelForCausalLM`:
- GPT-2 (all sizes)
- LLaMA / LLaMA-2 / LLaMA-3
- Mistral / Mixtral
- Phi-2 / Phi-3
- Qwen / Qwen2
- GPT-Neo / GPT-J
- OPT
- Falcon

## Key findings from research

- **Zero language cost**: triadic head adds no measurable degradation to language quality
- **49K params**: for GPT-2 (768D, 64 bits) — negligible overhead
- **Semantic ordering emerges at scale**: related concepts become more similar than unrelated ones only above ~40M parameters
- **InfoNCE + GPT-2 closes 48% of gap** to post-hoc projection (Triadic-Neurosymbolic-Engine)
- **Sharp Pareto cliff at alpha > 0.05**: do not exceed this value

## Citation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19206545.svg)](https://doi.org/10.5281/zenodo.19206545)

```bibtex
@article{ornelas2026triadic,
  title={End-to-End Prime Factorization in a Generative Language Model:
         Emergent Algebraic Semantics from Joint Training},
  author={Ornelas Brand, J. Arturo},
  year={2026},
  doi={10.5281/zenodo.19206545},
  publisher={Zenodo}
}
```

## License

Business Source License 1.1 (BUSL-1.1). Free for individuals, academics, and non-profits. See [LICENSE](./LICENSE).
