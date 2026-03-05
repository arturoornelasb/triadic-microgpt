---
description: Overview of the Triadic MicroGPT project and its neurosymbolic architecture
---

# Triadic MicroGPT — Project Overview

## What This Is

A neurosymbolic language model that combines standard GPT text generation with a **triadic prime-factor semantic encoding system**. Every token gets both:
1. A **language prediction** (standard next-token)
2. A **prime signature** (semantic fingerprint using prime number products)

## The Triadic System

```
Hidden State → Triadic Head → tanh(Wx) → bits [-1, +1, +1, -1, ...]
                                            │
                                    Positive bits activate primes:
                                    bit 0 → 2, bit 1 → 3, bit 2 → 5, ...
                                            │
                                    Product = semantic signature
                                    "King" → 2 × 3 × 5 × 11 = 330
                                    "Queen" → 2 × 5 × 7 × 11 = 770
                                    GCD(330, 770) = shared meaning
```

## Key Concepts

- **PrimeMapper** (`src/triadic.py`): converts tanh projections → prime products
- **TriadicValidator**: computes GCD-based similarity between prime signatures
- **Triadic Loss**: 3 objectives — coherence (adjacent agree), diversity (use all bits), contrastive (different docs differ)
- **Experiment Log** (`experiment_log.md`): tracks all training runs with metrics

## Current State (as of Run 7)

- **Model**: 8L / 384D / 8H / 48 triadic bits / ~16M params
- **Training**: 50K TinyStories, 20K steps, RTX 5060 Ti
- **Loss**: 1.65 (pretrain), 0.78 (fine-tune for chat)
- **Tokenizer**: HuggingFace `tokenizers` (Rust, 1000× faster than Python BPE)
- **Pipeline**: Full train+eval cycle in <15 min

## Repos in the Ecosystem

| Repo | Purpose |
|------|---------|
| `triadic-microgpt` | This repo: the model, training, evaluation |
| `Triadic-Neurosymbolic-Engine` | Core neurosymbolic engine library |
| `neurosym-client` | Client/SDK for the engine |
| `triadic-cloud` | Cloud deployment infrastructure |

## Workflow

See `.agents/workflows/training.md` for the step-by-step training workflow.
