---
description: Overview of the Triadic MicroGPT project — bitwise neurosymbolic architecture
---

# Triadic MicroGPT — Project Overview

## What This Is

A neurosymbolic language model that combines standard GPT text generation with a **triadic bitwise semantic encoding system**. Every token gets both:
1. A **language prediction** (standard next-token)
2. A **bit signature** (63-bit semantic fingerprint, verified via O(1) bitwise algebra)

The paper demonstrates that this system scales knowledge through a human-in-the-loop discovery cycle: train → discover bit semantics → correct → expand anchors → retrain.

## The Bitwise System

```
Hidden State → Triadic Head → iFSQ(Wx) → 63 bits [-1, 0, +1]
                                            │
                                   Bitmask = semantic signature
                                   "King"  → 0b...101101  (bits 0,2,3,5 active)
                                   "Queen" → 0b...101011  (bits 0,1,3,5 active)
                                   King & Queen = shared meaning
                                   King ^ Queen = what differs
                                            │
                                   BitwiseValidator: O(1) algebra
                                   - subsumes:  (A & B) == B
                                   - compose:   A | B
                                   - analogy:   (C & ~only_a) | only_b
                                   - similarity: popcount(A&B) / popcount(A|B)
```

The mathematical theory uses prime factorization (PrimeMapper). The implementation uses bitwise ops (BitwiseValidator). They are **isomorphic** — same algebra, O(1) vs O(n).

## Key Concepts

- **BitwiseMapper/BitwiseValidator** (`src/triadic.py`): O(1) bitwise algebra on bit signatures
- **PrimeMapper/TriadicValidator** (`src/triadic.py`): prime-factor algebra (paper theory, same results)
- **Triadic Loss**: diversity + contrastive + entropy + embedding alignment (**NEVER coherence — causes collapse**)
- **iFSQ activation**: `2 * sigmoid(1.6x) - 1` — better gradients than tanh
- **Anchors**: hand-factorized gold concepts (50 v1, 158 v2) from "La Danza Cósmica de los Opuestos"
- **reptimeline**: discovers bit semantics, duals, 3-way interactions from trained model
- **Discovery Loop**: train → discover → human corrects → expand anchors → retrain (core paper thesis)

## Current State (2026-03-19)

- **Production**: D-A14 v2 tanh — 40M params, 93% test, 98.3% subsumption, 158 anchors
- **Alternative**: D-A16 iFSQ+v2 — 93.2% test, R3=0.842, best LM loss
- **From-scratch**: Run 15 — PPL 7.69, gap +0.020 (no supervised anchors)
- **Scaling**: D-A17 GPT-2 355M (training), D-A18 unified (script ready)
- **Paper**: 16 pages compiled, all experiments included
- **Validation**: E1-E7 complete, 80 unit tests pass, 12 benchmarks done
- **Key evidence**: 50→158 anchors improved 87%→93% (discovery loop works)

## Repos in the Ecosystem

| Repo | Purpose | Status |
|------|---------|--------|
| `triadic-microgpt` | This repo: model, training, paper | Active |
| `Triadic-Neurosymbolic-Engine` | Parent library + paper (neurosym on PyPI) | Published |
| `neurosym-client` | Python SDK (neurosym-cloud on PyPI) | Published |
| `triadic-cloud` | Commercial SaaS API (PRIVATE) | Active |
| `triadic-head` | Standalone PyPI package from microgpt | Built, not published |

## Workflow

See `.agents/workflows/training.md` for the full training cycle (train → evaluate → discover → correct → retrain).
