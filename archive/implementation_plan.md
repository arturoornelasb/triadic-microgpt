> **⚠ ARCHIVED — High-level content migrated to [`EXPERIMENT_REFERENCE.md`](../EXPERIMENT_REFERENCE.md) Section 2.** This file preserved in `archive/` for the ranked evidence table, ASCII architecture diagram, implementation phases 2-5, and next actions checklist.

# Implementation Plan — Final Model for Paper

> Date: 2026-03-19 | Status: ACTIVE
> Goal: ONE paper, ONE model, ONE framework
> Based on: ALL experiment results in repo (29 runs + 15 experiments + 12 benchmarks)

---

## EVIDENCE BASE — What We Know

### What Works (ranked by evidence strength)

| Discovery | Evidence | Numbers | Source |
|-----------|----------|---------|--------|
| Zero LM cost | PPL 7.69 vs 7.56 ablation | delta = 0.13 (noise) | Run 15 vs Run 18 |
| Subsumption algebra | v2 98.3% test, D-A13 100% holdout | 1258/1280 correct | D-A14, D-A13 |
| Analogy verification | 98% on 51 quads (50/51) | Top-1 verification | E3 expanded |
| Bitwise isomorphism | 1000/1000 random tests pass | 5-78x faster | prime_vs_bitwise.py |
| R3 composition | 98.1% round-trip, sub-linear chains | p<0.001, d=6.64 | D-A5, D-A11 |
| iFSQ activation | Loss 0.924 (best LM) + 87.1% sub | Better than tanh on both | D-A10 |
| Free bits learn | 6/63 dead vs 26/63 supervised-only | 17 triadic interactions | D-A9 hybrid |
| ~42% sparsity convergence | 42.9% (D-A5), 41.3% (D-A14), 42.3% (BitNet) | Three independent sources | Multiple |
| 158 anchors >> 54 anchors | 93% vs 79.4% test accuracy | +13.6pp, 4x triadic | D-A14 vs D-A5 |
| Scaling to 355M | 100% subsumption holdout | From D-A13 training log | D-A13 |
| Exact king:queen analogy | Bitwise analogy = 0 bit difference | sim=0.913 | D-A14 bitwise |

### What Failed (and why)

| Attempt | Result | Root Cause | Lesson for Final Model |
|---------|--------|-----------|----------------------|
| Gradient decoupling | 49.6% = random | Per-bit gradient manipulation breaks joint optimization | Don't decouple — bits learn holistically |
| Absmean quantization | Loss 1.309 | Worse than FSQ for this architecture | Use iFSQ, not absmean |
| 49-bit structured (P15) | 17% test (memorization) | Structured system can't generalize | End-to-end > manual structure |
| Bootstrap self-play (D-A6) | Cycle 0 stuck | Confidence gate too strict with 54 anchors | More anchors (v2=158) > self-play |
| 8x compression claim | 13.3% accuracy (below random) | Bits don't compress like embeddings | Remove from claims |
| Sinusoidal head (P1) | +4 dead bits | Literal wave is wrong level | tanh/iFSQ valid structural |
| E10-v2 GPT-2 InfoNCE | tri_loss=NaN | Numerical overflow | Need temperature fix |

### What's Ambiguous

| Question | Evidence | Resolution |
|----------|----------|------------|
| Ternary vs binary at scale | D-A13 (355M): zeros -> 0% | **OPEN** — ternary may only work at small scale |
| iFSQ + v2 anchors together | **D-A16: tested 2026-03-19** | **RESOLVED** — matches v2 tanh on accuracy (93.2%), improves analogies (R3=0.842), but LM loss slightly worse (0.993 vs 0.946). v2 anchors dominate over activation choice |
| Discovery loop closed | Manual proof-of-concept works | OPEN — automate and test convergence |
| L11/L12 with v2 | **Tested 2026-03-19** | **RESOLVED** — PASS. v2 model correctly learns indifference as true opposite of love |

---

## TESTS THAT ARE MISSING

### Tier 1: Must run before paper (can do NOW)

| Test | Why critical | Script | Effort | Blocking |
|------|-------------|--------|--------|----------|
| **L11/L12 re-run with v2** | Model failed with old anchors, v2 has 3x more anchors — may now pass | Modify `test_indifference_and_false_opposites.py` to load v2 | 30 min CPU | YES — paper claim about indifference |
| **L15 Aristotelian types** | Script exists, never run, tests Cap. 11 claim | `test_aristotelian_types.py` | 30 min CPU | Semi — validates book |
| **L19 Enantiodromia** | Script exists, never run | `test_enantiodromia.py` | 30 min CPU | No — but free test |

### Tier 2: Must run before paper (needs GPU)

| Test | Why critical | Script | Effort | Blocking |
|------|-------------|--------|--------|----------|
| **L2 D-A13 formal eval** | 100% sub from training log but no formal benchmark | `test_d_a13_eval.py` | 1h GPU | YES — 355M scaling claim |
| **iFSQ + v2 anchors** | Best activation + best anchors never combined | New script needed | 2h GPU | YES — final model decision |

### Tier 3: Valuable, not blocking

| Test | What it proves | Effort |
|------|---------------|--------|
| L13: 1000 adversarial concepts | Stress test beyond 158 anchors | 4h GPU |
| L14: PCA real primitive count | How many independent dims exist? | 3h CPU |
| L16: Polisemia contextual | Same word different meaning | 1h CPU |
| Cross-corpus (WikiText2, LAMBADA) | Generalization beyond TinyStories | 2h GPU |

---

## THE FINAL MODEL — Based on Evidence

### Architecture Decision Table

| Component | Options tested | Winner | Evidence |
|-----------|---------------|--------|----------|
| **Activation** | tanh, iFSQ, FSQ, absmean, sigmoid | **iFSQ** | 0.924 loss (best), 87.1% sub |
| **Algebra backend** | PrimeMapper, BitwiseValidator | **BitwiseValidator** | O(1), 1000/1000 equiv, 5-78x |
| **Anchor strategy** | 54 (v1), 158 (v2), 0 (unsupervised) | **158+ (v2)** | 93% vs 79.4% test |
| **Bit allocation** | 63 supervised, 30+33 hybrid | **Hybrid recommended** | 6 dead vs 26 dead |
| **Discovery** | None, post-hoc, integrated | **Post-hoc (proven), integrated (proposed)** | 68 triadic in v2 |
| **Scale** | 1.3M, 5.8M, 15.9M, 40M, 355M | **40M (paper), 355M (bonus)** | Phase transition at ~20M |

### Final Architecture

```
Text -> BPE (4096) -> TriadicGPT (12L/512D/8H, 40M params)
                            |
                      +-----+-----+
                      |           |
                 LM Head    Triadic Head
                 (softmax)  (Linear -> iFSQ -> 63 bits)
                      |           |
                 next token  BitwiseValidator
                             - subsumes:  (A & B) == B       O(1)
                             - compose:   A | B              O(1)
                             - analogy:   (C & ~oA) | oB     O(1)
                             - gap:       A & ~B, B & ~A     O(1)
                             - similarity: popcount ratio    O(1)
                                  |
                             reptimeline (post-hoc)
                             - bit semantics
                             - dual pairs (7 found)
                             - 3-way interactions (68 found)
                             - 635 dependency edges
```

### What Changes vs Current

| Component | Current (v2) | Final Model | Why change |
|-----------|-------------|-------------|-----------|
| Activation | tanh | **iFSQ** | 0.924 < 0.946 loss, preserves gradients |
| Algebra | PrimeMapper | **BitwiseValidator** | 5-78x faster, scales to 1024+ bits |
| Bits | 63 all supervised | **30 sup + 33 free** (hybrid) | 6 dead vs 26, free bits learn composites |
| Discovery | manual reptimeline | **auto-hook post-training** | Same tool, just automated |
| Anchors | 158 (v2) | **158 + auto-generated** | Discovery can propose new ones |

### What Does NOT Change

- Transformer architecture (12L/512D/8H) — validated at this scale
- BPE tokenizer (4096 vocab) — works
- Dual-head design (LM + triadic) — zero cost proven
- 63 bits total — k=64 optimal (bits sweep)
- Training loop — warmup 80% LM, then add triadic loss
- `triadic_loss()` — differentiable, works as-is

---

## IMPLEMENTATION PHASES

### Phase 0: Run Missing Tests (TODAY, CPU)

```bash
# 1. L11/L12 with v2 checkpoint
python playground/audit_tests/test_indifference_and_false_opposites.py \
  --checkpoint checkpoints/danza_63bit_xl_v2/model_best.pt

# 2. L15 Aristotelian types
python playground/audit_tests/test_aristotelian_types.py

# 3. L19 Enantiodromia
python playground/audit_tests/test_enantiodromia.py
```

**Exit criteria:** Results documented. L11/L12 either pass with v2 or we document why not.

### Phase 1: iFSQ + v2 Anchors (KEY EXPERIMENT, 2h GPU)

The two best components have never been combined:
- iFSQ (best LM: 0.924) was trained with 54 anchors
- v2 (best accuracy: 93%) was trained with tanh

```bash
# New script or flag in danza_63bit.py
python playground/danza_63bit.py \
  --scale xl --steps 50000 --v2 \
  --activation ifsq --dtype bfloat16
```

| Expected outcome | Metric to beat |
|-----------------|---------------|
| LM loss < 0.95 | v2 tanh = 0.946, iFSQ = 0.924 |
| Test accuracy >= 93% | v2 tanh = 93.0% |
| Subsumption >= 98% | v2 tanh = 98.3% |
| Dead bits < 26 | v2 tanh = 26/63 |

**If this succeeds** -> this IS the final model.
**If accuracy drops** -> use v2 tanh (already validated, already good).

Files to modify:
- `playground/danza_63bit.py`: Add `--activation` flag with iFSQ option
- `src/torch_transformer.py`: Add iFSQ activation to TriadicGPT

### Phase 2: BitwiseValidator as Default (1 day, no GPU)

Replace PrimeMapper in ALL runtime paths:

| File | Change |
|------|--------|
| `src/torch_train.py` | Eval uses BitwiseValidator |
| `src/evaluate.py` | All algebraic checks use bitwise |
| `benchmarks/scripts/*.py` | Default to bitwise backend |
| `ui/model_interface.py` | UI uses BitwiseValidator |
| `tests/test_all.py` | Add dual-backend equivalence tests |

Validation: Run all 12 benchmarks, verify identical results.

### Phase 3: Hybrid Bits (2-3 days GPU)

Combine v2 anchors + iFSQ + hybrid bit allocation:

```
Bits 0-29:  Supervised (158 anchor concepts)
Bits 30-62: Free (learned via LM + adversarial disentanglement)
```

| From D-A9 (hybrid) | From D-A14 (v2) | Combined expectation |
|--------------------|-----------------|---------------------|
| 6 dead bits | 26 dead bits | <10 dead bits |
| 50 active bits | 48 active bits | ~55 active bits |
| 17 triadic 3-way | 68 triadic 3-way | 80+ triadic |
| 80% subsumption | 98.3% subsumption | >95% subsumption |

Files to create/modify:
- `playground/unified_training.py`: New script combining iFSQ + hybrid + v2 anchors
- Use `hybrid_adversarial.py` as base, swap tanh for iFSQ, use v2 anchors

### Phase 4: Discovery Loop (3-5 days)

Close the loop: after training, run discovery, generate new anchors, retrain.

```python
for cycle in range(N_CYCLES):
    # 1. Train
    train(model, anchors, steps=50000)

    # 2. Discover
    codes = extract_projections(model, all_words)
    report = BitDiscovery().discover(ConceptSnapshot(codes=codes))

    # 3. Generate new anchors from discovery
    new_anchors = generate_from_triadic_deps(report)
    anchors = merge(anchors, new_anchors)

    # 4. Log metrics
    log(cycle, report.n_triadic_deps, test_accuracy)
```

**Already works manually** — bootstrap does train->eval->re-anchor.
Need to: hook reptimeline discover, auto-generate anchors from 3-way deps.

### Phase 5: Scale to 355M (1-2 weeks, GPU-dependent)

Apply final model architecture to GPT-2 Medium:
- iFSQ activation
- BitwiseValidator backend
- v2+ anchors
- Run L2 formal eval

**Open question:** At 355M, zeros collapse to 0%. Document as finding.

---

## PAPER SECTION <-> EVIDENCE MAPPING

| Section | Claim | Evidence | Status |
|---------|-------|----------|--------|
| Abstract | Zero-cost triadic head | PPL 7.69 vs 7.56 | READY |
| 3.1 | Prime algebra | Formal proof in `src/triadic.py` | READY |
| 3.2 | Bitwise isomorphism | 1000/1000 tests, `prime_vs_bitwise.py` | READY |
| 3.3 | Scaling advantage | Primes IMPOSSIBLE >128 bits | READY |
| 4 | Architecture | 12L/512D/8H + triadic head | READY |
| 5.1 | Scaling study | 4-point (1.3M-40M), phase transition | READY |
| 5.2 | Bits sweep | k=8-128, optimal k=32-64 | READY |
| 5.3 | Ablation | Run 18 (no head), D-A15 (grad decoupling FAIL) | READY |
| 5.4 | Subsumption | 98.3% test (v2). 355M: 88% bit acc but sub=9-20%, analogy=0% | **NEEDS REVISION** |
| 5.5 | Analogy | 98% verification (51 quads), exact king:queen bitwise | READY |
| 5.6 | Composition | R3 98.1% round-trip, p<0.001 | READY |
| 5.7 | Domain separation | 1.21 sentence-level (+19%) | READY |
| 5.8 | Discovery | 68 triadic 3-way, 7 duals, 635 deps | READY |
| 5.9 | iFSQ activation | D-A10: 0.924 loss, 87.1% sub. D-A16: iFSQ+v2 = 93.2%, R3=0.842 | **READY** |
| 5.10 | Convergence | ~42% sparsity, three paths | READY |
| 6 | Discussion | 355M: bits accurate but algebra degrades. Zeros 6.2%. Collisions. | **NEEDS REVISION** |
| 7 | Future work | Discovery loop, scaling, cross-linguistic | READY |

**Status: 11/13 sections READY. 2 need revision (5.4 subsumption @ 355M, 6 discussion).**
**L2 formal eval DONE — 355M v1 has good bits (88%) but algebraic ops fail. BUT: D-A13 never had v2 anchors (158). The v2 anchor set was the breakthrough at 40M. Fair comparison requires 355M+v2 training (~4.5h GPU).**

---

## IMMEDIATE NEXT ACTIONS (Priority Order)

1. ~~Run L11/L12 with v2 checkpoint~~ — **DONE: PASS** (2026-03-19)
2. ~~Run L15 + L19~~ — **DONE: both FAIL** (2026-03-19)
3. ~~iFSQ+v2 decisive experiment~~ — **DONE: D-A16** (2026-03-19, 93.2% test, R3=0.842)
4. ~~Create iFSQ+v2 training script~~ — **DONE: added `--activation ifsq` to danza_63bit.py**
5. ~~Run iFSQ+v2 training~~ — **DONE: see step 3**
6. **Switch to BitwiseValidator default** — 1 day, no GPU
7. ~~Run L2 D-A13 eval~~ — **DONE: 88% bits, sub 9-20%, analogy 0% (but only v1 anchors!)**
8. **Train D-A17: GPT-2 Medium (355M) + v2 anchors** — ~4.5h GPU. Fair scaling comparison.
9. **Run L2 formal eval on D-A17** — compare with D-A13 (v1) and D-A14 (40M v2)
10. **Revise paper Sections 5.4 and 6** with honest scaling results
11. **Update paper with all final results** — 1 day

**Minimum for submission: Steps 1-5 DONE. Remaining: L2 eval (GPU) + paper update.**
**Final model: v2 tanh (D-A14) — confirmed by D-A16 ablation.**
