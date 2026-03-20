# Triadic MicroGPT — Unified Experiment Reference

> **Canonical reference for ALL experiments, results, and research decisions.**
> Created: 2026-03-19 | Consolidates: experiment_log.md, danza_experiment_log.md, PLAN.md, AUDIT.md, TEST_STATUS.md, EVOLUTION_PLAN.md, implementation_plan.md, PRIMITIVE_RECONCILIATION.md
>
> For detailed per-step training data, see `experiment_log.md` (preserved as data store).

---

## Table of Contents

- [1. Master Results Table](#1-master-results-table)
- [2. Current State & Optimal Model](#2-current-state--optimal-model)
- [3. Key Findings & Lessons Learned](#3-key-findings--lessons-learned)
- [4. Line F: Danza Cósmica (D-A series)](#4-line-f-danza-cósmica-d-a-series)
- [5. Line E: Validation (E1-E7, B1-B3)](#5-line-e-validation-e1-e7-b1-b3)
- [6. Line C: Playground Explorations (P-series)](#6-line-c-playground-explorations-p-series)
- [7. Line D: Transfer Learning (Experiment 10)](#7-line-d-transfer-learning-experiment-10)
- [8. Line A: Core Model Development (Runs 1-18)](#8-line-a-core-model-development-runs-1-18)
- [9. Line B: Scaling & Architecture](#9-line-b-scaling--architecture)
- [10. Tests & Benchmarks](#10-tests--benchmarks)
- [11. Technical Infrastructure](#11-technical-infrastructure)
- [12. Primitive Systems Reconciliation](#12-primitive-systems-reconciliation)
- [13. Reviewer FAQ](#13-reviewer-faq)
- [14. Paper Corrections](#14-paper-corrections)
- [15. Version History](#15-version-history)
- [Appendix A: Detailed Data Store](#appendix-a-detailed-data-store)
- [Appendix B: Reproducibility](#appendix-b-reproducibility)

---

## 1. Master Results Table

Every experiment in ONE table, newest to oldest.

### Danza Line (D-A series) — Supervised 63-bit

| ID | Date | Name | Key Result | Status | Checkpoint |
|---|---|---|---|---|---|
| **D-A16** | 03-19 | iFSQ + v2 (158 anchors) | 93.2% test, 98.3% sub, R3=0.842 | COMPLETE | `danza_63bit_xl_v2_ifsq/` |
| **D-A14** | 03-19 | v2 tanh (158 anchors) | **93% test, 98.3% sub, 68 triadic** | COMPLETE | `danza_63bit_xl_v2/` |
| D-A15 | 03-19 | Gradient decoupling | 49.6% = random | FAILED | `danza_grad_decoupling_xl/` |
| **D-A18** | 03-20 | Unified (iFSQ + hybrid 30+33 + v2) | 92.2% sup, 75.3% full, 96.5% sub, 15 dead, R3=0.851 | COMPLETE | `danza_unified_xl/` |
| **D-A19** | 03-20 | GPT-2 355M + full losses + sparsity | **97.1% bit, 76.9% sub, 16 dead — algebra restored at 355M** | COMPLETE | `danza_gpt2_355m_sparsity_v2/` |
| D-A17 | 03-20 | GPT-2 355M + v2 | **97.7% bit, 1.7% sub, 26 dead — algebra destroyed at scale** | COMPLETE | `danza_gpt2medium_ternary_v2/` |
| D-A9 | 03-19 | Hybrid adversarial (30+33) | 69.3% test, 13 dead bits (5 sup + 8 free), 17 triadic | COMPLETE | `danza_hybrid_adv_xl/` |
| D-A13 | 03-18 | GPT-2 355M ternary (v1) | 88% bits, sub 9-20%, analogy 0% | COMPLETE | `danza_gpt2medium_ternary/` |
| D-A10 | 03-18 | iFSQ binary ablation | loss 0.924 (BEST LM), sub 87.1% | COMPLETE | `danza_ifsq_binary_xl/` |
| D-A8 FSQ | 03-18 | Ternary FSQ | loss 0.951, sub 86.5% | COMPLETE | `danza_ternary_fsq_xl/` |
| D-A8 Abs | 03-18 | Ternary absmean | loss 1.309 (inferior) | COMPLETE | `danza_ternary_absmean_xl/` |
| D-A11 | 03-18 | Negative baselines | trivial 90.2%, real 90.7%, p<0.001 | COMPLETE | — (CPU) |
| D-A16e | 03-18 | Multi-quad ensemble | **90.6%** mean ensemble (was 94.6% stale) | COMPLETE | — (CPU) |
| D-A16f | 03-18 | Subsumption FPR test | FPR 24.1% (dead bits cause spurious) | COMPLETE | — (eval) |
| D-A6 | 03-18 | Bootstrap loop (3 cycles) | Converged cycle 0, 0 accepted | COMPLETE | `danza_bootstrap_v2_xl/` |
| D-A5 | 03-18 | Bootstrap half-anchor | holdout 87.2%, algebraic 90.7% | COMPLETE | `danza_bootstrap_xl/` |
| D-A12 | 03-18 | Dead-bit surgery | — | PREPARED (never run) | — |
| D-A2 | 03-17 | Full XL supervised (50K) | 89.5% test, 90% sub, 27 dead | COMPLETE | `danza_63bit_xl/` |
| D-A1 | 03-17 | Post-hoc analysis (0 GPU) | 22 anti-corr, DAG depth 2, R3 70.6% | COMPLETE | — (CPU) |
| D1 | 03-17 | Smoke test (base, 100 steps) | 88.7% test, R3 97.4% | COMPLETE | `danza_63bit_base/` |

### Validation Line (E/B series)

| ID | Date | Name | Key Result | Status |
|---|---|---|---|---|
| E1 | 03-17 | Multi-seed (3 seeds) | gap +0.038 ± 0.005, 100% analogy | COMPLETE |
| E2 | 03-17 | Alignment ablation | alignment = THE driver, entropy redundant | COMPLETE |
| E3 | 03-15 | Expanded analogy (51 quads) | 98% verification (up from 69.2%) | COMPLETE |
| E4 | 03-17 | Sub weight sweep (80% warmup) | best 92.3% @ w=2.0 | COMPLETE |
| E4b | 03-17 | Sub weight (50% warmup) | 76.9% sub, 24 dead (validates E4) | COMPLETE |
| E5 | 03-17 | Scale interpolation (25M/30M) | crossover ~20M, gradual | COMPLETE |
| E6 | 03-15 | Compression benchmark | "8x" REFUTED (13.3% acc) | COMPLETE |
| E7 | 03-16 | R3 at low k (6,8,12) | alive but gap -0.27 to -0.42 | COMPLETE |
| E7v2 | 03-17 | R3 low k clean words | E7 validated, no confound | COMPLETE |
| XL2 | 03-17 | Sigmoid+anneal temp=5 XL | PPL +110%, definitively negative | COMPLETE |
| B1 | 03-17 | Embedding gap baseline | triadic amplifies 2.6x in 8x fewer dims | COMPLETE |
| B2 | 03-17 | Pure language XL | cost +2% PPL; random gap +0.056 | COMPLETE |
| B3 | 03-17 | Frozen random head XL | gap ≠ value; algebra only from training | COMPLETE |

### Playground Line (P-series)

| ID | Date | Name | Key Result | Status |
|---|---|---|---|---|
| P15 | 03-15 | Concept GPT 49-bit | 86.2% acc, 0 dead, 97.3% sub | COMPLETE |
| P14 | 03-15 | Post-hoc concept projection | ~20% (negative) | COMPLETE |
| P13 | 03-15 | Cross-dataset eval | OOD expected (LAMBADA 345 PPL) | COMPLETE |
| P12 | 03-14 | XL Subsumption loss | 100% held-out @25K, PPL +47% | COMPLETE |
| P11 | 03-14 | Curriculum Sub→R3 | R3 erases Sub in 3K steps | COMPLETE |
| P10 | 03-14 | R3 entropy guard | 64/64 dead (unfixable) | COMPLETE |
| P9 | 03-14 | Info hierarchy | 93% bit reduction in hypernyms | COMPLETE |
| P8 | 03-14 | Phase-aware attention | negative (learned positions better) | COMPLETE |
| P7 | 03-14 | R3+Sub combo | Sub dominates, R3 collapses | COMPLETE |
| P6 | 03-14 | Subsumption loss | 100% train, 91.7% held-out (BREAKTHROUGH) | COMPLETE |
| P5 | 03-14 | Rule-of-Three loss | K=1.0 (perfect), but memorization | COMPLETE |
| P4 | 03-14 | XL Sigmoid+anneal | PPL +116% (overfitting) | COMPLETE |
| P3 | 03-13 | Soft signatures | sigmoid+anneal best (+0.039 gap, 0 dead) | COMPLETE |
| P2 | 03-13 | Random baseline | frozen random > trained at 5.8M | COMPLETE |
| P1 | 03-13 | Sinusoidal head | +0.021 gap, +4 dead bits | COMPLETE |
| P0 | 03-13 | K-constant analysis | mean K=1.21 (R3 approximately holds) | COMPLETE |

### Transfer Learning (Experiment 10)

| ID | Date | Name | Key Result | Status |
|---|---|---|---|---|
| E10-v3 | 03-19 | GPT-2 InfoNCE (Bug #7 fix) | gap +0.076 (corrected from +0.099) | COMPLETE |
| E10b | 03-08 | GPT-2 Rank alignment | gap +0.047, analogy 83.3% | COMPLETE |
| E10a | 03-08 | GPT-2 MSE alignment | gap +0.011 (negative for rich embeds) | COMPLETE |
| E10-v2 | 03-18 | GPT-2 InfoNCE (NaN bug) | tri_loss=NaN from step 300 | FAILED |
| Run 29 | 03-08 | Staged MSE→InfoNCE | negative (loss-embed structural) | COMPLETE |
| Run 28 | 03-08 | From-scratch Rank | broken ordering | COMPLETE |
| Run 27 | 03-08 | From-scratch InfoNCE | broken ordering | COMPLETE |

### Core Model Development (Runs 1-18)

| ID | Date | Name | Key Result | Status |
|---|---|---|---|---|
| Run 15 | 03-07 | **Strong alignment (PRODUCTION)** | **loss 0.946, PPL 7.69, gap +0.020** | **PRODUCTION** |
| Run 18 | 03-07 | Ablation (no triadic) | PPL 7.56 → cost = 0 | COMPLETE |
| Run 17 | 03-07 | Mid alignment (alpha=0.1) | ordering LOST (Pareto cliff) | COMPLETE |
| Run 16 | 03-07 | Max alignment (alpha=0.2) | ordering LOST | COMPLETE |
| Run 14 | 03-07 | Embedding alignment | entropy 0.720, partial ordering | COMPLETE |
| Run 13 | 03-07 | No coherence + entropy | diverse but random | COMPLETE |
| Run 12 | 03-07 | Entropy reg | COLLAPSED (coherence = root cause) | FAILED |
| Run 11 | 03-05 | Industrial audit | 98.5% acc, 0.96% FPR | COMPLETE |
| Run 10 | 03-05 | Knowledge distillation | distill@5x = COLLAPSE | COMPLETE |
| Run 9 | 03-05 | XL model (first 40M) | loss 1.277 | COMPLETE |
| Run 8 | 03-05 | Fixed tokenizer | clean text output | COMPLETE |
| Run 6 | 03-05 | Diversity+contrastive fix | differentiation starting | COMPLETE |
| Run 4 | 03-04 | PyTorch GPU validation | loss 1.55, coherent English | COMPLETE |
| Run 1-2 | 03-04 | CPU baseline | loss 1.75-3.23, too slow | COMPLETE |

### Scaling & Architecture

| ID | Date | Name | Key Result | Status |
|---|---|---|---|---|
| Exp 11 | 03-10 | Sentence aggregation | domain sep 1.02→1.21 (+19%) | COMPLETE |
| Exp 9 | 03-07 | Engine comparison (Table 7) | 5 methods, PCA=Contrastive best | COMPLETE |
| Runs 22-26 | 03-07 | Bits sweep (k=8-128) | optimal k=32-64, U-shaped loss | COMPLETE |
| Runs 19-21 | 03-07 | Scale study (1.3-15.9M) | gap crosses zero ~20M | COMPLETE |

### Audit Tests

| ID | Date | Test | Result | Status |
|---|---|---|---|---|
| L11/L12 v2 | 03-19 | Indifference + false opposites | **PASS** (was FAIL with Run 15) | COMPLETE |
| L15 v2 | 03-19 | Aristotelian types | FAIL (0/4 significant) | COMPLETE |
| L19 v2 | 03-19 | Enantiodromia | FAIL (2/8 confirmed) | COMPLETE |
| L1 | 03-19 | Bridge PFs (Q1-Q6) | 3/4 PASS (Q6 FAIL) | COMPLETE |
| L2 | 03-19 | D-A13 formal eval | 88% bits, sub 9-20%, analogy 0% | COMPLETE |
| L3 | 03-19 | Blind prime assignment | PASS (vacuous) | COMPLETE |
| F0 | 03-19 | Data validation | 93.7% bit acc (54 anchors) | COMPLETE |

### Technical Infrastructure

| ID | Date | Name | Key Result | Status |
|---|---|---|---|---|
| BitwiseValidator | 03-19 | O(1) isomorphic algebra | 1000/1000 equiv, 5-78x faster | COMPLETE |
| Convergence | 03-19 | Trits/BitNet/Bitwise | three paths → {+1,0,-1} ~42% sparsity | COMPLETE |
| R3 Formulas | 03-18 | 4 discrete vs continuous | Formula D ternary best | COMPLETE |
| R3 Chains | 03-18 | Composition depth test | 98.1% round-trip, sub-linear | COMPLETE |
| NSM Mapping | 03-18 | 7×7 vs Wierzbicka | 55% convergence (28 direct, 8 close) | COMPLETE |

---

## 2. Current State & Optimal Model

### Production Models

| Purpose | Model | Loss | Test Acc | Sub | Analogy | Checkpoint |
|---|---|---|---|---|---|---|
| **Paper (self-supervised)** | Run 15 (v1.4-strongalign) | 0.946 | — | 86.5% | 98% verif | `torch_run15_strongalign/` |
| **Best supervised** | D-A14 (v2 tanh, 158 anchors) | 0.946 | **93.0%** | **98.3%** | 68 triadic | `danza_63bit_xl_v2/` |
| **Best LM** | D-A10 (iFSQ binary) | **0.924** | ~80% | 87.1% | — | `danza_ifsq_binary_xl/` |
| **Best R3** | D-A16 (iFSQ+v2) | 0.993 | 93.2% | 98.3% | **R3=0.842** | `danza_63bit_xl_v2_ifsq/` |
| **Fewest dead bits** | D-A9 (hybrid adversarial) | ~1.0 | 69.3% | — | 17 triadic | `danza_hybrid_adv_xl/` |

**Final model decision** (from implementation_plan.md): **D-A14 (v2 tanh)** — confirmed by D-A16 ablation showing v2 anchors dominate over activation choice.

### Architecture Decision Table

| Component | Options Tested | Winner | Evidence |
|---|---|---|---|
| Activation | tanh, iFSQ, FSQ, absmean, sigmoid | **iFSQ** (production: tanh) | D-A10: 0.924 loss, 87.1% sub |
| Algebra backend | PrimeMapper, BitwiseValidator | **BitwiseValidator** | O(1), 1000/1000 equiv, 5-78x |
| Anchor strategy | 54 (v1), 158 (v2), 0 (unsupervised) | **158+ (v2)** | 93% vs 79.4% test |
| Bit allocation | 63 supervised, 30+33 hybrid | **Hybrid recommended** | 13 dead vs 26 dead |
| Scale | 1.3M-355M | **40M (paper), 355M (bonus)** | Phase transition ~20M |
| Bits (k) | 8-128 | **32-64** | U-shaped loss, gap peaks k=32 |

### Paper Section ↔ Evidence Mapping

| Section | Claim | Evidence | Status |
|---|---|---|---|
| Abstract | Zero-cost triadic head | PPL 7.69 vs 7.56 | READY |
| 3.1 | Prime algebra | Formal proof in `src/triadic.py` | READY |
| 3.2 | Bitwise isomorphism | 1000/1000 tests | READY |
| 4 | Architecture | 12L/512D/8H + triadic head | READY |
| 5.1 | Scaling | 4-point study, crossover ~20M | READY |
| 5.2 | Bits sweep | k=8-128, optimal k=32-64 | READY |
| 5.3 | Ablation | Run 18 (no head), D-A15 (FAIL) | READY |
| 5.4 | Subsumption | 98.3% (v2), 355M: 1.7% (D-A17), **76.9% (D-A19 fix)** | **UPDATED** |
| 5.5 | Analogy | 98% verification, exact king:queen | READY |
| 5.6 | Composition | R3 98.1% round-trip | READY |
| 5.7 | Domain separation | 1.21 sentence-level | READY |
| 5.8 | Discovery | 68 triadic 3-way, 9 duals (coherence 0.57-1.00) | **UPDATED** |
| 6 | Discussion | 355M destroys algebra (D-A17), **D-A19 restores it** — bugs not inherent scale limitation | **UPDATED** |

---

## 3. Key Findings & Lessons Learned

### Critical Findings (from AUDIT.md)

**3.1 Baseline trivial = 90.2%**
The gold bit distribution is highly unbalanced: 24 bits always OFF, 6 always ON, 33 variable. A model predicting the majority class per bit achieves 90.2%. ALL bit accuracy must compare against this, not 50%.

**3.2 Value of the triadic head = algebraic operations, NOT gap**
Random projections of backbone hidden states produce semantic gap (+0.056) higher than trained head (+0.038). BUT: analogies random 16.7% vs trained 100%. Subsumption: random 0% vs trained 92-100%. **The gap is necessary but NOT sufficient — algebraic capabilities only emerge from end-to-end training.** (B1-B3)

**3.3 Coherence loss = collapse (NEVER re-enable)**
Adjacent-token agreement drives all projections to identical. Run 12 proved this definitively — entropy dropped to 0.000 with 0.9% unique signatures. (Run 12)

**3.4 Embedding alignment = semantic teacher**
The wte embeddings transfer semantic structure to the triadic head. Without alignment, dead bits double (11→23). Entropy reg is redundant when alignment present. (E2)

**3.5 Loss-embedding interaction**
Optimal loss depends on embedding quality:
- Rich embeddings (GPT-2): InfoNCE (+0.076) >> Rank (+0.047) >> MSE (+0.011)
- Weak embeddings (from-scratch): MSE (+0.020) >> Rank (~0) ≈ InfoNCE (~0)
(Runs 27-29, E10)

**3.6 158 anchors >> 54 anchors**
D-A14 (v2, 158) vs D-A5 (v1, 54): test accuracy +13.6pp (93% vs 79.4%), subsumption +18.3pp, 4x more triadic interactions. The v2 anchor set was THE breakthrough. (D-A14 vs D-A5)

**3.7 R3 is dead at k=64**
Three independent experiments (P7 combo, P10 entropy guard, P11 curriculum) all produce 64/64 dead bits. R3's trivial global minimum (all bits identical) is unfixable. At k=6-12, R3 is alive but destroys semantic gap. (P7, P10, P11, E7)

**3.8 ~42% sparsity convergence**
Three independent paths arrive at the same target: D-A5 42.9%, D-A14 41.3%, BitNet b1.58 42.3%. The triadic framework predicts {+1, 0, -1} as the optimal discrete representation.

**3.9 Dead bits = BitNet third state**
Dead bits are NOT a bug — they are the model's `[0]` (vacío/irrelevant) state. D-A8 ternary head formalized this: 1.3% negative, 73.3% zero, 25.3% positive.

**3.10 True language cost = +2% PPL**
Pure language baseline PPL 10.65 vs triadic 10.86. The +38% comparison to Run 15 (7.69) was unfair — Run 15 used distillation. (B2)

### Known Bugs in Code

| Bug | File | Line | Severity | Status |
|---|---|---|---|---|
| `distill_target_tensor` undefined | `src/torch_finetune.py` | 270 | MEDIUM | OPEN — should be `gold_sequences` |
| GradScaler enabled without dtype check | `src/torch_finetune.py` | 172 | LOW | OPEN — no `--dtype` arg |
| Hardcoded tokenizer paths | `src/auditor.py`, `src/test_generalization.py` | various | LOW | OPEN |

---

## 4. Line F: Danza Cósmica (D-A series)

### 4.0 Central Thesis: Bits as a Semantic Operating System

> *Source: danza_experiment_log.md (UNIQUE)*

A normal LLM learns from raw text and must see billions of tokens to develop implicit semantic structure. We propose a different path:

**If you give a model 63 named semantic primitives as inductive bias, it should:**

1. **Learn more with less** — the bit structure provides skeleton, data provides flesh
2. **Infer what it hasn't seen** — regla de tres predicts new concepts algebraically
3. **Self-validate** — subsumption, dual axes, and dependency constraints detect errors
4. **Bootstrap knowledge** — use algebraic inference to expand its own training set

The key claim: **50 hand-factorized concepts + algebraic constraints can SEED a system that grows its own semantic knowledge without human intervention.**

This is NOT "training a bigger model" or "using more data". It's testing whether a mathematical structure (63 bits, 12 dual axes, 6 dependency layers) provides enough inductive bias to let a small model punch above its weight class.

### 4.1 Known Biases & Limitations

> *Source: danza_experiment_log.md (UNIQUE)*

**Gender-Element Mapping (CRITICAL):** The anchor system maps `hombre` → `tierra` (earth: solidity) and `mujer` → `agua` (water: fluidity). This is a cultural/philosophical mapping from the book's archetypal framework, **NOT a biological truth**. The regla de tres `hombre:mujer = rey:reina` works algebraically (Hamming=0) but its correctness depends on accepting the archetypal gender mapping.

**Other Biases:**
- `rico` (rich) includes `bien` (moral good) — conflates wealth with virtue
- `pobre` (poor) lacks `hacer` (agency) — implies poverty = passivity
- `lógico` includes `tierra` + `control` — associates logic with rigidity
- `creativo` includes `caos` — associates creativity with disorder

**TinyStories Coverage:** 4 anchors skipped (estasis_absoluta, hombre_vaciado, inercia_mental, amoral). Many anchors tokenize to multiple BPE tokens (50/54) → mean-pooled supervision.

### 4.2 D-A16: iFSQ + v2 Anchors — The Decisive Experiment (2026-03-19)

**Status**: COMPLETE | **GPU**: 110.8m | **Checkpoint**: `checkpoints/danza_63bit_xl_v2_ifsq/`

| Metric | Value |
|---|---|
| Bit acc test | 93.2% |
| Subsumption test | 98.3% |
| R3 king:queen | cos=0.959, 100% bits |
| R3 mean bit acc | 90.5% |
| LM loss | 0.993 (worse than tanh 0.946) |

**Finding**: iFSQ+v2 matches v2 tanh on accuracy/subsumption, improves analogies (R3=0.842), but slightly worse LM loss. **v2 anchors are the dominant factor**, not activation function. v2 tanh (D-A14) confirmed as final model.

> Full details: experiment_log.md, lines 3143-3207

### 4.3 D-A14: v2 — 158 Anchors + reptimeline Discovery (2026-03-19)

**Status**: COMPLETE | **GPU**: 129.2m | **Checkpoint**: `checkpoints/danza_63bit_xl_v2/`

| Metric | Value |
|---|---|
| Bit acc test | **93.0%** |
| Subsumption test | **98.3%** |
| Triadic 3-way interactions | **68** |
| king:queen bitwise | **EXACT MATCH** (0 bit difference) |
| Dead bits (discovery) | 15/63 |

**Finding**: **BEST MODEL** — 3x more anchors yields +13.6pp test accuracy, +18.3pp subsumption, 4x more triadic interactions vs D-A5. The man/woman encoding has near-minimal 1-bit gender difference.

> Full details: experiment_log.md, lines 3365-3469

#### reptimeline Analysis (2026-03-20)

**Command**: `python -m reptimeline --checkpoint-dir checkpoints/danza_63bit_xl_v2/ --primitives --overlay --max-checkpoints 5`

**Results**: `reptimeline/results/d_a14_v2_discovery.json` + 4 plots in `reptimeline/results/d_a14_v2_plots/`

**Timeline Summary** (5 checkpoints: 2.5K → 50K steps):

| Metric | Value |
|---|---|
| Concepts tracked | 53 (primitives) |
| Bit births | 2,725 |
| Bit deaths | 1,825 |
| Connections formed | 1,378 |
| Code churn | 0.000 → 0.245 |
| Code utilization | 0.981 → 0.717 |
| Mean entropy | 0.536 → 0.212 |
| Phase transitions | 0 (gradual, no sharp jumps) |

**Dual Axis Coherence** (learned oppositions):

| Dual Pair | Coherence | Exclusive activations |
|-----------|-----------|----------------------|
| placer ↔ dolor | **1.00** | 113 excl, 0 shared |
| libertad ↔ control | **0.98** | 47 excl, 1 shared |
| receptivo ↔ creador_obs | **0.94** | 62 excl, 4 shared |
| vida ↔ muerte | **0.91** | 158 excl, 15 shared |
| consciente ↔ ausente | **0.88** | 191 excl, 25 shared |
| bien ↔ mal | **0.85** | 100 excl, 17 shared |
| temporal_obs ↔ eterno_obs | 0.57 | 85 excl, 63 shared |
| verdad ↔ mentira | 0.57 | 92 excl, 70 shared |
| individual ↔ colectivo | 0.56 | 84 excl, 65 shared |

**Key Finding**: 6/9 dual axes have coherence > 0.85 — the model learns genuine semantic opposition without explicit dual supervision. The top 3 (placer/dolor, libertad/control, receptivo/creador) are near-perfect. The bottom 3 (temporal/eterno, verdad/mentira, individual/colectivo) have high shared activations, suggesting these concepts share more structure than they oppose.

**Layer Emergence**: All 6 ontological layers (L1 Punto → L6 Meta) activate by step 2,500 — the model learns the full ontological structure almost immediately. L6 Meta (consciousness/observer primitives) is the last to fully activate (median step 12,500).

**Bit Stability**:
- Most stable: bits 52, 42, 27, 35, 19 (locked in early, never change)
- Most unstable: bits 43, 54, 23, 62, 12 (high churn, candidates for dead bits)

**Visualizations** (in `reptimeline/results/d_a14_v2_plots/`):
- `swimlane.png` — concept × bit activation heatmap over training
- `phase_dashboard.png` — entropy, churn, utilization curves
- `churn_heatmap.png` — per-bit flip frequency across steps
- `layer_emergence.png` — when each ontological layer stabilizes

### 4.4 D-A15: Gradient Decoupling — FAILED (2026-03-19)

**Status**: FAILED | **GPU**: ~50m (killed early at step 500) | **Checkpoint**: `danza_grad_decoupling_xl/`

| Metric | Value |
|---|---|
| Bit acc test | 49.6% (random) |

**Finding**: Per-bit gradient manipulation is too invasive — the triadic head learns patterns holistically, not bit-by-bit. Approach abandoned.

> Full details: experiment_log.md, lines 3523-3562

### 4.5 D-A13: GPT-2 Medium 355M Ternary (2026-03-18)

**Status**: COMPLETE | **GPU**: 4.5h | **Checkpoint**: `danza_gpt2medium_ternary/`

| Metric | Value |
|---|---|
| Sub holdout (training log) | 100% |
| Sub holdout (formal eval) | 9.3% / 20.0% |
| Bit acc holdout | 88.0% |
| Analogy (formal R3) | 0.0% |
| Ternary collapse | 70.6% neg, **0% zero**, 25.9% pos |

**Finding**: 355M learns accurate bits but NOT algebraic structure. Ternary collapses to binary (zeros→0%). **v1 anchors (54) are insufficient at any scale** — fair comparison needs 355M + v2.

> Full details: experiment_log.md, lines 3049-3139

### 4.5b D-A17: GPT-2 Medium 355M + v2 Anchors (2026-03-20)

**Status**: COMPLETE | **GPU**: 289.7m | **Checkpoint**: `danza_gpt2medium_ternary_v2/`

| Metric | D-A19 (355M, fix) | D-A17 (355M, v2) | D-A14 (40M, v2) | D-A13 (355M, v1) |
|---|---|---|---|---|
| Bit accuracy (holdout) | 97.1% | **97.7%** | 93.0% | 88.0% |
| Subsumption (test) | **76.9%** | 1.7% | **98.3%** | 9.3% |
| Dead bits | **16/63** | 26/63 | 26/63 | 26/63 |
| Always-on bits | 4/63 | — | — | — |
| Ternary zeros (per-token) | 1.0% | 3.4% | **41.3%** | 0% |
| Ternary distribution | -1=70.8%, 0=1.0%, +1=28.2% | — | — | 70.6%/0%/25.9% |
| Unique signatures | 79.7% (126/158) | 84.2% (133/158) | 100% | — |
| Analogy (R3) | **100% (6/6)** cos 0.81-0.92 | 6.7% | ~100% | 0% |
| Training time | 301 min | 289.7 min | 129.2 min | 4.5h |

**Eval results**: `playground/audit_tests/results/f4_4_d_a17_eval.json`

**Key Finding**: **Scaling the backbone from 40M to 355M DESTROYS algebraic structure** while improving bit accuracy. The mechanism:

1. **Ternary zeros collapse** (41.3% → 3.4%) — the larger backbone provides enough signal that the model doesn't need the "irrelevant" state
2. **Without zeros, subsumption fails** — `(A & B) == B` is combinatorially unsatisfiable when all bits are ±1
3. **Signature collisions appear** — 20 concepts share identical signatures (bad/evil, absence/apathy/indifference, etc.)
4. **Analogy breaks** — 6.7% vs ~100% at 40M

**Conclusion**: ~42% ternary sparsity is **structurally necessary** for algebraic operations. More parameters ≠ better algebra. The 40M model (D-A14) IS the optimal architecture for the triadic system. This is evidence for the paper: the triadic head's value is algebraic composability, not bit accuracy. **UPDATE (D-A19)**: The root cause was bugs in D-A17's loss setup, not an inherent scale limitation. See D-A19 below.

### 4.5c D-A19: GPT-2 Medium 355M — Fix Scale-Algebra Tradeoff (2026-03-20)

**Status**: COMPLETE | **GPU**: 301 min | **Checkpoint**: `danza_gpt2_355m_sparsity_v2/model_best.pt` (saved at step 17500)

**Motivation**: D-A17 showed 355M destroying algebra (1.7% sub). D-A19 tests whether this was an inherent scale limitation or fixable bugs.

**Fixes applied**:
1. **Full 4-component triadic loss** (D-A17 had alignment-only)
2. **Differentiable subsumption** — `(x+1)/2` not `(x>0).float()`
3. **Sparsity target loss** (42% target zero rate)
4. **Earlier unfreeze** (10% not 50%)
5. **Triadic warmup** 0%

| Metric | D-A19 (355M, fix) | D-A17 (355M, v2) | D-A14 (40M, v2) |
|---|---|---|---|
| Bit accuracy (holdout) | 97.1% | **97.7%** | 93.0% |
| Subsumption (test) | 76.9% (895 pairs) | **1.7%** | **98.3%** |
| Analogy (R3) | **100% (6/6)** cos 0.81-0.92 | 6.7% | ~100% |
| Dead bits | **16/63** | 26/63 | 26/63 |
| Always-on bits | 4/63 | — | — |
| Ternary (per-token) | -1=70.8%, 0=1.0%, +1=28.2% | — | — |
| Unique signatures | 79.7% (126/158) | 84.2% (133/158) | 100% |

**Eval script**: `playground/eval_d_a19.py`

**Verdict**: **PASS — algebra restored at 355M scale.** The D-A17 failures were caused by bugs (alignment-only loss, non-differentiable subsumption), NOT an inherent scale limitation. D-A19 recovers 76.9% subsumption (vs 1.7%) and 100% analogy (vs 6.7%). Subsumption does not fully match D-A14's 98.3%, but the algebraic capability is clearly present. The scale-algebra tradeoff is real but mitigable, not a hard wall.

### 4.6 D-A11: Negative Baselines (2026-03-18)

**Status**: COMPLETE | **GPU**: 0 (CPU)

| Baseline | Accuracy |
|---|---|
| Random projections | 50.0% ± 2.1% |
| Shuffled labels | 81.4% ± 1.4% |
| Trivial (majority class) | 90.2% |
| **D-A5 Real R3** | **90.7%** |

p-value = 0.0000 (0/1000 permutations), Cohen's d = 6.64. **D-A5 claims validated.**

> Full details: experiment_log.md, lines 2758-2787

### 4.7 D-A10: iFSQ Binary Ablation (2026-03-18)

**Status**: COMPLETE | **GPU**: ~95m | **Checkpoint**: `danza_ifsq_binary_xl/`

| Metric | Value |
|---|---|
| Lang loss | **0.924** (BEST of all experiments) |
| Sub holdout | 87.1% |
| Dead bits | 30 |

**Finding**: **iFSQ activation is the critical innovation**, not ternary quantization. Binary iFSQ achieves the best language loss while maintaining subsumption.

> Full details: experiment_log.md, lines 2943-2969

### 4.8 D-A9: Hybrid Bits + Adversarial (2026-03-19)

**Status**: COMPLETE | **GPU**: ~95m | **Checkpoint**: `danza_hybrid_adv_xl/`

| Metric | Value |
|---|---|
| Bit acc test | 69.3% (30 supervised bits only) |
| Dead bits | **6/63** (lowest of any model) |
| Triadic 3-way | 17 (including cross-domain) |
| Active bits | 50/63 |

**Finding**: Free bits learn genuine semantic content via adversarial training. Cross-domain triadic interactions prove compositional features emerge unsupervised. Trade-off: coverage (50 active) vs accuracy (69.3%).

> Full details: experiment_log.md, lines 3473-3519

### 4.9 D-A8: Ternary Head Variants (2026-03-18)

**FSQ** (50K steps): loss 0.951, sub 86.5%, ternary distribution 1.3%/73.3%/25.3%. MAJOR positive — 3-state ontology works without destroying LM.

**Absmean** (25K steps): loss 1.309 (inferior). Not recommended.

> Full details: experiment_log.md, lines 2913-2997

### 4.10 D-A6: Bootstrap Loop (2026-03-18)

**Status**: COMPLETE | **GPU**: 132.4m

Converged at cycle 0 — 0 pseudo-anchors accepted. Confidence gate too strict. R3 helps individual concepts (reina +22.2%) but net +3.1% < +5% threshold. **Bootstrap hypothesis NOT confirmed at XL scale.**

> Full details: experiment_log.md, lines 3263-3361

### 4.11 D-A5: Bootstrap — Half-Anchor Algebraic Prediction (2026-03-18)

**Status**: COMPLETE | **GPU**: 103.4m | **Checkpoint**: `danza_bootstrap_xl/`

| Metric | Value | Target | Verdict |
|---|---|---|---|
| Holdout direct | 87.5% | >75% | PASS |
| Algebraic (R3) | 90.7% | >80% | PASS |
| Algebraic > direct +5% | +3.1% | +5% | FAIL |
| Reachable > unreachable | — | +10% | UNTESTED |
| reina via R3 | 100% | — | HIGHLIGHT |

**Finding**: R3 algebraic (90.7%) crosses trivial baseline (90.2%); reina achieves 100% via canonical analogy. 2/4 criteria PASS.

> Full details: experiment_log.md, lines 2274-2523

### 4.12 D-A2/D2: Full XL Supervised Run (2026-03-17)

> *Source: danza_experiment_log.md (UNIQUE — not in experiment_log.md)*

**Script**: `playground/danza_63bit.py --scale xl --steps 50000` | **Time**: 100.8 min

| Metric | Train | Test | Target | Verdict |
|---|---|---|---|---|
| Bit accuracy | 100.0% | **89.5%** | >85% | **PASS** |
| Subsumption | 100.0% | **90.0%** | >90% | **PASS** |
| Dead bits | 27/63 | — | <15 | **FAIL** |

**Regla de Tres** (6 quads):

| Quad | Cosine | Bit Acc |
|---|---|---|
| man:woman = king:queen | +0.917 | 90.5% |
| happy:sad = love:hate | +0.898 | 90.5% |
| open:close = free:prisoner | +0.817 | 88.9% |
| teach:learn = king:queen | +0.761 | 85.7% |
| cold:hot = quiet:loud | +0.619 | 79.4% |
| bright:dark = loud:quiet | +0.531 | 77.8% |
| **Mean** | **+0.757** | **85.4%** |

Per-concept worst: loud (78%), liquid (84%), still (87%). Best: slow (92%), happy (94%), sun (98%).

R3 dropped from 97.4% (100 steps) to 85.4% (50K) — known XL overfitting pattern.

### 4.13 D-A1: Post-Hoc Analysis of Self-Supervised Bits (2026-03-17)

> *Source: danza_experiment_log.md (UNIQUE — 5 detailed tests)*

**Script**: `playground/danza_posthoc_analysis.py` (CPU, ~30s)
**Checkpoint**: Run 15 (64 self-supervised bits, 40M params)
**Concepts**: 102 encoded (90 in 12 domains, 12 extra for analogies)

| Test | Key Metric | Value | Expected (real) | Expected (arbitrary) | Verdict |
|---|---|---|---|---|---|
| A1.1 Dual Axes | Anti-correlated pairs (r<-0.3) | **22** | ~10-15 | <5 | **PASS** |
| A1.2 Hierarchy | DAG depth | **2** | 4-6 | 0-1 | WEAK |
| A1.3 Abstraction | Foundation bits (>60%) | **20** | skewed | uniform | MIXED |
| A1.4 Semantic Probe | Bits with >50% purity | **1** | 10-20 | <5 | FAIL |
| A1.5 Regla de Tres | Bit accuracy | **70.6%** | >90% | ~50% | WEAK |

**A1.1 — Dual Axes: PASS.** 22 anti-correlated pairs. Strongest: bit 37↔39 (r=-0.55). Bit 28 has only 3 activations (love, hate, nose) but strongest coherence — emotions isolate in their own bit.

**A1.2 — Hierarchy: WEAK.** 61 dependency edges, DAG depth only 2 (vs expected 4-6). Model compresses hierarchy into fewer layers.

**A1.3 — Abstraction: MIXED.** 20 foundation bits (>60%), 6 rare (<10%). Mean 48.4%. Non-uniform (KL=7.68) but different shape than Sistema.

**A1.4 — Semantic Probe: FAIL.** Only 1 bit with >50% purity (bit 28 → emotions, n=3). Mean purity 0.189 (2.3× chance). Bits don't map to human categories.

**A1.5 — R3 Transfer: WEAK.** 70.6% mean (all quads fail >90% threshold). BUT gender transform is consistent: 7 bits (5,6,12,20,21,47,63) flip across ALL 4 gender quads.

**Overall conclusion**: Self-supervised model discovers PARTIAL structure (dual axes, consistent transforms, abstraction gradient) but does NOT reconstruct the full 6-layer hierarchy or category-pure bits. **The Sistema 7×7 is a USEFUL FRAMEWORK but does not emerge COMPLETE without supervision.**

### 4.14 D1: Smoke Test (2026-03-17)

> *Source: danza_experiment_log.md (UNIQUE)*

**Script**: `playground/danza_63bit.py --scale base --steps 100` | **Time**: 5 seconds

| Metric | Train | Test |
|---|---|---|
| Bit accuracy | 90.0% | 88.7% |
| Subsumption | 88.4% | 90.0% |
| R3 mean cosine | +0.974 | 97.4% bit acc |

Script validated. Ready for full XL run.

### 4.15 Planned Experiments (NOT YET RUN)

> *Source: danza_experiment_log.md (UNIQUE)*

**D-A3: Cross-Lingual Zero-Shot.** Train English → evaluate Spanish. Success: Hamming < 10 between same-concept pairs (84% match). Challenge: BPE is English-only. Status: PLANNED.

**D-A4: Fully Unsupervised Primitive Discovery.** Train k=63 self-supervised → Hungarian algorithm to align with gold. Success: post-permutation accuracy > 70% (strong) or > 50% (moderate). **The definitive test** of whether the Sistema is discoverable or constructed. Status: PLANNED.

**D-A7: Scale Test (307M).** If bootstrap works at 40M, test at 307M with all 2.1M stories. Estimated 4h GPU. Status: PLANNED (after bootstrap proof).

**D-A12: Dead-Bit Surgery.** Remap 30/63 dead bits to discriminative primitives from inventory. CPU-only. Status: PREPARED (never run — superseded by D-A9 hybrid approach).

---

## 5. Line E: Validation (E1-E7, B1-B3)

### 5.1 E1: Multi-Seed Validation (2026-03-17)
**GPU**: 7.3h | 3 seeds (42, 123, 777) | Gap: **+0.038 ± 0.005** | PPL: 10.86 ± 0.01 | Analogy: **100% ± 0%** | 95% CI: [+0.029, +0.046] entirely positive. **Reproducible.**
> experiment_log.md, lines 1794-1846

### 5.2 E2: Alignment Loss Ablation (2026-03-17)
**GPU**: 7.3h | 3 variants: FULL, NO_ALIGN, NO_ENTROPY | **Alignment is THE driver** — without it, dead bits 11→23 (+109%). Entropy reg is redundant when alignment present. wte embeddings = semantic teacher.
> experiment_log.md, lines 1905-1956

### 5.3 E3: Expanded Analogy Benchmark (2026-03-15)
**GPU**: 0 (eval) | 51 quads (up from 13) | Verification: **98.0%** (50/51) — revised UP from 69.2%. Discovery: ~0% (expected). Original benchmark was too small.
> experiment_log.md, lines 1686-1742

### 5.4 E4/E4b: Subsumption Weight Sweep (2026-03-17)
**GPU**: 12.5h + 4.2h | Best: 92.3% at w=2.0 (80% warmup). E4b (50% warmup): 76.9% sub, 24 dead bits (vs 44). **Warmup-subsumption paradox**: more active steps = better bits but worse sub. Both are valid tradeoff points.
> experiment_log.md, lines 2006-2102

### 5.5 E5: Scale Interpolation (2026-03-17)
**GPU**: 5.2h | 25M gap +0.010, 30M gap +0.043 | **Zero-crossing ~20M params, gradual transition** (NOT sharp phase transition). Paper's analogy should be softened.
> experiment_log.md, lines 1960-2002

### 5.6 E6: Compression Benchmark (2026-03-15)
**GPU**: 0 (eval) | Triadic centroid: 13.3%, Embedding centroid: 16.4% | **"8x compression" NOT supported.** Revise to "8x compression with no language-modeling cost" (which IS supported).
> experiment_log.md, lines 1746-1790

### 5.7 E7/E7v2: R3 at Low k (2026-03-16/17)
**GPU**: ~90m | k=6: 1/6 dead (no collapse). BUT gap -0.27 to -0.42 (destroyed). E7v2 validated — word overlap was NOT a confound. **R3 impractical at any k.**
> experiment_log.md, lines 1850-2167

### 5.8 XL2: Sigmoid+Anneal temp=5 (2026-03-17)
**GPU**: 5.2h | PPL 16.18 (+110%), gap -0.003 | **Sigmoid+anneal does NOT scale to XL** regardless of temperature. Overfitting inherent at 40M.
> experiment_log.md, lines 2106-2129

### 5.9 B1: Embedding Gap Baseline (2026-03-17)
**GPU**: 0 | Embedding gap +0.014, Triadic gap +0.038 | **Triadic amplifies 2.6x in 8x fewer dims.** Head is NOT merely copying — it concentrates and amplifies.
> experiment_log.md, lines 2171-2192

### 5.10 B2: Pure Language Baseline (2026-03-17)
**GPU**: 143m | Pure PPL 10.65, Triadic PPL 10.86 | **True cost = +2% PPL.** Random head gap +0.056 > trained +0.038, BUT random analogy 16.7% vs trained 100%.
> experiment_log.md, lines 2196-2223

### 5.11 B3: Frozen Random Head (2026-03-17)
**GPU**: 144m | Frozen gap +0.022, analogy 33.3% | Trained analogy 100%, ordering CORRECT vs WRONG. **Training is CRITICAL for algebraic ops.**
> experiment_log.md, lines 2227-2254

---

## 6. Line C: Playground Explorations (P-series)

### 6.1 P15: 49-Bit Concept GPT (2026-03-15)
**GPU**: 103m | Primary acc 86.2% (42x random), 0 dead bits, sub 97.3% | BUT: held-out 17% = memorization, not compositional generalization. Requires tanh + full supervised + subsumption. Sigmoid+subsumption collapses.
> experiment_log.md, lines 1552-1682

### 6.2 P12: XL Subsumption Loss (2026-03-14)
**GPU**: ~9h | **100% held-out @25K** (paper limitation RESOLVED). PPL +47% at XL. Base scale is "free lunch" (language improves). Early stopping at 25K optimal.
> experiment_log.md, lines 1369-1432

### 6.3 P6: Subsumption Loss — BREAKTHROUGH (2026-03-14)
**GPU**: ~5m | Train sub: 100% (from 0%!), held-out: 91.7%. Language improved (1.707 from 1.810). Emergent hypernym sparsity: 35→2.3 active bits.
> experiment_log.md, lines 1057-1127

### 6.4 P7/P10/P11: R3 Experiments (all failed)
- **P7** (R3+Sub combo): R3 generalizes BUT causes 64/64 dead bits. Sub-only dominates.
- **P10** (R3+entropy guard): Even 20x entropy weight can't prevent R3 collapse.
- **P11** (Curriculum Sub→R3): R3 destroys Sub's structure in 3K steps.
**Conclusion: R3 is fundamentally broken at k=64.**
> experiment_log.md, lines 1130-1365

### 6.5 Other P-series (condensed)

| ID | Finding | Detail |
|---|---|---|
| P14 | Post-hoc concept projection: ~20% (negative) | Run 15 embeddings insufficient for 7×7 |
| P13 | Cross-dataset: LAMBADA 345, WikiText 3033 PPL | OOD expected, triadic neutral |
| P9 | Info hierarchy: 93% bit reduction in hypernyms | Emergent set-theoretic structure |
| P8 | Phase-aware attention: negative | Learned positions are strictly better |
| P5 | Rule-of-Three loss: K=1.0 (perfect) | But memorization, not generalization |
| P4 | XL Sigmoid+anneal: PPL +116% | Overfitting at 40M, mixed results |
| P3 | Soft signatures: sigmoid+anneal best | +0.039 gap, 0 dead bits vs tanh |
| P2 | Random baseline: frozen > trained at 5.8M | Semantic ordering only at 40M+ |
| P1 | Sinusoidal head: +0.021 gap | +4 dead bits, periodicity interesting |
| P0 | K-constant: mean K=1.21 | R3 approximately holds (K=1 is perfect) |

### 6.6 Research Line Designs (from PLAN.md — UNIQUE)

> *These designs motivated the P-series experiments. Preserved for context.*

**Línea 1 — Conceptual Tokenizer**: Replace BPE (frequency-based) with concept-based tokenization. 3 experiments: vocabulary builder (P2 DONE — silhouette -0.059), hybrid tokenizer (PENDING), concept-token GPT (→ P15 DONE).

**Línea 2 — Wave Model**: Test sinusoidal activation from the book's wave theory. 2 experiments: sin head (P1 DONE — +0.021 gap), phase-aware attention (P8 DONE — negative).

**Línea 3 — Regla de Tres**: Algebraic loss component. 2 experiments: R3 loss (P5 DONE — memorization), K-constant analysis (P0 DONE — K=1.21).

**Línea 4 — Subsumption Recovery**: Fix 0% subsumption at k=64. 2 experiments: supervised bit inheritance (P6 DONE — BREAKTHROUGH), hierarchical bit allocation (redundant with P6).

**Línea 5 — Dead Bits**: Activate the ~15 dead bits. 2 experiments: L1 sparsity (P5.1 DONE — redundant with entropy reg), adaptive k via Gumbel (PENDING — high effort).

**Línea 6 — Quantum Superposition**: Soft signatures during training. 1 experiment: sigmoid→hard annealing (P3 DONE — best base result).

**Línea 7 — Cross-Validation**: Test generalization. 2 experiments: cross-dataset (P13 DONE — OOD expected), random baseline (P2 DONE — critical control).

#### Prioritization Table (28 experiments, from PLAN.md)

| # | Experiment | Impact | Effort | GPU hrs | Priority | Status |
|---|-----------|--------|--------|---------|----------|--------|
| 3.2 | K-Constant Analysis | medio | bajo | 0 | **P0** | DONE |
| 2.1 | Sin Head | medio | bajo | ~1h | **P1** | DONE |
| 5.1 | L1 Dead Bits | medio | bajo | ~1h | **P1** | DONE (redundant) |
| 6.1 | Soft Signatures | alto | bajo | ~2h | **P1** | DONE (BEST) |
| 7.2 | Random Baseline | alto | bajo | ~2h | **P1** | DONE (critical) |
| XL | Sigmoid+Anneal XL | alto | alto | ~3h | **P1** | DONE (PPL +116%) |
| 1.1 | Concept Vocab | alto | medio | ~2h | **P2** | DONE |
| 3.1 | Rule-of-Three Loss | alto | medio | ~30m | **P2** | DONE |
| 4.1 | Subsumption Loss | alto | medio | ~30m | **P2** | DONE BREAKTHROUGH |
| 2.2 | Phase Attention | medio | medio | ~30m | **P3** | DONE (negative) |
| R3+S | R3 + Subsumption combo | muy alto | bajo | ~30m | **P2** | DONE (Sub wins) |
| P9 | Info Hierarchy Analysis | alto | 0 | 0 | **P0** | DONE (93% reduction) |
| P10 | R3 Entropy Guard | alto | bajo | ~50m | **P2** | DONE (unfixable) |
| P11 | Curriculum Sub→R3 | alto | bajo | ~30m | **P2** | DONE (R3 erases Sub) |
| 1.2 | Hybrid Tokenizer | muy alto | alto | ~5h | **P3** | pending |
| 4.2 | Hierarchical Bits | medio | alto | ~4h | **P3** | redundant with 4.1 |
| 5.2 | Adaptive k | alto | alto | ~6h | **P3** | pending |
| XL2 | Sigmoid+Anneal (temp=5) | alto | alto | ~3h | **P3** | DONE (+110% PPL) |
| XL-Sub | XL Subsumption Loss | muy alto | alto | ~9h | **P1** | DONE (100% @25K) |
| P14 | Concept Head (Phase 4) | alto | medio | ~1min | **P2** | DONE (negative) |
| 1.3 | Concept GPT (49-bit e2e) | muy alto | muy alto | ~2h | **P1** | DONE 86.2% |
| 7.1 | Cross-Dataset Eval | medio | medio | ~2min | **P4** | DONE (OOD expected) |
| E1 | Multi-Seed (3 seeds) | critical | alto | ~7h | **P0** | DONE +0.038 +/- 0.005 |
| E2 | Alignment Ablation | alto | alto | ~7h | **P1** | DONE — alignment is driver |
| E3 | Expanded Analogy (51 quads) | alto | 0 | 0 | **P0** | DONE — 98% verification |
| E4 | Sub Weight Sweep (80% warmup) | alto | muy alto | ~12h | **P2** | DONE |
| E4b | Sub Weight Sweep (50% warmup) | critical | muy alto | ~4h | **P1** | DONE — 76.9% sub |
| E5 | Scale Interpolation (25M/30M) | alto | alto | ~5h | **P1** | DONE — crossover ~20M |
| E6 | Compression Benchmark | alto | 0 | 0 | **P0** | DONE — claim NOT supported |
| E7 | R3 at Low k (6/8/12) | medio | bajo | ~45m | **P2** | DONE — alive but kills gap |
| E7v2 | R3 Low k (clean words) | bajo | bajo | ~45m | **P2** | DONE — E7 validated |
| B1 | Embedding Gap Baseline | critical | 0 | 0 | **P0** | DONE — triadic amplifies 2.6x |
| B2 | Pure Language XL | critical | alto | ~2.5h | **P0** | DONE — +2% PPL |
| B3 | Frozen Random Head XL | critical | alto | ~2.5h | **P0** | DONE — training = algebraic ops |

---

## 7. Line D: Transfer Learning (Experiment 10)

### 7.1 E10-v3: GPT-2 + InfoNCE — Bug #7 Fix (2026-03-19)
**GPU**: 45.5m | Gap **+0.076** (corrected from +0.099). Gap closure: **48%** (not 72%). Bug #7: temp 0.1→0.5, eps in F.normalize, clamp logits ±30. Paper updated.
> experiment_log.md, lines 2640-2702

### 7.2 E10b: Rank Alignment (2026-03-08)
**GPU**: ~25m | Gap +0.047, **analogy 83.3%** (best). Rank outperforms InfoNCE on analogies because it preserves ordinal structure.
> experiment_log.md, lines 593-654

### 7.3 E10a: MSE Alignment (2026-03-08)
**GPU**: ~25m | Gap +0.011. NEGATIVE — MSE wastes capacity on absolute matching when embeddings are rich.
> experiment_log.md, lines 526-589

### 7.4 Runs 27-29: From-Scratch InfoNCE/Rank/Staged (2026-03-08)
All NEGATIVE — InfoNCE and Rank fail from-scratch because TinyStories embeddings lack structure for pos/neg mining. MSE is definitively best for weak embeddings. **The loss-embedding interaction is structural, not temporal.**
> experiment_log.md, lines 658-829

---

## 8. Line A: Core Model Development (Runs 1-18)

### 8.1 Run 15: Strong Alignment — PRODUCTION (2026-03-07)
**v1.4-strongalign** | 40M params | alpha=0.05, align=5.0 | loss **0.946**, entropy 0.749, PPL **7.69**, gap +0.020 | First run with correct semantic ordering (+21pt gap). **Pareto-optimal configuration.**
> experiment_log.md, lines 192-208

### 8.2 Runs 12-14: Coherence Collapse → Fix (2026-03-07)
- **Run 12**: Entropy reg → COLLAPSED (entropy 0.000, 0.9% unique). **Root cause: coherence loss.**
- **Run 13**: No coherence → diverse but random (entropy 0.521).
- **Run 14**: + Alignment → semantics emerging (entropy 0.720, +17pt gap).
> experiment_log.md, lines 136-188

### 8.3 Run 10-11: Distillation + Audit (2026-03-05)
- **Run 10**: Distillation at 5x weight = triadic collapse. Made configurable (default 1.0).
- **Run 11**: Industrial audit — 98.5% accuracy, 0.96% FPR on 2000 word pairs.
> experiment_log.md, lines 77-125

### 8.4 Run 9: First XL Model (2026-03-05)
40.17M params | loss 1.277 | 161.8 min | Best language loss to date. XL scale is where quality emerges.
> experiment_log.md, lines 60-74

### 8.5 Runs 1-8: Baseline Through GPU (2026-03-04/05)
CPU baseline (loss 1.75) → TinyStories (3.23) → PyTorch GPU (1.55) → Diversity fix (1.37) → Fixed tokenizer (1.59). Progression from 866K to 15.9M params.
> experiment_log.md, lines 2-57

### 8.6 Phase Narrative (from EVOLUTION_PLAN.md — UNIQUE)

| Phase | Focus | Outcome |
|---|---|---|
| 0 | Baseline audit | Run 9 was NOT collapsed; Run 10 distillation caused it |
| 0.5 | Diagnostic | XL Pure model has 97.3% unique signatures |
| 1 | Triadic quality | Coherence=collapse → alignment=teacher → Run 15 production |
| 2 | Language ablation | PPL 7.69 vs 7.56 → zero cost |
| 3 | Triadic benchmarks | Subsumption 0% at k=64 (→ fixed by P6) |
| 4 | Scaling study | Gap crosses zero ~20M, bits sweep k=32-64 optimal |
| 5 | Transfer (GPT-2) | InfoNCE +0.076, loss-embedding interaction |
| 6 | Paper preparation | 16+ pages, compiled |
| 7 | Staged training | Run 29 negative, confirms structural interaction |

---

## 9. Line B: Scaling & Architecture

### 9.1 Runs 19-21: Scale Study (2026-03-07)
Small (1.3M) gap -0.076, Base (5.8M) gap -0.040, Large (15.9M) gap -0.034, XL (40M) gap +0.020. **Semantic ordering is emergent only at XL.**
> experiment_log.md, lines 362-402

### 9.2 Runs 22-26: Bits Sweep k=8-128 (2026-03-07)
U-shaped language loss (minimum k=64), gap peaks k=32, analogy best k=48-64. **End-to-end training shifts optimal k from paper's k=6-12 to k=32-64.**
> experiment_log.md, lines 406-451

### 9.3 Experiment 11: Sentence Aggregation (2026-03-10)
Token-level sep ratio 1.02 → sentence-level **1.21 (+19%)**. Best: family (1.42), worst: emotions (1.11). **Domain structure exists but was invisible at token level.**
> experiment_log.md, lines 833-921

### 9.4 Experiment 9: Engine Comparison — Table 7 (2026-03-07)
5 methods compared. PCA = Contrastive best gap (+0.136). Subsumption 0% for ALL methods at k=64. **Embedding quality > projection method.**
> experiment_log.md, lines 493-522

---

## 10. Tests & Benchmarks

### Unit Tests

| Suite | Tests | Status |
|---|---|---|
| Core (autograd, transformer, triadic) | 37 | ALL PASS |
| triadic-head package | 33 | ALL PASS |
| reptimeline discovery | ~10 | ALL PASS |
| **Total** | **~80** | **ALL PASS** |

### Benchmark Suite (12/12 COMPLETE)

| Benchmark | Key Result |
|---|---|
| Scaling Study | Phase transition ~20M |
| Bit Entropy | Mean H=0.749 |
| Language Quality | PPL 7.69 vs 7.56 (cost=0) |
| Analogy | 98% verification (51 quads) |
| Subsumption | **0% recall** (Run 15); 98.3% (D-A14 v2); 100% train (P12 w/ sub loss) |
| Interpretability Probe | Linear probe |
| Engine Comparison | 5 methods (Table 7) |
| Domain Topology | 1.21 sentence-level |
| Bit Evolution | 50K trace |
| Bits Sweep | k=8-128 |
| Scaling Plots | Publication figures |
| Prime vs Bitwise | 1000/1000 equiv, 5-78x |

### Audit Tests Executed

| Test | Result | Date |
|---|---|---|
| L11/L12 v2: Indifference + false opposites | **PASS** | 03-19 |
| L15 v2: Aristotelian types | FAIL (0/4 significant) | 03-19 |
| L19 v2: Enantiodromia | FAIL (2/8) | 03-19 |
| L1: Bridge PFs | 3/4 PASS (Q6 FAIL) | 03-19 |
| L2: D-A13 formal eval | 88% bits, sub 9-20% | 03-19 |
| L3: Blind prime assignment | PASS (vacuous) | 03-19 |
| F0: Data validation | 93.7% bit acc | 03-19 |

### Book Corrections (L4-L10)

| Line | Correction | Evidence | Status |
|---|---|---|---|
| L4 | "40M" → "~20M" params threshold | E5: gap crosses zero at ~20M params, gradual crossover (NOT phase transition). Run 15 (40M) and scale interpolation confirm. | **READY** — use ~20M |
| L5 | Rewrite "8x compression" claim | E6 REFUTED: both triadic and raw are near-random for compression. Correct claim: "no PPL cost" (+2% PPL per B2), NOT "8x no info loss". | **READY** — replace with "no language cost" |
| L6 | Locate source for "108,694 discrepancies" | From Engine's original prime-factor audit on WordNet concepts. The number represents prime assignments that fail algebraic consistency when checked against human-labeled ontological relations. Exact source: `Triadic-Neurosymbolic-Engine` audit logs. | **READY** — cite Engine repo |
| L7 | Update "69.2%" → 98% analogies | E3: 98% verification on 51 quads (revised UP from 69.2% on 13 quads). Discovery ~0%. Paper already updated. | **READY** — use 98% (51 quads) |
| L8 | Add Placer/Dolor to primitives | reptimeline D-A14 v2: placer/dolor dual = 1.00 coherence (strongest axis). vida/muerte = 0.91. 9 total duals discovered. See `reptimeline/results/d_a14_v2_discovery.json`. | **READY** — duals confirmed by model |
| L9 | Include new results (iFSQ, BitNet, R3, composition) | iFSQ: D-A16 93.2% (tied best). BitNet connection: ternary {-1,0,+1} = same as BitNet 1.58-bit. R3: dead at k=64, alive at k=6-12 but destroys gap. Composition: 100% by construction (bitwise OR). | **READY** — 4 results to add |
| L10 | Document zeros→0% collapse at 355M | D-A17: 97.7% bit, 1.7% sub (algebra destroyed). **D-A19 fix: 97.1% bit, 76.9% sub, 100% R3** — root cause was bugs (alignment-only loss, non-diff sub), not inherent scale limitation. | **UPDATED** — D-A19 restores algebra |

### Pending Tests — Before Publication

> Updated: 2026-03-20. D-A19 COMPLETE (97.1% bit, 76.9% sub, 100% R3 — algebra restored at 355M, bugs confirmed as root cause). D-A18 COMPLETE (92.2% sup, 96.5% sub, 15 dead). D-A17 COMPLETE (algebra destroyed). D-A14 confirmed as production model.

#### Tier 0: Critical (blocks paper claims)

| ID | Test | Script | What it validates | Effort | Status |
|---|---|---|---|---|---|
| **D-A17** | GPT-2 355M + v2 anchors | `playground/gpt2_medium_ternary.py --v2` | Fair 355M scaling comparison (D-A13 only had v1 anchors) | ~4.8h GPU | **COMPLETE** — 97.7% bit, 1.7% sub, 26 dead. Algebra destroyed at scale. |
| **D-A17-eval** | Formal eval on D-A17 | `playground/audit_tests/test_d_a13_eval.py --v2` | Does v2 fix the algebra failure at 355M? **NO — 1.7% sub** | 5 min GPU | **COMPLETE** — ternary zeros collapsed (3.4% vs 41.3%), subsumption unsatisfiable |
| **Paper 5.4** | Revise subsumption section | — | Honest reporting: 40M=98.3%, 355M=1.7%. Ternary zeros collapse. | 1h writing | **DONE** (2026-03-19) |
| **Paper 6** | Revise discussion section | — | Scaling destroys algebra (ternary zeros 41.3%→3.4%). Discovery loop added. | 1h writing | **DONE** (2026-03-19) |

**D-A17 status** (updated 2026-03-20): **COMPLETE** — 97.7% bit accuracy, 1.7% subsumption, 26 dead bits. Algebra destroyed at scale (ternary zeros collapsed from 41.3%→3.4%). Formal eval done. Paper sections 5.4, 5.8, 6 updated with honest results. Checkpoint dir: `danza_gpt2medium_ternary_v2/`.

**D-A19 status** (updated 2026-03-20): **COMPLETE** — 97.1% bit accuracy, 76.9% subsumption (895 pairs), 100% R3 (6/6, cos 0.81-0.92), 16 dead bits, 4 always-on. **Algebra restored at 355M.** Root cause of D-A17 failure: alignment-only loss + non-differentiable subsumption. Fix: full 4-component triadic loss + differentiable sub + sparsity target. Best at step 17500, 301 min training. Checkpoint dir: `danza_gpt2_355m_sparsity_v2/`.

#### Tier 1: Valuable (strengthens paper, not blocking)

| ID | Test | Script | What it validates | Effort |
|---|---|---|---|---|
| D-A12 | Bootstrap confidence intervals | `playground/d_a12_bootstrap_ci.py` | 95% CIs for multi-quad ensemble and bootstrap results | 10 min CPU |
| L13 | 1000 adversarial concepts | NEW | Stress test beyond 158 anchors | 4h GPU |
| L14 | PCA real primitive count | NEW | How many independent semantic dimensions exist? | 3h CPU |
| L16 | Polisemia contextual | NEW | Same word, different meaning in context | 1h CPU |

#### Tier 2: Future work (post-publication)

L17: Categorical bits architecture (own paper) | L18: Semantic compositions depth 3-4 | L19: Enantiodromia detection | L20: Centering | L21: Cross-corpus (WikiText2, LAMBADA) | L22: Fourier head | L23: Cross-linguistic | L24: Dataset reconciliation | L25: Polisemia | L26: Wave model v2 | L27: UI tests | L28: 6/9 algebraic structures | L29: Hooke homeostasis | L30: Practice→life | L31: Scale 400M+

### Bug Fixes

| Bug | File | Fix | Status |
|---|---|---|---|
| `distill_target_tensor` undefined | `src/torch_finetune.py` | Changed to `gold_sequences` | **FIXED** |
| GradScaler always enabled | `src/torch_finetune.py` | Added `--dtype` arg, condition on `amp_dtype == float16` | **FIXED** |
| autocast missing dtype param | `src/torch_finetune.py` | Added `dtype=amp_dtype` to autocast | **FIXED** |
| Q4 hardcoded `cat5_start=28` | `test_pf_bridge.py` | Uses actual `ejes_duales` from primitivos.json (12 real dual pairs) | **FIXED** |
| Q5 `bit_idx // 7` categories | `test_pf_bridge.py` | Builds categories from actual `capa` field | **FIXED** |
| Q6 wrong observer bits (56-62) | `test_pf_bridge.py` | Uses real bits: consciente(36), ausente(37), temporal_obs(38), eterno_obs(39), receptivo(42), creador_obs(43) | **FIXED** |
| Hardcoded tokenizer paths | `src/auditor.py`, `src/test_generalization.py` | Migrated to BitwiseMapper + argparse | **FIXED** |

---

## FINAL MODEL PLAN — D-A18: Unified Optimal Architecture

> Updated: 2026-03-19
> Goal: ONE model that combines ALL proven optimal components

### Evidence Summary — What to Combine

Each component was validated independently. They have never been combined into a single model:

| Component | Best Result | Source | Key Number |
|---|---|---|---|
| **v2 anchors (158)** | +13.6pp test accuracy | D-A14 vs D-A5 | 93.0% vs 79.4% |
| **iFSQ activation** | Best LM loss | D-A10 | 0.924 (vs tanh 0.946) |
| **Hybrid bits (30+33)** | Fewest dead bits | D-A9 | 13/63 dead (5 sup + 8 free) (vs 26/63) |
| **BitwiseValidator** | O(1) algebra, 5-78x faster | benchmarks | 1000/1000 equivalent |
| **v2 + iFSQ** | Best R3, tied accuracy | D-A16 | R3=0.842, test 93.2% |

**What D-A16 proved**: iFSQ + v2 together ≈ v2 tanh on accuracy (93.2% vs 93.0%), better R3 (0.842 vs ~0.6), slightly worse LM (0.993 vs 0.946). **v2 anchors dominate over activation.**

**What hasn't been tested**: Hybrid bits (30 sup + 33 free) with v2 anchors. D-A9 only used v1 (54 anchors). The hybrid approach reduced dead bits from 26→13 but with inferior anchors.

### D-A18 Architecture

```
Text → BPE (4096) → TriadicGPT (12L/512D/8H, 40M params)
                          |
                    +-----+-----+
                    |           |
               LM Head    Hybrid Triadic Head
               (softmax)  |
                          +-- Supervised (30 bits) → iFSQ → gold labels (158 v2 anchors)
                          +-- Free (33 bits) → iFSQ → contrastive + adversarial disentanglement
                          |
                     BitwiseValidator (O(1))
                     - subsumes:  (A & B) == B
                     - analogy:   (C & ~oA) | oB
                     - compose:   A | B
                     - similarity: popcount ratio
                          |
                     reptimeline (post-hoc discovery)
```

### Expected Outcomes

| Metric | D-A14 (v2 tanh) | D-A9 (hybrid v1) | D-A18 Target | **D-A18 Actual** |
|---|---|---|---|---|
| Test accuracy | 93.0% | 69.3% | ≥90% | **92.2% sup / 75.3% full** |
| Subsumption | 98.3% | — | ≥95% | **96.5%** (864/895) |
| Dead bits | 26/63 | 6/63 | <10/63 | **15/63** (22 formal) |
| Active bits | 37/63 | 57/63 | >50/63 | **48/63** |
| LM loss | 0.946 | ~1.0 | <0.95 | TBD |
| Zero rate | ~42% | — | ~42% | **61.4%** (too sparse) |
| R3 round-trip | ~60% | — | >80% | **85.1%** cosine |
| Sig uniqueness | — | — | — | **56.3%** (89/158) |

### Implementation Steps

#### Phase 1: D-A18 Training Script (1 day, no GPU) — DONE

**Script**: `playground/unified_final.py` — `UnifiedTriadicGPT` model class.

Combines:
- Base: `hybrid_adversarial.py` (30+33 split, gradient reversal)
- Anchors: v2 from `danza_data/anclas_v2.json` (158 concepts)
- Activation: iFSQ (`2*sigmoid(1.6x)-1`) in both `sup_head` and `free_head`
- Eval: BitwiseValidator from `src/triadic.py`
- Mixed precision: `--dtype bfloat16` (Blackwell optimization)
- Resume support: `--resume <checkpoint>`

Verified: syntax OK, imports resolve, forward pass produces correct shapes (logits, proj, loss, sup_proj, adv_logits).

#### Phase 2: D-A18 Training (105 min GPU) — COMPLETE

```bash
python playground/unified_final.py --scale xl --steps 50000 --dtype bfloat16
```

**Results** (2026-03-20):

| Metric | D-A18 Result | Target | Pass? |
|---|---|---|---|
| Test sup accuracy | 92.2% | ≥90% | **PASS** |
| Full bit accuracy | 75.3% | — | — |
| Subsumption (formal) | 96.5% (864/895) | ≥95% | **PASS** |
| Dead bits | 15/63 (22 formal) | <10 | **FAIL** |
| Active bits | 48/63 | >50 | **FAIL** |
| Zero rate | 61.4% | ~42% | **FAIL** (too sparse) |
| Signature uniqueness | 56.3% (89/158) | — | Low |
| R3 cosine | +0.851 | >80% | **PASS** |
| Adversary accuracy | 92.7% | ~50% | **FAIL** (not converged) |
| Training time | 105.3 min | — | — |

**Analysis**: Hybrid 30+33 successfully reduces dead bits vs D-A14 (15 vs 26) and activates more bits (48 vs 37). However, zero rate 61.4% (vs D-A14's ~42%) means the model over-zeroes, hurting signature uniqueness (56.3% vs D-A14's higher). Adversary at 92.7% (target 50%) indicates the backbone still encodes supervised information — gradient reversal didn't fully decouple.

**reptimeline BitDiscovery** (2026-03-20):

| Metric | D-A18 | D-A14 v2 |
|---|---|---|
| Active bits | 41 | 49 |
| Dead bits | 22 | 14 |
| Duals | 5 | 14 |
| Dependencies | 855 | 526 |
| **Triadic interactions** | **11** | **328** |

The 30× reduction in triadic interactions is the strongest evidence against the hybrid approach. High dependency count (855) reflects trivial always-ON bit correlations, not genuine structure. D-A14's 328 triadic interactions emerge from a balanced ternary code; D-A18's binary +1/0 code cannot produce meaningful three-way relationships.

**Verdict**: D-A14 remains production model. D-A18 demonstrates the hybrid mechanism works directionally for subsumption but fundamentally impoverishes algebraic structure.

#### Phase 3: BitwiseValidator as Runtime Default (1 day, no GPU) — DONE

Added `DefaultMapper = BitwiseMapper` and `DefaultValidator = BitwiseValidator` aliases in `src/triadic.py`.

| File | Status |
|---|---|
| `src/triadic.py` | **DONE** — DefaultMapper/DefaultValidator aliases |
| `src/evaluate.py` | **DONE** — uses BitwiseMapper + BitwiseValidator |
| `playground/audit_tests/common.py` | **DONE** — exports defaults, added `proj_to_bitmask()` |
| `playground/audit_tests/test_pf_bridge.py` | **DONE** — uses DefaultMapper/DefaultValidator |
| `src/torch_train.py` | Unchanged (uses PrimeMapper for legacy compat) |
| `benchmarks/scripts/*.py` | Unchanged (historical, PrimeMapper results are canonical) |
| `ui/model_interface.py` | Unchanged (low priority, display-only) |

Playground experiments left as-is (historical, already ran with PrimeMapper). Critical runtime paths (eval, audit tests) migrated.

#### Phase 4: D-A18 Evaluation & reptimeline (1 day)

Run the full audit battery on D-A18:
1. `test_indifference_and_false_opposites.py` — L11/L12
2. `test_aristotelian_types.py` — L15 (may pass with more active bits)
3. `test_enantiodromia.py` — L19 (may improve with free bits)
4. `analyze_v2.py` — reptimeline discovery (target: >50 triadic 3-way)
5. Full benchmark suite (12 benchmarks)

#### Phase 5: D-A17/D-A19 Scaling Test — COMPLETE (NEGATIVE REVERSED)

1. **D-A17 training**: COMPLETE — 50K steps, 289.7 min. Best holdout 91.6%.
2. **D-A17 eval**: COMPLETE — **97.7% bit accuracy but 1.7% subsumption (vs 98.3% at 40M)**
3. **Key finding (D-A17)**: 355M destroys algebraic structure. Ternary zeros collapse 41.3% → 3.4%, making `(A & B) == B` unsatisfiable.
4. **D-A19 (355M fix)**: COMPLETE — **76.9% sub, 100% R3, 16 dead bits.** Root cause was bugs in D-A17 (alignment-only loss, non-differentiable subsumption), NOT an inherent scale limitation. Full 4-component triadic loss + differentiable sub + sparsity target restores algebra at 355M. 301 min, best at step 17500.

#### Phase 6: Paper Update — DONE

- Section 5.4: Added D-A17 row to table, rewrote "fourth result" with honest scaling findings
- Section 6: Updated subsumption FPR discussion with ternary sparsity bracket finding
- Conclusion: Changed to 8 core results, added discovery loop, corrected subsumption claim
- Future work: Replaced "scaling → perfect generalization" with scale-algebra tradeoff + discovery loop

### Timeline

| Phase | Task | Depends on | GPU | Status |
|---|---|---|---|---|
| 1 | Write `unified_final.py` | — | No | **DONE** |
| 2 | Train D-A18 | Phase 1 | **Yes** | **DONE** (105 min, 92.2% sup test, 96.5% sub, 15 dead bits) |
| 3 | BitwiseValidator default | — | No | **DONE** (critical paths migrated) |
| 4 | D-A18 eval + reptimeline | Phase 2 | Partial | **DONE** (formal eval + reptimeline 2026-03-20: 41 active, 5 duals, 11 triadic vs D-A14's 328) |
| 5a | D-A17 training | — | **Yes** | **DONE** (97.7% bit, 1.7% sub) |
| 5b | Eval D-A17 | Phase 5a | 5 min | **DONE** (algebra destroyed at scale) |
| 6 | Paper update | Phase 5b | No | **DONE** |

**All phases complete.**
**Production model**: D-A14 v2 tanh (93%, 98.3% sub, 328 triadic interactions). D-A18 reptimeline confirms: hybrid architecture collapses triadic structure (11 interactions vs 328).

### Fallback Strategy

If D-A18 (hybrid + iFSQ + v2) underperforms:
- **Accuracy <88%**: Use D-A16 (iFSQ + v2, all supervised) — already validated at 93.2%
- **Dead bits still >20**: Use D-A14 (v2 tanh) — already validated at 93.0%, accept dead bits
- **LM loss >1.1**: Use tanh instead of iFSQ — D-A14 proven at 0.946

**The paper can be published with D-A14 as-is.** D-A18 is the aspirational "best possible" model.

---

## 11. Technical Infrastructure

### BitwiseValidator — O(1) Isomorphic Alternative
1000/1000 random equivalence tests PASS. Speedup: analogy 5.2x, similarity 78x. Primes IMPOSSIBLE at >128 bits. **Formalize with primes, implement with bits.**
> experiment_log.md, lines 3566-3607

### Convergence: Trits / BitNet / Bitwise
Three independent paths → {+1, 0, -1} with ~42% sparsity: La Danza (philosophical), BitNet b1.58 (engineering), bitwise algebra (mathematical). The triadic framework predicts the optimal discrete representation independently.
> experiment_log.md, lines 3648-3661

### R3 Formula Comparison
4 discrete vs continuous variants tested. In ternary: Formula D (category-aware) best (90.3%). Ternary has direction via sign — XOR destroys directional info.
> experiment_log.md, lines 2817-2857

### R3 Chain & Fork Composition
Round-trip accuracy 98.1% (+16.2% over 1-step). 2-step chain 87.4% (+4.5% over multiplicative). Fork pairwise cosine ~0 (NOT word2vec-style). **Triadic space is a computational substrate.**
> experiment_log.md, lines 2861-2909

---

## 12. Primitive Systems Reconciliation

> *Source: PRIMITIVE_RECONCILIATION.md (UNIQUE — complete content)*

Three incompatible primitive counts:

| System | Primitives | Representation |
|---|---|---|
| Sistema 7×7 v3.5 | 51 | Floats tanh [-1,+1] |
| Inventario de opuestos | 63 | Binary (bits + primes) |
| TriadicGPT Run 15 | 64 | Learned emergent bits |
| Danza Bootstrap (v2) | 63 | Supervised binary |

**Key differences**: P15 fuses duals (Bien_Mal = 1 bit with 3 states), Inventario splits (bien + mal = 2 bits). P15 has 10 features (colors, textures), Inventario adds logic/quantity/agency (29 primitives). Run 15 bits are arbitrary (not ontologically mapped).

34 direct matches between P15 (49) and Inventario (63). 10 P15-only features. 29 Inventario-only features.

---

## 13. Reviewer FAQ

> *Source: AUDIT.md (UNIQUE)*

**Q1: "Why not use a standard encoder (BERT) instead of training from scratch?"**
A: The point is to test whether the triadic head can learn semantics END-TO-END during language modeling. BERT would confound the experiment.

**Q2: "TinyStories is a toy dataset. Do results generalize?"**
A: P13 shows expected OOD degradation. The triadic head is neutral on generalization — it neither helps nor hurts. Corpus is the bottleneck, not the method.

**Q3: "90.2% trivial baseline makes your 90.7% algebraic look weak."**
A: The margin is +0.5pp over trivial with D-A5. BUT: D-A16 ensemble amplifies to +4.3pp (8.6x). AND: D-A11 shows 0/1000 permutations reach 90.7% (p<0.001). The signal is real.

**Q4: "Subsumption fails at 355M. Doesn't this undermine scaling claims?"**
A: D-A17 (355M+v2) showed 1.7% subsumption — but this was caused by bugs (alignment-only loss, non-differentiable subsumption), not an inherent limitation. **D-A19 fixes these bugs and recovers 76.9% sub, 100% R3 at 355M.** The scale-algebra tradeoff is real (76.9% vs 98.3% at 40M) but mitigable, not a hard wall.

**Q5: "Why not use word2vec analogies as baseline?"**
A: R3 chain composition shows pairwise cosine ~0 — triadic space operates categorically (bit-flip), not vectorially. Direct comparison is inappropriate.

**Q6: "What about larger k (128, 256 bits)?"**
A: Bits sweep (Runs 22-26) shows k=128 produces WORSE language loss and gap than k=64. Diminishing returns at k>64.

**Q7: "Are the 63 primitives cherry-picked?"**
A: L3 (blind prime assignment) found 0% cherry-picking. The primitives come from a systematic philosophical framework (La Danza), not data-driven selection.

**Q8: "Ternary zeros collapse to 0% at 355M. Is this fixable?"**
A: **YES — D-A19 confirms.** Root cause was bugs in D-A17 (alignment-only loss, non-differentiable subsumption). Full 4-component triadic loss + differentiable subsumption + sparsity target loss restores algebraic structure: 76.9% sub (vs 1.7%), 100% R3 (vs 6.7%), 16 dead bits (vs 26). Per-token ternary: -1=70.8%, 0=1.0%, +1=28.2%.

---

## 14. Paper Corrections

### From AUDIT.md

| # | Current Claim | Correction | Source |
|---|---|---|---|
| 1 | "40M params" threshold | ~20M (E5 interpolation) | E5 |
| 2 | "8x compression" | REFUTED — "no LM cost" only | E6 |
| 3 | "69.2% analogy" | 98% (51 quads) | E3 |
| 4 | "108,694 discrepancies" | Source not located | L6 |
| 5 | Semantic gap +0.099, 72% closure | +0.076, 48% (Bug #7 fix) | E10-v3 |
| 6 | 355M validates scaling | D-A17: algebra fails (bugs). **D-A19: 76.9% sub, 100% R3 (fixed)** | D-A17, D-A19 |

### From TEST_STATUS.md (Book Corrections L4-L10)

See [Section 10: Book Corrections Pending](#book-corrections-pending-l4-l10).

---

## 15. Version History

> *Source: EVOLUTION_PLAN.md (UNIQUE)*

| Version | Date | Milestone |
|---|---|---|
| v0.1 | 2026-03-04 | CPU autograd working |
| v0.2 | 2026-03-04 | PyTorch GPU training |
| v1.0 | 2026-03-05 | XL model (40M params) |
| v1.4 | 2026-03-07 | Run 15 — production model |
| v2.0 | 2026-03-07 | Full benchmark suite |
| v3.0 | 2026-03-08 | GPT-2 transfer (Exp 10) |
| v4.0 | 2026-03-10 | Domain separation (Exp 11) |
| v5.0 | 2026-03-13 | Playground complete (P0-P15) |
| v5.1 | 2026-03-17 | Validation complete (E1-E7, B1-B3) |
| v6.0 | 2026-03-18 | Danza bootstrap (D-A5 through D-A16) |
| v7.0 | 2026-03-19 | v2 anchors, iFSQ, audit tests, consolidation |
| v7.1 | 2026-03-20 | D-A18 unified, D-A17 scaling (negative), D-A19 fix (algebra restored at 355M) |

---

## Appendix A: Detailed Data Store

For full per-step training data, detailed metric tables, and verbose analysis of each experiment, see **`experiment_log.md`** (~194 KB, ~3700 lines). That file is preserved as the raw data archive. Line numbers referenced throughout this document point to specific sections.

---

## Appendix B: Reproducibility

> *Source: danza_experiment_log.md + PLAN.md (UNIQUE)*

### Environment
```
conda activate triadic-microgpt
# Python 3.10, PyTorch CUDA 12.8
# GPU: RTX 5060 Ti 16GB (Blackwell) or equivalent
```

### Key Commands
```bash
# Core model training (Run 15 equivalent)
python src/torch_train.py --scale xl --steps 50000

# Danza supervised (D-A2)
python playground/danza_63bit.py --scale xl --steps 50000

# Danza v2 (D-A14, 158 anchors)
python playground/danza_63bit.py --scale xl --steps 50000 --v2

# Danza iFSQ+v2 (D-A16)
python playground/danza_63bit.py --scale xl --steps 50000 --v2 --activation ifsq

# Bootstrap (D-A5)
python playground/danza_bootstrap.py --phase train --train-anchors 25

# GPT-2 355M fix (D-A19)
python playground/gpt2_355m_sparsity.py --steps 50000 --v2 --sparsity-weight 2.0 --dtype bfloat16

# GPT-2 transfer (E10)
python experiment10/train_gpt2_triadic.py --alignment infonce

# Evaluation
python src/evaluate.py --model checkpoints/torch_run15_strongalign/model_best.pt

# Tests
python tests/test_all.py
```

### Data Sources
```
data/TinyStories-train.txt          # 1.8GB, 2.1M stories
data/gold_primes_64.json            # 10K WordNet concepts
playground/danza_data/primitivos.json # 63 primitives
playground/danza_data/anclas.json     # 54 v1 anchors
playground/danza_data/anclas_v2.json  # 158 v2 anchors
```

### Playground Conventions
- Each script is **autocontenido**: load model, run experiment, save to `playground/results/`
- Default `--steps 10000` for exploration before committing GPU for hours
- Results documented in this file under the appropriate research line
