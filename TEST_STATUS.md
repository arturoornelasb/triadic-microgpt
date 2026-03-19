# Test & Research Line Status

> Last updated: 2026-03-19

---

## Unit Tests

| Suite | Tests | Location | Status |
|-------|-------|----------|--------|
| Core (autograd, transformer, triadic) | 37 | `tests/test_all.py` | PASS |
| triadic-head package | 33 | `triadic-head/tests/` | PASS |
| reptimeline discovery | ~10 | `reptimeline/tests/` | PASS |
| **Total** | **~80** | | **ALL PASS** |

---

## Benchmark Suite

| Benchmark | Script | Key Result | Status |
|-----------|--------|------------|--------|
| Scaling Study | `benchmarks/scripts/scaling_study.py` | Phase transition ~20M | COMPLETE |
| Bit Entropy | `benchmarks/scripts/bit_entropy.py` | Mean H=0.749 | COMPLETE |
| Language Quality | `benchmarks/scripts/language_quality.py` | PPL 7.69 vs 7.56 (cost=0) | COMPLETE |
| Analogy | `benchmarks/scripts/analogy_benchmark.py` | 69.2% top-1 (Run15) | COMPLETE |
| Subsumption | `benchmarks/scripts/subsumption_benchmark.py` | 100% train, 86.5% held-out | COMPLETE |
| Interpretability Probe | `benchmarks/scripts/interpretability_probe.py` | Linear probe | COMPLETE |
| Engine Comparison | `benchmarks/scripts/engine_comparison.py` | 5 methods (Table 7) | COMPLETE |
| Domain Topology | `benchmarks/scripts/geometric_topology.py` | 1.21 sentence-level | COMPLETE |
| Bit Evolution | `benchmarks/scripts/bit_evolution.py` | 50K trace | COMPLETE |
| Bits Sweep Plots | `benchmarks/scripts/bits_sweep_plots.py` | k=8-128 | COMPLETE |
| Scaling Plots | `benchmarks/scripts/scaling_plots.py` | Publication figs | COMPLETE |
| Prime vs Bitwise | `benchmarks/scripts/prime_vs_bitwise.py` | 1000/1000 equiv, 5-78x | COMPLETE |

---

## Audit Tests (Research Lines)

### Executed

| Line | Test | Script | Result | Date |
|------|------|--------|--------|------|
| L1 | Bridge PFs (Q1,Q4,Q5,Q6) | `playground/audit_tests/test_pf_bridge.py` | 3/4 PASS (Q6 FAIL) | 03-19 |
| L3 | Blind prime assignment | `playground/audit_tests/test_blind_primes.py` | PASS (vacuous) | 03-19 |
| L11 | Indifference (Cap. 5) | `playground/audit_tests/test_indifference.py` | GOLD PASS, MODEL FAIL | 03-19 |
| L12 | False opposites | (included in L11 script) | GOLD PASS, MODEL FAIL | 03-19 |

### Pending — Critical (before publication)

| Line | Test | Script exists? | What's needed | Blocking |
|------|------|---------------|---------------|----------|
| L2 | D-A13 (355M) formal eval | `playground/audit_tests/eval_da13.py` | GPU time | Paper claim |
| L11 | Re-run with v2 model | Modify test_indifference.py | v2 checkpoint (DONE) | Paper claim |
| L12 | Re-run with v2 model | Same as L11 | v2 checkpoint (DONE) | Paper claim |

### Pending — Book Corrections (no experiments needed)

| Line | Correction | Status |
|------|-----------|--------|
| L4 | "40M" -> "~20M" params threshold | TODO |
| L5 | Rewrite "8x compression" (refuted) | TODO |
| L6 | Locate source for "108,694 discrepancies" | TODO |
| L7 | Update "69.2%" -> 98% analogies | TODO |
| L8 | Add Placer/Dolor to primitives | TODO |
| L9 | Include new results (iFSQ, BitNet, R3, composition) | TODO |
| L10 | Document zeros->0% collapse at 355M | TODO |

### Pending — Valuable but non-blocking

| Line | Test | Script exists? | Effort | Notes |
|------|------|---------------|--------|-------|
| L13 | 1000 adversarial concepts | NO | 4h | Stress test |
| L14 | PCA for real primitive count | NO | 3h | How many real dims? |
| L15 | Aristotelian types (Cap. 11) | YES (not run) | 1h | Script created |
| L16 | Polisemia contextual | NO | 1h | Same word, different context |
| L17 | Categorical Bits Architecture | NO | 6h GPU | Could be its own paper |

### Future Work (after publication)

| Line | Description |
|------|-------------|
| L18 | Semantic compositions depth 3-4 |
| L19 | Enantiodromia detection |
| L20 | Centering optimization |
| L21 | Cross-corpus validation (WikiText2, LAMBADA) |
| L22 | Fourier head exploration |
| L23 | Cross-linguistic validation |
| L24 | Dataset reconciliation (auto vs manual) |
| L25 | Formalize polisemia |
| L26 | Wave model v2 |
| L27 | UI end-to-end tests |
| L28 | 6/9 algebraic structures |
| L29 | Hooke homeostasis |
| L30 | Practice -> life applications |
| L31 | Scale to 400M+ |

---

## Experiment Status Summary

### Successful

| Experiment | Checkpoint | Key Result |
|-----------|-----------|------------|
| Run 15 (strongalign) | `torch_run15_strongalign/` | Production model, PPL 7.69, gap +0.020 |
| D-A5 (bootstrap) | `danza_bootstrap_xl/` | R3 90.7% > trivial 90.2%, p<0.001 |
| D-A8 FSQ (ternary) | `danza_ternary_fsq_xl/` | Loss 0.951, sub 86.5%, 3-state clean |
| D-A10 iFSQ (binary) | `danza_ifsq_binary_xl/` | Loss 0.924, sub 87.1%, best LM |
| D-A9 (hybrid) | `danza_hybrid_adv_xl/` | 6 dead bits, free bits learn, 17 triadic |
| D-A13 (GPT-2 355M) | `danza_gpt2medium_ternary/` | 100% sub holdout |
| **D-A14 (v2 158 anc)** | `danza_63bit_xl_v2/` | **93% test, 98.3% sub, 68 triadic** |
| BitwiseValidator | `src/triadic.py` | 1000/1000 equiv, 5-78x speedup |

### Failed

| Experiment | Checkpoint | What went wrong | Lesson |
|-----------|-----------|----------------|--------|
| D-A15 (grad decoupling) | `danza_grad_decoupling_xl/` | 49.6% = random | Per-bit gradient manipulation too invasive |
| D-A8 Absmean | `danza_ternary_absmean_xl/` | Loss 1.309, inferior | Absmean quantization worse than FSQ |
| P15 (49-bit concept) | `concept_gpt_49bit_xl/` | 88.5% train, 17% test | Pure memorization, no generalization |
| Bootstrap D-A6 | `danza_bootstrap_xl/` | Cycle 0 converged | Confidence gate too strict |
| E10-v2 (GPT-2 InfoNCE) | — | tri_loss=NaN from step 300 | Numerical instability in InfoNCE |

### Not Yet Evaluated

| Checkpoint | Notes |
|-----------|-------|
| `danza_bootstrap_v2_xl/` | 3 cycles complete, needs formal eval |
| `gpt2_medium_infonce/` | Experiment 10 transfer model |
| `torch_run29_staged/` | Staged MSE->InfoNCE |

---

## Coverage Summary

```
Unit tests:          ~80 (ALL PASS)
Benchmarks:          12/12 COMPLETE
Audit tests:         4/4 executed, 3 pending re-runs
Book corrections:    0/7 done
Experiments:         8 successful, 5 failed (documented), 3 not evaluated
Research lines:      4 executed, 3 critical pending, 5 valuable pending, 14 future
```
