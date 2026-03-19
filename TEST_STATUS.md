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
| L3 | Blind prime assignment | `playground/audit_tests/test_blind_prime_assignment.py` | PASS (vacuous) | 03-19 |
| F0 | Data + model validation | `playground/audit_tests/test_data_validation.py` | PASS (93.7% bit acc) | 03-19 |
| L11 | Indifference (Cap. 5) | `playground/audit_tests/test_indifference_and_false_opposites.py` | GOLD PASS, MODEL FAIL | 03-19 |
| L12 | False opposites | (included in L11 script) | GOLD PASS, MODEL FAIL | 03-19 |
| — | reptimeline hybrid analysis | `playground/audit_tests/analyze_hybrid.py` | 17 triadic, 50 active | 03-19 |
| — | reptimeline v2 analysis | `playground/audit_tests/analyze_v2.py` | 68 triadic, 48 active | 03-19 |

### Pending — Critical (before publication)

| Line | Test | Script exists? | What's needed | Blocking |
|------|------|---------------|---------------|----------|
| L2 | D-A13 (355M) formal eval | `playground/audit_tests/test_d_a13_eval.py` | GPU time | Paper claim |
| L11 | Re-run with v2 model | Modify test_indifference_and_false_opposites.py | v2 checkpoint (DONE) | Paper claim |
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
| L15 | Aristotelian types (Cap. 11) | `playground/audit_tests/test_aristotelian_types.py` | 1h | Script created, not run |
| L16 | Polisemia contextual | NO | 1h | Same word, different context |
| L17 | Categorical Bits Architecture | NO | 6h GPU | Could be its own paper |
| L19 | Enantiodromia detection | `playground/audit_tests/test_enantiodromia.py` | 1h | Script created, not run |

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

## Playground Results (46 files in `playground/results/`)

| File | Experiment | Key metric |
|------|-----------|-----------|
| `concept_gpt_49bit.json` | P15 structured 49-bit | 88.5% train, 17% test |
| `compression_benchmark.json` | E6 compression | 8.3% (refutes "8x") |
| `expanded_analogy_benchmark.json` | E3 51-analogy | 98% verification |
| `cross_dataset_eval.json` | P13 cross-corpus | WikiText2/LAMBADA |
| `subsumption_loss.json` | P6 subsumption | 100% train |
| `r3_subsumption_combo.json` | P7 R3+sub combo | Combined results |
| `random_baseline.json` | P2 random | 50% (chance) |
| `sin_head_experiment.json` | P1 sinusoidal | +4 dead bits (failed) |
| `dead_bit_regularization.json` | Dead bit entropy | Reg analysis |
| `embedding_gap_baseline.json` | B1 embedding gap | Baseline metric |
| `sub_weight_sweep/` | E4 4-weight sweep | w=2.0 best (92.3%) |
| `multi_seed/` | E1 3-seed validation | Reproducibility |
| `r3_low_k/`, `r3_low_k_v2/` | E7 R3 at k=6,8,12 | R3 collapses at low k |
| `scale_interpolation/` | E5 25M/30M | Gradual transition |
| `alignment_ablation/` | E2 ablation | Full/no-align/no-entropy |
| `xl_baselines/` | B2/B3 baselines | Language-only, frozen random |

## Data Files

| File | Location | Purpose |
|------|----------|---------|
| `primitivos.json` | `playground/danza_data/` | 63 primitive definitions |
| `anclas.json` | `playground/danza_data/` | 54 anchor concepts (v1) |
| `anclas_v2.json` | `playground/danza_data/` | 158 anchor concepts (v2) |
| `gold_primes_64.json` | `data/` | 10K concepts, 64-bit signatures |
| `gold_primes_32.json` | `data/` | 10K concepts, 32-bit signatures |
| `gold_primes.json` | `data/` | 100 gold signatures (original) |
| `TinyStories-train.txt` | `data/` | 1.8GB training corpus |
| `alpaca_data_cleaned.json` | `data/` | 43MB fine-tuning (70K examples) |

## Reports

| File | Content |
|------|---------|
| `reports/bias_audit_results.json` | Exp 8: 98.5% acc, 0.96% FPR |
| `reports/eval_report.json` | Evaluation metrics export |
| `reports/loss_curve.png` | Training loss visualization |

---

## Coverage Summary

```
Unit tests:          ~80 (ALL PASS)
Benchmarks:          12/12 COMPLETE
Audit tests:         7 executed, 3 pending (L2 GPU, L11/L12 re-run)
Audit scripts:       9 total (7 executed, 2 not run: aristotelian, enantiodromia)
Book corrections:    0/7 done
Experiments:         8 successful, 5 failed (documented), 3 not evaluated
Playground results:  46 files across 16 experiments
Research lines:      4 executed, 3 critical pending, 6 valuable pending, 14 future
Data files:          10 in data/, 3 in danza_data/
Reports:             3 files
```
