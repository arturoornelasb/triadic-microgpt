# Audit Tests — Pre-Publication Validation

Tests for the comprehensive audit of *La Danza Cosmica de los Opuestos* + TriadicGPT.

## Quick Start

```bash
cd C:\Github\triadic-microgpt

# Pre-flight (run first)
python playground/audit_tests/test_data_validation.py

# Run all tests
python playground/audit_tests/run_all.py

# Only the 3 indispensable tests
python playground/audit_tests/run_all.py --indispensable

# Only the 3 valuable tests
python playground/audit_tests/run_all.py --valuable

# A specific test
python playground/audit_tests/run_all.py --test f2.1
```

## Test Inventory

### PREFLIGHT (run first)

| ID | Script | Status | What it validates |
|----|--------|--------|-------------------|
| F0 | `test_data_validation.py` | **DONE** | Gold targets, model accuracy, anchor coverage |

**Results**: 54 anchors v1, 93.7% bit accuracy. GOLD confirms indifference thesis.

### INDISPENSABLE (block publication, ~7h)

| ID | Script | Status | What it validates |
|----|--------|--------|-------------------|
| F3.1 | `test_pf_bridge.py` | Created | 5 falsifiable predictions (PF-Q1/Q2/Q4-Q6) |
| F3.4 | `test_blind_prime_assignment.py` | Created | Prime assignment not cherry-picked (4 strategies + 100 random trials) |
| F4.4 | `test_d_a13_eval.py` | Created | D-A13 (355M) formal eval: bit acc, subsumption, ternary, uniqueness, R3 |

### VALUABLE (don't block, ~2.5h)

| ID | Script | Status | What it validates |
|----|--------|--------|-------------------|
| F2.1 | `test_indifference_and_false_opposites.py` | **DONE** | Book's central thesis (Hilo 6+8). GOLD PASS, MODEL FAIL |
| F2.2 | `test_aristotelian_types.py` | Created | 4 Aristotelian opposition types, Kruskal-Wallis (Cap. 11) |
| F2.5 | `test_enantiodromia.py` | Created | Extremes contain seed of opposite (Cap. 18) |

## Dual-Level Evaluation

All tests evaluate at TWO levels:

1. **GOLD level**: Uses manually assigned bit targets (`anclas.json` + `anclas_v2.json`). Tests whether the THEORY is correct.
2. **MODEL level**: Uses learned projections from the model. Tests whether the model LEARNED the theory.

A GOLD PASS + MODEL FAIL means the theory is sound but the model needs more supervision. This is the current state for F2.1.

## Anchor Data

| File | Concepts | Purpose |
|------|----------|---------|
| `danza_data/anclas.json` | 54 (v1) | Original supervised anchors |
| `danza_data/anclas_v2.json` | 104 (v2) | Additional anchors for audit tests |
| **Merged** | **158 total** | Via `load_all_anchors()` in `danza_63bit.py` |

v2 includes all concepts needed by the tests: socialism, capitalism, tyranny, fear, shame, stone, carbon, predator, prey, cause, effect, etc. Each has manually reasoned bit assignments using the same 63 primitives.

The v2 anchors are NOT yet used for training — only for gold-level evaluation. To retrain:
```python
from danza_63bit import load_primitives, load_all_anchors
prim_data = load_primitives()
anchors, _ = load_all_anchors(prim_data)  # 158 concepts
```

## Model Requirements

| Model | Checkpoint | Tokenizer | max_tokens |
|-------|-----------|-----------|------------|
| Run 15 (40M) | `checkpoints/danza_bootstrap_xl/model_best.pt` | Custom BPE (`tokenizer.json`) | **4** |
| D-A13 (355M) | `checkpoints/danza_gpt2medium_ternary/model_best.pt` | GPT2Tokenizer (HuggingFace) | **8** |

## File Structure

```
playground/audit_tests/
  common.py                              # Shared: model loading, metrics, gold targets
  run_all.py                             # Runner with --indispensable/--valuable/--test
  test_data_validation.py                # F0:   Pre-flight
  test_indifference_and_false_opposites.py  # F2.1: Indifference + false opposites
  test_aristotelian_types.py             # F2.2: 4 Aristotelian types
  test_enantiodromia.py                  # F2.5: Enantiodromia
  test_pf_bridge.py                      # F3.1: PF bridge test
  test_blind_prime_assignment.py         # F3.4: Blind prime assignment
  test_d_a13_eval.py                     # F4.4: D-A13 355M evaluation
  results/                               # JSON output from each test
```

## Results

All results saved as JSON in `results/`. Current:
- `f0_data_validation.json` — 54 anchors, 93.7% acc
- `f2_1_indifference_false_opposites.json` — GOLD PASS, MODEL FAIL

## Dependencies

- PyTorch, NumPy, SciPy
- transformers (only for F4.4 / D-A13)

## Next Steps

1. **Retrain** model with 158 anchors (vs 54) to improve from 93.7%
2. **Re-run** F2.1 to get MODEL PASS
3. **Execute** F3.1 + F3.4 + F4.4 (indispensable, scripts ready)
4. **Execute** F2.2 + F2.5 (valuable, scripts ready)
