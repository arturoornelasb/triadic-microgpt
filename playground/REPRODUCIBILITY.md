# Reproducibility Guide — Triadic MicroGPT Experiments

> All experiments run on: RTX 5060 Ti 16GB | CUDA 12.8 | Python 3.10 | Conda `triadic-microgpt`
> All commands assume: `conda activate triadic-microgpt` and working dir = project root

---

## Prerequisites

```bash
conda activate triadic-microgpt
# Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

**Required checkpoints:**
- Production model: `checkpoints/torch_run15_strongalign/model_L12_D512_B64_best.pt`
- Tokenizer: `checkpoints/torch_run15_strongalign/tokenizer.json`
- Training data: `data/TinyStories-train.txt`

---

## Phase 1: Production Model (Run 15)

### Pre-training (XL, 40M params, ~76 min)
```bash
python src/torch_train.py --scale xl --steps 50000 --alpha 0.05 \
  --entropy-weight 1.0 --align-weight 5.0 --align-mode mse \
  --triadic-warmup-pct 0.8 --batch-size 64 --no-distill
```

### Evaluation
```bash
python src/evaluate.py \
  --model checkpoints/torch_run15_strongalign/model_L12_D512_B64_best.pt \
  --tokenizer checkpoints/torch_run15_strongalign/tokenizer.json
```

**Expected:** PPL ~7.69, semantic gap ~+0.020, 15 dead bits, entropy 0.749

---

## Phase 2: Playground Experiments (P0-P15)

### P0 — K-Constant Analysis (0 GPU)
```bash
python playground/k_constant_analysis.py
```
**Expected:** Mean K = 1.21, algebraic exact match 0/15

### P1 — Sinusoidal Head (~10 min)
```bash
python playground/sin_head_experiment.py
```
**Expected:** SIN gap +0.016 vs TANH -0.005

### P2 — Random Baseline (~10 min)
```bash
python playground/random_baseline.py
```
**Expected:** Frozen random head beats trained at 5.8M (gap +0.008 vs -0.013)

### P3 — Soft Signatures (~20 min)
```bash
python playground/soft_signatures.py
```
**Expected:** sigmoid+anneal and gumbel achieve 0 dead bits, gap -0.003

### P4 — XL Sigmoid+Anneal (~76 min)
```bash
python playground/xl_sigmoid_anneal.py
```
**Expected:** PPL 16.6 (+116%), dead bits 12, does NOT scale

### P5 — Phase-Aware Position Encoding (~15 min)
```bash
python playground/phase_attention.py
```
**Expected:** Negative result. Learned positions strictly better.

### P6 — Subsumption Loss (~15 min base, ~9h XL)
```bash
# Base scale
python playground/subsumption_loss.py
# XL scale
python playground/xl_subsumption.py
```
**Expected base:** sub(5.0) → 91.7% held-out, language improves
**Expected XL:** 100% held-out at 25K, degrades at 50K

### P7 — R3 + Subsumption Combo (~15 min)
```bash
python playground/r3_subsumption_combo.py
```
**Expected:** Sub-only wins. R3 causes 64/64 dead bits.

### P9 — Info Hierarchy Analysis (0 GPU)
```bash
python playground/info_hierarchy.py
```
**Expected:** Hypernym bits 35.3 → 2.3 (93% reduction)

### P10 — R3 Entropy Guard (~50 min)
```bash
python playground/r3_entropy_guard.py
```
**Expected:** ALL R3 variants: 64/64 dead bits regardless of entropy weight

### P11 — Curriculum Sub→R3 (~15 min)
```bash
python playground/curriculum_sub_r3.py
```
**Expected:** R3 erases 7K steps of Sub structure in 3K steps

### P12 — XL Subsumption Loss (~9h)
```bash
python playground/xl_subsumption.py --sub-weight 5.0
```
**Expected:** 25K optimal: PPL 11.35, sub test 100%. 50K degrades.

### P13 — Cross-Dataset Eval (~2 min)
```bash
python playground/cross_dataset_eval.py
```
**Expected:** TinyStories PPL 6.60, LAMBADA 345, WikiText-2 3033

### P14 — Concept Head Phase 4 (~1 min)
```bash
python playground/concept_head_phase4.py
```
**Expected:** Negative. Train 100%, test ~20%. Post-hoc projection fails.

### P15 — 49-Bit Concept GPT (~3h)
```bash
# v3 (T1-only supervision) — 86.2% accuracy
python playground/concept_gpt_49bit.py --scale xl --steps 50000 \
  --batch-size 64 --sub-weight 5.0 --sup-weight 10.0 \
  --entropy-weight 2.0 --activation tanh

# v4 (T1+T2 supervision, 80/20 split) — 88.5% train, 17% test
# Same command (script updated to include T2)
```
**Expected v3:** 86.2% primary acc, 0/49 dead bits, sub test 97.3%
**Expected v4:** 88.5% train, 17.0% held-out (memorization, not compositionality)

---

## Phase 3: Validation Experiments (E1-E7)

### E1 — Multi-Seed Validation (~7h)
```bash
# All 3 seeds sequential
python playground/multi_seed_validation.py

# Or individual seeds (~76 min each)
python playground/multi_seed_validation.py --seed 42
python playground/multi_seed_validation.py --seed 123
python playground/multi_seed_validation.py --seed 777

# Aggregate only (after all seeds done)
python playground/multi_seed_validation.py --aggregate-only
```
**Expected:**
| Metric | Mean ± Std |
|--------|-----------|
| PPL | 10.86 ± 0.01 |
| Semantic Gap | +0.038 ± 0.005 |
| Analogy Verif | 100% ± 0% |
| Dead Bits | 11.0 ± 1.0 |
| Ordering | 3/3 correct |

**Results:** `playground/results/multi_seed/aggregate.json`

### E2 — Alignment Loss Ablation (~7h)
```bash
# All 3 variants
python playground/alignment_ablation.py --all

# Or individual variants (~2.5h each)
python playground/alignment_ablation.py --variant full
python playground/alignment_ablation.py --variant no_align
python playground/alignment_ablation.py --variant no_entropy

# Aggregate
python playground/alignment_ablation.py --aggregate-only
```
**Expected:**
| Variant | Semantic Gap | Dead Bits | Entropy |
|---------|-------------|-----------|---------|
| FULL (control) | +0.025 | 11 | 0.624 |
| NO_ALIGN | +0.018 | 23 | 0.519 |
| NO_ENTROPY | +0.023 | 12 | 0.623 |

PPL identical (~10.87) across all. Alignment is the driver.
**Results:** `playground/results/alignment_ablation/`

### E3 — Expanded Analogy Benchmark (0 GPU, ~30s)
```bash
python playground/expanded_analogy_benchmark.py
```
**Expected:**
- Verification rate: 98.0% (50/51) — revised up from 69.2%
- Top-1 retrieval: 0% (prime), 3.9% (vector)
- Easy (same-domain): 100%, Hard (cross-domain): 92.3%
- Only failure: tree:forest::star:sky

**Results:** `playground/results/expanded_analogy_benchmark.json`

### E4 — Subsumption Weight Sweep (~12h)
```bash
# All weights
python playground/sub_weight_sweep.py --all

# Or individual weights (~3h each)
python playground/sub_weight_sweep.py --weight 0.5
python playground/sub_weight_sweep.py --weight 1.0
python playground/sub_weight_sweep.py --weight 2.0
python playground/sub_weight_sweep.py --weight 5.0

# Aggregate
python playground/sub_weight_sweep.py --aggregate-only
```
**Expected:**
| Weight | Sub Test@50K | Dead@50K | Gap@50K |
|--------|-------------|----------|---------|
| 0.5 | 84.6% | 30 | +0.015 |
| 1.0 | 53.8% | 38 | +0.013 |
| 2.0 | **92.3%** | 44 | +0.006 |
| 5.0 | 76.9% | 33 | +0.008 |

**Critical note:** 0% sub at 25K for ALL weights due to 80% warmup. Use 50% warmup for subsumption.
**Results:** `playground/results/sub_weight_sweep/aggregate.json`

### E5 — Scale Interpolation (~5h)
```bash
# Both configs
python playground/scale_interpolation.py --all

# Or individual (~2-3h each)
python playground/scale_interpolation.py --config 25m
python playground/scale_interpolation.py --config 30m

# Aggregate
python playground/scale_interpolation.py --aggregate-only
```
**Expected:**
| Config | Params | PPL | Semantic Gap | Dead Bits | Ordering |
|--------|--------|-----|-------------|-----------|----------|
| 25M | 26.1M | 7.75 | +0.010 | 9 | ❌ |
| 30M | 29.8M | 8.38 | +0.043 | 9 | ❌ |

Gap crosses zero between 16M and 26M. Transition is gradual, not sharp.
**Results:** `playground/results/scale_interpolation/`

### E6 — Compression Benchmark (0 GPU, ~30s)
```bash
python playground/compression_benchmark.py
```
**Expected:**
| Task | Triadic 64D | Embedding 512D | Random |
|------|------------|---------------|--------|
| Centroid | 13.3% | 16.4% | 11.1% |
| Spearman rho | 0.398 | 1.000 | 0.000 |
| Separation | 1.010 | 1.004 | 1.000 |
| k-NN (k=3) | 11.7% | 8.6% | 11.1% |

**Verdict:** "8x compression without info loss" is NOT supported. Both near random.
**Results:** `playground/results/compression_benchmark.json`

### E7 — R3 Loss at Low k (~45 min)
```bash
# All k values
python playground/r3_low_k.py --all

# Or individual
python playground/r3_low_k.py --k 6
python playground/r3_low_k.py --k 8
python playground/r3_low_k.py --k 12

# Aggregate
python playground/r3_low_k.py --aggregate-only
```
**Expected:**
- R3 does NOT collapse at k=6-12 (dead bits 1-3, vs 64/64 at k=64)
- R3 train: 100% for all k. Test: 25-50%
- R3 DESTROYS semantic gap (-0.27 to -0.42 vs baseline -0.01 to -0.08)
- Entropy actually IMPROVES with R3 at low k

**Results:** `playground/results/r3_low_k/`

---

## Summary of All Results

### Confirmed Claims (strong evidence)
1. **Semantic ordering is reproducible** — 3 seeds, gap +0.038 ± 0.005, 95% CI entirely positive (E1)
2. **Zero language cost** — PPL unaffected by triadic head (E1, E2)
3. **Analogy verification: 98%** on 51 analogies (E3, revised from 69.2%)
4. **Alignment loss is the primary driver** — 2x dead bits without it (E2)
5. **Subsumption loss works** — up to 92.3% held-out at XL (E4), 100% with proper warmup (P12)
6. **7×7 Sistema learnable end-to-end** — 88.5% known vocabulary (P15)
7. **Coherence loss = collapse** — confirmed, permanently removed (Run 12-13)
8. **R3 dead at k=64** — 3 independent experiments (P7, P10, P11)
9. **R3 alive at k=6-12** but destroys semantic gap (E7)
10. **Emergent information hierarchy** from subsumption loss — 93% bit reduction (P9)

### Corrected Claims (paper needs revision)
1. **"8x compression without info loss"** → "8x compression without PPL cost" (E6)
2. **"Phase transition"** → "Gradual emergence, crossover at ~20M params" (E5)
3. **"Analogy verification 69.2%"** → "98.0%" with proper benchmark (E3)
4. **Entropy regularization** → Redundant when alignment present (E2)

### Open Questions (not yet resolved)
1. Does it scale beyond 40M params / TinyStories?
2. Can compositional generalization be achieved? (17% held-out in P15 v4)
3. What is the killer application? (verification is the best candidate)
4. Warmup interaction with subsumption needs systematic study

---

## File Index

| File | Experiment | GPU Time |
|------|-----------|----------|
| `playground/k_constant_analysis.py` | P0 | 0 |
| `playground/sin_head_experiment.py` | P1 | ~10m |
| `playground/random_baseline.py` | P2 | ~10m |
| `playground/soft_signatures.py` | P3 | ~20m |
| `playground/xl_sigmoid_anneal.py` | P4 | ~76m |
| `playground/phase_attention.py` | P5 | ~15m |
| `playground/subsumption_loss.py` | P6 base | ~15m |
| `playground/r3_subsumption_combo.py` | P7 | ~15m |
| `playground/info_hierarchy.py` | P9 | 0 |
| `playground/r3_entropy_guard.py` | P10 | ~50m |
| `playground/curriculum_sub_r3.py` | P11 | ~15m |
| `playground/xl_subsumption.py` | P12 | ~9h |
| `playground/cross_dataset_eval.py` | P13 | ~2m |
| `playground/concept_head_phase4.py` | P14 | ~1m |
| `playground/concept_gpt_49bit.py` | P15 | ~3h |
| `playground/multi_seed_validation.py` | E1 | ~7h |
| `playground/alignment_ablation.py` | E2 | ~7h |
| `playground/expanded_analogy_benchmark.py` | E3 | 0 |
| `playground/sub_weight_sweep.py` | E4 | ~12h |
| `playground/scale_interpolation.py` | E5 | ~5h |
| `playground/compression_benchmark.py` | E6 | 0 |
| `playground/r3_low_k.py` | E7 | ~45m |

**Total GPU time (all experiments):** ~46h
**Results directory:** `playground/results/`
**Experiment log:** `experiment_log.md`
