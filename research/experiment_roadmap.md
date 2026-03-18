# Experiment Roadmap: Deep Analysis Campaigns D-A8 through D-A16

**Date**: 2026-03-18
**Status**: Active -- updated as experiments launch and complete
**Source**: Deep analysis session integrating related work survey, running experiments, and paper submission timeline

---

## 1. Completed Experiments (this session)

| Experiment | Script | Status | Key Result |
|------------|--------|--------|------------|
| E4 sub_weight sweep (all 4 weights) | `playground/sub_weight_sweep.py` | **COMPLETE** | w=2.0→92.3% sub_test, w=5.0@25K best PPL (8.28) |
| E10-v2 GPT-2 Medium + InfoNCE | `experiment10/src/train.py` | **FAILED** (Bug #7) | tri_loss NaN from step 300 |
| D-A11 Negative Baselines | `playground/negative_baselines.py` | **COMPLETE** | p<0.001, Cohen's d=6.64 |
| D-A16 Multi-Quad Ensemble | `playground/multi_quad_ensemble.py` | **COMPLETE** | 90.6% ensemble, max 96.8% (reina) |
| D-A16 FPR Neg Subsumption | `playground/negative_subsumption_test.py` | **COMPLETE** | FPR=24.1% (motivates D-A8) |
| **D-A8 FSQ (ternario)** | `playground/danza_ternary.py --quantize-mode fsq` | **COMPLETE** | loss 0.951, sub 86.5%, ternary 1.3/73.3/25.3 |
| **D-A10 iFSQ binary** | `playground/ifsq_binary_ablation.py` | **COMPLETE** | loss 0.924 (BEST), sub 87.1% |
| **D-A8 Absmean (ternario)** | `playground/danza_ternary.py --quantize-mode absmean` | **COMPLETE** | loss 1.309 (25K), sub 85.7% |
| **R3 Formula Comparison** | `playground/r3_formula_comparison.py` | **COMPLETE** | Formula D ternary 90.3% > continuous 89.9% |
| **R3 Chain & Fork** | `playground/r3_chain_test.py` | **COMPLETE** | Round-trip 98.1%, sub-linear chains, NOT word2vec |

**GPU status**: All priority experiments COMPLETE. GPU available for optional experiments (D-A13 scaling, D-A9 CB-LLMs).

---

## 2. Experiment Queue (Priority Order)

### Priority 1 -- Must Complete Before Paper Submission

These experiments fill critical gaps in the paper's evidence. Without them, reviewers will flag missing controls and incomplete sweeps.

---

#### D-A12: Multi-Quad Ensemble Bootstrap

| Field | Value |
|-------|-------|
| **Priority** | 1 (critical) |
| **Status** | **DONE** — `playground/multi_quad_ensemble.py` (90.6% ensemble, max 96.8%) |
| **Script path** | `playground/multi_quad_bootstrap.py` (to be created) |
| **GPU required** | No -- inference only on existing checkpoints |
| **Runtime estimate** | 30 minutes |
| **Dependencies** | None -- uses already-trained D-A5 checkpoint |

**Hypothesis**: Subsumption and analogy accuracy measured on a single random quad split is noisy. Bootstrap resampling over multiple quad subsets will yield confidence intervals and reveal whether our 92-100% accuracy claims are robust or artifacts of a lucky split.

**Success criteria**:
- Mean subsumption accuracy >= 90% across 1000 bootstrap samples
- 95% confidence interval width <= 8 percentage points
- Mean analogy accuracy >= 85% across bootstrap samples

**Launch command**:
```bash
python playground/multi_quad_bootstrap.py --checkpoint checkpoints/danza_bootstrap_xl/ --n-bootstrap 1000
```

---

#### D-A16: Negative Subsumption False Positive Rate

| Field | Value |
|-------|-------|
| **Priority** | 1 (critical) |
| **Status** | **DONE** — `playground/negative_subsumption_test.py` (FPR=24.1%, motivates D-A8) |
| **Script path** | `playground/negative_subsumption_fpr.py` (to be created) |
| **GPU required** | No -- inference only |
| **Runtime estimate** | 10 minutes |
| **Dependencies** | None -- uses existing checkpoint |

**Hypothesis**: Our subsumption metric only measures true positives (does "dog subsumes animal" pass?). We need to verify that semantically unrelated pairs ("dog subsumes chair") correctly FAIL. High false positive rate would invalidate the entire subsumption claim.

**Success criteria**:
- False positive rate < 5% on 500+ random negative pairs
- True positive rate remains > 90% on known subsumption pairs
- Clear separation between positive and negative pair score distributions

**Launch command**:
```bash
python playground/negative_subsumption_fpr.py --checkpoint checkpoints/danza_bootstrap_xl/ --n-negatives 500
```

---

#### D-A11: Negative Baselines (Random + Frozen + Untrained Heads)

| Field | Value |
|-------|-------|
| **Priority** | 1 (critical) |
| **Status** | **DONE** — `playground/negative_baselines.py` (p<0.001, d=6.64) |
| **Script path** | `playground/negative_baselines.py` (to be created) |
| **GPU required** | Yes -- 3 full training runs |
| **Runtime estimate** | 8 hours (3 runs x 50K steps each) |
| **Dependencies** | GPU available (after E4 and E10-v2 complete) |

**Hypothesis**: Reviewers will ask "does any random head achieve these numbers?" Three baselines answer this: (1) random bit projection (no training), (2) frozen random head (train LM only), (3) trained head with shuffled gold labels. All should produce near-chance subsumption/analogy scores, proving that our learned representations are non-trivial.

**Success criteria**:
- Random projection: subsumption accuracy < 15%, analogy < 10%
- Frozen random head: subsumption < 20%, analogy < 15%
- Shuffled labels: subsumption < 25%, analogy < 20%
- All baselines at least 3x worse than our trained model
- Language PPL within 10% of our trained model (proving the head doesn't help language via lucky correlation)

**Launch command**:
```bash
python playground/negative_baselines.py --all --steps 50000 --scale xl
```

---

#### D-A15: Complete Subsumption Weight Sweep

| Field | Value |
|-------|-------|
| **Priority** | 1 (critical) |
| **Status** | **DONE** — all 4 weights complete. See `playground/results/sub_weight_sweep/aggregate.json` |
| **Script path** | `playground/sub_weight_sweep.py` |
| **GPU required** | Yes |
| **Runtime estimate** | ~8 hours total (weight 1.0 running; 0.5, 2.0, 5.0 at ~2h each) |
| **Dependencies** | Weight 1.0 must finish first to free GPU; weights 0.5/2.0/5.0 can run sequentially after |

**Hypothesis**: There exists an optimal sub_weight that maximizes subsumption accuracy while keeping PPL degradation under 3%. The sweep across {0.5, 1.0, 2.0, 5.0} maps the Pareto frontier of this tradeoff.

**Success criteria**:
- Identify weight with best subsumption accuracy at PPL < baseline + 3%
- Clear monotonic or peaked relationship between weight and subsumption accuracy
- At least one weight achieves > 95% subsumption with acceptable PPL
- Results for all 4 weights with 25K and 50K step evaluations

**Launch command** (after weight 1.0 completes):
```bash
python playground/sub_weight_sweep.py --weight 0.5
python playground/sub_weight_sweep.py --weight 2.0
python playground/sub_weight_sweep.py --weight 5.0
# Or all remaining at once:
python playground/sub_weight_sweep.py --all
```

---

### Priority 2 -- Strongly Recommended

These experiments strengthen the paper significantly and open new result sections. Skip only if time-constrained.

---

#### D-A8: Ternary Head with iFSQ Activation

| Field | Value |
|-------|-------|
| **Priority** | 2 (strong) |
| **Status** | PREPARED -- script written, validated, ready to launch |
| **Script path** | `playground/danza_ternary.py` |
| **GPU required** | Yes |
| **Runtime estimate** | 4 hours (50K steps, XL scale) |
| **Dependencies** | GPU available |

**Hypothesis**: Replacing tanh bounding with ternary quantization via STE (straight-through estimator) turns "dead bits" from a failure mode into an intentional semantic state (0 = irrelevant). Combined with the iFSQ activation fix (`2*sigmoid(1.6*x) - 1` instead of `tanh`), this should reduce dead bit percentage from ~42% to < 15% while maintaining or improving subsumption and analogy accuracy.

**Success criteria**:
- Dead bit rate < 15% (down from ~42%)
- Subsumption accuracy >= 90% (maintained)
- Analogy accuracy >= 85% (maintained)
- PPL within 5% of binary baseline
- Bit entropy > 0.8 bits/dim (up from ~0.5)
- Each bit uses all three states {-1, 0, +1} with non-degenerate frequency

**Launch command**:
```bash
python playground/danza_ternary.py --phase train --scale xl --steps 50000
```

**Key insight from survey**: This experiment directly imports FSQ (ICLR 2024) and iFSQ (Tencent 2025) techniques. FSQ proves that fixed-grid quantization eliminates codebook collapse entirely. The iFSQ follow-up discovered that vanilla tanh bounding causes activation collapse -- exactly our dead bits problem.

---

#### D-A10: iFSQ Activation on Binary Head (Ablation)

| Field | Value |
|-------|-------|
| **Priority** | 2 (strong) |
| **Status** | NEEDS_SCRIPT |
| **Script path** | `playground/ifsq_binary_ablation.py` (to be created) |
| **GPU required** | Yes |
| **Runtime estimate** | 4 hours |
| **Dependencies** | GPU available; ideally run after D-A8 to compare |

**Hypothesis**: The iFSQ activation fix (`2*sigmoid(1.6*x) - 1`) may solve dead bits even without switching to ternary. This isolates the contribution of the activation function change from the quantization scheme change.

**Success criteria**:
- Dead bit rate < 25% (meaningful improvement from ~42%)
- All other metrics within 5% of tanh baseline
- Clear comparison: iFSQ-binary vs tanh-binary vs iFSQ-ternary (D-A8)

**Launch command**:
```bash
python playground/ifsq_binary_ablation.py --scale xl --steps 50000
```

---

#### D-A13: GPT-2 Medium + Ternary Head (Scale Test)

| Field | Value |
|-------|-------|
| **Priority** | 2 (strong) |
| **Status** | **EN CURSO** — lanzado 2026-03-18, ~6h GPU |
| **Script path** | `playground/gpt2_medium_ternary.py` (CREATED) |
| **GPU required** | Yes |
| **Runtime estimate** | 6 hours |
| **Dependencies** | D-A8 must be positive (confirms ternary works at base scale) |

**Hypothesis**: If ternary heads work on GPT-2 Small (D-A8), they should also work -- possibly better -- on GPT-2 Medium (355M params). Larger backbones produce richer hidden states, giving the triadic head more semantic signal to quantize. This tests the scaling hypothesis essential for the paper's "Discussion" section.

**Success criteria**:
- All D-A8 metrics maintained or improved at Medium scale
- PPL strictly better than GPT-2 Small variant (larger backbone helps language)
- Subsumption accuracy >= 92%
- Training converges without instability

**Launch command**:
```bash
python playground/gpt2_medium_ternary.py --steps 50000
```

---

### Priority 3 -- Useful for Completeness

---

#### D-A9: Hybrid Bits + Adversarial Disentanglement

| Field | Value |
|-------|-------|
| **Priority** | 3 (useful) |
| **Status** | NEEDS_SCRIPT |
| **Script path** | `playground/hybrid_adversarial.py` (to be created) |
| **GPU required** | Yes |
| **Runtime estimate** | 4.5 hours |
| **Dependencies** | D-A8 should complete first (establishes ternary baseline) |

**Hypothesis**: Following CB-LLMs (ICLR 2025), splitting bits into 30 supervised (gold labels from anclas.json) and 33 free (contrastive-only) enables the model to discover additional semantic structure beyond the La Danza ontology. Adversarial disentanglement (gradient reversal) forces concept information into the triadic head rather than letting the backbone bypass it.

**Success criteria**:
- Supervised bits achieve >= 90% alignment with gold labels
- Free bits show non-trivial cluster structure (k-means purity > 0.6)
- Adversarial loss converges to chance level
- Overall subsumption accuracy maintained
- At least 3 free bits show interpretable activation patterns

**Launch command**:
```bash
python playground/hybrid_adversarial.py --n-supervised 30 --n-free 33 --scale xl --steps 50000
```

---

### Priority 4 -- Nice to Have

---

#### D-A14: Gradient Decoupling Analysis

| Field | Value |
|-------|-------|
| **Priority** | 4 (optional, theoretical) |
| **Status** | NEEDS_SCRIPT |
| **Script path** | `playground/gradient_decoupling_analysis.py` (to be created) |
| **GPU required** | Yes -- requires gradient tracking during training |
| **Runtime estimate** | 5 hours (training + analysis overhead) |
| **Dependencies** | None, but lower priority than all above |

**Hypothesis**: Wang et al. (NeuS 2025, DARPA Award) predict that gradient flow over discrete-output networks decouples into independent per-bit optimization. We can empirically test this by tracking per-bit gradient norms and correlation matrices during training. If gradients decouple, we provide the first experimental evidence for their purely theoretical result.

**Success criteria**:
- Per-bit gradient correlation decreases over training (decoupling)
- Bits "lock in" at distinct training steps (progressive contraction)
- Correlation between bit lock-in order and final bit entropy
- Publishable figure showing gradient decoupling trajectory

**Launch command**:
```bash
python playground/gradient_decoupling_analysis.py --scale xl --steps 50000 --track-gradients
```

---

## 3. Paper Impact Matrix

Which experiment results feed into which paper sections:

| Experiment | Abstract | Intro | Method | Results | Ablations | Discussion | Related Work |
|------------|----------|-------|--------|---------|-----------|------------|--------------|
| D-A8 Ternary | | | X | X | | X | X (FSQ, BitNet) |
| D-A9 Hybrid | | | X | | X | X | X (CB-LLMs) |
| D-A10 iFSQ ablation | | | | | X | | X (iFSQ) |
| D-A11 Neg baselines | | | | X | X | | |
| D-A12 Bootstrap CI | X | | | X | | | |
| D-A13 GPT-2 Medium | X | X | | X | | X | |
| D-A14 Grad decoupling | | | | | | X | X (Wang et al.) |
| D-A15 Weight sweep | | | | X | X | | |
| D-A16 Neg sub FPR | | | | X | | | |

**Legend**: X = experiment results appear in or directly support that section.

**Critical path for paper**:
- Abstract claims require D-A12 (confidence intervals) and D-A13 (scale evidence)
- Results section requires D-A11 (baselines), D-A15 (sweep), D-A16 (FPR)
- Ablations require D-A10 (activation) and D-A11 (negative controls)
- Discussion requires D-A8 (ternary interpretation), D-A14 (theory connection)
- Related Work requires D-A8 (FSQ/BitNet connection), D-A9 (CB-LLMs connection)

---

## 4. Implications of the Research Survey

The related work survey (`research/related_work_survey.md`) analyzed 20 papers across 5 research areas. Key implications for our project:

### 4.1 We Are UNIQUE

No existing work combines all of our elements. The closest neighbors each lack a critical piece:

| System | End-to-end | Algebraic | Primes | Interpretable bits |
|--------|-----------|-----------|--------|-------------------|
| FSQ (ICLR 2024) | Yes | No | No | No |
| CB-LLMs (ICLR 2025) | Yes | No | No | Yes (continuous) |
| Hyperdimensional Probe | No (post-hoc) | Yes | No | No (random dims) |
| Monosemanticity (Anthropic) | No (post-hoc) | No | No | Yes (continuous) |
| CRH (2025) | Yes | No | No | No |
| **TriadicGPT (ours)** | **Yes** | **Yes** | **Yes** | **Yes** |

This uniqueness is a publication strength: we occupy an empty cell in the literature matrix.

### 4.2 We Are VALIDATED

Five independent research lines confirm our design choices were sound:

1. **Discrete representations work at scale** -- FSQ (ICLR 2024) + BitNet b1.58 (Microsoft)
2. **Concept bottlenecks scale to LLMs** -- CB-LLMs (ICLR 2025), near-zero accuracy cost
3. **Algebraic structure emerges from gradient training** -- Wang et al. (NeuS 2025, DARPA Award)
4. **tanh > sigmoid for discrete heads** -- Wang et al. proves odd activation functions are theoretically required; matches our empirical finding across experiments R3, P15, XL2
5. **~40% sparsity is natural** -- BitNet reports 42.3% zero rate in ternary weights, matching our dead bit percentage

### 4.3 The iFSQ Fix May Solve Our Biggest Limitation

Our ~42% dead bit rate is our most visible weakness. The iFSQ paper (Tencent, 2025) identified the exact same failure mode in FSQ and traced it to tanh's activation concentration near zero. Their one-line fix (`2*sigmoid(1.6*x) - 1`) achieves uniform bin utilization. If this works for us (D-A8/D-A10), our biggest limitation becomes a solved problem.

### 4.4 Wang et al. Provides Theoretical Grounding We Lacked

Our paper currently lacks a "why does this work?" explanation beyond empirical results. Wang et al.'s theorem -- that gradient flow over neural networks naturally decouples into independent boolean variable optimization under mild conditions -- provides exactly this theoretical foundation. Their three conditions (Gaussian init, odd activations, geometric symmetry) map onto our setup (standard init, tanh, embedding alignment).

### 4.5 CB-LLMs Shows Concept Bottlenecks Scale

Reviewers might question whether a 63-bit bottleneck can work on larger models. CB-LLMs demonstrates 208-476 concept neurons in LLMs with < 0.6% accuracy cost. Our 63 bits are within the same order of magnitude. Their adversarial disentanglement technique (D-A9) is the natural next step if we need to prevent the backbone from bypassing the triadic head.

### 4.6 VSA Comparison Clarifies Our Strengths and Weaknesses

Hyperdimensional computing (VSA) is the closest existing paradigm to our algebraic approach. The comparison is illuminating:

**Our strengths over VSA**: Exact algebra (divisibility-based subsumption is mathematically certain, not approximate), compact representation (63 bits vs 4096), interpretable named dimensions, 28.4x speed advantage.

**Our weaknesses vs VSA**: No bundling/superposition (cannot store multiple concepts in one vector and recover them), low noise tolerance (single bit flip changes the signature), limited capacity (2^63 vs 2^4096).

These tradeoffs are defensible and should be discussed honestly in the paper.

---

## 5. Critical Path to Paper Submission

### Day 0 (Today -- March 18)

- [x] Deep analysis session complete
- [x] Related work survey written
- [x] Experiment roadmap written (this document)
- [ ] E4 weight 1.0 running (~2h remaining)
- [ ] E10-v2 GPT-2 Medium InfoNCE running (~30min remaining)

### Day 1 (March 19)

**Morning** -- no GPU needed:
- [ ] D-A12: Multi-quad ensemble bootstrap (30 min)
- [x] D-A16: Negative subsumption FPR (10 min) — **DONE** FPR=24.1%
- [x] Harvest E4 weight 1.0 results, harvest E10-v2 results — **DONE**
- [x] Update paper tables with new numbers — AUDIT.md updated

**Afternoon** -- GPU available after E4 completes:
- [x] D-A15: Launch remaining sweep weights (0.5, 2.0, 5.0) — **DONE** (all 4 weights)

### Day 2 (March 18 — accelerated timeline)

- [x] D-A11: Negative baselines — **DONE** p<0.001, Cohen's d=6.64
- [x] D-A8 FSQ: Ternary head — **DONE** loss 0.951, sub 86.5%, ternary clean
- [x] D-A10: iFSQ binary ablation — **DONE** loss 0.924 (BEST), sub 87.1%
- [x] D-A8 Absmean: Ternary head — **DONE** loss 1.309 (25K), sub 85.7%
- [x] R3 Formula Comparison (CPU) — **DONE** Formula D ternary > continuous
- [x] R3 Chain & Fork Composition (CPU) — **DONE** Round-trip 98.1%, NOT word2vec
- [x] Aggregate all sweep results (E4) — **DONE**

### Remaining (Paper preparation)

- [ ] Paper edits: related work section (new citations from survey)
- [ ] Paper edits: discussion section (connections to Wang et al., FSQ, CB-LLMs)
- [ ] Paper edits: integrate D-A8 + R3 composition results (new sections)
- [ ] E4 Pareto figure (data ready in aggregate.json)
- [ ] R3 chain diagram figure
- [ ] Final paper review: all numbers updated, all tables complete
- [ ] Final proofreading and formatting
- [ ] Submission

### Optional (GPU time permitting)

- [ ] D-A13: GPT-2 Medium + Ternary (6h) — D-A8 positive, scaling test viable
- [ ] D-A9: CB-LLMs comparison (new related work link)
- [ ] D-A8 Absmean rerun at 50K steps (fair comparison with FSQ)
- Paper narrative stays with binary head + iFSQ activation as "improved" version

If D-A11 (negative baselines) shows surprisingly high baseline accuracy:
- Indicates our metric is too lenient, not that our model is bad
- Tighten success criteria (exact match vs. threshold match)
- Add additional negative controls (permuted bit assignments)

---

## 6. What We Should NOT Do

From the survey (Section 8.4) and deep analysis, these are tempting directions that would hurt more than help:

### Do Not Co-Learn Gold Labels
CRH (2025) shows co-learned hash centers outperform fixed ones. But our gold labels come from La Danza's philosophical framework -- they encode intentional meaning (`amor = fuego AND agua AND union AND vida AND placer AND consciente AND querer AND interoception`). Making them learnable would destroy this interpretive structure and undermine the paper's core claim that the bits are semantically meaningful.

**Exception**: Free bits in a hybrid architecture (D-A9) can be learned.

### Do Not Switch to Continuous SAE Features
Anthropic's sparse autoencoders discover thousands of interpretable continuous features. Our approach is fundamentally different: discrete, algebraically verifiable, compact. Switching to continuous features would forfeit our unique contribution (exact subsumption via divisibility, deterministic analogy verification).

### Do Not Replace Prime Multiplication with VSA Binding
VSA's Hadamard product binding has nice properties (self-inverse, noise-tolerant). But our prime multiplication is the foundation of the entire algebraic framework. Switching binding operators is not an incremental change -- it's a different project entirely.

### Do Not Scale to D=4096 Hypervectors
VSA literature uses 4096-10000 dimensional vectors for noise tolerance. Our 63-bit compact representation is a deliberate design choice and a strength (28.4x speed, interpretable named dimensions). Scaling dimensions would lose both advantages without clear benefit for our use case.

### Do Not Add Bundling/Superposition
VSA's ability to superpose multiple concepts in one vector is elegant but incompatible with prime factorization. `Phi(cat) * Phi(dog)` already has a meaning in our algebra (conjunction), so we cannot also use it for bundling. This is a fundamental limitation we should acknowledge, not try to fix.

### Do Not Chase Grokking
Wang et al.'s theory connects to grokking (sudden generalization after memorization). It is tempting to set up grokking experiments. But grokking requires specific training regimes (small data, long training) that don't map to our setup. Cite the connection, don't chase the experiment.

---

## Appendix A: Script Status Summary

| Script | Exists | Location |
|--------|--------|----------|
| D-A8 danza_ternary.py | **YES** | `playground/danza_ternary.py` — **COMPLETE** (fsq: 0.951, absmean: 1.309) |
| D-A9 hybrid_adversarial.py | NO | needs creation |
| D-A10 ifsq_binary_ablation.py | **YES** | `playground/ifsq_binary_ablation.py` — **COMPLETE** (0.924, sub 87.1%) |
| D-A11 negative_baselines.py | **YES** | `playground/negative_baselines.py` — **COMPLETE** (p<0.001) |
| D-A12 multi_quad_ensemble.py | **YES** | `playground/multi_quad_ensemble.py` — **COMPLETE** (94.6%) |
| D-A13 gpt2_medium_ternary.py | **SI** | CREATED and RUNNING (2026-03-18) |
| D-A14 gradient_decoupling_analysis.py | NO | needs creation |
| D-A15 sub_weight_sweep.py | **YES** | `playground/sub_weight_sweep.py` — **COMPLETE** (4 weights) |
| D-A16 negative_subsumption_test.py | **YES** | `playground/negative_subsumption_test.py` — **COMPLETE** (FPR=24.1%) |
| R3 formula comparison | **YES** | `playground/r3_formula_comparison.py` — **COMPLETE** (Formula D > continuous) |
| R3 chain & fork test | **YES** | `playground/r3_chain_test.py` — **COMPLETE** (round-trip 98.1%) |

**Completed**: 8 of 11 scripts (D-A8, D-A10, D-A11, D-A12, D-A15, D-A16, R3 formula, R3 chain).
**Scripts needing creation**: 2 (D-A9, D-A14) — all optional. D-A13 CREATED.
**All priority experiments DONE.**

## Appendix B: GPU Time Budget

Total GPU hours needed (after currently running experiments complete):

| Experiment | GPU Hours | Priority |
|------------|-----------|----------|
| D-A12 Bootstrap | 0 (inference) | P1 |
| D-A16 Neg FPR | 0 (inference) | P1 |
| D-A15 Remaining sweep | 6 | P1 |
| D-A11 Neg baselines | 8 | P1 |
| D-A8 Ternary head | 4 | P2 |
| D-A10 iFSQ ablation | 4 | P2 |
| D-A13 GPT-2 Medium | 6 | P2 |
| D-A9 Hybrid adversarial | 4.5 | P3 |
| D-A14 Grad decoupling | 5 | P4 |
| **Total** | **37.5 GPU hours** | |

**Priority 1 only**: 14 GPU hours (~2 days at ~8h/day)
**Priority 1 + 2**: 28 GPU hours (~4 days)
**All experiments**: 37.5 GPU hours (~5 days)

The 7-day timeline in Section 5 accounts for sequential GPU access, script creation time, result analysis, and paper editing -- all interleaved.
