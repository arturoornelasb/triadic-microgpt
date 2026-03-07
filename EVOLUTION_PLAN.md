# Triadic MicroGPT — Evolution Plan & Research Roadmap

> **Goal**: Produce a publishable research paper demonstrating that end-to-end triadic training yields interpretable, algebraically verifiable semantic representations in a generative language model — with rigorous, industry-standard benchmarks at every phase.

---

## Phase 0: Baseline Audit (COMPLETED)
**Objective**: Establish current model capabilities and identify critical gaps.

### Achieved
| Metric | Value | Status |
|--------|-------|--------|
| Model Size | 40M params (12L/512D/8H/64bits) | Done |
| Language Loss | 1.03 (Run 10, Knowledge Distillation) | Done |
| Perplexity (TinyStories val) | 2.80 | Done |
| Bias Audit Accuracy | 98.5% (FPR 0.96%) | Done |
| Coherent Generation | Yes — multi-sentence narratives | Done |

### Phase 0.5: Diagnostic Results (2026-03-06)

**CRITICAL FINDING**: The original diagnosis was WRONG. The triadic collapse is NOT in the base model — it was CAUSED by knowledge distillation.

| Metric | XL Puro (Run 9) | GoldPrimes (Run 10) |
|--------|-----------------|---------------------|
| Unique signatures | **109/112 (97.3%)** | 1/112 (0.9%) COLLAPSED |
| Mean bit entropy | **0.381** | 0.000 |
| King↔Queen similarity | **88.9%** | 100% (all same) |
| King↔Dog similarity | **60.5%** | 100% (all same) |
| Happy↔Sad similarity | **61.5%** | 100% (all same) |
| Fire↔Water similarity | **78.8%** | 100% (all same) |

**Key Insights**:
1. The XL model (no distillation) already differentiates: related pairs (89%) > unrelated pairs (60%)
2. Knowledge distillation with 5× alpha weight DESTROYED all differentiation
3. 97.3% unique signatures means the model is NOT collapsed — it's working
4. Per-bit entropy (0.38) is below ideal (0.8) — ~15 bits are nearly dead (always positive)
5. ~20 bits have good entropy (>0.5) and carry most of the semantic information

**New Baseline (XL Pure model)**:
- [x] Bit Entropy: 0.381 (partial — many bits biased but functional)
- [x] Heatmap: Shows clear per-concept variation in ~30 active bits
- [x] Diversity: 97.3% unique (PASS)
- [x] Differentiation: Related > Unrelated (PASS)

---

## Phase 1: Improve Triadic Quality (4/6 targets MET)
**Objective**: The model ALREADY differentiates concepts. The goal is now to:
1. ~~Increase per-bit entropy from 0.38 → >0.6~~ **DONE (0.749)**
2. ~~Improve separation: push unrelated pairs below 40%~~ **DONE (30%)**
3. ~~Fix the knowledge distillation approach~~ **DONE (configurable --dist-weight, --no-distill)**
4. ~~Maintain or improve language quality~~ **DONE (loss 0.946, best ever)**

### 1.1 Key Findings (Runs 12-15, 2026-03-07)

**Root Cause of Triadic Collapse: Coherence Loss (Run 12)**
The coherence loss component (forcing adjacent tokens to agree) is the root cause of all triadic collapse. With warmup=0.3 (35K triadic steps), the model fully collapsed. Run 9's warmup=0.8 (10K steps) was only partially collapsed. **Coherence loss permanently removed.**

**Embedding Alignment as Semantic Teacher (Runs 14-15)**
Without semantic signal, entropy regularization alone produces diverse but random projections (Run 13). The model's own trained embeddings serve as a teacher: `L_align = MSE(cosine_sim_triadic, cosine_sim_embed)` on sampled token pairs. This transfers semantic structure to the triadic head.

**Architecture Changes Made**:
- Removed coherence loss from `triadic_loss()` (was the 4th loss component)
- Added `L_entropy`: penalizes low per-bit entropy, activates dead bits
- Added `L_align`: aligns triadic similarity with embedding similarity
- Added CLI args: `--entropy-weight`, `--align-weight`, `--dist-weight`, `--no-distill`

### 1.2 Progress Table

| Metric | Run 9 | Run 12 | Run 13 | Run 14 | Run 15 | Target | Status |
|--------|-------|--------|--------|--------|--------|--------|--------|
| Bit Entropy | 0.381 | 0.000 | 0.521 | 0.720 | **0.749** | > 0.6 | PASS |
| Unique Sigs | 97.3% | 0.9% | 100% | 100% | **100%** | > 95% | PASS |
| Unrelated Sim | 60% | — | 51% | 56% | **30%** | < 40% | PASS |
| Semantic Gap | +29pt | — | -10pt | +17pt | **+21pt** | positive | PASS |
| Language Loss | 1.277 | 1.036 | 0.981 | 0.980 | **0.946** | < 1.40 | PASS |
| Sep. Ratio | 1.01 | 1.00 | 1.01 | 1.00 | **1.02** | > 1.5 | PENDING |

### 1.3 Remaining
- [ ] **Domain separation ratio** > 1.5 — Run 16 (alpha=0.2, align=10) in progress
- [ ] **Language quality benchmark** (`language_quality.py`) on best checkpoint
- [ ] Decide: is sep. ratio 1.5 achievable, or should target be revised?

---

## Phase 2: Industry-Standard Language Benchmarks
**Objective**: Quantify language quality using standard NLP evaluation suites, establishing a publishable baseline comparable to other small LMs.

### 2.1 Intrinsic Metrics (Automated)
| Benchmark | Tool | Target | Notes |
|-----------|------|--------|-------|
| **Perplexity** (TinyStories val) | `evaluate.py` | < 5.0 | Already at 2.80 on train; need proper held-out split |
| **Perplexity** (WikiText-103) | Custom eval script | Report only | Cross-domain generalization |
| **BLEU-4** (story completion) | `sacrebleu` | Report only | Given prefix, generate continuation, compare to reference |
| **MAUVE Score** | `mauve-text` | > 0.7 | Measures distribution similarity between model & human text |
| **Distinct-n** (n=1,2,3) | Custom | Report only | Lexical diversity of generations |
| **Repetition Rate** | Custom | < 10% | % of 4-grams that repeat within a generation |

### 2.2 Downstream Task Evaluation (via lm-evaluation-harness)
| Benchmark | Category | Target | Notes |
|-----------|----------|--------|-------|
| **HellaSwag** | Commonsense reasoning | Report (expect low for 40M) | Standard LM benchmark |
| **ARC-Easy** | Science reasoning | Report | Multiple-choice QA |
| **PIQA** | Physical intuition | Report | Physical commonsense |
| **BoolQ** | Reading comprehension | Report | Yes/no questions |
| **LAMBADA** | Long-range dependency | Report | Last-word prediction |

> **Note**: For a 40M-param model trained on TinyStories, downstream tasks will show low absolute scores. The value is in showing that triadic training does NOT degrade performance vs. a matched non-triadic baseline.

### 2.3 Ablation: Triadic vs Non-Triadic Baseline
Train an identical architecture (12L/512D/8H) WITHOUT the triadic head (α=0) on the same data for the same steps. Compare:
- Perplexity: must be within 5% to prove triadic head is not a tax on language quality
- Generation quality: blind human preference comparison
- Training speed: wall-clock time comparison

---

## Phase 3: Triadic-Specific Benchmarks (Novel Contribution)
**Objective**: Define and execute benchmarks that are UNIQUE to neurosymbolic prime-factor representations — these form the core contribution of the paper.

### 3.1 Taxonomic Consistency (WordNet Hierarchy Preservation)
**Setup**: Use WordNet hypernym/hyponym pairs. If "Dog" is-a "Animal", then Φ(Animal) | Φ(Dog) (subsumption).
| Metric | Definition | Target |
|--------|-----------|--------|
| **Subsumption Recall** | % of true hypernym pairs where Φ(hyper) \| Φ(hypo) | > 60% |
| **Subsumption FPR** | % of unrelated pairs falsely showing subsumption | < 5% |
| **Taxonomic F1** | Harmonic mean of precision and recall | > 0.5 |

### 3.2 Semantic Analogy via Prime Algebra
**Setup**: Analogies like "King:Queen :: Man:Woman". In prime space:
```
Φ(Queen) = lcm(Φ(King), Φ(Woman)) / gcd(Φ(King), Φ(Man))
```
| Metric | Definition | Target |
|--------|-----------|--------|
| **Analogy Accuracy (top-1)** | Correct concept retrieved by prime algebra | > 10% (paper baseline: 2-10%) |
| **Analogy Accuracy (top-5)** | Correct concept in top-5 by prime similarity | > 25% |

### 3.3 Compositional Reasoning
**Setup**: Given concepts A and B, compute lcm(Φ(A), Φ(B)) and verify the composed concept is semantically meaningful.
| Metric | Definition | Target |
|--------|-----------|--------|
| **Composition Coherence** | Human-rated meaningfulness of composed concepts (1-5) | > 3.0 |
| **Composition Consistency** | compose(A,B) == compose(B,A) always | 100% |

### 3.4 Interpretability Probing
**Setup**: Train a linear probe on frozen triadic bits to predict WordNet supersenses (26 categories: noun.animal, noun.person, etc.).
| Metric | Definition | Target |
|--------|-----------|--------|
| **Probe Accuracy** | Linear classifier on triadic bits → supersense | > 40% |
| **Probe vs Embedding Baseline** | Compare to same probe on hidden states | Report delta |
| **Bit-Feature Correlation** | Mutual info between each bit and each supersense | Heatmap |

### 3.5 Relational Bias Audit (Extended from Run 11)
Scale up the existing auditor to a full benchmark:
| Metric | Definition | Target |
|--------|-----------|--------|
| **Subsumption Accuracy** | On 10K pairs from gold primes | > 95% |
| **FPR** | False subsumption rate | < 2% |
| **Cross-Domain Transfer** | Audit on domains NOT in training data | Report |

### 3.6 Geometric Concept Topology (Experimental — UHRT-inspired)
**Origin**: Adapted from the Unified Holographic Resonance Theory (UHRT) project. The idea is that concepts don't exist in isolation — they form geometric structures:
- **Point** (0-simplex): A single concept's prime signature Φ(x) and its informational complexity UBS(x) = log2(Φ(x)) + H(bits)
- **Line** (1-simplex): Pairwise relationships via GCD — shared prime factors create "edges" between concepts
- **Triangle** (2-simplex): Three concepts sharing a common factor form a coherent triple
- **Volume** (3-simplex+): Clusters of related concepts forming "semantic bubbles" — domains like {animals}, {emotions}, {family}

**Hypothesis**: Semantically related concepts should form dense simplicial complexes in prime space, while unrelated concepts remain topologically disconnected. If the triadic head works, intra-domain similarity >> inter-domain similarity.

**Metrics**:
| Metric | Definition | Target |
|--------|-----------|--------|
| **Separation Ratio** | avg_intra_similarity / avg_inter_similarity per domain | > 1.5 |
| **Triangle Coherence** | % of intra-domain triples sharing GCD > 1 | > 50% |
| **Bubble Entropy** | Shannon entropy of factor distribution within domain | Report |
| **UBS Variance** | Std dev of concept complexity across vocabulary | > 0 (against collapse) |

**Script**: `benchmarks/scripts/geometric_topology.py`

**Status**: Exploratory — results will determine if this becomes a full paper contribution or is dropped.

### 3.7 Comparison: End-to-End vs Post-Hoc
Compare Triadic MicroGPT's learned projections against the parent Engine's post-hoc PCA projections on identical concept sets:
| Metric | MicroGPT (end-to-end) | Engine (post-hoc PCA) |
|--------|----------------------|----------------------|
| Subsumption F1 | Measure | Measure |
| Analogy accuracy | Measure | Measure |
| Bit entropy | Measure | Measure |
| Inference speed | Measure (single forward pass) | Measure (embed + project) |

---

## Phase 4: Scaling Study
**Objective**: Demonstrate scaling laws for the triadic approach across model sizes.

### 4.1 Model Size Sweep
| Config | Params | Train Steps | Expected Time |
|--------|--------|-------------|---------------|
| Small: 4L/128D/4H/16bits | ~1M | 20K | ~2 min |
| Medium: 6L/256D/8H/32bits | ~6M | 30K | ~5 min |
| Large: 8L/384D/8H/48bits | ~16M | 40K | ~15 min |
| XL: 12L/512D/8H/64bits | ~40M | 50K | ~76 min |

For each size, measure: language loss, perplexity, triadic entropy, subsumption F1, bit utilization.
Plot scaling curves: metric vs log(params).

### 4.2 Triadic Bits Sweep
Fix model at XL size, vary only n_triadic_bits: 8, 16, 32, 48, 64, 128.
Measure: expressiveness (unique primes), collision rate, subsumption accuracy.

### 4.3 Alpha (Triadic Weight) Sweep
Fix model at XL size, vary α: 0.01, 0.05, 0.1, 0.15, 0.3, 0.5.
Measure: language loss vs triadic quality tradeoff curve (Pareto frontier).

---

## Phase 5: Data & Training Improvements
**Objective**: Improve training data and methodology for publication-quality results.

### 5.1 Curated Concept Training Data
- [ ] Extend beyond TinyStories: add Wikipedia Simple English, children's encyclopedias
- [ ] Create concept-enriched training examples that naturally expose semantic relationships
- [ ] Build evaluation-only held-out sets (never seen during training)

### 5.2 Training Methodology
- [ ] Proper train/val/test split (80/10/10)
- [ ] Learning rate finder experiment
- [ ] Gradient accumulation for effective larger batch sizes
- [ ] Mixed-precision profiling and optimization

### 5.3 Reproducibility
- [ ] Fix all random seeds (torch, numpy, python random)
- [ ] Log full environment (conda list, pip freeze, GPU info, CUDA version)
- [ ] Deterministic DataLoader (num_workers=0 or seed workers)
- [ ] Publish training configs as YAML files

---

## Phase 6: Paper Preparation
**Objective**: Write and submit a research paper with all results.

### 6.1 Paper Structure (Proposed)
1. **Abstract**: End-to-end triadic training for interpretable LMs
2. **Introduction**: The interpretability gap in LLMs; prime factorization as solution
3. **Background**: Triadic Neurosymbolic Engine (parent work); knowledge distillation
4. **Method**: Architecture, dual-objective training, triadic loss components
5. **Experiments**:
   - 5.1 Language quality (Phase 2 benchmarks)
   - 5.2 Triadic differentiation (Phase 3 benchmarks)
   - 5.3 Scaling laws (Phase 4)
   - 5.4 Ablation studies (α sweep, bits sweep, baseline comparison)
   - 5.5 Comparison with post-hoc projection
6. **Analysis**: What the bits learn, interpretability probing, failure cases
7. **Discussion**: Limitations, future work (larger models, multilingual)
8. **Conclusion**

### 6.2 Figures and Tables
- Training loss curves (language + triadic + distillation)
- Triadic heatmaps (concepts × bits)
- Scaling law curves
- Pareto frontier (language quality vs triadic quality)
- Confusion matrix for subsumption audit
- Analogy examples with prime algebra
- Comparison table: end-to-end vs post-hoc

### 6.3 Code Release
- Clean repository with reproducible training scripts
- Pre-trained model weights on HuggingFace Hub
- Benchmark suite as standalone scripts
- Docker/conda environment file

---

## Execution Priority

```
Phase 1 (Triadic Quality)    ████████████████░░░░  4/6 targets MET, Run 16 in progress
Phase 2 (Language Benchmarks) ████████████░░░░░░░░  Pending — language_quality.py ready
Phase 3 (Triadic Benchmarks)  ██████████████░░░░░░  Core paper contribution
Phase 4 (Scaling Study)       ██████████░░░░░░░░░░  Strengthens claims
Phase 5 (Data/Training)       ████████░░░░░░░░░░░░  Improves results
Phase 6 (Paper)               ██████░░░░░░░░░░░░░░  Final deliverable
```

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| v0.1 | 2026-03-04 | Initial pure-Python prototype (866K params) |
| v0.5 | 2026-03-04 | PyTorch GPU training, 5.8M params, loss 1.37 |
| v0.8 | 2026-03-05 | XL model (40M), loss 1.27, HuggingFace tokenizer |
| v1.0-GoldPrimes | 2026-03-05 | Knowledge Distillation, loss 1.03, audit 98.5% |
| v1.1-Industrial | 2026-03-05 | 10K WordNet dictionary, 64-bit primes, FPR 0.96% |
| v1.1-entropy | 2026-03-07 | Run 12: entropy reg, COLLAPSED (coherence loss root cause found) |
| v1.2-nocoherence | 2026-03-07 | Run 13: coherence removed, 100% unique, entropy 0.521, random |
| v1.3-align | 2026-03-07 | Run 14: embedding alignment, entropy 0.720, semantics emerging |
| **v1.4-strongalign** | **2026-03-07** | **Run 15: correct semantic ordering, entropy 0.749, loss 0.946** |
| v1.5-maxalign (NEXT) | TBD | Run 16: alpha=0.2, align=10, push domain separation |
| **v2.0** | TBD | Full benchmark suite + scaling study |
| **v3.0** | TBD | Paper submission |
