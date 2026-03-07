# Benchmark Specification — Triadic MicroGPT

This document defines every benchmark we will run, how to run it, and what constitutes a pass.

## A. Language Quality Benchmarks

### A.1 Perplexity (Intrinsic)
- **Tool**: `src/evaluate.py`
- **Data**: TinyStories val split (last 10% of corpus, never used in training)
- **Metric**: exp(avg_cross_entropy_loss)
- **Baseline**: 2.80 (Run 10)
- **Pass**: < 5.0 on held-out data

### A.2 MAUVE Score
- **Tool**: `pip install mauve-text`
- **Data**: Generate 500 stories (128 tokens each), compare to 500 reference stories
- **Metric**: MAUVE(P_model, P_human) in [0, 1]
- **Pass**: > 0.7

### A.3 Distinct-n
- **Tool**: Custom script
- **Data**: Generate 1000 stories (64 tokens each)
- **Metric**: |unique n-grams| / |total n-grams| for n=1,2,3
- **Pass**: Distinct-1 > 0.5, Distinct-2 > 0.7

### A.4 Repetition Rate
- **Tool**: Custom script
- **Data**: Same 1000 generations
- **Metric**: % of generated texts containing a repeated 4-gram
- **Pass**: < 15%

### A.5 lm-evaluation-harness (Optional for 40M model)
- **Tool**: `pip install lm-eval`
- **Tasks**: hellaswag, arc_easy, piqa, boolq, lambada_openai
- **Purpose**: Establish baseline for comparison with other small LMs
- **Pass**: No absolute target — purpose is comparison

## B. Triadic-Specific Benchmarks

### B.1 Bit Entropy
- **Setup**: Encode 1000 WordNet concepts through the model
- **Metric**: Per-bit entropy H(bᵢ) = -p log p - (1-p) log(1-p), averaged over bits
- **Pass**: Mean entropy > 0.8 (bits used roughly equally)
- **Failure mode**: H ≈ 0 means all bits saturated (triadic collapse)

### B.2 Signature Diversity
- **Setup**: Encode 1000 concepts, count unique prime signatures
- **Metric**: |unique signatures| / 1000
- **Pass**: > 0.80 (at least 800 unique primes out of 1000 concepts)

### B.3 Taxonomic Consistency (WordNet)
- **Setup**: Extract hypernym/hyponym pairs from WordNet (e.g., Animal→Dog)
- **Test set**: 5000 positive pairs + 5000 negative (random) pairs
- **Metric**:
  - Subsumption Recall: % of true hypernym pairs where Φ(hyper) | Φ(hypo)
  - FPR: % of random pairs falsely showing subsumption
  - F1: harmonic mean
- **Pass**: F1 > 0.5, FPR < 5%

### B.4 Semantic Analogy
- **Setup**: Use standard analogy datasets (Google analogy, BATS)
- **Method**: Given a:b :: c:?, compute candidate = compose(c, gap(a→b))
- **Metric**: top-1 and top-5 accuracy in retrieving correct d
- **Pass**: top-5 > 40% (ambitious for 40M model)

### B.5 Interpretability Probe
- **Setup**: Freeze model, extract triadic bits for 5000 WordNet nouns
- **Train**: Linear classifier (bits → 26 WordNet supersenses)
- **Metric**: Classification accuracy (5-fold CV)
- **Pass**: > 40% (random baseline ~4%)

### B.6 Relational Bias Audit (Extended)
- **Setup**: Full 10K gold primes dictionary
- **Test set**: 5000 positive + 5000 negative subsumption pairs
- **Metric**: Accuracy, FPR, Precision, Recall, F1
- **Pass**: Accuracy > 95%, FPR < 2%
- **Important**: Only valid on models NOT trained with distillation on the same gold primes

### B.7 Cross-Domain Generalization
- **Setup**: Train on TinyStories (children's domain), test triadic quality on:
  - Scientific terms (medicine, physics)
  - Legal terms
  - Technical terms (programming, engineering)
- **Metric**: Same as B.3 but on out-of-domain concepts
- **Pass**: Report only — expected degradation is informative

### B.8 Geometric Concept Topology (Experimental — UHRT-inspired)
- **Origin**: Adapted from Unified Holographic Resonance Theory layered structure
- **Setup**: Organize 100+ concepts into 12 semantic domains (royalty, animals, family, emotions, etc.)
- **Metrics**:
  - **0-simplex (Point)**: UBS(concept) = log2(Phi(x)) + H(bits) — informational complexity per concept
  - **1-simplex (Line)**: Pairwise connectivity rate — % of concept pairs sharing prime factors
  - **2-simplex (Triangle)**: Coherence rate — % of intra-domain triples sharing GCD > 1
  - **Bubble Analysis**: Separation ratio = avg_intra_sim / avg_inter_sim per domain
- **Pass**: Separation ratio > 1.5 (domains form distinct clusters in prime space)
- **Script**: `benchmarks/scripts/geometric_topology.py`
- **Note**: This is EXPLORATORY. If separation ratio ~1.0, triadic collapse is confirmed from a topological perspective. If >> 1.5, we have evidence of emergent semantic geometry.

## C. Ablation Benchmarks

### C.1 Triadic vs Non-Triadic Baseline
- **Setup**: Train identical model with α=0 (no triadic loss)
- **Compare**: Perplexity, MAUVE, generation quality
- **Pass**: Triadic model within 5% of baseline on all language metrics

### C.2 Alpha Sweep
- **Values**: α ∈ {0.01, 0.05, 0.1, 0.15, 0.3, 0.5}
- **Measure**: Language perplexity AND triadic F1 for each
- **Output**: Pareto curve showing optimal tradeoff

### C.3 Bits Sweep
- **Values**: bits ∈ {8, 16, 32, 48, 64, 128}
- **Measure**: Signature diversity, subsumption F1, collision rate
- **Output**: Curve showing expressiveness vs overhead

### C.4 Scaling Sweep
- **Sizes**: 1M, 6M, 16M, 40M params
- **Measure**: All metrics from A.1 + B.1-B.3
- **Output**: Log-scale curves showing scaling behavior

## D. Comparison Benchmarks

### D.1 End-to-End (MicroGPT) vs Post-Hoc (Engine)
- **Setup**: Encode same 1000 concepts through both systems
  - MicroGPT: single forward pass → triadic bits
  - Engine: embed with all-MiniLM-L6-v2 → PCA projection → bits
- **Compare**:
  - Subsumption F1
  - Analogy accuracy
  - Bit entropy
  - Inference latency (ms per concept)
- **Output**: Side-by-side comparison table for paper

## Results Storage

All benchmark results are stored as JSON in:
```
benchmarks/
  results/
    v{VERSION}_{BENCHMARK}_{DATE}.json
  figures/
    {BENCHMARK}_{TYPE}.png
```

Example:
```
benchmarks/results/v2.0_bit_entropy_2026-03-10.json
benchmarks/results/v2.0_taxonomic_consistency_2026-03-10.json
benchmarks/figures/scaling_curves.png
benchmarks/figures/pareto_frontier.png
```
