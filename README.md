# TriadicGPT

[![CI](https://github.com/arturoornelasb/triadic-microgpt/actions/workflows/ci.yml/badge.svg)](https://github.com/arturoornelasb/triadic-microgpt/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/triadic-head)](https://pypi.org/project/triadic-head/)
[![Python](https://img.shields.io/pypi/pyversions/triadic-head)](https://pypi.org/project/triadic-head/)
[![License](https://img.shields.io/badge/license-BUSL--1.1-blue)](LICENSE)
[![HF Model](https://img.shields.io/badge/%F0%9F%A4%97-triadic--gpt--40m-yellow)](https://huggingface.co/arturoornelasb/triadic-gpt-40m)
[![HF Model](https://img.shields.io/badge/%F0%9F%A4%97-triadic--gpt2--medium-yellow)](https://huggingface.co/arturoornelasb/triadic-gpt2-medium)
[![DOI Paper](https://zenodo.org/badge/DOI/10.5281/zenodo.19206545.svg)](https://doi.org/10.5281/zenodo.19206545)
[![DOI Repo](https://zenodo.org/badge/DOI/10.5281/zenodo.19207845.svg)](https://doi.org/10.5281/zenodo.19207845)

**End-to-end prime factorization in a generative language model.**

TriadicGPT is a 40M-parameter GPT that learns discrete prime-factor signatures alongside standard next-token prediction. A lightweight *triadic projection head* maps each token's hidden state to a binary vector, which encodes as a prime composite &Phi;(x) = &prod; p&#x1D62;. The result: algebraically verifiable semantic representations that emerge as a side effect of language modeling, at zero cost to language quality.

```
King  = 2 x 3 x 5        (Royalty x Male x Authority)
Queen = 2 x 5 x 7        (Royalty x Authority x Female)

Shared:     gcd  -> {2, 5}    Royalty, Authority
Difference: div  -> {3} vs {7}   Male vs Female
Analogy:    factor transfer   king:queen :: man:woman
```

## Key Results

| Metric | Value |
|--------|-------|
| Language cost | **Zero** (PPL 7.69 vs 7.56 ablation, +1.7%) |
| Analogy verification | **98%** (51 analogies) |
| Subsumption accuracy | **98.3%** held-out (158 supervised anchors) |
| Domain separation | **1.21** mean (12 domains, sentence-level) |
| GPT-2 transfer gap closure | **48%** toward Engine PCA upper bound |
| Signature uniqueness | **100%** across all evaluated concepts |
| Algebraic ops | **O(1)** bitwise, 5-78x faster than prime arithmetic |
| Scale crossover | Semantic ordering emerges at **~20M params** |

## Architecture

```
Text -> BPE (4096 vocab) -> TriadicGPT (12L / 512D / 8H)
                                    |
                              +-----+-----+
                              |           |
                         LM Head    Triadic Head
                              |           |
                        next-token   tanh(Wx) -> bits -> Phi(x) = prod(p_i)
                              |           |
                              +-----+-----+
                                    |
                         L = L_lang + alpha * L_triadic
```

The triadic loss combines four components: diversity (bits fire ~50%), contrastive (sequences differ), entropy (no dead bits), and **embedding alignment** (triadic similarity tracks embedding similarity).

### Model Scales

| Scale | Layers | Dim | Heads | Bits | Params |
|-------|--------|-----|-------|------|--------|
| small | 4 | 128 | 4 | 16 | 1.3M |
| base | 6 | 256 | 8 | 32 | 5.8M |
| large | 8 | 384 | 8 | 48 | 15.9M |
| xl | 12 | 512 | 8 | 64 | 40M |

### Algebraic Operations

Eight operations over prime composites, all O(1) via bitwise arithmetic:

| Operation | Definition | Example |
|-----------|-----------|---------|
| **Subsumption** | A &sube; B iff &Phi;(A) divides &Phi;(B) | animal &sube; dog |
| **Composition** | A &cup; B = lcm(&Phi;(A), &Phi;(B)) | king + female = queen |
| **Intersection** | A &cap; B = gcd(&Phi;(A), &Phi;(B)) | shared features |
| **Difference** | A &setminus; B = &Phi;(A) / gcd(&Phi;(A), &Phi;(B)) | unique to A |
| **Symmetric diff** | A &Delta; B | features in exactly one |
| **Analogy (R3)** | A:B :: C:D via factor transfer | king:queen :: man:woman |
| **Negation** | &not;A = complement bits | invert all features |
| **Projection** | &pi;(A, mask) | extract feature subset |

## Installation

**Requirements**: Python 3.10, CUDA 12.8+ (for GPU training)

```bash
# Option A: conda (recommended)
conda env create -f environment.yml
conda activate triadic-microgpt

# Option B: pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

**Git LFS**: Model checkpoints (`.pt` files) are tracked with Git LFS. After cloning:

```bash
git lfs install
git lfs pull
```

**Data**: Download TinyStories (~1.8 GB) before training. See [`data/README.md`](data/README.md) for instructions.

```bash
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('roneneldan/TinyStories')
with open('data/TinyStories-train.txt', 'w', encoding='utf-8') as f:
    for story in ds['train']:
        f.write(story['text'] + '\n')
"
```

## Usage

### Train

```bash
# XL model (40M params, ~76 min on RTX 5060 Ti)
python src/torch_train.py --scale xl --steps 50000

# Reproduce the paper's production model (Run 15, v1.4-strongalign)
python src/torch_train.py \
  --scale xl --steps 50000 \
  --alpha 0.05 --entropy-weight 1.0 --align-weight 5.0 \
  --triadic-warmup-pct 0.3 --no-distill \
  --checkpoint-dir checkpoints/torch_run15_strongalign
```

Additional flags: `--override-bits N` (decouple bits from scale), `--dtype bfloat16` (default on Blackwell), `--grad-checkpoint` (save VRAM), `--no-compile` (disable torch.compile), `--dist-weight W` (knowledge distillation weight).

### Pre-tokenize (Optional)

Encode the corpus to a binary `.npy` cache once, eliminating the tokenization bottleneck at training startup:

```bash
python src/pre_tokenize.py --corpus data/TinyStories-train.txt --output data/tokens_30k.npy
```

### Evaluate

```bash
# Full evaluation (perplexity, generation, triadic analysis)
python src/evaluate.py --model checkpoints/torch_run15_strongalign/model_best.pt

# Relational bias audit (Experiment 8)
python src/auditor.py
```

### Chat

```bash
python src/chat.py
```

### Fine-tune

```bash
# Instruction fine-tuning on Alpaca
python src/torch_finetune.py --model checkpoints/torch_run15_strongalign/model_best.pt
```

### Tests

```bash
python tests/test_all.py   # 42 unit tests (autograd, transformer, triadic, integration)
```

## Desktop UI

A PySide6 desktop application for exploring, auditing, and chatting with TriadicGPT. Three backends: native `.pt` checkpoints, GPT-2 Transfer (Experiment 10), and any HuggingFace model via `triadic-head`.

| Tab | Description |
|-----|-------------|
| **Encoder** | Tokenize text and visualize bit vectors, prime composites, projection bars |
| **Compare** | Algebraic comparison — Jaccard similarity, subsumption, clickable prime factor chips with Prime Inspector |
| **Explore** | Pairwise similarity heatmap for N concepts with ranked pairs |
| **Analogy** | Solve A:B::C:? via prime algebra with top-10 vocabulary matches |
| **Validate** | Semantic quality audit — diversity, active bits, ordering checks per word group |
| **Chat** | Conversational interface with real-time triadic signature analysis (native + GPT-2 backends) |
| **Benchmarks** | Browse stored benchmark results (27 JSON files + charts) |

```bash
pip install PySide6
python ui/app.py
```

See [ui/README.md](ui/README.md) for the full feature reference.

## Benchmark Suite

12 evaluation scripts in `benchmarks/scripts/`:

| Script | Purpose |
|--------|---------|
| `scaling_study.py` | 4-point scaling study (1.3M-40M), PPL + triadic quality |
| `analogy_benchmark.py` | Analogy verification (51 analogies) |
| `subsumption_benchmark.py` | Taxonomic subsumption accuracy |
| `bit_entropy.py` | Per-bit entropy analysis (dead bit detection) |
| `bit_evolution.py` | Longitudinal bit activation tracking across checkpoints |
| `language_quality.py` | Perplexity, Distinct-n, repetition, MAUVE |
| `interpretability_probe.py` | Linear probe on triadic bits vs embeddings |
| `geometric_topology.py` | Domain clustering (token vs sentence aggregation) |
| `engine_comparison.py` | TriadicGPT vs Triadic Engine (5 projection methods) |
| `prime_vs_bitwise.py` | BitwiseValidator equivalence proof + scaling to 1024+ bits |
| `scaling_plots.py` | Publication-quality scaling figures |
| `bits_sweep_plots.py` | Publication-quality bits sweep figures |

```bash
python benchmarks/scripts/scaling_study.py \
  --model checkpoints/torch_run15_strongalign/model_best.pt
```

Results are stored in `benchmarks/results/` (27 JSON files) and `benchmarks/figures/` (12 PNGs).

## Supervised Experiments (Danza Cosmica)

Training TriadicGPT with 158 hand-factorized anchor concepts (63 semantic primitives from *La Danza Cosmica de los Opuestos*) produces the strongest algebraic results. 19 experiments (D-A1 through D-A19) explore supervised training, bootstrap discovery, ternary quantization, iFSQ activation, hybrid bits, and GPT-2 scaling.

| Model | Test Acc | Subsumption | Dead Bits | R3 | Scale |
|-------|----------|-------------|-----------|-----|-------|
| **D-A14 v2 tanh** | **93.0%** | **98.3%** | 26/63 | 90.7% | 40M |
| D-A16 iFSQ+v2 | 93.2% | 98.3% | -- | 84.2% | 40M |
| D-A18 Unified | 92.2% | 96.5% | 15/63 | 75.3% | 40M |
| **D-A19 GPT-2 355M** | **97.1%** | **76.9%** | 16/63 | 100% | 355M |

```bash
# Train supervised model (158 anchors)
python playground/danza_63bit.py --scale xl --steps 50000 --v2 --dtype bfloat16

# GPT-2 355M with restored algebra
python playground/gpt2_355m_sparsity.py --steps 50000
```

Data: `playground/danza_data/` (primitivos.json with 63 primitives, anclas.json with 50 anchors, anclas_v2.json with 100+ additional anchors).

### Audit Tests

8 formal falsifiable tests in `playground/audit_tests/` validating book predictions and model integrity:

| Test | What It Validates |
|------|-------------------|
| **F0** Data Validation | Pre-flight: anchor coverage, bit accuracy, gold/model agreement |
| **F2.1** Indifference | True opposite of love = indifference (not hate); false opposites detection |
| **F2.2** Aristotelian Types | 4 opposition types (contraries, contradictories, privatives, relatives) produce distinct patterns |
| **F2.5** Enantiodromia | Extremes are closer to their opposite than moderate versions |
| **F3.1** PF Bridge | 5 falsifiable predictions: Hamming~similarity, GCD=1~opposites, dual exclusion, category coverage, observer bits |
| **F3.4** Blind Assignment | Cherry-picking audit: original prime assignment vs random/frequency/semantic alternatives |
| **F4.4** 355M Eval | Formal evaluation of GPT-2 355M scale-up (bit accuracy, subsumption, ternary distribution) |
| **Reptimeline** | BitDiscovery analysis on v2 and hybrid models (duals, dependencies, 3-way interactions) |

```bash
python playground/audit_tests/run_all.py              # All 7 tests
python playground/audit_tests/run_all.py --test f3.1   # Single test
```

Results stored in `playground/audit_tests/results/` (12 JSON files).

## Transfer Learning (Experiment 10)

Attaching the triadic head to pre-trained GPT-2 Small (124M) with two-phase training (frozen backbone, then partial unfreeze) reveals a loss-embedding interaction:

| Alignment Loss | Semantic Gap | Analogy | Best For |
|----------------|-------------|---------|----------|
| MSE | +0.011 | 75.0% | From-scratch (weak embeddings) |
| Rank | +0.047 | **83.3%** | Analogy tasks |
| **InfoNCE** | **+0.076** | 100% | Pre-trained models (rich embeddings) |

Self-contained code in `experiment10/src/` (model.py, train.py, evaluate.py) with separate checkpoints for each alignment mode.

## reptimeline

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19208628.svg)](https://doi.org/10.5281/zenodo.19208628)

A discovery module that tracks how discrete representations evolve during training. Backend-agnostic core with a triadic-specific extractor. Standalone repo: [github.com/arturoornelasb/reptimeline](https://github.com/arturoornelasb/reptimeline).

**Capabilities:**
- **Timeline tracking**: bit births, deaths, connections, phase transitions (exploration, consolidation, crystallization, stabilization)
- **Bottom-up discovery**: bit semantics, dual pairs (anti-correlated opposites), dependency hierarchies, 3-way AND-gate interactions
- **Auto-labeling**: three strategies (embedding centroid, contrastive, LLM)
- **Reconciliation**: compare discovered ontology vs manual theory, suggest corrections for both
- **Visualization**: swimlane grids, phase dashboards, churn heatmaps, layer emergence plots

```bash
python -m reptimeline \
  --checkpoint-dir checkpoints/danza_63bit_xl_v2/ \
  --primitives --overlay --plot
```

Results in `reptimeline/results/` (timeline JSON, discovery reports, autolabel output).

## triadic-head (Standalone Package)

A drop-in triadic projection head for any HuggingFace transformer. BUSL-1.1 licensed. Published on [PyPI](https://pypi.org/project/triadic-head/) (v0.1.0).

```python
from triadic_head import TriadicWrapper

model = TriadicWrapper("gpt2", n_bits=64, align_mode="infonce")
model.freeze_backbone()

# Forward + loss
logits, proj, lang_loss = model(input_ids, labels=labels)
tri_loss = model.triadic_loss(proj, input_ids=input_ids)

# Encode, compare, validate, explore
sigs = model.encode(["king", "queen", "prince"])
comparison = model.compare("king", "queen")
report = model.validate()
model.explore(["king", "queen", "dog", "cat"], show_factors=True)
```

The algebra module (`triadic_head.algebra`) is pure Python with zero dependencies — `PrimeMapper` and `TriadicValidator` work without PyTorch.

Supports: GPT-2, LLaMA, Mistral, Phi, Qwen, GPT-Neo, OPT, Falcon. See [triadic-head/README.md](triadic-head/README.md).

## Conceptual Tokenizer (Experimental)

A meaning-based tokenization system that decomposes language into 49 semantic primitives (7 categories x 7 primitives) instead of BPE frequency tokens. Based on the *Sistema 7x7* framework. Each word maps to 49 activation values with three states: [+] active, [0] negated, [null] irrelevant.

Modules: config, primitives, states, seed_lexicon (462 curated words), prime_encoder, triadic_bridge. Post-hoc projection failed (Phase 4: overfit), but end-to-end training works — `playground/concept_gpt_49bit.py` achieves **86.2% classification accuracy** at XL scale.

## Key Checkpoints

| Checkpoint | Description | Size |
|------------|-------------|------|
| `checkpoints/torch_run15_strongalign/` | **Production** — Run 15, 40M, PPL 7.69, gap +0.020 | 4.7 GB |
| `checkpoints/danza_63bit_xl_v2/` | Best supervised — D-A14, 93% test, 98.3% sub | 3.2 GB |
| `checkpoints/danza_gpt2_355m_sparsity_v2/` | GPT-2 355M — D-A19, 97.1% bits, 76.9% sub | 8.0 GB |
| `checkpoints/concept_gpt_49bit_xl/` | Concept GPT — 49-bit Sistema, 86.2% accuracy | 3.3 GB |
| `experiment10/checkpoints_infonce/` | GPT-2 transfer — InfoNCE, gap +0.076 | 950 MB |
| `checkpoints/chat_run8/` | Chat fine-tuned on Alpaca | 61 MB |

48 checkpoint directories total (~235 GB) spanning 29 training runs, scale sweeps, bits sweeps, and ablations.

## Repository Structure

```
src/                               # Core source (26 modules)
  torch_transformer.py             # TriadicGPT model (nn.Module) — core architecture
  torch_train.py                   # GPU pretraining (dual-loss: language + triadic)
  torch_finetune.py                # Instruction fine-tuning on Alpaca
  triadic.py                       # PrimeMapper, BitwiseValidator, 8 algebraic operations
  evaluate.py                      # Perplexity, generation, triadic analysis, loss curves
  fast_tokenizer.py                # HuggingFace BPE tokenizer (Rust backend)
  chat.py                          # Interactive chat REPL
  auditor.py                       # Relational bias audit (FPR measurement)
  evolution_hook.py                # Per-checkpoint triadic snapshots during training
  pre_tokenize.py                  # Corpus -> .npy token cache (eliminates startup bottleneck)
  concept_trainer.py               # Head-only semantic alignment fine-tuning
  evaluation_script.py             # Automated diagnostic suite (collapse detection, alignment)
  test_generalization.py           # Zero-shot held-out generalization test
  graph_builder.py                 # Inverted prime index for O(N*k) neighbor search
  apply_pca.py                     # PCA initialization for triadic head weights
  autograd.py                      # Pure-Python autograd engine (educational)
  transformer.py                   # Pure-Python GPT (educational)
  train.py                         # CPU training loop (educational, pure-Python path)
  fast_transformer.py              # NumPy-based GPT (intermediate path, functional)
  tensor_ops.py                    # NumPy forward/backward primitives (Adam, attention, etc.)
  fast_train.py                    # NumPy training loop (faster than autograd, still legacy)
  pretrain.py                      # Legacy pretraining pipeline (NumPy path)
  finetune.py                      # Legacy Alpaca fine-tuning (NumPy path)
  inference.py                     # Legacy generation and triadic explanation
  tokenizer.py                     # Pure-Python BPE tokenizer (zero external deps)

ui/                                # PySide6 desktop application
  app.py                           # Entry point
  main_window.py                   # QMainWindow + 7-tab interface
  model_interface.py               # Unified API (native, GPT-2 transfer, HuggingFace)
  model_panel.py                   # Top bar: 3 backends, tokenizer selector, progress
  tabs/                            # encoder, compare, explore, analogy, validate, chat, benchmarks
  widgets/                         # BitVectorWidget, PrimeDisplayWidget, PrimeInspector, MplCanvas
  workers/                         # Async QThread workers for inference
  resources/style.qss              # Dark theme (Catppuccin Mocha)

triadic-head/                      # PyPI package: pip install triadic-head (v0.1.0, BUSL-1.1)
  triadic_head/algebra.py          # PrimeMapper, TriadicValidator (pure Python, zero deps)
  triadic_head/wrapper.py          # TriadicWrapper for HuggingFace models
  examples/train_gpt2.py           # Full training pipeline example
  tests/                           # 33 unit tests
  pyproject.toml                   # Package metadata

reptimeline/                       # Representation timeline discovery module
  core.py, tracker.py              # Timeline tracking (births, deaths, phase transitions)
  discovery.py                     # Bit semantics, duals, dependencies, 3-way interactions
  autolabel.py                     # 3 auto-labeling strategies (embedding, contrastive, LLM)
  reconcile.py                     # Compare discovered vs manual ontology, suggest corrections
  extractors/                      # Backend-specific (triadic extractor, extensible ABC)
  overlays/                        # Domain-specific analysis (primitive overlay)
  viz/                             # Swimlane, phase dashboard, churn heatmap, layer emergence
  tests/                           # 3 test files (smoke, discovery, reconcile)
  results/                         # Discovery JSON outputs

benchmarks/
  scripts/                         # 12 evaluation scripts
  results/                         # 27 JSON benchmark results
  figures/                         # 12 publication-quality PNGs

playground/                        # 50+ experimental scripts
  danza_63bit.py                   # Supervised training (63 primitives, 158 anchors)
  danza_bootstrap.py               # Bootstrap discovery loop (train -> discover -> retrain)
  danza_ternary.py                 # Ternary {-1, 0, +1} quantization (BitNet-style)
  danza_posthoc_analysis.py        # Post-hoc analysis of self-supervised bits
  gpt2_355m_sparsity.py            # GPT-2 355M with restored algebra (D-A19)
  gpt2_medium_ternary.py           # GPT-2 Medium ternary experiments
  concept_gpt_49bit.py             # 49-bit Sistema 7x7 end-to-end training
  hybrid_adversarial.py            # Supervised + free bits with adversarial disentanglement
  unified_final.py                 # Combined best components (D-A18)
  multi_seed_validation.py         # E1: 3-seed confidence intervals
  alignment_ablation.py            # E2: alignment loss driver analysis
  expanded_analogy_benchmark.py    # E3: 51-analogy benchmark
  scale_interpolation.py           # E5: gradual crossover at ~20M
  compression_benchmark.py         # E6: compression ratio analysis
  r3_low_k.py                      # E7: R3 at k=6-12
  embedding_gap_baseline.py        # B1: embedding vs triadic gap
  subsumption_loss.py              # P6: subsumption loss at base scale
  xl_subsumption.py                # P12: subsumption at XL (needs early stopping)
  sin_head_experiment.py           # P1: sinusoidal triadic head
  random_baseline.py               # P1: rigorous random control
  soft_signatures.py               # P1: sigmoid vs hard threshold
  sub_weight_sweep.py              # E4: subsumption weight tradeoff
  cross_dataset_eval.py            # P13: WikiText-2 + LAMBADA evaluation
  eval_d_a18.py, eval_d_a19.py     # Formal evaluation scripts
  run_reptimeline_d_a18.py         # BitDiscovery on D-A18
  audit_tests/                     # 8 formal falsifiable tests + orchestrator + 12 result JSONs
  danza_data/                      # primitivos.json (63), anclas.json (50), anclas_v2.json (100+)
  results/                         # 74 JSON result files from all E/P/B experiments
  REPRODUCIBILITY.md               # Step-by-step reproduction guide (~46h GPU total)

conceptual_tokenizer/              # Experimental 49-primitive semantic tokenizer
  config.py                        # 49 primes (2..227), 7 categories, state thresholds
  primitives.py                    # ConceptToken, PrimitiveActivation, State enum
  states.py                        # StateResolver: projection -> discrete [+]/[0]/[null]
  prime_encoder.py                 # Activations -> composite prime signatures
  triadic_bridge.py                # Bridge to src/triadic.py TriadicValidator
  seed_lexicon.py                  # 462 curated words with hand-factorized decompositions
  training/                        # Phase 4 supervised training (negative result)
  CONTEXT.md                       # Quickstart and phase status
  los_tres_reinos.md               # Theoretical framework (Four Realms)

experiment10/                      # GPT-2 transfer learning (self-contained)
  src/model.py                     # GPT2TriadicModel (freeze/unfreeze, 3 alignment modes)
  src/train.py                     # Two-phase training (frozen -> partial unfreeze)
  src/evaluate.py                  # Evaluation vs baselines (111 concepts, 10 categories)
  results/                         # experiment10_results.json

paper/                             # Academic publication
  triadic_microgpt.tex             # LaTeX source (27 pages, 11 experiments)
  triadic_microgpt.pdf             # Compiled PDF
  figures/                         # 12 embedded figures

research/                          # Design documents (9 files)
  algebraic_operations.md          # Formal spec of 8 ops with proofs + Boolean lattice theory
  relational_prime_chains.md       # Proposed: relational primes for anti-hallucination (O(1) fact verification)
  bitnet_b158_analysis.md          # BitNet b1.58 parallels (~42% sparsity convergence)
  convergence_trits_bitnet_bitwise.md  # Three independent paths → same {+1, 0, -1} representation
  nsm_mapping.md                   # 55% overlap with Wierzbicka's cross-linguistic semantic primes
  related_work_survey.md           # 20 papers: FSQ, CB-LLMs, Wang et al., position statement
  unified_model_architecture.md    # Proposed architecture combining all validated components
  experiment_roadmap.md            # D-A8–D-A16 roadmap with paper impact matrix + GPU budget
  trits_vs_bits_63.md              # Decision: keep 63 primitives (ternary handles dimensionality)

tests/test_all.py                  # 42 unit tests
  TestAutograd (15)                # Forward, backward, activations, gradients
  TestTransformer (8)              # Shapes, softmax, RMSNorm, triadic projection
  TestTriadic (12)                 # Primes, PrimeMapper, TriadicValidator, R3
  TestIntegration (2)              # Training smoke test, projection consistency

scripts/                           # Utilities (7 scripts)
  run_bits_sweep.py                # Orchestrate k={8,16,32,48,64,128} training runs
  generate_gold_primes.py          # WordNet -> deterministic prime signatures (10K concepts)
  build_vocab.py                   # Extract top 10K words from TinyStories
  recover_tokenizer.py             # Regenerate Run 15 tokenizer (vocab 4096, seed 42)
  _check_vocab.py                  # Verify embedding matrix shapes
  _find_tokenizer.py               # Find compatible tokenizer for checkpoint
  _smoke_test.py                   # Quick model sanity check

data/                              # Datasets (gitignored except concepts.txt)
  concepts.txt                     # 31 semantic categories for debugging

reports/                           # Evaluation outputs
  eval_report.json                 # Run 15 full evaluation (PPL, samples, triadic analysis)
  bias_audit_results.json          # Experiment 8 (98.5% accuracy, 0.96% FPR)
  loss_curve.png                   # Training loss visualization

archive/                           # 7 archived docs (evolution plan, audit tables, reconciliation)
experiments/                       # Speculative experiments (quaternion_probe.py)

EXPERIMENT_REFERENCE.md            # Master consolidated reference (1,271 lines, 15 sections, 60+ experiments)
experiment_log.md                  # Detailed raw logs for all 29 training runs (194 KB, deprecated data store)
ROADMAP.md                         # Future improvements and cleanup tasks
COMMERCIAL.md                      # Commercial licensing terms (participation model)
TERMS.md                           # Contribution terms
environment.yml                    # Conda environment specification
requirements.txt                   # pip dependencies
```

## Reproducing Results

See [`playground/REPRODUCIBILITY.md`](playground/REPRODUCIBILITY.md) for step-by-step instructions covering all phases: pre-training (Run 15), playground experiments (P0-P15), validation (E1-E7), and Danza (D-A1 through D-A19). Total GPU time for full reproduction: ~46 hours on RTX 5060 Ti.

### Bits Sweep (Runs 22-26)

```bash
python scripts/run_bits_sweep.py   # k={8,16,32,48,64,128}, ~6 hours
```

### Engine Comparison (Table 7)

Requires [Triadic-Neurosymbolic-Engine](https://github.com/arturoornelasb/Triadic-Neurosymbolic-Engine) cloned alongside this repo:

```bash
python benchmarks/scripts/engine_comparison.py \
  --model checkpoints/torch_run15_strongalign/model_best.pt \
  --engine-path ../Triadic-Neurosymbolic-Engine
```

## Experiment Documentation

- **[EXPERIMENT_REFERENCE.md](EXPERIMENT_REFERENCE.md)**: Master consolidated reference — all 11 experiments organized by research line, results tables, key findings, reviewer FAQ.
- **[experiment_log.md](experiment_log.md)**: Detailed raw logs for all 29 training runs (deprecated as primary reference, preserved as data store).
- **[playground/REPRODUCIBILITY.md](playground/REPRODUCIBILITY.md)**: Step-by-step reproduction guide with exact commands and expected metrics.
- **[research/](research/)**: 9 design documents — algebraic operation specs, NSM convergence analysis, BitNet parallels, relational prime chains proposal, and architecture decisions.

## Paper

> **End-to-End Prime Factorization in a Generative Language Model: Emergent Algebraic Semantics from Joint Training**
> J. Arturo Ornelas Brand, 2026

27-page paper covering 11 experiments across 29 training runs: scaling study, bits sweep, transfer learning, subsumption recovery, compositionality, ternary representations, and domain separation.

```bash
cd paper && pdflatex triadic_microgpt.tex && pdflatex triadic_microgpt.tex
```

### Cite

**Paper:**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19206545.svg)](https://doi.org/10.5281/zenodo.19206545)

Ornelas Brand, J. A. (2026). *End-to-End Prime Factorization in a Generative Language Model: Emergent Algebraic Semantics from Joint Training*. Zenodo. https://doi.org/10.5281/zenodo.19206545

**Repository:**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19207845.svg)](https://doi.org/10.5281/zenodo.19207845)

Ornelas Brand, J. A. (2026). *End-to-End Prime Factorization in a Generative Language Model: Emergent Algebraic Semantics from Joint Training (triadic-microgpt)* (Repository) (0.1.0). Zenodo. https://doi.org/10.5281/zenodo.19207845

### Companion Repos

**Triadic Neurosymbolic Engine** (parent library):

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19205805.svg)](https://doi.org/10.5281/zenodo.19205805) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18748671.svg)](https://doi.org/10.5281/zenodo.18748671)

Ornelas Brand, J. A. (2026). *Triadic Neurosymbolic Engine: Prime Factorization as a Neurosymbolic Bridge: Projecting Continuous Embeddings into Discrete Algebraic Space for Deterministic Verification*. Zenodo. https://doi.org/10.5281/zenodo.19205805

**reptimeline** (training dynamics):

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19208672.svg)](https://doi.org/10.5281/zenodo.19208672) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19208628.svg)](https://doi.org/10.5281/zenodo.19208628)

Ornelas Brand, J. A. (2026). *reptimeline: Tracking Discrete Representation Evolution During Neural Network Training*. Zenodo. https://doi.org/10.5281/zenodo.19208672

**Triadic Emergent Duality** (ontological framework):

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19374914.svg)](https://doi.org/10.5281/zenodo.19374914)

Ornelas Brand, J. A. (2026). *Triadic Emergent Duality: 14+ Candidate Dualities Across 6 Algebraic Layers*. Zenodo. https://doi.org/10.5281/zenodo.19374914

## Credits

- **Initial scaffolding**: Inspired by [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)
- **Triadic algebra**: Based on the [Triadic-Neurosymbolic-Engine](https://github.com/arturoornelasb/Triadic-Neurosymbolic-Engine) (Ornelas Brand, 2026)

## License

**Business Source License 1.1 (BUSL-1.1)** — see [LICENSE](./LICENSE), [TERMS.md](./TERMS.md), and [COMMERCIAL.md](./COMMERCIAL.md).

Individuals, academics, and non-profits: free. Companies: participation agreement required.
**Change Date:** 2030-03-22 — auto-converts to AGPL-3.0.

Copyright J. Arturo Ornelas Brand, 2026

Contact: arturoornelas62@gmail.com
