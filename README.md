# TriadicGPT

**End-to-end prime factorization in a generative language model.**

TriadicGPT is a 40M-parameter GPT augmented with a *triadic projection head* that produces discrete prime-factor signatures alongside standard next-token predictions. Unlike post-hoc approaches that project frozen embeddings into prime space, TriadicGPT learns algebraically verifiable semantic representations end-to-end as a side effect of language modeling.

```
King  = 2 x 3 x 5        (Royalty x Male x Authority)
Queen = 2 x 5 x 7        (Royalty x Authority x Female)

Shared:     gcd  -> {2, 5}    Royalty, Authority
Difference: div  -> {3} vs {7}   Male vs Female
Analogy:    factor transfer   king:queen :: man:woman
```

## Key Results

| Finding | Result |
|---------|--------|
| Language cost of triadic head | **Zero** (PPL 7.69 vs 7.56 ablation, within noise) |
| Semantic ordering emergence | Phase transition at **40M params** (gap: -0.076 -> +0.020) |
| Optimal bit width | **k=32-64** (shifted from k=6-12 post-hoc) |
| Analogy verification | **69.2%** (random baseline 50%) |
| Semantic compression | **8x** (64 bits = 512D embedding probe accuracy) |
| Signature uniqueness | **100%** across all evaluated concepts |
| GPT-2 transfer (InfoNCE) | Gap **+0.099**, closing **72%** of gap to Engine PCA (+0.136) |
| Domain separation (sentence-level) | **1.21** mean across 12 domains (+19% vs token-level) |

## Phase 5: Transfer Learning & Alignment Loss Ablation (Experiment 10)

Attaching the triadic head to pre-trained GPT-2 Small (124M) and testing three alignment loss
formulations reveals a **loss-embedding interaction**: the optimal loss depends on embedding quality.

| Alignment Loss | Semantic Gap | Analogy Verif | Notes |
|----------------|-------------|---------------|-------|
| MSE | +0.011 | 75.0% | Weak — absolute value matching wastes capacity |
| Rank | +0.047 | **83.3%** | Best analogies; ordering-based |
| **InfoNCE** | **+0.099** | 66.7% | **Closes 72% of gap to Engine PCA** |
| From-scratch (MSE) | +0.020 | 66.7% | Baseline |
| Engine PCA | +0.136 | 91.7% | Upper bound (post-hoc, MiniLM) |

**Key finding — Loss-Embedding Interaction:**
- **Rich embeddings** (GPT-2, 768D, WebText): Use **InfoNCE** — structured pos/neg mining leverages
  tight semantic clusters. MSE's absolute matching wastes capacity on scale mismatch.
- **Weak embeddings** (from-scratch, 512D, TinyStories): Use **MSE** — dense local gradients
  work even without global cluster structure. InfoNCE and Rank fail (no meaningful negatives).

The bottleneck is the **loss formulation**, not embedding quality: same model, same embeddings,
9× gap difference from changing only the alignment loss (MSE→InfoNCE).

## Experiment 11: Sentence-Level Domain Separation

Token-level projections (isolated words) show separation ratio ~1.02 — domains are indistinguishable.
Sentence-level aggregation (mean-pool triadic projections across 3 contextual sentences per concept)
reveals that the model **does** encode domain structure:

| Domain | Token | Sentence | Delta |
|--------|-------|----------|-------|
| family | 1.03 | **1.42** | +38% |
| colors | 1.05 | **1.25** | +19% |
| royalty | 1.04 | **1.24** | +19% |
| food | 1.01 | **1.23** | +22% |
| emotions | 1.00 | **1.11** | +11% |
| **Mean** | **1.02** | **1.21** | **+19%** |

```bash
python benchmarks/scripts/geometric_topology.py \
  --model checkpoints/torch_run15_strongalign/model_L12_D512_B64_best.pt \
  --aggregate sentence --version v6.0-sentence
```

## Architecture

```
Text -> BPE Tokenizer (4096 vocab) -> Token IDs -> TriadicGPT (12L/512D/8H)
                                                          |
                                                    +-----+-----+
                                                    |           |
                                               LM Head    Triadic Head
                                                    |           |
                                              Next-Token   tanh(Wx) -> bits
                                              Prediction   [+, -, +, +, ...]
                                                    |           |
                                              L_lang       PrimeMapper
                                              (CE)         Phi = 2 x 5 x 7
                                                    |           |
                                                    +-----+-----+
                                                          |
                                              L = L_lang + alpha * L_triadic
```

The triadic loss has four components: diversity (bits fire ~50%), contrastive (sequences differ), entropy (no dead bits), and **embedding alignment** (triadic similarity matches embedding similarity -- the key innovation).

## Quick Start

### Environment

```bash
conda create -n triadic-microgpt python=3.10
conda activate triadic-microgpt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install tokenizers numpy matplotlib
```

### Train

```bash
# Train XL model (40M params, ~76 min on RTX 5060 Ti)
python src/torch_train.py --scale xl --steps 50000

# Train with custom triadic bits
python src/torch_train.py --scale xl --override-bits 32 --steps 50000
```

### Evaluate

```bash
# Full evaluation (perplexity, generation, triadic analysis)
python src/evaluate.py --model checkpoints/torch/model_best.pt

# Run benchmark suite
python benchmarks/scripts/scaling_study.py --model checkpoints/torch_run15_strongalign/model_best.pt
```

### Chat

```bash
python src/chat.py
```

### Desktop Explorer (GUI)

A full desktop UI for exploring, auditing, and chatting with TriadicGPT interactively. Supports three backends: native `.pt` checkpoints, GPT-2 Transfer (Experiment 10), and any HuggingFace model.

```bash
pip install PySide6
python ui/app.py
```

7 tabs: Encoder, Compare (with clickable prime chips + Prime Inspector), Explore (heatmap), Analogy, Validate, Chat, and Benchmarks. See **[Desktop Explorer docs →](ui/README.md)** for the full feature reference.

## Repository Structure

```
src/
  torch_transformer.py     # TriadicGPT model (nn.Module) -- core architecture
  torch_train.py           # GPU training (dual-loss: language + triadic)
  torch_finetune.py        # Chat fine-tuning on Alpaca
  triadic.py               # PrimeMapper, TriadicValidator, algebraic operations
  evaluate.py              # Perplexity, generation, triadic analysis
  auditor.py               # Relational bias audit (FPR measurement)
  fast_tokenizer.py        # HuggingFace BPE tokenizer (Rust backend)
  chat.py                  # Interactive chat REPL
  autograd.py              # Pure-Python autograd (educational)
  transformer.py           # Pure-Python GPT (educational)

benchmarks/
  scripts/
    scaling_study.py       # 4-point scaling study benchmark
    scaling_plots.py       # Publication-quality scaling figures
    bits_sweep_plots.py    # Bits sweep figures
    engine_comparison.py   # TriadicGPT vs Engine (5-method Table 7)
    bit_entropy.py         # Per-bit entropy analysis
    analogy_benchmark.py   # Analogy verification benchmark
    subsumption_benchmark.py
    interpretability_probe.py
    language_quality.py
    geometric_topology.py  # Domain clustering (--aggregate token|sentence)
  results/                 # All benchmark JSON results
  figures/                 # Generated figures

paper/
  triadic_microgpt.tex     # Full paper (LaTeX)
  figures/                 # Paper figures

scripts/
  run_bits_sweep.py        # Orchestrator for k={8,16,32,48,64,128} sweep
  generate_gold_primes.py  # WordNet gold prime dictionary generator
  build_vocab.py           # BPE vocabulary builder

tests/
  test_all.py              # 37 unit tests

triadic-head/              # Standalone PyPI package (triadic algebra + HF wrapper)
  triadic_head/            # triadic_head.TriadicHead, TriadicWrapper, algebra
  tests/                   # 33 unit tests

ui/
  app.py                   # Entry point: python ui/app.py
  model_interface.py       # Unified API for native .pt, GPT-2 Transfer, and HF TriadicWrapper
  model_panel.py           # Top bar: 3 backends (native, GPT-2 transfer, HuggingFace)
  main_window.py           # QMainWindow + 7-tab interface
  tabs/                    # Encoder, Compare, Explore, Analogy, Validate, Chat, Benchmarks
  widgets/                 # BitVectorWidget, PrimeDisplayWidget, PrimeInspectorDialog, MplCanvas
  workers/                 # Async QThread workers for inference
  resources/style.qss      # Dark theme (Catppuccin Mocha)

experiment_log.md          # Complete record of all 29 runs + 11 experiments
EVOLUTION_PLAN.md          # Research roadmap and phase tracking
```

## Reproducing Results

### Data

Download [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) and place `TinyStories-train.txt` in `data/`.

### Training Runs

The paper's production model is **Run 15** (`v1.4-strongalign`):

```bash
python src/torch_train.py \
  --scale xl \
  --steps 50000 \
  --alpha 0.05 \
  --entropy-weight 1.0 \
  --align-weight 5.0 \
  --triadic-warmup-pct 0.25 \
  --no-distill \
  --checkpoint-dir checkpoints/torch_run15_strongalign
```

### Bits Sweep (Runs 22-26)

```bash
python scripts/run_bits_sweep.py
```

### Experiment 9 (Table 7: Full Comparison)

Requires [Triadic-Neurosymbolic-Engine](https://github.com/arturoornelasb/Triadic-Neurosymbolic-Engine) cloned alongside this repo:

```bash
python benchmarks/scripts/engine_comparison.py \
  --model checkpoints/torch_run15_strongalign/model_best.pt \
  --engine-path ../Triadic-Neurosymbolic-Engine
```

## Paper

> **End-to-End Prime Factorization in a Generative Language Model: Emergent Algebraic Semantics from Joint Training**
> Arturo Ornelas Brand, 2026

The full paper is in `paper/triadic_microgpt.tex`. Compile with:

```bash
cd paper && pdflatex triadic_microgpt.tex && pdflatex triadic_microgpt.tex
```

## Credits

- **Initial scaffolding**: Inspired by [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)
- **Triadic algebra**: Based on the [Triadic-Neurosymbolic-Engine](https://github.com/arturoornelasb/Triadic-Neurosymbolic-Engine) (Ornelas Brand, 2026)

## License

MIT
