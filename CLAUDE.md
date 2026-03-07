# Triadic MicroGPT — Agent Configuration

## Project Identity
**Triadic MicroGPT** is a neurosymbolic language model that trains a GPT transformer with an integrated triadic projection head end-to-end. Unlike the parent library (Triadic-Neurosymbolic-Engine) which applies prime factorization post-hoc to frozen embeddings, this model learns to produce semantically meaningful prime signatures jointly with language modeling.

## Environment
```bash
conda activate triadic-microgpt
# Python 3.10 | PyTorch (CUDA 12.8) | HuggingFace tokenizers | numpy | matplotlib
# GPU: RTX 5060 Ti 16GB
```

## Repository Structure
```
src/                       # All source code
  torch_transformer.py     # TriadicGPT model (nn.Module) — THE core architecture
  torch_train.py           # GPU pretraining script (dual-loss: language + triadic)
  torch_finetune.py        # Chat fine-tuning on Alpaca
  triadic.py               # PrimeMapper, TriadicValidator, algebraic operations
  evaluate.py              # Perplexity, generation, triadic analysis, loss curves
  auditor.py               # Experiment 8: Relational Bias Audit (FPR measurement)
  fast_tokenizer.py        # HuggingFace BPE tokenizer (Rust backend)
  chat.py                  # Interactive chat REPL
  autograd.py              # Pure-Python autograd engine (educational, not used in PyTorch path)
  transformer.py           # Pure-Python GPT (educational, not used in PyTorch path)
data/
  TinyStories-train.txt    # Primary training corpus
  alpaca_data_cleaned.json # Fine-tuning data
  gold_primes_64.json      # 10K WordNet concepts with deterministic 64-bit prime signatures
  gold_primes_32.json      # 32-bit version
checkpoints/               # All model checkpoints (organized by run)
  torch/                   # Latest production checkpoints (Run 10/11)
reports/                   # Evaluation outputs (JSON, PNG)
tests/test_all.py          # 37 unit tests (autograd, transformer, triadic, integration)
scripts/                   # Utility scripts (vocab building, gold prime generation)
experiment_log.md          # Historical record of ALL training runs with metrics
benchmarks/                # Industry-standard evaluation suite (NEW)
.claude/                   # Agent guide and workflow documentation
```

## Key Commands
```bash
# Pre-train XL model (40M params, ~76 min)
python src/torch_train.py --scale xl --steps 50000

# Evaluate a checkpoint
python src/evaluate.py --model checkpoints/torch/model_best.pt --tokenizer checkpoints/torch/tokenizer.json

# Relational Bias Audit (Experiment 8)
python src/auditor.py

# Run tests
python tests/test_all.py

# Interactive chat
python src/chat.py
```

## Architecture Overview
```
Text → FastBPETokenizer → Token IDs → TriadicGPT (12L/512D/8H) → Two Heads:
  1. LM Head → next-token prediction (cross-entropy)
  2. Triadic Head → tanh(Wx) → bits → PrimeMapper → Φ(x) = ∏ pᵢ
  Total Loss = L_lang + α · L_triadic [+ α_dist · L_distillation]
```

## Current State (as of Phase 0.5 Diagnostic, 2026-03-06)
- **Best Language Loss**: 1.27 (Run 9, XL pure) / 1.03 (Run 10, GoldPrimes — but triadic collapsed)
- **Bias Audit**: 98.5% accuracy, FPR 0.96% (using distillation model)
- **Triadic Quality (XL pure)**: 97.3% unique signatures, King↔Queen=89%, King↔Dog=60%
- **Language Quality**: Excellent — coherent multi-sentence TinyStories generation

## Known Issues (MUST READ)
1. **Knowledge Distillation Destroys Triadic Head**: GoldPrimes model (Run 10) has COMPLETE triadic collapse (1 unique signature, all similarities=100%). The 5× distillation weight overwhelmed all emergent differentiation. Use XL pure model (Run 9) as base.
2. **Partial Bit Entropy**: XL model has mean entropy 0.381 — ~15 bits are dead (always positive), ~20 bits carry most semantic info. Needs entropy regularization to activate dead bits.
3. **Tokenizer Compatibility**: Runs 1-6 use Python BPE; runs 7+ use HuggingFace. NOT interchangeable.
4. **eval_report.json is MISLEADING**: It was generated from the GoldPrimes model and shows 100% similarity for all pairs. This does NOT represent the XL model's actual capability.

## Coding Conventions
- All PyTorch training code is in `src/torch_*.py`
- Legacy pure-Python code is in `src/autograd.py`, `src/transformer.py`, `src/train.py` (DO NOT modify unless asked)
- CSV training logs go in checkpoint directories
- Evaluation reports go in `reports/`
- Benchmark results go in `benchmarks/`
- Every training run MUST be documented in `experiment_log.md`

## Paper Context
This project implements and extends the research from:
- **Paper**: "Prime Factorization as a Neurosymbolic Bridge" (Ornelas Brand, 2026)
- **Parent Library**: github.com/arturoornelasb/Triadic-Neurosymbolic-Engine
- **Goal**: Demonstrate that end-to-end triadic training produces interpretable, algebraically verifiable semantic representations without post-hoc projection
