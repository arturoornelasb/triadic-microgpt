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

## Current State (2026-03-10 — 11 Experiments, 29 Runs, Paper Ready)
- **Production Model**: Run 15 (v1.4-strongalign), 40M params, loss 0.946, entropy 0.749
- **Checkpoint**: `checkpoints/torch_run15_strongalign/model_L12_D512_B64_best.pt`
- **Tokenizer**: `checkpoints/torch_run15_strongalign/tokenizer.json` (different from `checkpoints/torch/`)
- **Language Cost**: Zero (PPL 7.69 vs 7.56 ablation, within noise)
- **Semantic Gap**: +0.020 from-scratch; +0.099 GPT-2 transfer (InfoNCE, closes 72% to Engine PCA)
- **Domain Separation**: 1.21 mean (sentence-level aggregation; was ~1.02 token-level) — Experiment 11
- **Paper**: 16 pages compiled (`paper/triadic_microgpt.pdf`), all 11 experiments included
- **PyPI**: `triadic-head` v0.1.0 built & validated (signal +8.5% above random), not yet published

## Known Issues (MUST READ)
1. **Knowledge Distillation at 5× weight = collapse**: GoldPrimes model (Run 10) has complete triadic collapse. Made configurable (`--dist-weight`, default 1.0). Use Run 15 as production model.
2. **Dead Bits**: ~15 of 64 bits have low entropy (< 0.3). Entropy regularization mitigates but doesn't eliminate.
3. **Tokenizer Compatibility**: Runs 1-6 use Python BPE; runs 7+ use HuggingFace. NOT interchangeable. Run 15's tokenizer differs from `checkpoints/torch/tokenizer.json` — always use the one in the same checkpoint directory.
4. **Subsumption at k=64**: RESOLVED via subsumption loss (Exp P12). 100% held-out at 25K steps with sub_weight=5.0. PPL cost +47% at XL scale (early stopping required). Base scale is "free lunch" (language improves).
5. **Coherence loss = collapse**: NEVER re-enable. Adjacent-token agreement drives all projections to identical.

## GPU Optimization Standard (RTX 5060 Ti — Blackwell)

**All training scripts MUST use bfloat16 mixed precision.** This is not optional.

```python
# CORRECT — bfloat16 on Blackwell Tensor Cores (2-8x faster than float32)
amp_dtype = torch.bfloat16
use_scaler = False  # bfloat16 has 8-bit exponent like float32, no underflow risk
scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=(device.type == 'cuda')):
    logits, proj, loss = model(x, targets=y)

# WRONG — float16 wastes Tensor Core throughput, needs GradScaler overhead
with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):  # defaults to float16!
    ...
```

**Why bfloat16 over float16:**
- Blackwell Tensor Cores execute bfloat16 matmuls at peak throughput (4th gen)
- Same exponent range as float32 (8 bits) → no gradient underflow → no GradScaler needed
- GradScaler adds CPU-GPU sync overhead on every step — removing it is free speedup

**Checklist for new training scripts:**
1. Add `--dtype` arg with `default='bfloat16'`
2. Map to `torch.bfloat16` via dict lookup
3. Pass `dtype=amp_dtype` to `torch.amp.autocast()`
4. Disable `GradScaler` when dtype != float16
5. Print precision in training log header for verification
6. Add `torch.set_float32_matmul_precision('high')` — uses TF32 for residual float32 ops
7. Add `torch.backends.cudnn.benchmark = True` — autotuning for kernel selection
8. Add `torch.compile(model)` — fuses CUDA kernels (10-30% speedup). **Requires Triton (Linux only).** On Windows, guard with `try: import triton` and skip gracefully
9. Add `--no-compile` flag for debugging, `--grad-checkpoint` for VRAM savings

```python
# Full Blackwell optimization boilerplate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
amp_dtype = {'float32': torch.float32, 'float16': torch.float16,
             'bfloat16': torch.bfloat16}[args.dtype]
use_scaler = (device.type == 'cuda' and amp_dtype == torch.float16)

if device.type == 'cuda':
    torch.set_float32_matmul_precision('high')   # TF32 for residual ops
    torch.backends.cudnn.benchmark = True         # kernel autotuning

model = TriadicGPT(config).to(device)
if args.grad_checkpoint:
    model.gradient_checkpointing_enable()         # trade compute for VRAM
if device.type == 'cuda' and not args.no_compile:
    model = torch.compile(model)                  # fuse kernels (10-30% faster)

scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

# Training loop
with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=(device.type == 'cuda')):
    logits, proj, loss = model(x, targets=y)
```

**Conda environment:** Must use `triadic-microgpt` (PyTorch 2.12+cu128), NOT `base` (CPU-only).

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
