---
description: How to train, evaluate, discover, and retrain the Triadic MicroGPT model — full bitwise pipeline
---

# Triadic MicroGPT — Training Workflow

## Core Thesis

The triadic head produces **bits** — binary semantic fingerprints verified through O(1) bitwise algebra. The training cycle is:

```
Train → Evaluate → Discover (reptimeline) → Human Corrects → Expand Anchors → Retrain
  50 anchors → 87%     158 anchors → 93%      300+? → ...
```

This loop IS the paper's evidence: bits scale knowledge through human-in-the-loop discovery.

## Environment

```powershell
conda activate triadic-microgpt
# Python 3.10 | PyTorch 2.12+ (CUDA 12.8) | HuggingFace tokenizers | numpy | matplotlib
# GPU: RTX 5060 Ti 16GB (Blackwell, bfloat16 mandatory)
```

## Architecture Overview

```
Text → FastBPETokenizer → Token IDs → TriadicGPT → Two Heads:
  1. LM Head → next-token prediction (cross-entropy)
  2. Triadic Head → iFSQ(Linear(h)) → 63 bits → BitwiseValidator (O(1))
       - subsumes:   (A & B) == B
       - compose:    A | B
       - analogy:    (C & ~only_a) | only_b
       - gap:        A & ~B, B & ~A
       - similarity: popcount(A & B) / popcount(A | B)
```

Total loss = `L_lang + α × L_triadic`

Triadic loss components (to prevent collapse):
- **Diversity**: bits should be ~50% active across batch
- **Contrastive**: different sequences → different projections
- **Entropy**: per-bit entropy regularization (prevents dead bits)
- **Embedding alignment**: backbone wte → triadic head (THE driver of quality)

**NEVER use coherence loss** — adjacent-token agreement drives all projections to identical. Proven to collapse in every experiment.

## File Map

| File | Purpose |
|------|---------|
| `src/torch_transformer.py` | TriadicGPT model (nn.Module) — core architecture |
| `src/triadic.py` | BitwiseMapper, BitwiseValidator (+ PrimeMapper for paper theory) |
| `src/evaluate.py` | Perplexity, generation, triadic analysis, loss curves |
| `src/torch_train.py` | GPU pretraining (from-scratch, TinyStories) |
| `src/torch_finetune.py` | Chat fine-tuning on Alpaca |
| `src/fast_tokenizer.py` | HuggingFace BPE tokenizer (Rust, 1000× faster) |
| `playground/danza_63bit.py` | Danza 63-bit training (supervised anchors) |
| `playground/unified_final.py` | D-A18: iFSQ + hybrid 30+33 bits + v2 anchors + adversarial |
| `playground/audit_tests/common.py` | Shared evaluation utilities (bitwise) |
| `playground/audit_tests/test_d_a13_eval.py` | Formal model evaluation (--v2 for 158 anchors) |
| `EXPERIMENT_REFERENCE.md` | Master experiment reference (all results) |
| `experiment_log.md` | Detailed data store (raw logs) |

## The Training Cycle

### Step 1: Train

**Danza 63-bit** (supervised anchors, primary path):
```powershell
conda run -n triadic-microgpt python playground/danza_63bit.py `
  --scale xl --steps 50000 --dtype bfloat16 `
  --v2 `
  --checkpoint-dir checkpoints/danza_63bit_xl_vN
```

**Unified model** (iFSQ + hybrid + adversarial, experimental):
```powershell
conda run -n triadic-microgpt python playground/unified_final.py `
  --scale xl --steps 50000 --dtype bfloat16
```

**From-scratch** (no supervised anchors, baseline):
```powershell
conda run -n triadic-microgpt python src/torch_train.py `
  --scale xl --steps 50000 --dtype bfloat16 `
  --checkpoint-dir checkpoints/torch_runN
```

### Step 2: Evaluate

```powershell
# Formal eval with v2 anchors (158)
conda run -n triadic-microgpt python playground/audit_tests/test_d_a13_eval.py --v2

# Standard eval (perplexity, generation, triadic analysis)
conda run -n triadic-microgpt python src/evaluate.py `
  --model checkpoints/danza_63bit_xl_vN/model_best.pt `
  --tokenizer checkpoints/danza_63bit_xl_vN/tokenizer.json
```

Key metrics:
- **Test bit accuracy**: target > 90%
- **Subsumption**: target > 95%
- **Dead bits**: target < 30/63
- **Perplexity**: lower = better (Run 15 baseline: 7.69)

### Step 3: Discover (reptimeline)

```powershell
# Analyze what the bits learned — discover semantics, duals, 3-way interactions
conda run -n triadic-microgpt python playground/danza_63bit.py `
  --checkpoint checkpoints/danza_63bit_xl_vN/model_best.pt --analyze-only
```

reptimeline discovers:
- **Bit semantics**: what each bit encodes (e.g., bit 5 = "vida")
- **Dual detection**: anti-correlated pairs (e.g., bien↔mal)
- **3-way interactions**: compositional structure (bit_i + bit_j → bit_r)
- **Dependency chains**: which bits predict others

### Step 4: Human Corrects

Review reptimeline output. For each discovery:
- Is "bit 23 = vida" correct or noise?
- Are the detected duals real semantic oppositions?
- Do 3-way interactions reflect genuine composition?

Valid discoveries become new anchors in `playground/danza_data/anclas_v3.json`.

### Step 5: Retrain with Expanded Anchors

```powershell
# Train with v3 anchors (300+?)
conda run -n triadic-microgpt python playground/danza_63bit.py `
  --scale xl --steps 50000 --dtype bfloat16 `
  --v2 `  # or --v3 when anchors expand
  --checkpoint-dir checkpoints/danza_63bit_xl_v3
```

This closes the loop. Evidence so far:
- v1 (50 anchors) → 87% test accuracy
- v2 (158 anchors) → 93% test accuracy
- v3 (300+?) → ?

### Step 6: Document

After each run, update `EXPERIMENT_REFERENCE.md` with:
- ID, date, config, key metrics
- What changed from previous cycle
- Discovery results that informed the changes

## Mixed Precision (MANDATORY)

**All training MUST use bfloat16** on the RTX 5060 Ti (Blackwell, 4th gen Tensor Cores).

### Optimization stack

| Optimization | Speedup | What it does |
|---|---|---|
| **bfloat16** | 2-8x vs float32 | Tensor Core peak throughput, no GradScaler needed |
| **torch.compile** | 10-30% | Fuses CUDA kernels (Linux/Triton only) |
| **TF32 matmul** | 5-15% | TensorFloat-32 for residual float32 ops |
| **cudnn.benchmark** | ~5% | Kernel autotuning |
| **Gradient checkpointing** | Saves VRAM | Trade recompute for memory |
| **Flash Attention** | Already active | `F.scaled_dot_product_attention(is_causal=True)` |

### Boilerplate

```python
parser.add_argument('--dtype', default='bfloat16', choices=['float32','float16','bfloat16'])
parser.add_argument('--grad-checkpoint', action='store_true')
parser.add_argument('--no-compile', action='store_true')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
amp_dtype = {'float32': torch.float32, 'float16': torch.float16,
             'bfloat16': torch.bfloat16}[args.dtype]
use_scaler = (device.type == 'cuda' and amp_dtype == torch.float16)
if device.type == 'cuda':
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

model = TriadicGPT(config).to(device)
if args.grad_checkpoint:
    model.gradient_checkpointing_enable()
scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=(device.type == 'cuda')):
    logits, proj, loss = model(x, targets=y)
```

## Key Hyperparameters

| Param | Danza 63-bit | From-scratch | Notes |
|-------|-------------|-------------|-------|
| `lr` | 3e-4 | 3e-4 | |
| `alpha` | 0.05 | 0.15 | Lower for supervised (anchors do the work) |
| `triadic-warmup-pct` | 0.3 | 0.3 | Start triadic loss after 30% of steps |
| `dropout` | 0.1 | 0.1 | |
| `batch-size` | 32 | 32 | |
| `dtype` | bfloat16 | bfloat16 | Always |
| `--v2` | Yes | N/A | Use 158 anchors (default: 50) |

## Bitwise Algebra (Implementation)

PrimeMapper exists for the paper's mathematical theory. BitwiseValidator is the runtime implementation. They are **isomorphic** (1000/1000 tests, proven):

| Operation | PrimeMapper O(n) | BitwiseValidator O(1) | Speedup |
|-----------|-------------------|----------------------|---------|
| Subsumption | `GCD(A,B) == B` | `(A & B) == B` | 1.3x |
| Composition | `LCM(A,B)` | `A \| B` | 5x |
| Similarity | Set intersection | `popcount(A&B)/popcount(A\|B)` | 78x |
| Analogy | GCD + LCM | `(C & ~only_a) \| only_b` | 5x |

**All new code MUST use BitwiseMapper/BitwiseValidator** (aliased as DefaultMapper/DefaultValidator in `src/triadic.py`).

## Production Models

| Model | Params | Accuracy | Subsumption | Checkpoint |
|-------|--------|----------|-------------|------------|
| **D-A14 v2 tanh** | 40M | 93.0% | 98.3% | `checkpoints/danza_63bit_xl_v2/` |
| D-A16 iFSQ+v2 | 40M | 93.2% | R3=0.842 | `checkpoints/danza_ifsq_v2/` |
| Run 15 (from-scratch) | 40M | N/A | PPL 7.69 | `checkpoints/torch_run15_strongalign/` |
| D-A17 GPT-2 355M | 355M | TBD | TBD | `checkpoints/danza_gpt2medium_ternary/` |

## Known Issues

1. **Coherence loss = COLLAPSE**: NEVER re-enable. Drives all projections to identical.
2. **Dead bits**: ~26/63 bits have low entropy (~42% sparsity). This appears structural, not a bug.
3. **Tokenizer compatibility**: Each checkpoint has its OWN tokenizer.json. Always use the one in the same directory.
4. **Bootstrap (D-A6) failed**: Fully automated discover→retrain doesn't work. Human-in-the-loop is required.
5. **iFSQ vs tanh**: iFSQ has better LM loss (0.924 vs 0.946) but slightly lower subsumption. Both valid.
6. **Pareto cliff at alpha > 0.05**: alpha=0.1+ kills semantic ordering. Stay at 0.05.

## Scaling Guidelines

| Size | Layers | Dim | Heads | Params | GPU Time (50K steps) |
|------|--------|-----|-------|--------|---------------------|
| Small | 4 | 128 | 4 | ~1M | ~5 min |
| Base | 6 | 256 | 8 | ~6M | ~10 min |
| Large | 8 | 384 | 8 | ~16M | ~30 min |
| XL | 12 | 512 | 8 | ~40M | ~76 min |
| XXL | 24 | 1024 | 16 | ~307M | ~4h (est.) |
