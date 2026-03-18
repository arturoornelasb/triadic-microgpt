# BitNet b1.58: Ternary LLMs and Connections to Triadic Representations

**Date**: 2026-03-18
**Context**: Research notes for the Triadic MicroGPT project
**Source**: Microsoft Research — "The Era of 1-bit LLMs" (Ma et al., 2024)

---

## 1. BitNet b1.58 Architecture

### Overview
BitNet b1.58 is a ternary-weight LLM where every parameter in the linear layers is constrained to **{-1, 0, +1}**. The "1.58" refers to log2(3) ≈ 1.58 bits of information per weight (vs 16 bits in fp16).

### Model Specs (2B variant)
| Parameter | Value |
|-----------|-------|
| Layers | 30 |
| Hidden dim | 2560 |
| Attention heads | 20 |
| Parameters | 2.0B |
| Weight values | {-1, 0, +1} only |
| Activations | Full precision (fp16/bf16) |

### Key Insight
Only the **weights** are quantized to ternary. Activations remain in full precision. This is critical — the model maintains rich intermediate representations while using minimal storage for learned parameters.

---

## 2. BitLinear: The Core Mechanism

### Standard Linear Layer
```python
y = x @ W.T + b    # W is fp16, millions of multiply-accumulate ops
```

### BitLinear Replacement
```python
class BitLinear(nn.Linear):
    def forward(self, x):
        # 1. Quantize weights to {-1, 0, +1}
        gamma = W.abs().mean()                          # absmean scale
        W_q = (W / gamma).round().clamp(-1, 1)          # RoundClip

        # 2. Quantize activations to 8-bit (for efficiency)
        alpha = x.abs().max()
        x_q = (x / alpha * 127).round().clamp(-128, 127)

        # 3. Integer matmul + rescale
        y = x_q @ W_q.T * (alpha * gamma / 127)
        return y
```

### Straight-Through Estimator (STE)
The quantization step (`round()`) has zero gradient everywhere. BitNet uses the STE trick:

```python
W_q = W + (quantize(W) - W).detach()
# Forward: uses quantized values
# Backward: gradient flows through as if quantization didn't happen
```

This is the same pattern we already use in the triadic head with `tanh` — the continuous function allows gradients to flow while the output approaches discrete values.

**UPDATE (2026-03-18)**: The iFSQ paper (Tencent, arXiv 2601.17124) discovered that `tanh` causes **activation collapse** — concentrating values near 0, causing inner bins to be over-used and outer bins under-used. Their fix: replace `tanh(x)` with `2*sigmoid(1.6*x) - 1`, which distributes activations more uniformly. This is directly relevant to our dead bits problem. See `research/related_work_survey.md` Section 1.3 for details.

### Absmean Quantization
```
gamma = mean(|W|)
W_scaled = W / gamma
W_q = RoundClip(W_scaled, -1, +1)
```

The division by `gamma` centers the distribution so that `round()` naturally produces a balanced mix of -1, 0, +1. Empirically, **~42.3% of weights become zeros** — the model learns sparsity as a natural consequence of the quantization scheme.

---

## 3. Benchmark Results

### BitNet b1.58 2B vs Full-Precision Models

| Model | Params | PPL (wiki2) | ARC-C | HellaSwag | MMLU |
|-------|--------|-------------|-------|-----------|------|
| Llama 3B | 3.0B | 8.14 | 40.4 | 69.1 | 44.2 |
| Qwen 1.5B | 1.5B | 11.2 | 32.8 | 58.7 | — |
| **BitNet 2B** | **2.0B** | **9.72** | **38.5** | **64.3** | **40.1** |
| Gemma 2B | 2.0B | 10.8 | 36.1 | 61.3 | 38.4 |

Key takeaway: BitNet 2B with ternary weights **matches or beats** full-precision models of similar size. The 1.58 bits/weight is sufficient for competitive language modeling.

### Efficiency Gains
- **Memory**: 5-7× reduction (1.58 bits vs 16 bits per weight)
- **Compute**: Integer add replaces FP multiply — 37× more energy efficient on some hardware
- **CPU inference**: Viable on commodity hardware (no GPU needed for inference)
- **Latency**: Comparable to or better than fp16 at same model size

---

## 4. Native Training vs Post-Hoc Quantization

### Critical Finding
BitNet trains ternary weights **from scratch** (native). This dramatically outperforms post-training quantization (PTQ):

| Approach | Quality |
|----------|---------|
| fp16 → PTQ 2-bit | Significant degradation, especially at small scale |
| fp16 → GPTQ/AWQ 4-bit | Reasonable but still lossy |
| **Native 1.58-bit** | **Matches fp16 quality** |

The model learns to **work with** the constraint, distributing information across more weights and developing robust ternary representations.

### Parallel to Triadic MicroGPT
This exactly mirrors our finding:

| BitNet | Triadic MicroGPT |
|--------|------------------|
| Native ternary training >> PTQ | End-to-end triadic >> Engine post-hoc |
| 2B native ≥ 7-8B PTQ | From-scratch gap +0.020 > post-hoc on same embeddings |
| Model adapts to constraint | Model learns to produce algebraically valid projections |

The lesson is universal: **if the model trains with the constraint, it learns to exploit it.**

---

## 5. Deep Structural Parallels

### 5.1 Dead Bits ≈ Zero Weights

| System | "Dead" proportion | Mechanism |
|--------|-------------------|-----------|
| BitNet b1.58 | 42.3% zeros | Absmean quantization naturally produces zeros |
| TriadicGPT D2 (63-bit) | 27/63 = 42.9% dead bits | Low entropy bits that don't activate |
| TriadicGPT base (64-bit) | ~15/64 = 23.4% dead bits | Similar phenomenon at smaller scale |

The convergence to ~42% inactive units is striking. Both systems independently discover that **sparsity is efficient** — not all dimensions need to carry information. The model allocates capacity to the dimensions that matter and lets others go dormant.

### 5.2 Three States = Ternary Values

The three-way parallel is the deepest connection:

| La Danza Cosmica | BitNet b1.58 | Triadic Head |
|------------------|--------------|--------------|
| [+] Presencia | +1 (active positive) | tanh → +1 (bit ON) |
| [0] Vacío | 0 (zero/dormant) | dead bit (entropy < 0.3) |
| [∅] Ausencia | -1 (active negative) | tanh → -1 (bit OFF) |

La Danza's philosophical framework describes three fundamental states:
- **[+] Presencia**: The quality is actively present (fuego in caliente)
- **[0] Vacío**: The quality is absent, a void (vacío in oscuridad)
- **[∅] Ausencia**: The quality is actively negated (the opposite pole)

BitNet's weights encode exactly this: +1 (positive contribution), 0 (no contribution), -1 (negative contribution). And our triadic head's tanh output with dead bits creates the same three-state system.

### 5.3 Information Density

| System | Bits per element | Effective bits |
|--------|-----------------|----------------|
| BitNet weight | log2(3) = 1.58 | 1.58 |
| Triadic bit (binary) | 1.0 | 1.0 |
| Triadic bit (ternary) | log2(3) = 1.58 | 1.58 |
| 63 binary bits | 63.0 | 63.0 |
| **63 ternary bits** | **63 × 1.58 = 99.5** | **99.5** |

If we move the triadic head from binary (tanh → {-1, +1}) to ternary ({-1, 0, +1}), each concept's signature gains 58% more information capacity without adding dimensions.

---

## 6. Proposed Experiment: Ternary Triadic Head (D-A8)

### Hypothesis
Replacing the triadic head's `tanh` activation with BitNet-style `absmean + STE` quantization will:
1. Explicitly model three states instead of two
2. Reduce dead bits (the model can intentionally use zero as a meaningful state)
3. Improve subsumption (zero bits won't interfere with bit-AND operations)
4. Maintain or improve algebraic operations (regla de tres, subsumption)

### Implementation Sketch

```python
def ternary_quantize(x):
    """Absmean quantization to {-1, 0, +1}"""
    gamma = x.abs().mean(dim=-1, keepdim=True) + 1e-8
    x_scaled = x / gamma
    x_q = x_scaled.round().clamp(-1, 1)
    # STE: forward uses quantized, backward passes through
    return x + (x_q - x).detach()

class TernaryTriadicHead(nn.Module):
    def __init__(self, d_model, n_bits):
        super().__init__()
        self.proj = nn.Linear(d_model, n_bits)

    def forward(self, x):
        logits = self.proj(x)
        return ternary_quantize(logits)  # {-1, 0, +1}
```

### Changes to Triadic Math
- **Prime signature**: Currently `Phi(x) = prod(p_i for bit_i > 0)`. With ternary: `Phi(x) = prod(p_i^state_i)` where state ∈ {0, 1} (treating -1 as a separate semantic signal, not affecting the prime product).
- **Subsumption**: `A ⊂ B` iff all +1 bits of A are +1 in B (zeros are neutral, -1 is explicit absence).
- **Regla de tres**: Vector arithmetic in ternary space: `D = C + (B - A)`, then re-quantize.

### Key Design Decision
What does -1 mean semantically?
- Option A: **Active negation** (the concept explicitly excludes this primitive). E.g., "muerto" has -1 on "vida".
- Option B: **Irrelevant** (the dimension doesn't apply). E.g., "rojo" has 0 on "gusto" (not applicable) vs -1 on "frío" (actively not cold).
- Option C: **Polarity** (each primitive has +/- poles). E.g., +1 on "temperatura" = hot, -1 = cold.

Option A aligns best with La Danza's framework and BitNet's weight semantics.

### Experimental Plan
1. Modify `DanzaTriadicGPT` to use `ternary_quantize` instead of `tanh`
2. Update gold labels to include explicit -1 states (currently binary: present/absent)
3. Train at base scale (smoke test) then XL
4. Compare: dead bits, R3 accuracy, subsumption, bit entropy distribution
5. Measure if ~42% zero rate emerges naturally (as in BitNet)

---

## 7. Actionable Improvements for TriadicGPT

Three concrete techniques from BitNet that can directly improve our project, ordered by impact.

### 7.1 Ternary Triadic Head (Experiment D-A8) — HIGH IMPACT

**Problem it solves**: 27/63 bits are dead (42.9%) — they don't carry information. With binary `tanh`, a dead bit is ambiguous: is it stuck, or does the model not need it?

**Solution**: Replace `tanh` with absmean + STE quantization. The output becomes exactly {-1, 0, +1} instead of continuous values near ±1.

```
Current (tanh):      bit = -0.97 (OFF) | +0.02 (dead?) | +0.99 (ON)
Proposed (ternary):  bit = -1 (ausente) | 0 (no aplica) | +1 (presente)
```

**Why this matters**:
- Zero becomes a **real semantic state** ("this primitive doesn't apply") instead of a failure mode
- Each concept goes from 63 binary bits (63 bits info) to 63 ternary trits (**99.5 bits info**) — +58% capacity without adding dimensions
- Aligns with La Danza's three states: [+] presencia, [0] vacío, [∅] ausencia
- Code change is surgical (~20 lines): replace `tanh(self.proj(x))` with `ternary_quantize(self.proj(x))`

**Implementation**:
```python
def ternary_quantize(x):
    """Absmean quantization to {-1, 0, +1} with STE"""
    gamma = x.abs().mean(dim=-1, keepdim=True) + 1e-8
    x_scaled = x / gamma
    x_q = x_scaled.round().clamp(-1, 1)
    return x + (x_q - x).detach()  # STE trick
```

**Gold label update**: Currently anclas.json is binary (bit present or absent). Ternary labels would add explicit -1 for negated primitives:
- `muerto`: vida = -1 (actively negated), gusto = 0 (irrelevant), muerte = +1 (present)
- `caliente`: fuego = +1, agua = 0, tierra = -1 (explicitly not-earth)

**Success criteria**: Fewer dead bits, comparable or better R3 accuracy, natural emergence of ~40% zero rate.

### 7.2 STE for Cleaner Gradients — MEDIUM IMPACT

**Problem it solves**: `tanh` distorts gradients near saturation. A bit at +0.99 has gradient tanh'(0.99) ≈ 0.02 — almost zero. Once a bit saturates, it effectively freezes, contributing to dead bits.

**Solution**: The STE pattern passes gradients through unchanged regardless of the output value:

```python
# tanh: gradient shrinks as |x| grows (vanishing gradient)
grad_tanh = 1 - tanh(x)^2   # → 0 as x → ±∞

# STE: gradient is always 1 (identity in backward pass)
forward:  y = quantize(x)    # discrete output
backward: dy/dx = 1          # full gradient always
```

**Why this matters**:
- Bits can always be updated, even after thousands of steps
- No gradient vanishing → faster convergence
- Model can "change its mind" about a bit assignment late in training
- Works with or without the ternary upgrade (can use STE + binary {-1, +1} too)

**Risk**: STE gradients are biased (the true gradient of `round()` is zero, not one). BitNet shows this works at 2B scale, but behavior at our 40M scale needs validation.

### 7.3 Embrace Natural Sparsity — LOW IMPACT (mindset shift)

**Problem it solves**: We've been treating dead bits as a defect and fighting them with entropy regularization. BitNet shows ~42% sparsity is **optimal**, not pathological.

**Evidence**:
| System | Inactive proportion | Emerged from |
|--------|-------------------|--------------|
| BitNet 2B | 42.3% zero weights | Absmean quantization |
| TriadicGPT D2 | 42.9% dead bits (27/63) | Tanh + entropy reg |
| TriadicGPT base | 23.4% dead bits (15/64) | Tanh + entropy reg |

**Practical implications**:
- Stop penalizing dead bits (reduce or remove entropy regularization)
- Accept that ~36 active bits out of 63 is the natural operating point
- Focus evaluation on the **active** bits: are they semantically meaningful?
- Monitor which primitives survive vs die — this tells us what the model considers important
- The effective representation is ~36 ternary trits = 57 bits, still far richer than post-hoc k=6-12

**Caveat**: This applies only if we adopt the ternary head. With binary tanh, dead bits are genuinely wasted capacity. With ternary, zero is information.

---

## 8. Broader Implications

### For the Triadic Project
1. **Validation of discrete representations**: BitNet proves that extreme discretization (1.58 bits) is viable even for large-scale language modeling. Our 63-bit triadic head is not a limitation — it's an advantage.
2. **Native training is essential**: Both projects confirm that the representation must be learned end-to-end, not applied post-hoc.
3. **Sparsity is natural**: Both systems converge to ~40% inactive units. This isn't a bug — it's efficient coding.
4. **CPU inference feasibility**: If ternary weights enable GPU-free inference, ternary triadic signatures enable GPU-free semantic operations. The triadic head's output is already discrete — verification, subsumption, and composition are all integer operations.

### For the Paper
The BitNet parallel strengthens the paper's argument:
- Section on "discrete representations in neural networks" can cite BitNet as independent evidence that extreme discretization works
- The dead bits phenomenon has a parallel explanation (natural sparsity under quantization constraints)
- The three-states framework gains computational credibility beyond philosophical motivation
- D-A8 results (if positive) would be a strong new experiment: "ternary head inspired by BitNet b1.58 reduces dead bits from 43% to X%"

### Philosophical Connection
La Danza Cosmica's three states ([+], [0], [∅]) were formulated as a metaphysical framework. BitNet arrives at the same structure from pure engineering optimization. When philosophy and engineering converge on the same pattern independently, it suggests the pattern captures something real about information representation.

### Experiment Priority
```
D-A5/D-A6 (bootstrap)  →  currently running, finish first
D-A8 (ternary head)    →  highest ROI from BitNet research, ~20 lines change
D-A7 (scale 307M)      →  test if ternary head changes scale crossover point
```

---

## References

1. Ma, S. et al. (2024). "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits." Microsoft Research. arXiv:2402.17764
2. Wang, H. et al. (2023). "BitNet: Scaling 1-bit Transformers for Large Language Models." arXiv:2310.11453
3. Microsoft (2024). "bitnet.cpp" — Official inference framework for 1-bit LLMs. github.com/microsoft/BitNet
4. Ornelas Brand, A. (2026). "Prime Factorization as a Neurosymbolic Bridge." (This project)
5. Ornelas Brand, A. (2026). "La Danza Cosmica de los Opuestos." (Theoretical framework)
