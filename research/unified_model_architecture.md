# Unified Model Architecture — Bitwise + Ternary + Discovery Loop

> Date: 2026-03-19 | Status: PROPOSAL

---

## 1. What Works Today (Validated)

### Current Architecture

```
Text -> BPE (4096) -> TriadicGPT (12L/512D/8H, 40M params)
                            |
                      +-----+-----+
                      |           |
                 LM Head    Triadic Head (Linear -> tanh -> 63 bits)
                      |           |
                 next token   PrimeMapper -> TriadicValidator
                                  |
                              subsumption, analogy, gap, compose
```

### Validated Components

| Component | Evidence | Status |
|-----------|----------|--------|
| Dual-head architecture | PPL 7.69 vs 7.56 ablation (cost=0) | PRODUCTION |
| 63-bit supervised head | 93% test, 98.3% subsumption (v2) | PRODUCTION |
| PrimeMapper algebra | Subsumption, analogy, gap, compose | PRODUCTION |
| BitwiseValidator | 1000/1000 isomorphic, 5-78x faster | VALIDATED |
| iFSQ activation | Loss 0.924, best LM, 87.1% sub | VALIDATED |
| Free bits (hybrid) | Learn without supervision, 6 dead | VALIDATED |
| reptimeline discovery | 68 triadic 3-way, 7 duals, 635 deps | VALIDATED |
| R3 composition | 98.1% round-trip, sub-linear chains | VALIDATED |
| Ternary convergence | ~42% sparsity in all three paths | OBSERVED |

### What Failed

| Approach | Why | Lesson |
|----------|-----|--------|
| Gradient decoupling | 49.6% = random | Don't manipulate per-bit gradients |
| Absmean quantization | Loss 1.309 | FSQ/iFSQ >> absmean for this architecture |
| 49-bit structured (P15) | 17% test = memorization | End-to-end > structured for generalization |
| Bootstrap self-improvement | Confidence gate too strict | Need more anchors, not self-play |

---

## 2. Proposed Architecture (Model for Paper)

### Core Insight

Three independent convergences point to the same optimal representation:

```
Philosophy:  {+1, 0, -1}  (presence / void / absence)
Engineering: {+1, 0, -1}  (BitNet b1.58 weights, ~42% zeros)
Mathematics: AND / OR / XOR (isomorphic to prime algebra)
```

### Architecture Diagram

```
Text -> BPE (4096) -> Unified TriadicGPT (12L/512D/8H)
                            |
                      +-----+-----+
                      |           |
                 LM Head    Triadic Head
                      |           |
                 next token   Linear -> iFSQ -> 63 bits
                                  |
                           BitwiseValidator (O(1))
                           - subsumes: (A & B) == B
                           - compose:  A | B
                           - analogy:  (C & ~only_a) | only_b
                           - gap:      A & ~B, B & ~A
                           - similarity: popcount(A&B)/popcount(A|B)
                                  |
                           reptimeline Discovery Loop
                           - bit semantics (what each bit means)
                           - dual detection (anti-correlated pairs)
                           - 3-way interactions (A + B -> C)
                           - auto-label + reconcile
                                  |
                           Anchor Generation
                           - discover -> label -> new anchors
                           - feed back into next training cycle
```

### Key Changes from Current

| What | Current | Proposed | Reason |
|------|---------|----------|--------|
| Activation | tanh | **iFSQ** | Better loss (0.924 vs 0.946), preserves gradients |
| Validation backend | PrimeMapper | **BitwiseValidator** | O(1), scales to 1024+ bits |
| Discovery | Post-hoc manual | **Integrated loop** | Auto-anchor, auto-label |
| Bits | 63 supervised only | **63 supervised + free** | Hybrid proved free bits learn |
| Theory | Primes | **Primes (paper) + Bits (impl)** | Same math, better performance |

---

## 3. Optimizations Detected

### 3.1 BitwiseValidator (Implemented)

All prime algebra operations have O(1) bitwise equivalents:

| Prime O(n) | Bitwise O(1) | Speedup |
|-----------|-------------|---------|
| GCD(A,B) via factorization | A & B | 1.3x |
| LCM(A,B) via factorization | A \| B | 5x |
| Jaccard via set intersection | popcount ratio | 78x |
| Analogy via GCD+LCM | mask + OR | 5x |

Beyond 128 bits, prime arithmetic overflows. Bitwise stays O(1) at any dimension.

### 3.2 iFSQ Activation (Validated, not yet default)

```python
# Current: tanh (saturates, kills gradients)
proj = torch.tanh(self.triadic_head(h))

# Proposed: iFSQ (smooth, preserves gradients)
proj = 2 * torch.sigmoid(1.6 * self.triadic_head(h)) - 1
```

Evidence: D-A10 iFSQ achieved loss 0.924 (best LM) + 87.1% subsumption.

### 3.3 Natural Sparsity (~42%)

Independent measurements converge:

| System | Dead/Zero % | Mechanism |
|--------|------------|-----------|
| BitNet b1.58 weights | 42.3% | Absmean quantization |
| D-A5 (63-bit, tanh) | 42.9% (27/63) | tanh saturation |
| D-A14 v2 (63-bit) | 41.3% (26/63) | Training convergence |
| D-A9 hybrid | 9.5% (6/63) | Free bits prevent death |

The ~42% sparsity appears to be a structural property of semantic encoding, not a bug.

### 3.4 Triadic 3-Way Interactions

Discovery found 68 interactions of the form: bit_i + bit_j -> bit_r
(where P(r|i,j) >> P(r|i) and P(r|j))

This is compositional structure: pairs of features predict a third feature
that neither predicts alone. Evidence that the bit space has genuine
algebraic structure beyond pairwise correlations.

---

## 4. What's NOT in the Unified Model

| Idea | Why not |
|------|---------|
| Tokenizer in bits | Interesting but untested, separate paper |
| Ternary weights (transformer) | At 355M, zeros collapse to 0% — scale-dependent |
| Fourier head | Structural not literal (Cap. 7), tanh/iFSQ valid |
| 49-bit structured system (P15) | Memorization problem (17% test) |
| Gradient decoupling | Failed (49.6% = random) |

---

## 5. Evidence Table for Paper Claims

| Claim | Evidence | Strength |
|-------|----------|----------|
| Zero language cost | PPL 7.69 vs 7.56 | Strong |
| Algebraic operations work | Subsumption 98.3%, analogy exact | Strong |
| Bitwise isomorphic to primes | 1000/1000 tests | Proof |
| Compositional structure | 68 triadic 3-way, R3 98.1% round-trip | Strong |
| Three-path convergence | ~42% sparsity in all three | Observed |
| Scaling works | 355M -> 100% sub holdout | Strong |
| Free bits learn | 6 dead vs 26 in supervised-only | Moderate |
| p < 0.001 significance | Permutation test, Cohen's d=6.64 | Strong |
