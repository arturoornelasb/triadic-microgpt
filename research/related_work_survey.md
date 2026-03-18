# Related Work Survey: Deep Dive into Papers Relevant to Triadic MicroGPT

**Date**: 2026-03-18
**Purpose**: Identify techniques from recent literature that can improve the triadic head, validate our approach, or open new research directions.
**Depth**: Full technical analysis — formulas, architectures, benchmark numbers, and concrete borrowable techniques.

---

## Executive Summary

| # | Paper | Key Borrowable Technique | Impact | Effort |
|---|-------|--------------------------|--------|--------|
| 1 | FSQ (Google DeepMind, ICLR 2024) | Ternary quantization + iFSQ activation fix | HIGH | ~20 LOC |
| 2 | CB-LLMs (ICLR 2025) | Adversarial disentanglement + hybrid bits | HIGH | ~100 LOC |
| 3 | Wang et al. (NeuS 2025, DARPA Award) | Theoretical justification — cite in paper | HIGH | 0 LOC |
| 4 | Hyperdimensional Probe (2025) | VSA analogy methodology for benchmarking | MEDIUM | ~50 LOC |
| 5 | CRH (2025) | Co-learned hash centers | MEDIUM | ~80 LOC |
| 6 | Monosemanticity (Anthropic 2023-24) | L1 sparsity loss for dead bits | LOW | ~5 LOC |
| 7 | LARS-VSA (2024) | Differentiable rule learning | LOW | future |

**Our position**: No existing work combines end-to-end transformer training with algebraically-verifiable discrete semantic signatures via prime factorization. We are unique, and the literature validates our design choices from multiple angles.

---

## 1. Finite Scalar Quantization (FSQ) — HIGHEST PRIORITY

**Paper**: "Finite Scalar Quantization: VQ-VAE Made Simple"
**Authors**: Mentzer, Minnen, Agustsson, Tschannen (Google DeepMind)
**Venue**: ICLR 2024 | **arXiv**: 2309.15505
**Follow-up**: iFSQ (Tencent, 2025, arXiv 2601.17124)

### 1.1 Core Mechanism

FSQ replaces VQ-VAE's learned codebook with per-dimension scalar rounding to fixed levels:

```python
def fsq_quantize(z, L):
    """Quantize each dimension to L levels."""
    half = floor(L / 2)
    z_bounded = half * tanh(z)          # bound to [-half, +half]
    z_q = round(z_bounded)              # snap to integers
    return z + (z_q - z).detach()       # STE: forward=discrete, backward=continuous
```

For ternary (L=3): output is {-1, 0, +1}. For 5-level: {-2, -1, 0, 1, 2}.

The implicit codebook is the Cartesian product of all per-dimension levels:
- 63 ternary dims → |C| = 3^63 ≈ 10^30 possible codes
- No codebook to learn, update, or collapse

### 1.2 Key Results

| Metric | FSQ | VQ-VAE |
|--------|-----|--------|
| Codebook utilization | **~100%** at all sizes | Degrades above 2^10 |
| Auxiliary losses needed | **Zero** | Commitment loss + EMA + reseeding |
| Training stability | **Rock solid** | Prone to codebook collapse |
| FID (ImageNet 256, MaskGIT) | Within 0.5-3% of VQ | Baseline |
| Dead codes | **Zero by construction** | Significant at large sizes |

### 1.3 The iFSQ Fix — CRITICAL for Us

The iFSQ follow-up (Tencent, 2025) discovered that vanilla FSQ's `tanh` bounding causes **activation collapse**: tanh concentrates values near 0, so inner quantization bins are over-used and outer bins are under-used.

**This is exactly our dead bits problem.** Our `tanh` projects bit activations, and ~42% end up as "dead" (stuck near 0, low entropy).

iFSQ's fix — one line of code:

```python
# BEFORE (FSQ / our current approach):
z_bounded = half * tanh(z)                          # concentrates near 0

# AFTER (iFSQ fix):
z_bounded = half * (2 * sigmoid(1.6 * z) - 1)      # uniform bin utilization
```

The `sigmoid(1.6*z)` function has a flatter middle region than `tanh`, distributing activations more uniformly across quantization bins.

### 1.4 What We Borrow

1. **D-A8**: Replace `tanh(proj(x))` with `ternary_quantize(proj(x))` using STE
2. **iFSQ activation**: Use `2*sigmoid(1.6*x) - 1` instead of `tanh(x)` as bounding function
3. **Remove entropy regularization**: FSQ proves it's unnecessary with fixed-grid quantization
4. **Remove diversity loss**: The grid structure prevents collapse by construction

### 1.5 Key Difference From Us

FSQ targets image VQ-VAE latents — their codes have no algebraic structure. Our codes must support subsumption (`Phi(A) | Phi(B)`) and analogy (`D = C + B - A`). We add algebraic constraints on top of their quantization insight.

### 1.6 Level Configurations Tested

| Target |C| | dims | Levels | Actual |C| |
|---------|------|--------|-----------|
| ~256 | 3 | [8, 6, 5] | 240 |
| ~1024 | 4 | [8, 5, 5, 5] | 1000 |
| ~4096 | 5 | [7, 5, 5, 5, 5] | 4375 |
| ~16K | 6 | [8, 8, 8, 5, 5, 5] | 16000 |
| ~64K | 7 | [8, 8, 8, 5, 5, 5, 5] | 80000 |

**Optimal per-dimension**: ~4 bits (16 levels) for image tasks. For semantic bits, ternary (3 levels = 1.58 bits) aligns with La Danza's three states.

---

## 2. Concept Bottleneck Large Language Models (CB-LLMs) — HIGH PRIORITY

**Paper**: "Concept Bottleneck Large Language Models"
**Authors**: Sun, Oikarinen, Ustun, Weng (UC San Diego)
**Venue**: ICLR 2025 | **arXiv**: 2412.07992
**Background**: Original CBM by Koh et al. (ICML 2020)

### 2.1 Architecture

The CBL sits **after the final transformer block**, parallel to an unsupervised layer:

```
Input → LLM backbone (LoRA fine-tuned) → final hidden states
                                              |
                                      +-------+-------+
                                      |               |
                                     CBL      Unsupervised Layer
                                      |               |
                                    ReLU            (raw)
                                      |               |
                                      +-------+-------+
                                              |
                                        Concatenation
                                              |
                                    LM Head → next token
```

**Critical insight**: The CBL and unsupervised layer operate in **parallel**, not series. The unsupervised layer preserves language modeling capacity while the CBL provides interpretability.

### 2.2 Adversarial Disentanglement — THE Key Innovation

Without this, the model bypasses the CBL entirely (pushes all information into the unsupervised layer, making CBL dead).

**Mechanism**: A linear classifier tries to predict concept activations FROM the unsupervised layer. The unsupervised layer is simultaneously trained to make this classifier fail (gradient reversal).

```
Unsupervised output → Linear Classifier → tries to predict CBL concepts
                  ↑
        gradient reversal (adversarial)
```

At equilibrium: concept information lives ONLY in the CBL, not in the unsupervised layer.

**Ablation**: Without adversarial training, steerability drops to near-random. This is **essential**.

### 2.3 Relevance to Our Architecture

**Our triadic head IS a concept bottleneck.** But we don't have adversarial disentanglement. This raises a critical question: is our backbone's hidden state encoding triadic information redundantly, bypassing the triadic head?

| Aspect | CB-LLMs | TriadicGPT (ours) |
|--------|---------|-------------------|
| Bottleneck | CBL (continuous neurons) | Triadic head (binary/ternary bits) |
| Unsupervised | Parallel layer | The backbone itself |
| Disentanglement | Adversarial (explicit) | None (implicit via dual loss) |
| Language cost | 0.3-0.6% accuracy | +2% PPL |
| Dead concepts | Pruned via ReLU + L1 | ~42% dead bits |
| Concept count | 208-476 (auto-generated) | 63 (from Sistema 7×7) |

### 2.4 What We Borrow

**D-A9: Hybrid bits + adversarial disentanglement**

```python
class HybridTriadicHead(nn.Module):
    def __init__(self, d_model, n_supervised=30, n_free=33):
        self.sup_proj = nn.Linear(d_model, n_supervised)   # gold labels
        self.free_proj = nn.Linear(d_model, n_free)        # contrastive only
        # Adversarial: prevent backbone from encoding concepts
        self.adversary = nn.Linear(d_model, n_supervised)  # gradient reversal

    def forward(self, h):
        sup = ternary_quantize(self.sup_proj(h))
        free = ternary_quantize(self.free_proj(h))
        adv_pred = self.adversary(h.detach())  # detached from backbone
        return torch.cat([sup, free], dim=-1), adv_pred
```

Training adds adversarial loss: `L_adv = -CrossEntropy(adv_pred, sup_targets)` with gradient reversal on the backbone.

### 2.5 Concept Intervention

At test time, they can manually set concept neuron values to steer generation. We could do the same: manually set specific triadic bits to control output semantics.

Example: Set bit 0 (`fuego`) = +1 and bit 1 (`agua`) = -1 → steer generation toward fire-related concepts.

### 2.6 Numbers

| Dataset | CB-LLM Accuracy | Black-box Accuracy | Delta |
|---------|----------------|-------------------|-------|
| SST2 | 0.9407 | 0.9462 | -0.6% |
| Yelp | 0.9806 | 0.9803 | +0.03% |
| AG News | 0.9453 | 0.9478 | -0.3% |
| DBpedia | 0.9928 | 0.9922 | +0.06% |

Near-zero language cost, validating the concept bottleneck approach at LLM scale.

---

## 3. Algebraic Emergence Theory — HIGH PRIORITY (Theoretical)

**Paper**: "Why Neural Networks Can Discover Symbolic Structures with Gradient-based Training"
**Authors**: Peihao Wang, Zhangyang "Atlas" Wang (UT Austin)
**Venue**: NeuS 2025 (PMLR v288) | **arXiv**: 2506.21797
**Recognition**: DARPA Disruptive Idea Paper Award

### 3.1 Core Theorem

Under three conditions, Wasserstein gradient flow over neural network parameters **decouples** into independent coordinate-wise optimization over discrete boolean variables:

```
∂_t ρ_ri[μ_t] = -C_i(t) · ∂/∂ρ_ri L(ρ)
```

where C_i(t) > 0 is a scalar depending only on ρ_ri itself.

**The three conditions:**
1. **Gaussian initialization**: μ_0 = N(0, I)
2. **Odd-degree monomials**: activation functions must be odd (degree ≥ 3)
3. **O(d)-equivariant velocity field**: geometric symmetry preserved

**Consequence**: Training a continuous neural network is equivalent to solving a boolean satisfiability problem. Each monomial potential independently converges to its 0/1 assignment.

### 3.2 Why This Matters for Us — 5 Direct Connections

**Connection 1: tanh works because it's odd.**
Their theory requires odd-degree monomials. `tanh` is an odd function (tanh(-x) = -tanh(x)). `sigmoid` is NOT odd. This provides the theoretical explanation for why tanh works in our triadic head and sigmoid causes collapse — a pattern we discovered empirically across multiple experiments (R3, P15, XL2).

**Connection 2: Dead bits = progressive contraction.**
Their Theorem 4.4 proves that during training, eigenvalues of the Hessian cross zero at discrete times, each crossing permanently "locking" one dimension. Dead bits are dimensions that got locked to a trivial value early in training.

**Connection 3: Our embedding alignment = their O(d)-equivariance.**
Their theory requires geometric symmetry constraints. Our embedding alignment loss teaches the triadic head to respect the structure of pre-trained embeddings — a soft form of the geometric invariance they require.

**Connection 4: Commutative semi-ring ↔ prime product space.**
Their measure space forms a commutative semi-ring under addition and multiplication of measures. Our prime product space is a commutative monoid under multiplication. The algebraic structure is analogous.

**Connection 5: They have no experiments — we fill the gap.**
The paper is purely theoretical (26 pages, zero experiments). Our 29 training runs across 11 experiments provide the empirical validation their theory lacks.

### 3.3 The Semi-Ring Structure

```
Elements: probability measures over parameter space
Addition: μ₁ + μ₂ (fuse mass)
Multiplication: μ₁ * μ₂ (element-wise product of samples)
Identity: δ_{1_d} (point mass at all-ones)

Key property (Theorem 3.3): Monomial potentials are ring homomorphisms:
  ρ_r(μ₁ + μ₂) = ρ_r(μ₁) + ρ_r(μ₂)
  ρ_r(μ₁ * μ₂) = ρ_r(μ₁) · ρ_r(μ₂)
```

This means complex solutions can be composed from simple ones — exactly what our subsumption and analogy operations do.

### 3.4 Connection to Grokking

The progressive contraction (eigenvalue zero-crossings at discrete times t₁ < t₂ < ... < tₘ) provides a mechanism for grokking:
- **Memorization phase**: measure has many degrees of freedom
- **Structure discovery**: eigenvalues cross zero, locking in algebraic structure
- **Sudden generalization**: once confined to algebraic submanifold, solution generalizes

This may explain why our subsumption accuracy sometimes "jumps" mid-training.

### 3.5 What We Borrow

- **Cite prominently** in paper's "Why does it work?" section
- **Their theory predicts minimum anchors for bootstrap** — relevant to D-A5
- **Analysis idea**: track per-bit optimization trajectories during training to check if they decouple as predicted

### 3.6 Limitations

- Requires O(d)-equivariance (standard transformers don't satisfy this)
- Only covers Abelian groups (not non-Abelian)
- Requires quadratic activations (theory), though tanh works (practice)
- Mean-field limit (infinite width) — our network is finite (40M params)
- Purely theoretical — no experimental validation in the paper itself

---

## 4. Hyperdimensional Computing / VSA — MEDIUM PRIORITY

### 4.1 Hyperdimensional Probe (Bronzini et al., 2025)

**Paper**: "Hyperdimensional Probe: Decoding LLM Representations via VSA"
**arXiv**: 2509.25045

**Architecture**: Post-hoc. A 3-layer MLP (55-71M params) maps frozen LLM residual stream activations to bipolar hypervectors {-1, +1}^4096.

**VSA operations**:
- **Binding**: Hadamard product (element-wise multiply). Associates roles with fillers.
- **Bundling**: Element-wise addition + sign(). Superposes multiple bindings.
- **Unbinding**: Multiply again (self-inverse in bipolar).

**Analogy example** ("Denmark:krone = Mexico:?"):
```python
# Encode knowledge
record = sign(phi_denmark * phi_krone + phi_mexico * phi_peso)

# Query: "What plays the role of peso for Denmark?"
answer = record * phi_peso * phi_denmark  # → recovers phi_krone
# Rank codebook by cosine similarity → find "krone"
```

**Results**:

| Model | Analogy Probing@1 | LLM next-token@1 |
|-------|-------------------|-------------------|
| GPT-2 (355M) | 60% | 8% |
| Pythia (1.4B) | ~75% | ~25% |
| Llama 3.1 (8B) | **85%** | 48% |
| OLMo-2 (32B) | ~83% | ~45% |
| Llama 4 Scout (109B) | ~83% | ~45% |

The probe consistently outperforms raw LLM output for analogy extraction.

### 4.2 LARS-VSA (Mejri et al., 2024)

**Paper**: "A Vector Symbolic Architecture For Learning with Abstract Rules"
**arXiv**: 2405.14436

**Key innovation**: Replaces transformer attention with HDSymbolicAttention operating in bipolar space.

```python
# Standard attention: softmax(QK^T/√d) · V
# LARS-VSA: sign(O_i ⊕ O_j) ⊗ S   (bind objects, then bind with symbol vectors)
```

**Results**:

| Task | LARS-VSA vs best alternative |
|------|------------------------------|
| Order relations | 1.07x better, needs only 200 samples |
| SET classification | 1.05x better |
| 5-element sorting | 1.66-2.25x better |
| 6-element sorting | 1.56-3.33x better |
| Memory efficiency | Up to 17x more efficient |
| Attention speed | ~25x faster than standard |

### 4.3 Head-to-Head: VSA vs Our Triadic Approach

| Dimension | VSA / HDC | Triadic MicroGPT |
|-----------|-----------|------------------|
| Vector type | Bipolar {-1,+1}^D, D=4096 | Binary/ternary, k=63 |
| Binding | Hadamard product | Prime multiplication |
| Bundling (superposition) | **Yes** — sign(A+B), recoverable | **No** — cannot superpose |
| Analogy accuracy | 83% probing (post-hoc) | 100% verification (end-to-end) |
| Subsumption | Not native (FactorHD adds it at 92%) | **Native via divisibility** (92-100%) |
| Noise tolerance | **High** — D/3 bits can flip | Low — 1 bit flip changes Φ(x) |
| Interpretability | Opaque (random dimensions) | **Each bit = named primitive** |
| Speed | O(D), D~4096 | **O(k), k~64** — 28.4x faster |
| Integration | Post-hoc probe OR attention replacement | **End-to-end dual-loss** |
| Capacity | 2^D ≈ 10^1233 | 2^63 ≈ 10^19 |

### 4.4 What VSA Has That We Don't — Bundling

VSA can **superpose** multiple concepts in a single vector and recover them:

```python
# Bundle: store both concepts in one vector
composite = sign(phi_cat + phi_dog)

# Unbind: recover either one
recovered_cat = composite * phi_dog  # ≈ phi_cat (noisy but recoverable)
```

Our prime products cannot do this. `Φ(cat) × Φ(dog)` is irreversible without knowing one factor. This is a fundamental capability gap.

### 4.5 What We Have That VSA Doesn't — Exact Algebra

Our subsumption is **mathematically exact**: if `Φ(A) | Φ(B)`, it's a provable fact, not a similarity threshold. VSA containment is always approximate.

Our analogy verification is **deterministic**: `Φ(king)/Φ(man) × Φ(woman) == Φ(queen)` is a yes/no check. VSA analogies require cosine similarity ranking.

### 4.6 Related HDC Work

**FactorHD (2025)**: Hierarchical containment in HDC via bundling/factorization. 92.48% accuracy on CIFAR-10 with 5667x speedup. Closest existing work to our subsumption, but via bundling not primes.

**Resonator Networks (Frady et al., 2020)**: Solves the VSA factorization problem — given a composite vector, recover the factors. Mathematically analogous to our prime factorization. Could potentially be adapted to recover which primes are active in a signature.

**Hrrformer (ICML 2023)**: Replaces self-attention with Holographic Reduced Representations (circular convolution). O(TH log H) vs O(T²H). 280x faster training, 10x fewer epochs. Shows VSA operations can serve as attention — but ours can't (primes lack additive structure).

**No existing work combines HDC with number-theoretic prime factorization.** Our approach is novel in this space.

### 4.7 What We Borrow

- **Benchmarking methodology**: Use Hyperdimensional Probe's analogy probing protocol as a standardized evaluation
- **Cosine similarity trick (LARS-VSA Lemma 1)**: Binarized cosine via AND + popcount — could speed up our bit comparisons further
- **Conceptual**: VSA's noise tolerance suggests we could add redundancy (duplicate important bits) if noise becomes an issue

---

## 5. Monosemanticity / Sparse Autoencoders (Anthropic, 2023-2024)

**Papers**: "Towards Monosemanticity" (2023) + "Scaling Monosemanticity" (2024)
**Authors**: Bricken, Templeton et al. (Anthropic)
**Source**: transformer-circuits.pub

### 5.1 Architecture

Train sparse autoencoders (SAEs) on transformer residual stream activations:

```
h → Encoder(W_enc · h + b_enc) → ReLU → sparse features → Decoder(W_dec · features) → h_reconstructed
Loss = ||h - h_reconstructed||² + λ · ||features||₁
```

The L1 penalty forces most features to be zero (sparse), encouraging each feature to represent a single concept (monosemantic).

### 5.2 Results

- 16x expansion on GPT-2 Small → ~15,000 features
- **70% map cleanly to single concepts** (monosemantic)
- Scaled to Claude 3 Sonnet: features are multilingual, multimodal, abstract
- Dead features: ~5-10% (features that never activate)

### 5.3 Comparison

| | SAE Features | Triadic Bits |
|---|---|---|
| Type | Continuous activations | Discrete {-1, 0, +1} |
| Count | 15,000+ (overcomplete) | 63 (compact) |
| Composition | Linear combination | Prime multiplication |
| Verifiable? | No formal guarantee | **Yes** (algebraic) |
| Dead units | ~5-10% | ~42% |
| Training | Post-hoc (frozen model) | End-to-end (joint) |
| Interpretability | Statistical (activation patterns) | **Structural** (named primitives) |

### 5.4 What We Borrow

**L1 sparsity on bit activations** (different from entropy regularization):

```python
# Entropy reg (current): encourages each bit to be active ~50% of time
L_entropy = -mean(H(sigmoid(proj)))

# L1 sparsity (new): encourages each bit to be zero for most tokens
L_sparse = lambda_sparse * mean(|proj|)
```

L1 sparsity would encourage bits to activate only when genuinely relevant, reducing noise in inactive bits. This complements (not replaces) our alignment loss.

---

## 6. Codebook-Centric Deep Hashing (CRH) — MEDIUM PRIORITY

**Paper**: "Codebook-Centric Deep Hashing"
**Authors**: Shuo Yin et al. | **Year**: 2025 | **arXiv**: 2511.12162

### 6.1 Key Idea

Instead of fixing hash center assignments randomly (as in CSQ/OrthoHash), CRH **co-learns** the hash centers alongside the hash function. Semantically similar classes get closer hash centers.

### 6.2 Relevance

We fix gold prime signatures in `anclas.json` — hand-factorized, frozen during training. CRH suggests **jointly optimizing target signatures** could yield better metrics.

### 6.3 Why We Probably Shouldn't Use This

Our gold labels come from La Danza's philosophical framework. Making them learnable could produce better numbers but lose interpretive power. The labels MEAN something:

```
amor = fuego ∧ agua ∧ unión ∧ vida ∧ placer ∧ consciente ∧ querer ∧ interocepción
```

This isn't an arbitrary hash — it's a reasoned decomposition. Co-learning would destroy this meaning.

**Exception**: Could co-learn the FREE bits (from D-A9 hybrid architecture) while keeping supervised bits fixed.

---

## 7. Additional Papers

### HASH-RAG (ACL 2025)
Binary hash codes for RAG retrieval. 90% time reduction, 1.4-4.3% accuracy improvement. Our 28.4x speedup aligns with their findings. Validates practical utility of learned binary codes.

### T5-VQVAE (EACL 2024)
Discrete latent codes control T5 cross-attention. Alternative to our dual-head: use triadic codes to modulate attention. Would require major architectural change.

### Disentangled Representation via Modular Compositional Bias (2025)
Grid structure of categorical distributions enables disentanglement. Theoretical justification for why binary/ternary codes disentangle better than continuous embeddings.

### Concept Bottleneck Sparse Autoencoders (CB-SAE, 2025)
Combines SAEs with concept bottlenecks. +32.1% interpretability, +14.5% steerability over SAEs alone. Validates bottleneck approach.

### Overmann's Triadic Memory (2021)
Tridirectional associative memory for sparse binary vectors. Different "triadic" than ours — structural (triples of vectors), not algebraic (prime products). Stores and recovers 1M triples at 1% sparsity.

---

## 8. Synthesis: Our Position in the Literature

### 8.1 We Are Unique

No existing work does what we do. The closest neighbors each lack a key element:

```
FSQ               = ternary quantization    + NO algebra
CB-LLMs           = concept bottleneck      + NO primes, NO discrete
Hyperdimensional   = algebraic operations    + post-hoc, NOT end-to-end
Monosemanticity    = interpretable features  + continuous, NO verification
CRH               = learned hash codes      + NO compositional algebra

TriadicGPT         = ternary quantization    + concept bottleneck
                   + algebraic operations    + end-to-end training
                   + interpretable bits      + prime factorization
                   + UNIQUE
```

### 8.2 We Are Validated

Multiple independent research lines confirm our design choices:

| Our Design Choice | Validated By |
|-------------------|-------------|
| Discrete representations work | FSQ (ICLR 2024), BitNet (Microsoft) |
| Concept bottleneck scales to LLMs | CB-LLMs (ICLR 2025) |
| Algebra emerges from gradient training | Wang et al. (NeuS 2025, DARPA Award) |
| tanh > sigmoid for discrete heads | Wang et al. (odd functions required) |
| Sparse features are interpretable | Anthropic Monosemanticity (2023-24) |
| End-to-end > post-hoc | BitNet, CB-LLMs, our own P4 experiment |
| ~40% sparsity is natural | BitNet (42.3%), FSQ (dead codes → 0%) |

### 8.3 Concrete Experiment Queue

| Priority | Experiment | Source | Change | Expected Impact |
|----------|-----------|--------|--------|-----------------|
| 1 | D-A8: Ternary head | FSQ + BitNet | ~20 LOC | Solves dead bits |
| 2 | iFSQ activation | iFSQ (Tencent) | 1 LOC | Reduces dead bits independently |
| 3 | D-A9: Hybrid bits | CB-LLMs | ~100 LOC | Frees capacity for model-discovered features |
| 4 | Adversarial disentanglement | CB-LLMs | ~50 LOC | Forces triadic info into head, not backbone |
| 5 | L1 sparsity loss | Anthropic SAE | ~5 LOC | Cleaner bit activations |
| 6 | Gradient decoupling analysis | Wang et al. | Analysis only | Theoretical validation for paper |

### 8.4 What We Should NOT Do

- **Co-learn gold labels** → destroys philosophical interpretability (CRH)
- **Switch to continuous SAE features** → loses algebraic guarantees
- **Replace prime multiplication with VSA binding** → too fundamental a change
- **Add bundling/superposition** → incompatible with prime factorization
- **Scale to D=4096 hypervectors** → our compact 63-bit representation is a feature, not a limitation

### 8.5 What We Should Cite in the Paper

**Essential citations** (strengthen our contribution):
1. Wang et al. (2025) — theoretical foundation for algebraic emergence from gradient training
2. FSQ (2024) — validates fixed-grid ternary quantization
3. CB-LLMs (2025) — validates concept bottleneck at LLM scale

**Supporting citations** (position in literature):
4. BitNet b1.58 (2024) — ternary representations in neural nets
5. Monosemanticity (2023-24) — interpretable sparse features
6. Hyperdimensional Probe (2025) — post-hoc algebraic probing of LLMs

**Differentiating citations** (show what we do that others don't):
7. HASH-RAG (2025) — binary codes for retrieval, but no algebra
8. FactorHD (2025) — hierarchical HDC, but approximate not exact

---

## References

### Core Papers (Deep-Dived)
1. Mentzer et al. (2024). "Finite Scalar Quantization: VQ-VAE Made Simple." ICLR 2024. arXiv:2309.15505
2. Sun et al. (2025). "Concept Bottleneck Large Language Models." ICLR 2025. arXiv:2412.07992
3. Wang & Wang (2025). "Why Neural Networks Can Discover Symbolic Structures." NeuS 2025. arXiv:2506.21797
4. Bronzini et al. (2025). "Hyperdimensional Probe: Decoding LLM Representations via VSA." arXiv:2509.25045
5. Mejri et al. (2024). "LARS-VSA: Learning with Abstract Rules." arXiv:2405.14436

### Follow-Up / Related
6. iFSQ (Tencent, 2025). arXiv:2601.17124 — activation collapse fix for FSQ
7. Koh et al. (2020). "Concept Bottleneck Models." ICML 2020
8. Ma et al. (2024). "The Era of 1-bit LLMs." (BitNet b1.58). arXiv:2402.17764
9. Bricken et al. (2023). "Towards Monosemanticity." Anthropic. transformer-circuits.pub
10. Anthropic (2024). "Scaling Monosemanticity." transformer-circuits.pub
11. Cunningham et al. (2024). "SAEs Find Interpretable Features." ICLR 2024. arXiv:2309.08600
12. Yin et al. (2025). "Codebook-Centric Deep Hashing." arXiv:2511.12162
13. Guo et al. (2025). "HASH-RAG." Findings of ACL 2025
14. FactorHD (2025). arXiv:2507.12366
15. Frady et al. (2020). "Resonator Networks." Neural Computation
16. Alam et al. (2023). "Hrrformer." ICML 2023
17. Tian (2024). "Composing Global Optimizers." arXiv:2410.01779
18. Nanda et al. (2023). "Progress Measures for Grokking." arXiv:2301.05217
19. He et al. (2025). "Survey on Deep Text Hashing." arXiv:2510.27232
20. Overmann (2021). "Triadic Memory." peterovermann.com
