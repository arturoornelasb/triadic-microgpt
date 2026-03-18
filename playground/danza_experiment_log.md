# Danza Cósmica — Experiment Log

Expansion experiments connecting TriadicGPT with the 63-primitive system from
"La Danza Cósmica de los Opuestos" (Sistema 7×7 v3.4).

**This is SEPARATE from the paper's `experiment_log.md`.** The paper uses
self-supervised triadic training (no gold labels). These experiments use
supervised gold signatures from the book's manually-factorized anchor concepts,
and post-hoc analysis of self-supervised models to test whether the primitive
structure emerges independently — and ultimately whether it can BOOTSTRAP itself.

---

## Central Thesis: Bits as a Semantic Operating System

A normal LLM learns from raw text and must see billions of tokens to develop
implicit semantic structure. We propose a different path:

**If you give a model 63 named semantic primitives as inductive bias, it should:**

1. **Learn more with less** — the bit structure provides skeleton, data provides flesh
2. **Infer what it hasn't seen** — regla de tres predicts new concepts algebraically
3. **Self-validate** — subsumption, dual axes, and dependency constraints detect errors
4. **Bootstrap knowledge** — use algebraic inference to expand its own training set

The key claim: **50 hand-factorized concepts + algebraic constraints can SEED a system
that grows its own semantic knowledge without human intervention.**

### What this is NOT

This is not "training a bigger model" or "using more data". It's testing whether
a mathematical structure (63 bits, 12 dual axes, 6 dependency layers) provides
enough inductive bias to let a small model punch above its weight class.

---

## Research Roadmap

| Phase | Experiment | What it proves | GPU | Status |
|-------|-----------|---------------|-----|--------|
| **Baselines** | D-A1 Post-hoc analysis | Self-supervised bits find PARTIAL structure | 0 | **DONE** |
| | D-A2 Full supervised (D2) | Model CAN learn 63 primitives | 101 min | **DONE** ✅ |
| **Bootstrap** | D-A5 Half-anchor inference | Algebra predicts unseen concepts | ~76 min | **DESIGNED** |
| | D-A6 Bootstrap loop | System expands its own knowledge | ~3×76 min | **DESIGNED** |
| **Validation** | D-A3 Cross-lingual zero-shot | Bits transfer across languages | ~76 min | PLANNED |
| | D-A4 Unsupervised discovery | Primitives emerge without labels | ~76 min | PLANNED |
| **Scale** | D-A7 307M + bf16 + full data | Structure holds at scale | ~4h | PLANNED |

---

## Source Data

- **Primitives**: 63 semantic atoms from `danza_data/primitivos.json`
  - 6 dimensional layers (Point → Line → Time → Plane → Volume → Meta)
  - 12 dual axes (bien↔mal, orden↔caos, vida↔muerte, etc.)
  - Transitive dependency expansion (e.g., `consciente` expands to 18+ prereqs)
- **Anchors**: 50 manually-factorized concepts from `danza_data/anclas.json`
  - Each bit justified with written reasoning
  - Source: inventario_de_opuestos/toolkit in la-danza-cosmica-de-los-opuestos
  - Copied here for reproducibility (does NOT require the external repo)
- **Training corpus**: TinyStories (English, 50K stories)
- **Anchor translations**: Spanish → English mappings hardcoded in `danza_63bit.py`

---

## Known Biases & Limitations

### Gender-Element Mapping (CRITICAL)

The anchor system maps:
- `hombre` (man) → `tierra` (earth): solidity, foundation
- `mujer` (woman) → `agua` (water): fluidity, adaptation

This comes from the book's archetypal framework (classical elements, yin/yang
tradition). **This is a cultural/philosophical mapping, NOT a biological truth.**

Implications:
1. The system embeds gender essentialism from Western/Eastern mythological traditions
2. "Tierra" and "agua" carry connotations beyond the intended primitive definitions
3. A man:woman analogy that resolves via tierra↔agua will propagate this bias
4. The regla de tres `hombre:mujer = rey:reina` works algebraically (Hamming=0)
   but its correctness depends on accepting the archetypal gender mapping

**Future work**: Test alternative factorizations where gender differs on
social/biological primitives rather than elemental ones. Compare downstream
metrics to quantify the impact of this design choice.

### Other Biases

- `rico` (rich) includes `bien` (moral good) — conflates wealth with virtue
- `pobre` (poor) lacks `hacer` (agency) — implies poverty = passivity
- `lógico` includes `tierra` + `control` — associates logic with rigidity
- `creativo` includes `caos` — associates creativity with disorder

These reflect the book's philosophical framework, not empirical claims.
They should be treated as ONE POSSIBLE factorization, not ground truth.

### TinyStories Coverage

- 4 anchors skipped (won't appear in children's stories): estasis_absoluta,
  hombre_vaciado, inercia_mental, amoral
- Many anchors tokenize to multiple BPE tokens (50/54), which means
  supervision is via mean-pooled multi-token representations
- English translations may not perfectly capture Spanish concept boundaries
  (e.g., "still" for "inmóvil" is ambiguous)

---

## Research Question: Can the Primitive Structure Self-Generate?

The Sistema 7×7 was designed by a human (50 hand-factorized anchors, 63
primitives with dependency chains, 12 dual axes). The central question:

**Is this structure a human construction, or a real property of semantic space
that neural models can discover independently?**

### What's already proven

1. **Self-supervised Run 15** (no gold labels): discovers 64 anonymous bits that
   capture semantic gap (+0.038), 98% analogy verification, 92-100% subsumption.
   The bits WORK but we don't know what they "mean".

2. **Supervised concept_gpt_49bit** (462 seed words): learns to reproduce 49
   named primitives at 86.2% accuracy. The bits CAN be grounded to named primitives.

3. **Unsupervised clustering** (500 concepts × 51 primitives): K-means discovers
   15 natural categories that cross human taxonomy. The 4 elements (fuego, agua,
   tierra, aire) emerge as natural attractors. Structure exists in the data.

### The path to auto-generation

See **Research Roadmap** at top of document for full experiment list and status.

---

## D-A1: Post-Hoc Analysis of Self-Supervised Bits (0 GPU)

### Purpose

Run 15 learned 64 bits without ANY supervision about what those bits should mean.
If the bit structure independently mirrors the Sistema 7×7 properties (dual axes,
dependency hierarchy, abstraction gradient), then the primitives are REAL, not arbitrary.

### Script: `playground/danza_posthoc_analysis.py`

### Tests

#### Test A1.1 — Emergent Dual Axes (anti-correlation)

**Hypothesis**: The 12 dual axes of the Sistema 7×7 predict that certain semantic
dimensions are mutually exclusive. If self-supervised bits independently discover
anti-correlated pairs, it validates the dual-axis structure.

**Method**:
1. Load Run 15 checkpoint (64 bits, self-supervised)
2. Run 113 evaluation concepts through the model → 113 × 64 bit matrix
3. Compute bit-bit correlation matrix (64 × 64)
4. Find the N most anti-correlated bit pairs (correlation < -0.3)
5. For each anti-correlated pair, examine which concepts activate each bit
6. Check if anti-correlated pairs correspond to semantic oppositions

**Expected if primitives are real**: We should find ~10-15 strongly anti-correlated
bit pairs, and the concepts activating each side should map to known oppositions
(e.g., one bit fires for positive emotions, its anti-pair fires for negative ones).

**Expected if primitives are arbitrary**: Anti-correlations will be sparse, random,
and not map to recognizable semantic categories.

**Metrics**:
- N pairs with correlation < -0.3
- N pairs that map to recognizable dual axes (manually judged)
- Alignment score: (recognizable dual pairs) / (total anti-correlated pairs)

#### Test A1.2 — Emergent Dependency Hierarchy (conditional activation)

**Hypothesis**: The Sistema 7×7 has a strict dependency hierarchy — `consciente`
requires `vida` requires `creación` requires `hacer`, etc. If self-supervised bits
independently show that some bits ONLY activate when other bits are already active,
it validates the layered dependency structure.

**Method**:
1. Same 113 × 64 bit matrix from A1.1
2. For each bit pair (i, j), compute P(bit_j=1 | bit_i=1) and P(bit_j=1 | bit_i=0)
3. If P(j|i) >> P(j|¬i) AND P(i|j) ≈ P(i|¬j), then j depends on i (not vice versa)
4. Build a directed dependency graph from these asymmetric conditionals
5. Check if this graph has layers (DAG depth) matching the 6 dimensional layers

**Expected if primitives are real**: A DAG with 4-6 depth levels. Foundation bits
(highly common, few dependencies) correspond to Layer 1-2 primitives. Deep bits
(rare, many prerequisites) correspond to Layer 5-6.

**Expected if primitives are arbitrary**: Flat graph, no clear hierarchy, random
conditional activation patterns.

**Metrics**:
- DAG depth (expected: 4-6 if structure matches)
- Bit activation frequency distribution (expected: bimodal — foundation vs. specific)
- Correlation between DAG depth and bit activation frequency (expected: negative)

#### Test A1.3 — Abstraction Gradient (activation frequency)

**Hypothesis**: In the Sistema 7×7, abstract primitives (vacío, información, uno)
activate for 70-80% of concepts, while specific ones (gusto, olfato, tal_vez)
activate for <5%. If self-supervised bits show a similar gradient, it validates
the abstraction layering.

**Method**:
1. Same 113 × 64 bit matrix
2. Compute per-bit activation frequency across all concepts
3. Sort bits by frequency → activation histogram
4. Compare shape to the known primitive frequency distribution from the 500-concept
   dataset (where Observador.Presencia = 78.6%, Porque = 0.4%)

**Expected if primitives are real**: Skewed distribution — a few bits activate for
>60% of concepts (foundation layer), most bits in 10-40% range (specific features),
a tail of rare bits <5%.

**Expected if primitives are arbitrary**: Uniform distribution around 50% (entropy
maximization from the diversity loss would push all bits toward 50/50).

**Metrics**:
- Distribution skewness
- Number of "foundation" bits (>60% activation)
- Number of "rare" bits (<10% activation)
- KL divergence between self-supervised distribution and the Sistema 7×7 distribution

#### Test A1.4 — Semantic Probing (concept → bit alignment)

**Hypothesis**: Even without supervision, some self-supervised bits might correspond
to recognizable semantic dimensions. We can probe this by checking if specific bits
activate consistently for concepts in the same semantic category.

**Method**:
1. Group the 113 concepts by category (13 categories: animal, person, feeling, etc.)
2. For each bit, compute category purity: does it activate predominantly for one
   category or spread uniformly?
3. Find bits with high purity (>70% of activations from 1-2 categories)
4. Name these bits by their dominant category
5. Compare to the named primitives (e.g., if a bit activates for all emotions,
   does it correspond to `consciente` + `interocepción`?)

**Expected if primitives are real**: 10-20 bits with high category purity, mapping
to recognizable dimensions (emotion, spatial, moral, temporal, sensory).

**Metrics**:
- N bits with purity > 70%
- Named bit → primitive alignment (manual judgment)
- Mean purity across all 64 bits

#### Test A1.5 — Regla de Tres Transfer (algebraic structure)

**Hypothesis**: If the self-supervised bits have discovered the same algebraic
structure as the primitives, then the regla de tres should work on the self-supervised
bits WITHOUT any primitive supervision. We already know it works (98% analogy
verification), but now we test if the TRANSFORMATIONS align.

**Method**:
1. Compute self-supervised bit vectors for: hombre, mujer, rey, reina
2. Compute the transform: bits_only_in_mujer - bits_only_in_hombre
3. Identify which bits flip (the "gender transform" in self-supervised space)
4. Apply same transform to rey → predicted_reina
5. Compare predicted_reina to actual reina bits
6. ALSO: check if the flipping bits in the self-supervised gender transform
   correspond to any of the 12 dual axes

**Expected if primitives are real**: The gender transform flips 1-3 bits consistently,
and those bits map to a recognizable semantic dimension (though not necessarily
tierra↔agua — the model might find a different, perhaps less biased, factorization).

**Metrics**:
- Hamming distance of the transform
- Consistency across analogy quads (same bits flip for man:woman, king:queen, boy:girl)
- Alignment with known dual axes (manual judgment)

### Results (2026-03-17)

**Script**: `playground/danza_posthoc_analysis.py` (CPU, ~30s)
**Checkpoint**: Run 15 (64 self-supervised bits, 40M params, loss 0.946)
**Concepts**: 102 encoded (90 in 12 domains, 12 extra for analogies)
**Method**: sentence-level aggregation (3 sentences/concept, mean-pool target tokens)

| Test | Key Metric | Value | Expected (real) | Expected (arbitrary) | Verdict |
|------|-----------|-------|-----------------|---------------------|---------|
| A1.1 | Anti-correlated pairs (r<-0.3) | **22** | ~10-15 | <5 | **PASS** |
| A1.2 | DAG depth | **2** | 4-6 | 0-1 | WEAK |
| A1.3 | Foundation bits (>60%) | **20** | skewed | uniform | MIXED |
| A1.4 | Bits with >50% purity | **1** | 10-20 | <5 | FAIL |
| A1.5 | Regla de tres bit accuracy | **70.6%** | >90% | ~50% | WEAK |

#### A1.1 — Dual Axes: PASS

22 anti-correlated bit pairs (r < -0.3). Strongest: bit 37↔39 (r=-0.55). The model
independently discovers mutually exclusive semantic dimensions. Notably, bit 28 has only
3 activations (love, hate, nose) but shows the strongest semantic coherence — emotions
isolate in their own bit, just like the Sistema's `consciente` + `interocepción`.

Correlation statistics: mean=0.00, std=0.12, 111 pairs below -0.2, 9 pairs above +0.3.

#### A1.2 — Hierarchy: WEAK

61 directed dependency edges found, but DAG depth is only 2 (vs expected 4-6).
The model finds SOME conditional activation (e.g., bit 28→13 with strength 0.89),
but doesn't reconstruct the full 6-layer hierarchy. The depth↔frequency correlation
is essentially zero (r=-0.004).

**Interpretation**: The self-supervised model discovers dependencies but compresses them
into fewer layers. The 6-layer hierarchy may be an artifact of human conceptual layering
rather than an inherent property of the embedding space.

#### A1.3 — Abstraction: MIXED

20 bits fire for >60% of concepts (foundation), 6 bits for <10% (specific).
Mean frequency 48.4%, NOT the expected 50% uniform distribution from entropy maximization.
But skewness is slightly negative (-0.12), not the positive skew of the Sistema.
KL divergence from uniform = 7.68 — significantly non-uniform, but shaped differently
than the Sistema's distribution.

**Interpretation**: The model differentiates between general and specific features,
but the gradient has a different shape than the human-designed hierarchy.

#### A1.4 — Semantic Probing: FAIL

Only 1 bit with >50% category purity (bit 28 → emotions, n=3). Mean purity 0.189
vs chance 0.083 — 2.3× above chance, so there IS structure, but bits don't map
cleanly to human-recognizable semantic categories.

**Interpretation**: Self-supervised bits capture statistical regularities, not
human-interpretable categories. This is expected — without named labels, the model
has no reason to align with human taxonomies.

#### A1.5 — Regla de Tres: WEAK

Mean bit accuracy 70.6% (all quads FAIL the >90% threshold that supervised training
achieves). BUT the gender transform is **consistent**: 7 bits (5, 6, 12, 20, 21, 47, 63)
flip across ALL 4 gender quads (man:woman, father:mother, king:queen, prince:princess).

**Interpretation**: The algebraic structure EXISTS in self-supervised bits but is noisier
than supervised versions. The model captures SOMETHING about gender that transfers
algebraically, but with 27 bits flipping (vs 1-3 expected), the transform is distributed
across too many dimensions.

#### Overall Conclusion

The self-supervised model discovers **partial structure** that mirrors the Sistema 7×7:
- Dual axes emerge independently (22 anti-correlated pairs)
- Gender transforms are consistent across analogy families (7 common bits)
- Foundation/specific gradient exists (20 foundation, 6 rare)

But it does NOT reconstruct:
- The deep dependency hierarchy (depth 2 vs 6)
- Category-pure bits (no clean primitives)
- Clean algebraic transforms (70.6% vs 97.4%)

**The Sistema 7×7 provides a USEFUL FRAMEWORK for organizing structure that neural
models partially discover, but is not a structure that emerges COMPLETE without
human design.** Supervised training (D-A2) is needed to ground the bits to specific
primitives.

---

## D1: Smoke Test (base, 100 steps) ✅ DONE

**Date**: 2026-03-17
**Script**: `playground/danza_63bit.py --scale base --steps 100 --stories 5000`
**Time**: 5 seconds

### Results

| Metric | Train | Test |
|--------|-------|------|
| Bit accuracy | 90.0% | 88.7% |
| Subsumption | 88.4% | 90.0% |
| Dead bits | 61/63 | — |
| Entropy | 0.026 | — |

**Regla de Tres** (all 6 quads):

| Quad | Cosine | Bit Acc |
|------|--------|---------|
| man:woman = king:queen | +0.978 | 96.8% |
| cold:hot = quiet:loud | +0.985 | 96.8% |
| happy:sad = love:hate | +0.980 | 100.0% |
| open:close = free:prisoner | +0.978 | 98.4% |
| bright:dark = loud:quiet | +0.970 | 95.2% |
| teach:learn = king:queen | +0.955 | 96.8% |
| **Mean** | **+0.974** | **97.4%** |

### Verdict

Script validated. Ready for full XL run.

---

## D2 / D-A2: Full XL Run ✅ DONE

**Date**: 2026-03-17
**Script**: `playground/danza_63bit.py --scale xl --steps 50000`
**Time**: 100.8 min
**Checkpoint**: `checkpoints/danza_63bit_xl/`

### Results

| Metric | Train | Test | Target | Verdict |
|--------|-------|------|--------|---------|
| Bit accuracy | 100.0% | **89.5%** | >85% | **PASS** |
| Subsumption | 100.0% | **90.0%** | >90% | **PASS** |
| Dead bits | 27/63 | — | <15 | **FAIL** |
| Entropy | 0.392 | — | — | — |
| Best test accuracy | — | **90.5%** | — | — |

**Regla de Tres** (all 6 quads):

| Quad | Cosine | Bit Acc |
|------|--------|---------|
| man:woman = king:queen | +0.917 | 90.5% |
| happy:sad = love:hate | +0.898 | 90.5% |
| open:close = free:prisoner | +0.817 | 88.9% |
| teach:learn = king:queen | +0.761 | 85.7% |
| cold:hot = quiet:loud | +0.619 | 79.4% |
| bright:dark = loud:quiet | +0.531 | 77.8% |
| **Mean** | **+0.757** | **85.4%** |

**Per-concept accuracy**:
- Worst test: loud (78%), liquid (84%), still (87%)
- Best test: slow (92%), happy (94%), sun (98%)

### Analysis

**Bit accuracy (89.5%)**: PASS. Comparable to concept_gpt_49bit's 86.2% with 14 MORE
bits (63 vs 49). The additional bits (elements, senses, meta-layer) are learnable.

**Subsumption (90.0%)**: PASS. Barely meets threshold. The dependency hierarchy is
mostly respected — the model learns that `consciente` requires `vida` requires
`creación` etc.

**Dead bits (27/63)**: FAIL. 43% of bits are dead (entropy < 0.3). This is the same
pattern as earlier runs — diversity loss alone isn't enough to activate all bits.
The dead bits are likely the rare/specific primitives (gusto, olfato, tal_vez) that
activate for very few anchor concepts. With only 50 anchors, rare primitives get
very few supervision signals.

**Regla de tres (85.4%)**: FAIL vs 95% target, but PASS vs the self-supervised
baseline (70.6% from D-A1). Supervision improved algebraic transfer by +14.8%.
The weak quads (loud/quiet at 77-79%) involve concepts with poor TinyStories coverage.

**Comparison to smoke test (D1)**: Regla de tres DROPPED from 97.4% (100 steps) to
85.4% (50K steps). This is the known XL overfitting pattern — auxiliary losses degrade
in the second half of training. Early stopping at ~25K steps may yield better R3.

### Verdict

The model learns 63 real primitives at 89.5% accuracy — sufficient for the bootstrap
experiment (D-A5). The algebraic structure is noisy (85.4% R3) but has strong signal.
Dead bits are a concern but won't block bootstrap testing since the active bits
carry the semantic information.

**D-A5 can proceed with this checkpoint as baseline.**

---

## D-A3: Cross-Lingual Zero-Shot (PLANNED)

**Purpose**: Train on English TinyStories with 50 English anchors. Then evaluate
on Spanish words NEVER seen during training. If "amor" produces the same bits as
"love" without any Spanish training data, the system auto-generates cross-lingual
representations.

**Method**:
1. Train D2 model (English, 63 bits, 50 anchors)
2. Create a Spanish evaluation set: the 50 anchor words in Spanish
3. Tokenize Spanish words (the BPE tokenizer was trained on English, so Spanish
   words will be split into character-level tokens)
4. Run through model → get 63-bit signatures
5. Compare Spanish signatures to English gold signatures via Hamming distance

**Challenge**: BPE tokenizer is English-only. Spanish words will be poorly tokenized.
Options: (a) accept noisy tokenization, (b) retrain tokenizer on mixed corpus,
(c) use a multilingual tokenizer (loses comparability).

**Success criteria**: Mean Hamming < 10 between English and Spanish versions of
the same concept (out of 63 bits = 84% match).

---

## D-A4: Fully Unsupervised Primitive Discovery (PLANNED)

**Purpose**: Train TriadicGPT with k=63 bits and NO supervision (no anchors,
no supervised loss — only diversity + contrastive + entropy + alignment, like Run 15).
Then analyze whether the discovered bits correspond to the Sistema 7×7 primitives.

**Method**:
1. Train self-supervised model with k=63 bits (same as Run 15 but k=63 instead of 64)
2. Run ALL 50 anchor words through the model
3. For each anchor, get the 63-bit signature
4. Find the optimal bit permutation that maximizes alignment with gold signatures
   (Hungarian algorithm on the 63×63 alignment matrix)
5. After permutation, measure per-anchor bit accuracy

**This is the definitive test**: if a model with NO knowledge of the primitives
independently produces bits that, after relabeling, match the gold factorizations
at >70%, then the Sistema 7×7 is an empirically discoverable structure, not just
a philosophical construction.

**Success criteria**:
- Post-permutation bit accuracy > 70% (strong evidence)
- Post-permutation bit accuracy > 50% (moderate evidence, above chance)
- Identifiable dual axes in anti-correlation matrix (from D-A1)

---

## D-A5: Half-Anchor Algebraic Inference (THE KEY TEST)

### Purpose

This is the central experiment of the bootstrap thesis. If a model trained on
HALF the anchors can algebraically PREDICT the other half, then the 63-bit system
is generative — it doesn't just describe, it PRODUCES new knowledge.

### Method

#### Phase 1 — Strategic Split

Split the 50 anchors into 25 TRAIN and 25 HOLDOUT, with a critical constraint:
the holdout set must contain concepts that are REACHABLE via regla de tres from
the training set.

**Reachable holdout examples**:
- Train: man, woman, king → Holdout: queen (predicted via regla de tres)
- Train: happy, sad, love → Holdout: hate (predicted via emotional opposition)
- Train: hot, cold, fire → Holdout: water (predicted via elemental opposition)
- Train: alive, dead, good → Holdout: evil (predicted via moral opposition)

**Unreachable holdout examples** (controls):
- Concepts with no algebraic path from training set (e.g., `liquid` if no
  elemental pairs are in training)
- These should FAIL — if they succeed, something is wrong

The split must be pre-registered (chosen before seeing results) to prevent
cherry-picking. Use a deterministic algorithm:
1. Build regla de tres graph from all 50 anchors
2. For each potential split, count how many holdout concepts are reachable
3. Choose the split that maximizes reachable holdout concepts
4. Record which holdout concepts are "algebraically reachable" vs "unreachable"

#### Phase 2 — Train with Partial Supervision

```bash
python playground/danza_bootstrap.py --phase train --train-anchors 25 --steps 50000
```

- Train D2 model but with ONLY 25 anchor supervision signals
- The model still sees all TinyStories text (language modeling loss on everything)
- The model sees the holdout WORDS in text, but gets NO bit supervision for them
- All other losses remain: diversity, contrastive, entropy, alignment, subsumption

#### Phase 3 — Algebraic Prediction

After training:
1. Encode all 50 anchor words through the model → get 50 × 63 bit vectors
2. For each holdout concept, attempt algebraic inference:
   a. **Direct encoding**: just read the model's bits for the holdout word
   b. **Regla de tres**: predict from known anchors
      - queen = king XOR (man XOR woman)  [bit-flip from man→woman applied to king]
   c. **Subsumption inference**: if `amor ⊇ placer` and we know amor's bits,
      we know placer must have a subset of amor's bits
   d. **Dual axis inference**: if we know `happy` and the happy↔sad dual axis,
      we can predict `sad` by flipping the axis bits
3. Compare ALL predictions to gold signatures

#### Phase 4 — Measurement

**Primary metric**: Holdout bit accuracy
- Direct encoding accuracy: model just encodes the word → compare to gold
- Algebraic prediction accuracy: regla de tres / dual axis → compare to gold
- Algebraic IMPROVEMENT: |algebraic_acc - direct_acc|

**Secondary metrics**:
- Reachable vs unreachable holdout accuracy (should be significantly different)
- Per-concept breakdown: which anchors are easy/hard to predict?
- Subsumption consistency: do predicted signatures respect the dependency hierarchy?

### Success Criteria

| Metric | Threshold | What it means |
|--------|-----------|---------------|
| Holdout direct encoding > 75% | The model learned useful bits even without supervision |
| Algebraic prediction > 80% | The algebra can predict unseen concepts |
| Algebraic > direct + 5% | The algebra ADDS information beyond what the model learned |
| Reachable > unreachable + 10% | The algebra targets the right concepts |

### Script: `playground/danza_bootstrap.py` (TO BE WRITTEN)

---

## D-A6: Bootstrap Loop (Self-Improvement)

### Purpose

If D-A5 shows that algebraic prediction works, D-A6 tests whether the system
can ITERATIVELY expand its own knowledge. This is the self-improvement test.

### Method

```
Cycle 0: Train with 25 gold anchors
         → Evaluate all 50 concepts
         → Algebraically predict holdout signatures
         → Accept predictions with confidence > threshold
         → Metric: holdout accuracy

Cycle 1: Train with 25 gold + N pseudo-anchors from Cycle 0
         → Evaluate remaining holdout concepts
         → Predict more
         → Metric: holdout accuracy (should INCREASE)

Cycle 2: Train with 25 gold + N₁ + N₂ pseudo-anchors
         → ...

Convergence: When no new predictions pass the confidence threshold
```

### Confidence Gating

A predicted signature is accepted as a pseudo-anchor only if ALL of:
1. **Bit certainty**: all 63 tanh outputs are > 0.8 or < -0.8 (no ambiguous bits)
2. **Subsumption consistency**: the predicted signature respects known subsumptions
3. **Dual axis consistency**: if the concept has a known dual, the predicted bits
   flip on the correct axes
4. **Cross-validation**: at least 2 independent algebraic paths (e.g., regla de tres
   AND dual axis) agree on the prediction

### Success Criteria

| Metric | Threshold | What it means |
|--------|-----------|---------------|
| Accuracy increases per cycle | The system genuinely self-improves |
| Converges to > 85% on all 50 | Bootstrapping recovers most of the gold structure |
| Cycle 0 → Cycle 2 improvement > 10% | Each iteration adds real value |
| No false positives in pseudo-anchors | Confidence gating prevents error propagation |

### Risk: Error Cascading

If Cycle 0 accepts a WRONG prediction, Cycle 1 trains on a wrong signal, which
could propagate errors. The confidence gating is designed to prevent this, but
we must monitor:
- Per-cycle accuracy (must be monotonically increasing)
- If accuracy drops, the gate threshold is too low
- Fallback: use only regla de tres predictions (highest algebraic certainty)

### Script: `playground/danza_bootstrap.py --phase bootstrap --cycles 5`

---

## D-A7: Scale Test (PLANNED — after Bootstrap proof)

### Purpose

If D-A5/D-A6 prove the bootstrap thesis at 40M params with 50K stories,
test whether the effect AMPLIFIES with:
- **Model**: 307M params (24L/1024D/16H) — 7.7× larger
- **Data**: all 2.1M TinyStories (470M tokens) — 42× more data
- **Precision**: bfloat16 on Blackwell Tensor Cores

### Hypothesis

A larger model with more data should:
1. Learn better base representations → higher direct encoding accuracy
2. The algebraic IMPROVEMENT should be maintained or grow
3. Bootstrap loop should converge faster (fewer cycles)

### Hardware

```
GPU: RTX 5060 Ti 16GB (Blackwell, compute 12.0)
Model: 307M params = ~10 GB VRAM at batch_size=32 with bf16
Data: 2.1M stories = ~470M tokens
Estimated time: ~4 hours
```

### Success Criteria

- Holdout direct encoding > 85% (vs D-A5 baseline)
- Bootstrap converges in ≤ 3 cycles (vs D-A6 baseline)
- All D-A2 criteria maintained (bit acc, regla de tres, subsumption)

---

## Reproducibility

### Requirements
```
conda activate triadic-microgpt
# Python 3.10, PyTorch CUDA 12.8
# GPU: RTX 5060 Ti 16GB (Blackwell) or equivalent
```

### Data files (included in repo)
```
playground/danza_data/
  primitivos.json    # 63 primitives (from La Danza toolkit v1.0)
  anclas.json        # 50 gold anchors (manually factorized)
```

### Scripts
```bash
# D-A1: Post-hoc analysis (0 GPU, ~30s)
python playground/danza_posthoc_analysis.py

# D-A2: Supervised training (smoke test, ~5s)
python playground/danza_63bit.py --scale base --steps 100 --stories 5000

# D-A2: Supervised training (full XL, ~76 min)
python playground/danza_63bit.py --scale xl --steps 50000

# D-A5: Bootstrap half-anchor test (TO BE WRITTEN)
python playground/danza_bootstrap.py --phase train --train-anchors 25

# D-A6: Bootstrap loop (TO BE WRITTEN)
python playground/danza_bootstrap.py --phase bootstrap --cycles 5

# D-A7: Scale test (307M, bf16, all data)
python playground/danza_63bit.py --scale xxl --steps 50000 --stories 0
```

### Source
Data originates from: github.com/arturoornelasb/la-danza-cosmica-de-los-opuestos
- `inventario_de_opuestos/toolkit/primitivos.json`
- `inventario_de_opuestos/toolkit/anclas.json`
- Toolkit tests: `inventario_de_opuestos/toolkit/tests.py` (35/35 pass)
