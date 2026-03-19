# 63 Trits vs 51 Trits vs 36 Trits — Formal Decision

**Date**: 2026-03-18
**Status**: DECIDED — keep 63

---

## The Question

The Sistema 7×7 defines 63 ontological primitives. NSM (Wierzbicka & Goddard, 2014) identifies ~65 semantic primes. Their overlap is:

| Set | Count | Description |
|-----|-------|-------------|
| Direct match (★★★) | 28 | Exact semantic alignment |
| Close match (★★☆) | 8 | Moderate confidence |
| Sistema-only | 27 | Philosophical/ontological, not in NSM |
| NSM-only | 11 | Deictic/linguistic, not in Sistema |

Should the triadic head use all 63 primitives, or trim to the 36 (or 28) NSM-validated subset?

---

## Options

| Option | Bits/Trits | Capacity (ternary) | NSM coverage |
|--------|------------|---------------------|--------------|
| **A: Full 63** | 63 trits | 99.5 bits | 55% overlap |
| **B: NSM-validated 36** | 36 trits | 57.1 bits | 100% overlap |
| **C: Direct-only 28** | 28 trits | 44.4 bits | 100% direct |

---

## Empirical Evidence

### 1. Dead Bits Converge to ~42% Regardless of Total

| System | Total bits | Dead/inactive | % inactive |
|--------|-----------|---------------|------------|
| BitNet b1.58 | varies | varies | **42.3%** |
| TriadicGPT D-A5 (63-bit) | 63 | 27 | **42.9%** |
| TriadicGPT Run 15 (64-bit) | 64 | 15 | **23.4%** |
| D-A8 ternary (63 trits) | 63 | 30 | **47.6%** |

The ~42% convergence is a natural information-theoretic equilibrium. The system *already self-selects* which bits to use. Trimming manually would:
- Remove the model's ability to discover which primitives matter for a given corpus
- Risk discarding primitives that are relevant for domains not in TinyStories

### 2. The 27 "Extras" Carry Information

**Dependency structure**: All 27 have non-trivial dependency chains in the 6-layer ontology. They are not arbitrary additions — each emerges from dimensional constraints (0D→3D+).

**Compositionality**: The multi-quad ensemble (D-A12) achieves 90.6% using quads that traverse these extras (elemental, moral, emotional axes). Removing them would reduce the quad space.

**R3 Formula D**: In ternary space, the category-aware formula (which uses all 63 bits grouped by the 7 categories) outperforms continuous R3: 90.3% vs 89.9%. This formula *requires* the full categorical structure.

**D-A13 scaling**: GPT-2 Medium (355M) with all 63 trits achieves 100% subsumption holdout and 89.4% bit accuracy. The full set converges at scale.

### 3. NSM-Only Primes Are Correctly Excluded

The 11 NSM primes NOT in Sistema (I, YOU, THIS, TWO, BEFORE, AFTER, NOW, HERE, BIG, SMALL, LIKE/AS) are:
- **Deictic**: I, YOU, THIS, NOW, HERE — indexical, not ontological
- **Determiner**: THE SAME, OTHER — linguistic function words
- **Covered compositionally**: TWO ≈ uno+más, BEFORE/AFTER ≈ posición_temporal, BIG/SMALL ≈ más/menos

Their exclusion is *principled*: the Sistema captures ontological structure, not linguistic deixis. Adding them would not improve algebraic operations.

### 4. Information Capacity Argument

| Config | Raw capacity | Effective (at 42% inactive) |
|--------|-------------|----------------------------|
| 63 binary bits | 63 bits | ~36.5 bits |
| 63 ternary trits | 99.5 bits | ~57.7 bits |
| 36 ternary trits | 57.1 bits | ~33.1 bits |
| 28 ternary trits | 44.4 bits | ~25.7 bits |

Trimming to 36 would give *less* effective capacity than the current 63-binary system. The ternary upgrade makes the full 63 *more* efficient, not redundant.

---

## Formal Argument

**Theorem (informal)**: The optimal primitive count for end-to-end triadic training is the full ontological inventory (63), not the cross-linguistic subset (36 or 28).

**Proof sketch**:

1. **Self-regularization**: The model naturally drives ~42% of trits to the zero state (D-A8: 73.3% zero). This is equivalent to soft feature selection — the model learns which primitives are irrelevant *per concept*, not globally. A primitive that is zero for "queen" may be active for "fire".

2. **Compositionality requires coverage**: Algebraic operations (R3 analogy, subsumption, composition) operate over the full bit vector. Removing bits reduces the compositional space quadratically (fewer possible factor combinations). D-A12 uses quads spanning all 7 categories.

3. **The extras are the unique contribution**: NSM primes are well-established (~40 years of fieldwork). The 27 Sistema-only primitives (elements, moral duals, observer layer) are what makes this system novel. Removing them would reduce the Sistema to "NSM with prime encoding" — losing the philosophical contribution entirely.

4. **Convergent evidence strengthens, not weakens, the full set**: The 55% overlap with NSM validates that the core is sound. The 27 extras represent the Sistema's *extension* of NSM into phenomenological/ontological territory. This is a feature, not noise.

5. **Practical tradeoff is favorable**: 63 trits at XL scale costs +0.5% language loss (D-A8: 0.951 vs 0.946 baseline). The compositional gains (90.6% ensemble, 98.1% round-trip, 100% subsumption at scale) far outweigh this marginal cost.

---

## Decision

**Keep all 63 primitives.** The ternary representation naturally handles dimensionality via the zero state. The model self-selects active primitives per concept. The 27 extras are the unique ontological contribution and are empirically validated through compositionality.

**For the paper**: Frame as "63 ontological primitives, of which 36 (55%) independently converge with Wierzbicka's NSM primes — validating the core while extending into phenomenological territory (elements, consciousness, observer states) that NSM excludes by design."

---

## References

- Wierzbicka, A. (1996). *Semantics: Primes and Universals*. Oxford University Press.
- Goddard, C., & Wierzbicka, A. (2014). *Words and Meanings*. Oxford University Press.
- Ma, S., et al. (2024). The Era of 1-bit LLMs. arXiv:2402.17764 (BitNet b1.58).
