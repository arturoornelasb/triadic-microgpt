# Algebraic Operations — Formal Specification

**Date**: 2026-03-18
**Status**: Complete (8/8 formalized)

---

## Universe

Let **U** = {p₁, p₂, ..., p_k} be the first k primes (k = 63 for Danza, k = 64 for Run 15).

A **concept** Φ(x) is a squarefree positive integer whose prime factorization encodes active primitives:

```
Φ(x) = ∏ᵢ pᵢ^{bᵢ}    where bᵢ ∈ {0, 1}
```

The **full universe** Ω = ∏ᵢ pᵢ (product of all k primes). The **identity** is 1 (no active primitives).

The set of all valid concepts forms a **Boolean lattice** under divisibility, isomorphic to the power set 2^U.

---

## Operation 1: Subsumption (⊇)

**Definition**: A ⊇ B ⟺ B | A (B divides A)

**Notation**: `subsumes(A, B) → bool`

**Semantics**: A contains ALL semantic features of B. "King subsumes Male" means every feature of Male is present in King.

**Algebraic properties**:
- Reflexive: A ⊇ A
- Antisymmetric: A ⊇ B ∧ B ⊇ A → A = B
- Transitive: A ⊇ B ∧ B ⊇ C → A ⊇ C
- Ω ⊇ X for all X (universe subsumes everything)
- X ⊇ 1 for all X (everything subsumes identity)

**Complexity**: O(1) — single modulo operation.

**NSM connection**: Maps to NSM's KIND-OF relation. "A dog is a KIND OF animal" ≈ Φ(animal) | Φ(dog).

---

## Operation 2: Composition (⊔)

**Definition**: A ⊔ B = lcm(A, B)

**Notation**: `compose(A, B) → int`

**Semantics**: Create a concept with ALL features from BOTH inputs. The algebraic union.

**Algebraic properties**:
- Commutative: A ⊔ B = B ⊔ A
- Associative: (A ⊔ B) ⊔ C = A ⊔ (B ⊔ C)
- Idempotent: A ⊔ A = A
- Identity: A ⊔ 1 = A
- Absorption: A ⊔ Ω = Ω

**Complexity**: O(log min(A,B)) — Euclidean GCD.

**NSM connection**: Combines features. compose(THINK, FEEL) creates a concept with both cognitive and emotional primitives active.

---

## Operation 3: Intersection (⊓)

**Definition**: A ⊓ B = gcd(A, B)

**Notation**: `intersect(A, B) → int`

**Semantics**: Extract ONLY the features SHARED between two concepts. The common semantic backbone.

**Algebraic properties**:
- Commutative: A ⊓ B = B ⊓ A
- Associative: (A ⊓ B) ⊓ C = A ⊓ (B ⊓ C)
- Idempotent: A ⊓ A = A
- Identity: A ⊓ Ω = A
- Annihilator: A ⊓ 1 = 1 (if coprime)
- Distributive: A ⊓ (B ⊔ C) = (A ⊓ B) ⊔ (A ⊓ C)

**Complexity**: O(log min(A,B)) — Euclidean GCD.

**NSM connection**: "What do king and queen have in common?" → gcd(King, Queen) = shared royalty features. Related to NSM's THE SAME.

---

## Operation 4: Difference (∖)

**Definition**: A ∖ B = A / gcd(A, B)

**Notation**: `difference(A, B) → int`

**Semantics**: Features present in A but NOT in B. What makes A different from B.

**Algebraic properties**:
- NOT commutative: A ∖ B ≠ B ∖ A (in general)
- A ∖ A = 1 (nothing unique)
- A ∖ 1 = A (everything is unique relative to identity)
- (A ∖ B) * (A ⊓ B) = A (reconstruction)

**Complexity**: O(log min(A,B)).

**NSM connection**: Explains the gap. "What does king have that queen doesn't?" → Male-specific primitives. Maps to NSM's OTHER ("something other than...").

---

## Operation 5: Symmetric Difference (△)

**Definition**: A △ B = (A ∖ B) × (B ∖ A)

**Notation**: `symmetric_difference(A, B) → int`

**Semantics**: Features present in EXACTLY ONE of the two concepts. The total semantic distance as a composite.

**Algebraic properties**:
- Commutative: A △ B = B △ A
- A △ A = 1 (identical concepts have no difference)
- A △ 1 = A (all features are "different" from identity)
- Metric-like: satisfies triangle inequality on factor count

**Complexity**: O(log min(A,B)).

**NSM connection**: The full "difference" between two concepts. Related to explain_gap but returns a single composite encoding all distinguishing features.

---

## Operation 6: Analogy (R3)

**Definition**: Given A:B :: C:?, compute D = (C ∖ gcd(C, A∖gcd(A,B))) × (B∖gcd(A,B)) / gcd(...)

**Simplified**: Remove A-specific features from C, add B-specific features.

```
shared_AB = gcd(A, B)
only_A = A / shared_AB     # features to remove
only_B = B / shared_AB     # features to add
C_reduced = C / gcd(C, only_A)
D = lcm(C_reduced, only_B)
```

**Notation**: `analogy(A, B, C) → D`

**Semantics**: "A is to B as C is to D." Transfer the A→B transformation onto C.

**Algebraic properties**:
- analogy(A, B, A) = B (trivial case)
- analogy(A, A, C) = C (identity transformation)
- NOT necessarily invertible: analogy(B, A, D) may ≠ C (information loss)

**Complexity**: O(log max(A,B,C)).

**Empirical performance**: 98% verification accuracy (E3, 51 quads), 90.7% algebraic prediction (D-A5).

**NSM connection**: Captures the LIKE/AS relation. "King is to queen LIKE man is to woman."

---

## Operation 7: Negation (¬)

**Definition**: ¬A = Ω / A (where Ω = product of all k primes)

**Notation**: `negate(A, universe=Ω) → int`

**Semantics**: Concept with ALL features that A LACKS. The ontological complement.

**Algebraic properties**:
- Involution: ¬(¬A) = A
- De Morgan: ¬(A ⊔ B) = ¬A ⊓ ¬B
- De Morgan: ¬(A ⊓ B) = ¬A ⊔ ¬B
- ¬Ω = 1, ¬1 = Ω
- A ⊔ ¬A = Ω, A ⊓ ¬A = 1

**Complexity**: O(k) — division by each active prime.

**NSM connection**: NOT. The direct algebraic encoding of negation. But NOTE: semantic negation is not always set complement. "Not-hot" ≠ "has every feature except hot" in practice. This operation is *algebraically* correct but *semantically* approximate.

**Caveat**: Useful for formal completeness but should be used cautiously. In practice, the dual-axis structure of La Danza (bien/mal, vida/muerte) handles negation through explicit opposing primitives rather than set complement.

---

## Operation 8: Projection (π)

**Definition**: π_S(A) = ∏{pᵢ : pᵢ | A ∧ pᵢ ∈ S}

**Notation**: `project(A, category_primes) → int`

**Semantics**: Extract ONLY the features from a specific category (e.g., "show me only the sensory features of this concept").

**Categories in Sistema 7×7**:
- Layer 1 (Existence): bits 0-1, 44 → primes {2, 3, 197}
- Layer 2 (Comparison): bits 2, 7-10, 26-27, 59, 61-62 → primes {5, 19, 23, 29, 31, 103, 107, 281, 293, 307}
- Layer 3 (Causality): bits 11-12, 22-25, 48, 50, 54-55 → primes {37, 41, 83, 89, 97, 101, 227, 233, 257, 263}
- Layer 4 (Morality): bits 13-18, 20-21, 28-31, 60 → primes for moral/truth/freedom axes
- Layer 5 (Body): bits 3-6, 15-17, 19, 32-37, 40-43, 49, 51-53 → sensory, life, consciousness
- Layer 6 (Meta): bits 38-39, 42-43 → observer primitives
- Quantification: bits 44-47, 61-62 → uno, muchos, todo, algunos, más, menos

**Algebraic properties**:
- π_S(A ⊔ B) = π_S(A) ⊔ π_S(B) (homomorphism)
- π_S(A ⊓ B) = π_S(A) ⊓ π_S(B) (homomorphism)
- π_U(A) = A (full projection = identity)
- π_∅(A) = 1 (empty projection = identity element)
- π_S(π_T(A)) = π_{S∩T}(A) (composition of projections)

**Complexity**: O(|S|) — one modulo check per prime in the category.

**NSM connection**: Maps to domain-specific analysis. "What are the FEEL-related features of this concept?" or "What THINK-related primitives are active?"

---

## Summary Table

| # | Operation | Symbol | Formula | In Code | Formally Specified |
|---|-----------|--------|---------|---------|-------------------|
| 1 | Subsumption | A ⊇ B | B \| A | `subsumes()` | Yes |
| 2 | Composition | A ⊔ B | lcm(A, B) | `compose()` | Yes |
| 3 | Intersection | A ⊓ B | gcd(A, B) | `intersect()` | **NEW** |
| 4 | Difference | A ∖ B | A / gcd(A, B) | `difference()` | **NEW** |
| 5 | Symmetric Diff | A △ B | (A∖B)(B∖A) | `symmetric_difference()` | **NEW** |
| 6 | Analogy | A:B::C:? | R3 transform | `analogy()` | Yes |
| 7 | Negation | ¬A | Ω / A | `negate()` | **NEW** |
| 8 | Projection | π_S(A) | ∏{pᵢ∈S∩A} | `project()` | **NEW** |

**Algebraic completeness**: These 8 operations, together with the lattice structure (⊔, ⊓, ¬), form a **complete Boolean algebra** over the set of squarefree composites. This means any set-theoretic statement about semantic features can be expressed and verified algebraically.

---

## Similarity (Derived)

Similarity is not an independent operation but is **derived** from intersection and composition:

```
sim(A, B) = |factors(A ⊓ B)| / |factors(A ⊔ B)|    (Jaccard)
```

Already implemented as `similarity()`. Range [0, 1].

---

## References

- Birkhoff, G. (1967). *Lattice Theory*. AMS. (Boolean lattice structure)
- Wierzbicka, A. (1996). *Semantics: Primes and Universals*. (NSM framework)
- Ornelas Brand, A. (2026). Prime Factorization as a Neurosymbolic Bridge. (This work)
