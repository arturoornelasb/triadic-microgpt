# Algebraic Fact Verification via Prime Relational Chains

**Status**: Pending — documented, not started
**Date**: 2026-03-19
**Author**: Arturo Ornelas Brand

## Motivation

triadic-microgpt (Paper 1) achieves **ontological transparency**: each token's prime
composite Phi(x) reveals what semantic category it belongs to. But ontological
properties ("Paris is a place") are necessary yet insufficient to prevent
hallucinations. The dominant class of LLM hallucinations involves **factual
relations** ("Paris is the capital of Germany") that are ontologically plausible
but factually wrong.

The key insight: the same algebraic framework that verifies ontological
properties can be extended to verify factual triples — if we encode
**relations** as primes alongside **properties**.

## From Ontological Primes to Relational Primes

### Paper 1 (current): Ontological Level

Each concept x has a composite Phi(x) encoding what it IS:

```
Phi(king)   = p_alive * p_human * p_male * p_powerful * p_social
Phi(queen)  = p_alive * p_human * p_female * p_powerful * p_social
Phi(Paris)  = p_place * p_cultural * p_important * p_human
Phi(France) = p_place * p_large * p_order * p_social
```

Verification is modular arithmetic:
- king is alive? Phi(king) % p_alive == 0 -> YES, O(1)
- king subsumes human? Phi(king) % Phi(human) == 0 -> YES, O(1)

### Paper 2 (proposed): Relational Level

Define a separate set of **relational primes** {r_1, r_2, ...} disjoint from
property primes. Each relation gets its own prime:

```
r_capital    = q_1    (e.g., 313)
r_part_of    = q_2    (e.g., 317)
r_born_in    = q_3    (e.g., 331)
r_invented   = q_4    (e.g., 337)
r_spouse_of  = q_5    (e.g., 347)
r_language   = q_6    (e.g., 349)
...
```

A **factual triple** (subject, relation, object) is encoded as a composite:

```
T(Paris, capital_of, France) = Phi(Paris) * r_capital * Phi(France)
```

## Verification Protocol

### Triple Verification (O(1) per check)

Given a generated statement "X is the R of Y":

```python
def verify_triple(subject, relation, object, fact_base):
    """Check if a factual triple is known to be true."""
    triple_composite = Phi(subject) * r(relation) * Phi(object)
    return triple_composite in fact_base
```

### Inconsistency Detection

Given a statement "Paris is the capital of Germany":

```python
def detect_inconsistency(subject, relation, object, fact_base):
    """Check if a triple contradicts known facts."""
    candidate = Phi(Paris) * r_capital * Phi(Germany)

    # 1. Does this exact triple exist?
    if candidate in fact_base:
        return False  # consistent

    # 2. Does a COMPETING triple exist?
    #    (same subject + relation, different object)
    for known_triple in fact_base:
        if known_triple % (Phi(Paris) * r_capital) == 0:
            # Found: Paris IS capital of something else
            known_object = known_triple // (Phi(Paris) * r_capital)
            if known_object != Phi(Germany):
                return True  # INCONSISTENCY: Paris is capital of {known}, not Germany

    # 3. No information -> unknown (not inconsistent)
    return None
```

### Chain Verification (Multi-Hop)

Chains enable transitive reasoning:

```
T1: Paris capital_of France       -> Phi(Paris) * r_capital * Phi(France)
T2: France part_of Europe         -> Phi(France) * r_part_of * Phi(Europe)

Chain: Paris in Europe?
  GCD(T1, T2) contains Phi(France) -> they share a link
  T1 * T2 / Phi(France) encodes the transitive relation
  Verify: Phi(Paris) * r_capital * r_part_of * Phi(Europe) is derivable
```

This is **algebraic inference**: deriving new facts from existing ones using
prime arithmetic, with each step verifiable via modular division.

## What Already Exists (Building Blocks)

All core algebraic operations are implemented in `src/triadic.py`:

| Operation | Code | Role in Fact Verification |
|-----------|------|---------------------------|
| Composition: LCM(a,b) | `TriadicValidator.compose()` | Combine subject + relation + object |
| Subsumption: a % b == 0 | `TriadicValidator.subsumes()` | Check if triple contains a component |
| Intersection: GCD(a,b) | `TriadicValidator.intersect()` | Find shared links between chains |
| Difference: a // GCD(a,b) | `TriadicValidator.difference()` | Extract unknown component from chain |
| Analogy: C + (B - A) | Regla de Tres (Danza) | Relational transfer across domains |
| Negation: omega // a | `TriadicValidator.negate()` | Contrapositive verification |
| Projection | `TriadicValidator.project()` | Extract only relational primes from composite |

The Regla de Tres IS relational encoding:
- king:queen = man:woman encodes the "gender swap" relation
- cold:hot = quiet:loud encodes the "opposition" relation
- teach:learn = king:queen encodes a structural analogy

The step from analogical relations to factual relations is:
1. Make relation primes EXPLICIT (not implicit in vector arithmetic)
2. Store verified triples as composites
3. Check generated text against stored triples at inference time

## Architecture Proposal

```
             PAPER 1 (done)                    PAPER 2 (proposed)
           Ontological Level                  Relational Level

Text -> GPT Backbone -> Triadic Head -----> Relational Head
                              |                     |
                         Phi(token)            R(subj, rel, obj)
                         63 property           K relational primes
                         primes                + triple composites
                              |                     |
                         Ontological           Fact Store
                         Verification          (verified triples)
                              |                     |
                         "king IS alive"       "Paris IS capital OF France"
                              |                     |
                         Category errors       Factual errors
                         detected              detected
```

### Relational Head Design

```python
class RelationalHead(nn.Module):
    """Maps token-pair hidden states to relational prime composites."""

    def __init__(self, d_model, n_relations):
        super().__init__()
        # Bilinear interaction: captures subject-object relationship
        self.bilinear = nn.Bilinear(d_model, d_model, n_relations)
        # tanh activation (NOT sigmoid — same lesson as Paper 1)

    def forward(self, h_subject, h_object):
        # h_subject, h_object: hidden states from GPT backbone
        relation_logits = self.bilinear(h_subject, h_object)
        return torch.tanh(relation_logits)  # [-1, +1] per relation
```

### Fact Store

```python
class FactStore:
    """Knowledge base of verified triples as prime composites."""

    def __init__(self):
        self.triples = set()        # set of composite integers
        self.by_subject = {}        # subject_phi -> set of triples
        self.by_relation = {}       # relation_prime -> set of triples

    def add(self, subject_phi, relation_prime, object_phi):
        triple = subject_phi * relation_prime * object_phi
        self.triples.add(triple)
        self.by_subject.setdefault(subject_phi, set()).add(triple)
        self.by_relation.setdefault(relation_prime, set()).add(triple)

    def verify(self, subject_phi, relation_prime, object_phi):
        """O(1) membership check."""
        return (subject_phi * relation_prime * object_phi) in self.triples

    def find_conflicts(self, subject_phi, relation_prime, object_phi):
        """Find triples with same subject+relation but different object."""
        key = subject_phi * relation_prime
        conflicts = []
        for t in self.by_subject.get(subject_phi, set()):
            if t % key == 0 and t != key * object_phi:
                conflicts.append(t)
        return conflicts
```

## Training Strategy

### Phase 1: Ontological (Paper 1 — DONE)

Train Phi(x) for property primes end-to-end with language modeling.
Result: 63-bit ontological composites per token.

### Phase 2: Relational (Paper 2)

Two possible approaches:

**A. Supervised from Knowledge Graph:**
- Source: Wikidata, ConceptNet, or curated facts
- Extract (subject, relation, object) triples
- Train RelationalHead to predict relation primes given subject/object hidden states
- Loss: MSE on relation bits (same as Danza supervised loss)
- Advantage: ground truth available, directly measurable
- Disadvantage: coverage limited to training triples

**B. Self-Supervised from Text:**
- Parse subject-relation-object from training text using dependency parsing
- Train RelationalHead on extracted triples
- Use algebraic consistency as auxiliary loss (if A->B and B->C, then A->C)
- Advantage: scales with data
- Disadvantage: noisy extraction, no ground truth

**C. Hybrid (recommended):**
- Seed with ~500 gold triples from Wikidata (analogous to Danza's 50 anchors)
- Bootstrap additional triples from text (analogous to D-A6b bootstrap)
- Verify bootstrapped triples algebraically before accepting
- The D-A6b confidence gate mechanism transfers directly

## Anti-Hallucination Pipeline

At inference time:

```
1. Model generates token sequence
2. Dependency parser extracts (subject, relation, object) triples
3. For each triple:
   a. Compute Phi(subject) * r(relation) * Phi(object)
   b. Check FactStore.verify() -> O(1)
   c. If not verified, check FactStore.find_conflicts()
   d. If conflict found -> FLAG: factual inconsistency
   e. If no info -> WARN: unverified claim (not necessarily wrong)
4. Options on detection:
   a. Reject and regenerate
   b. Add caveat ("I'm not certain about...")
   c. Log for human review
```

### What This Catches vs. What It Misses

**Catches:**
- Direct factual contradictions ("Paris is the capital of Germany")
- Transitive inconsistencies ("Paris is in Asia" when chain Paris->France->Europe exists)
- Relation type errors ("Einstein painted the Mona Lisa" — wrong relation for subject)
- Ontological impossibilities ("the rock felt sadness" — Level 1 catch)

**Does NOT catch:**
- Novel true facts not in FactStore (treats as "unverified", not "wrong")
- Subtle numerical errors ("population of France is 68 million" vs 67.75 million)
- Temporal facts without timestamps ("X is president of Y" — may be outdated)
- Opinions or subjective statements (not factual triples)

## Advantages Over Existing Approaches

| Approach | Verification | Speed | Guarantees |
|----------|-------------|-------|------------|
| RAG (retrieval) | Probabilistic (cosine sim) | Slow (embedding search) | None |
| Chain-of-thought | Self-reported | Slow (extra generation) | None |
| Knowledge graph lookup | Exact match | Medium (graph traversal) | Completeness-dependent |
| **Prime relational chains** | **Algebraic (modular arithmetic)** | **O(1) per triple** | **Mathematical: a%b==0 is deterministic** |

The key differentiator: verification is not statistical or probabilistic.
`a % b == 0` is a **mathematical guarantee**, not a confidence score.
No threshold to tune, no false positive rate to manage (within the fact base).

## Estimated Scope

### Phase A: Post-hoc proof of concept (NO retraining, NO new head)

Uses existing checkpoint + ConceptNet triples. Demonstrates algebraic
verification works. This is what goes in the paper.

| Component | Effort | Dependencies |
|-----------|--------|--------------|
| Map ConceptNet relations → relational primes | 1 day | Domain knowledge |
| FactStore class in `src/triadic.py` | 1 day | Existing code |
| Build FactStore from ConceptNet for 158 anchors | 1 day | ConceptNet download |
| Verification experiment: generate → extract triples → check | 1-2 days | spaCy for extraction |
| Paper section "Relational Extension" (1-2 pages) | 1 day | Results |
| **Total Phase A** | **~5 days, 0 GPU** | |

### Phase B: End-to-end (future, AFTER paper submission)

| Component | Effort | Dependencies |
|-----------|--------|--------------|
| RelationalHead module | 2-3 days | `torch_transformer.py` |
| Training pipeline with triple loss | 1-2 weeks | Phase A checkpoint |
| Bootstrap loop for relational triples | 1 week | D-A6b mechanism |
| Evaluation on TruthfulQA / FEVER | 1 week | Benchmarks |

## Connection to Existing Work

### La Danza Cosmica Framework

The Danza's 63 primitives already include relational concepts:
- hacer (to do), tener (to have), ser (to be) — these ARE relations
- The triadic structure {+, 0, -} maps to {subject, relation, object}
- "Every triple is a dance between two poles and their mediator" — La Danza

### Regla de Tres as Proto-Relational Encoding

The Regla de Tres (A:B = C:D) already captures relational structure:
```
king:queen = man:woman    -> relation: "gender counterpart"
teach:learn = give:receive -> relation: "directional complement"
fire:water = love:hate    -> relation: "elemental opposition"
```

Paper 2 makes the implicit relation EXPLICIT as a prime, enabling:
- Storage in FactStore
- O(1) lookup
- Chain composition
- Conflict detection

### Bootstrap Mechanism (D-A6b)

The bootstrap loop that just succeeded for ontological primes transfers
directly to relational primes:
1. Seed with gold triples (supervised)
2. Model predicts unseen triples
3. Confidence gate filters high-quality predictions
4. Accepted triples become new training data
5. Iterate until convergence

D-A6b proved this works: 21 pseudo-anchors accepted in cycle 0, improving
from 25 to 52 training concepts. The same mechanism on relational triples
could bootstrap a fact base from a small seed.

## Key Research Questions

1. **Scalability**: How many relational primes before composites overflow int64?
   - With 63 property primes + 50 relation primes = 113 primes
   - Products require Python bigint (already the case for 63 primes)
   - Verification via modular arithmetic is still O(1) regardless of size

2. **Relation granularity**: "capital_of" vs "administrative_center_of" vs "seat_of_government_of"
   - Start coarse (20-30 relations), refine if needed
   - Similar to starting with 63 ontological primitives

3. **Compositionality**: Can complex facts be composed from simpler ones?
   - "Marie Curie won the Nobel Prize in Physics in 1903"
   - = T(Curie, won, Nobel) * T(Nobel, field, Physics) * T(event, year, 1903)
   - Chain verification: each link independently verifiable

4. **Open-world assumption**: What to do with facts not in FactStore?
   - Cannot assume "not verified" = "false" (closed-world fallacy)
   - Solution: three-valued output {verified, contradicted, unknown}
   - Maps naturally to ternary {+, 0, -} framework

## ConceptNet Relations → Relational Primes (Proposed Mapping)

Use primes beyond p_63 (the 64th prime = 311 onward) to avoid collision
with the 63 ontological primes:

| ConceptNet Relation | Relational Prime | Example |
|---------------------|-----------------|---------|
| IsA | q_1 = 313 | (dog, IsA, animal) |
| PartOf | q_2 = 317 | (wheel, PartOf, car) |
| HasProperty | q_3 = 331 | (fire, HasProperty, hot) |
| Causes | q_4 = 337 | (rain, Causes, wet) |
| UsedFor | q_5 = 347 | (knife, UsedFor, cut) |
| CapableOf | q_6 = 349 | (bird, CapableOf, fly) |
| AtLocation | q_7 = 353 | (book, AtLocation, library) |
| Desires | q_8 = 359 | (dog, Desires, food) |
| CreatedBy | q_9 = 367 | (painting, CreatedBy, artist) |
| Antonym | q_10 = 373 | (hot, Antonym, cold) |
| HasPrerequisite | q_11 = 379 | (cook, HasPrerequisite, heat) |
| MotivatedByGoal | q_12 = 383 | (study, MotivatedByGoal, learn) |
| CausesDesire | q_13 = 389 | (hunger, CausesDesire, eat) |
| MadeOf | q_14 = 397 | (table, MadeOf, wood) |
| ReceivesAction | q_15 = 401 | (ball, ReceivesAction, throw) |

15 relations cover ~85% of ConceptNet English triples.
Ontological primes use p_1..p_63 (primes 2..307).
Relational primes use p_64+ (primes 311+).
No collision by construction.

### Data Pipeline

```
ConceptNet5 (CSV, ~3.4M English triples)
    |
    v
Filter: keep only concepts present in our 158 anchors
    |  (reduces to ~2K-5K triples)
    v
Map: subject/object → Phi(x) from trained model
     relation → relational prime from table above
    |
    v
FactStore: set of {Phi(subj) * q_rel * Phi(obj)} composites
    |
    v
Verification experiment:
  1. Model generates 1000 sentences
  2. spaCy extracts (subj, rel, obj) triples
  3. Map to composites
  4. Check against FactStore
  5. Report: verified / contradicted / unknown
```

### Why ConceptNet and not Wikidata

- ConceptNet has COMMONSENSE relations (Causes, UsedFor, CapableOf)
  → these overlap with TinyStories vocabulary and Danza primitives
- Wikidata has ENCYCLOPEDIC facts (capitalOf, bornIn, population)
  → requires world knowledge the 40M model doesn't have
- ConceptNet triples are simpler: (dog, IsA, animal) vs (Q183, P36, Q64)
- Start with commonsense (ConceptNet), extend to encyclopedic (Wikidata) later
