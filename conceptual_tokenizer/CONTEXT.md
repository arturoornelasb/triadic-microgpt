# Project Context — Conceptual Tokenizer

> This document captures all the context needed to continue development.
> Read it before working on the tokenizer.

---

## What this is

A tokenizer that converts text into **concepts** (not into statistical fragments like BPE). Each word is decomposed into the 49 primitives of the 7x7 System, each primitive has an assigned prime, and relationships are verified with exact arithmetic.

## Current state (2026-03-14)

### Completed (Phases 1-3, no GPU)

| File | What it does |
|------|--------------|
| `config.py` | 49 primitives, primes (2..227), categories, thresholds |
| `primitives.py` | ConceptToken, ConceptSequence, subsumes(), compose(), gap() |
| `states.py` | StateResolver: continuous projection [-1,1] -> [+]/[0]/[null] + intensity + polarity |
| `seed_lexicon.py` | 462 words mapped (349 T1 + 95 T2 + 18 T3), 49/49 primitives used |
| `prime_encoder.py` | Resolved states -> composite primes (dual-channel: active + zero) |
| `triadic_bridge.py` | Connects ConceptTokens to the TriadicValidator from `src/triadic.py` |

### Pending (Phases 4-6, requires GPU)

| Phase | What's missing | Files to create |
|-------|----------------|-----------------|
| 4 | Contextual encoder + Projection Head (49-dim, sigmoid+annealing) | `encoder.py`, `projection_head.py` |
| 5 | End-to-end tokenizer + supervised training | `tokenizer.py`, `training/trainer.py`, `training/losses.py` |
| 6 | Self-supervised on natural text at 40M+ scale | `training/dataset.py` |

---

## Technical decisions made (with evidence)

| Decision | Chosen | Evidence |
|----------|--------|----------|
| Activation | **Sigmoid + annealing** (not tanh) | `playground/soft_signatures.py`: gap +0.039 vs tanh |
| Main loss | **Subsumption loss** | `playground/subsumption_loss.py`: breakthrough, forces bit inheritance |
| Rule of Three loss | **Only after subsumption, or don't use** | `playground/r3_subsumption_combo.py`: R3 erases subsumption gains |
| Granularity | **Whole words** (not BPE subwords) | `playground/concept_tokenizer.py`: BPE silhouette -0.059, not semantic |
| Minimum scale | **40M params** | `playground/random_baseline.py`: semantic ordering emerges only at this scale |
| Number of bits | **49** (one per primitive of the 7x7) | Instead of the arbitrary 32/64 from the current model |
| States | **Dual-channel** (active_composite + zero_composite) | Preserves distinction between silence vs stone-that-doesn't-hear |

---

## The 7x7 System v2.1 — The 49 primitives

```
Cat 1 ELEMENTS:         Fuego=2, Tierra=3, Agua=5, Aire=7, Vacío=11, Información=13, Fuerza=17
Cat 2 CHARACTERISTICS:  Color=19, Textura=23, Forma=29, Material=31, Brillo=37, Transparencia=41, Estado=43
Cat 3 SPACE:            Arriba=47, Abajo=53, EnMedio=59, Adelante=61, Atrás=67, IzqDer=71, DentroFuera=73
Cat 4 TIME:             Presente=79, Pasado=83, Futuro=89, Pausa=97, IrPasado=101, IrFuturo=103, Play=107
Cat 5 SENSES:           Vista=109, Oído=113, Tacto=127, Gusto=131, Olfato=137, Equilibrio=139, Interocepción=149
Cat 6 PRINCIPLES:       BienMal=151, OrdenCaos=157, CreaciónDestrucción=163, UniónSeparación=167,
                        VerdadMentira=173, LibertadControl=179, VidaMuerte=181
Cat 7 OBSERVERS:        Consciente=191, Temporal=193, Eterno=197, Individual=199, Colectivo=211,
                        Ausente=223, Creador=227
```

### 3 states
- `[+]` Active: prime in active_composite, projection > 0
- `[0]` Zero (active absence): prime in zero_composite, projection < 0
- `[null]` N/A: prime absent, |projection| < threshold (0.1)

### 6 operations
Fusion, Modification, Opposition, Sequence, Nesting, **Representation** (new — distinguishes real/simulated)

### Dual Principles — polarity by sign
- Positive projection = first pole (Good, Order, Creation, Union, Truth, Freedom, Life)
- Negative projection = second pole (Evil, Chaos, Destruction, Separation, Lie, Control, Death)

---

## How the training plan works

### Phase 4: Encoder + Projection Head
```python
# Contextual encoder wrapper (frozen)
# Uses distilbert or the frozen TriadicGPT encoder
encoder = ContextualEncoder(model_name="distilbert-base-uncased")

# Projection head: n_embd -> 49 with sigmoid + annealing
# Based on SigmoidAnnealHead from playground/soft_signatures.py
head = ProjectionHead(n_embd=768, n_primitives=49, anneal_steps=10000)

# Output: 49 values in [0, 1] (sigmoid) or [-1, 1] (tanh)
```

### Phase 5: Supervised training
```python
# Phase 1: MSE with seed lexicon (warm-start, ~2000 steps)
# Phase 2: Subsumption loss adapted from playground/subsumption_loss.py (64->49 bits)
# Phase 3 (optional): Rule of Three loss after subsumption
```

### Phase 6: Self-supervised
```python
# Next-concept prediction on natural text
# Scale to 40M+ params
# Evaluate vs BPE baseline
```

---

## Theoretical framework: The Four Realms

The project is best understood through the framework of The Four Realms:

```
IMAGINARY (ideas)        -> The 7x7 System as vision
CONCEPTUAL (structure)   -> The 49 primitives, rules, operations
EXACT (verification)     -> Prime arithmetic, TriadicValidator
MATERIAL (construction)  -> This tokenizer, the code that runs
```

The first three are solid. The fourth (this code) is what we are building.

The Rule of Three from the book predicts: with the first three well-defined, the fourth is derived. Building is "the relatively easy part" — we already have everything.

---

## Key files in other repos

| File | Repo | What it contains |
|------|------|------------------|
| `sistema-7x7/Sistema_7x7_v2.md` | la-danza-cosmica | Full spec of the 7x7 (69KB) |
| `sistema-7x7/los_tres_reinos.md` | la-danza-cosmica | Theoretical framework (Four Realms) |
| `sistema-7x7/edge_cases_7x7.md` | la-danza-cosmica | Stress test: 49 cases, 65% works |
| `sistema-7x7/borrador_cap_tres_reinos.md` | la-danza-cosmica | Book chapter draft |
| `playground/PLAN.md` | triadic-microgpt | Original experiment plan |
| `experiment_log.md` | triadic-microgpt | Log of 41 runs |

---

## Author's notes

- J. Arturo Ornelas Brand, architect
- The architectural design process (imagine -> organize -> calculate -> build) is exactly the Four Realms
- "With strong base concepts, building is the relatively easy part"
- The book "La Danza Cosmica de los Opuestos" documents the complete theory (32 chapters, 3 rounds of editorial review)
- Papers on Zenodo: DOI 10.5281/zenodo.15169702 and 10.5281/zenodo.15384498 (not peer-reviewed)
