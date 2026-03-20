> **⚠ ARCHIVED — Summary migrated to [`EXPERIMENT_REFERENCE.md`](../EXPERIMENT_REFERENCE.md) Section 12.** This file preserved in `archive/` for the complete 34-row match table, categorized primitive lists, and dual representation explanation.

# Reconciliation of Primitive Systems: 49 vs 63 vs 64 bits

## Three Systems

| System | Count | Source | Used in |
|--------|-------|--------|---------|
| P15 conceptual_tokenizer | 49 | Sistema 7x7 (7 categories x 7) | concept_gpt_49bit.py |
| Inventario de opuestos | 63 | primitivos.json (6 geometric layers) | danza_63bit.py, danza_bootstrap.py |
| TriadicGPT Run 15 | 64 | Emergent PCA from Engine | Run 15, paper |

## Sistema 7x7 (51) vs Inventario (63)

The 63 primitives ARE all 51 from Sistema 7x7. The count difference comes from how dual axes are handled:
- Sistema 7x7: 12 dual axes counted as 1 each (e.g., "Bien/Mal" = 1 axis)
- Inventario: each pole is a separate bit (e.g., "bien" = bit 20, "mal" = bit 21)
- 51 base + 12 additional poles from splitting = 63

There are NO primitives in the inventario that don't exist in Sistema 7x7.

## P15 (49) vs Inventario (63)

### Direct matches: 34 primitives

| P15 Name | Inventario Name | Notes |
|----------|----------------|-------|
| Fuego | fuego | Exact |
| Tierra | tierra | Exact |
| Agua | agua | Exact |
| Aire | aire | Exact |
| Vacio | vacio | Exact |
| Informacion | informacion | Exact |
| Fuerza | fuerza | Exact |
| Arriba/Abajo | eje_vertical | P15 splits into 2, inventario uses 1 axis |
| Adelante/Atras | eje_profundidad | P15 splits into 2, inventario uses 1 axis |
| Izquierda_Derecha | eje_lateral | Exact |
| Dentro_Fuera | contencion | Different name, same concept |
| Presente/Pasado/Futuro | posicion_temporal | P15 has 3 states, inventario has 1 primitive |
| Pausa/Play | flujo_temporal | P15 has 2 states, inventario has 1 primitive |
| Vista | vista | Exact |
| Oido | oido | Exact |
| Tacto | tacto | Exact |
| Gusto | gusto | Exact |
| Olfato | olfato | Exact |
| Equilibrio | equilibrio | Exact |
| Interocepcion | interocepcion | Exact |
| Bien_Mal | bien + mal | P15 fused, inventario split |
| Orden_Caos | orden + caos | P15 fused, inventario split |
| Creacion_Destruccion | creacion + destruccion | P15 fused, inventario split |
| Union_Separacion | union + separacion | P15 fused, inventario split |
| Verdad_Mentira | verdad + mentira | P15 fused, inventario split |
| Libertad_Control | libertad + control | P15 fused, inventario split |
| Vida_Muerte | vida + muerte | P15 fused, inventario split |
| Consciente | consciente | Exact |
| Temporal | temporal_obs | Exact |
| Eterno | eterno_obs | Exact |
| Individual | individual | Exact |
| Colectivo | colectivo | Exact |
| Ausente | ausente | Exact |
| Creador | creador_obs | Exact |

### P15-only: 10 primitives (no inventario equivalent)

All 7 CARACTERISTICAS + 3 others:
- Color, Textura, Forma, Material, Brillo, Transparencia, Estado_materia
- En_medio (spatial)
- Ir_al_pasado, Ir_al_futuro (temporal)

**Why they're absent from inventario:** CARACTERISTICAS are perceptual qualities that the inventario considers derivable from elements + senses. The temporal composites (Ir_al_pasado, Ir_al_futuro) are considered composite operations in the inventario.

### Inventario-only: 29 primitives (not in P15)

**Logical/Modal (6):** porque, si_entonces, puede, debe, tal_vez, tipo_de
**Quantity (4):** uno, algunos, muchos, todo, parte_de
**Agency (6):** mover, hacer, querer, saber, pensar, decir
**Magnitude (2):** mas, menos
**Affective (2):** placer, dolor
**Dual poles (7):** mal, caos, destruccion, separacion, mentira, control, muerte
**Observer (1):** receptivo

**Why they're absent from P15:** P15 was designed from Sistema 7x7 v3.0 which had 7x7=49 primitives. The inventario expanded to 63 by splitting duals into separate bits and adding logic, quantity, and agency categories.

### Structural difference: Dual representation

The biggest structural difference is how duals are handled:
- **P15:** Each dual pair is ONE primitive with 3 states (+/0/null). E.g., "Bien_Mal" is one bit.
- **Inventario:** Each dual pole is a SEPARATE bit. E.g., "bien" (bit 20) and "mal" (bit 21) are independent.

This means: 7 dual pairs in P15 = 7 bits. Same 7 pairs in inventario = 14 bits. This accounts for 7 of the 14-bit difference.

## Run 15 (64 bits)

Run 15's 64 bits are **arbitrary/emergent** — they do NOT correspond to named primitives. They are the output of a linear projection head initialized with PCA of the backbone's hidden states. The gold_primes_64.json file maps 10,006 English words to 64-bit binary signatures, but these signatures come from the Triadic Engine (a separate system), not from any human-defined ontology.

**Bottom line:** Run 15's 64 bits are a learned compression, not a mapping to philosophical primitives. P15's 49 bits and the inventario's 63 bits are human-defined ontological mappings.

## Implications for the paper

1. The paper should clearly distinguish between "arbitrary k-bit learned signatures" (Run 15) and "ontology-mapped signatures" (P15, D-A5)
2. P15's 49-bit system proves the ontology IS learnable (86.2% accuracy)
3. The inventario's 63-bit system is a superset that adds logic, quantity, agency — untested at scale
4. A unified "best system" might use 63 bits with the inventario structure, but would need P15-style supervised training
