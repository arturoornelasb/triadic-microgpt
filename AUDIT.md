# Audit Report — triadic-microgpt
**Auditor:** Claude Opus 4.6 | **Fecha:** 2026-03-18
**Scope:** Computational validation pipeline, paper claims, code quality, experiment coverage

---

## 1. ESTADO ACTUAL

| Componente | Estado | Bloqueo |
|---|---|---|
| TriadicGPT (Run 15) | Produccion, 40M params, loss 0.946 | Paper no integra P12/P15 |
| Paper LaTeX | 15pp, 830 lineas, 40+ citas | 1 claim (72%->70.4%) |
| triadic-head (PyPI) | v0.1.0, bugs #2-#4 **FIXED** | API divergence map/encode (BAJO) |
| Bootstrap D-A5 | XL **COMPLETADO** 50K steps | R3 algebraic 90.7% > trivial 90.2% |
| Experiment Log | 29 runs + P1-P15 + E1-E7 + B1-B3 | Completo |
| Reconciliacion | 51/63/64 **RESUELTO** | `PRIMITIVE_RECONCILIATION.md` |

**Paper readiness: 7/10** | **Computational evidence: 8/10**

---

## 2. HALLAZGOS CRITICOS

### 2.1 Baseline trivial = 90.2%

La distribucion de bits gold de los 63 primitivos es muy desbalanceada:
- 24 bits siempre OFF en holdout (logica, emociones complejas, sentidos)
- 6 bits siempre ON (fuerza, posicion_temporal, uno, mover, mas)
- 33 bits variable (discriminativos)

Un modelo que predice la clase mayoritaria por bit logra **90.2%** de accuracy. El modelo base (5M, 100 steps triadicos) logra 90.4% con 61/63 dead bits — es trivial.

**Implicacion:** Todo resultado de bit accuracy debe compararse contra 90.2%, no contra 50%.

Always OFF (24 bits): eje_lateral, tacto, gusto, olfato, equilibrio, bien, union, placer,
temporal_obs, eterno_obs, colectivo, creador_obs, muchos, todo, algunos, querer, decir,
porque, si_entonces, puede, debe, tal_vez, parte_de, tipo_de

Always ON (6 bits): fuerza, eje_profundidad, posicion_temporal, uno, mover, mas

Variable / discriminativos (33 bits): vacio (50%), consciente (58%), vista (62%), vida (73%),
eje_vertical (77%), creacion (77%), hacer (77%), contencion (85%), separacion (85%),
flujo_temporal (88%), orden (88%), informacion (92%), etc.

### 2.2 Tres conteos de primitivos incompatibles

| Sistema | Primitivos | Representacion |
|---|---|---|
| Sistema 7x7 v3.5 | 51 | Flotantes tanh [-1,+1] |
| Inventario de opuestos | 63 | Binarios (bits + primos) |
| TriadicGPT Run 15 | 64 | Bits aprendidos emergentes |
| Danza Bootstrap | 63 | Bits supervisados |

**RESUELTO:** Mapping completo en `PRIMITIVE_RECONCILIATION.md`.
- P15 (49) y Inventario (63) comparten 34 primitivos; P15 tiene 10 extra (CARACTERISTICAS), Inventario tiene 29 extra (logica, cantidad, agencia, polos duales separados)
- Run 15 (64) son PCA emergentes, sin correspondencia a primitivos nombrados
- Diferencia estructural clave: P15 funde dualidades (Bien_Mal = 1 bit), Inventario las separa (bien + mal = 2 bits)

### 2.3 P15 XL ya completo — resultados clave

P15 (49-bit structured, `concept_gpt_49bit.py`) corrio a escala XL (40M, 50K steps) y ya termino:
- 86.2% primary accuracy (v3, T1-only) / 88.5% train, 17% test (v4, T1+T2)
- 97.3% subsumption test (75 pairs)
- 0/49 dead bits (vs 15/64 en Run 15)
- Lang loss 0.785 (mejor que Run 15's 0.946)
- **Composicionalidad NO generaliza** — 88.5% train / 17% test = memorization pura
- Resultados en: `playground/results/concept_gpt_49bit.json`
- **NO integrado en paper** (hallazgo mas importante del proyecto)

### 2.4 Resultados enterrados en playground

Los mejores resultados del proyecto NO estan en el paper:

| Resultado | Experimento | Impacto |
|---|---|---|
| 100% subsumption held-out k=64 | P12 | Resuelve limitacion declarada |
| Gap +0.342 (17x mejor que produccion) | P15 | Mejor resultado de gap |
| Analogia 100% con 49 bits estructurados | P15 | Perfeccion algebraica |
| R3 loss colapsa a k=64 | P5/P7/P10/P11 | Resultado negativo importante |

### 2.4 "Emergence" necesita re-enmarcar

Lo que se llama "emergent semantic ordering" es realmente transferencia de embeddings via alignment loss. E5 muestra gradualidad, no phase transition. Usar "gradual transfer of semantic structure" o similar.

---

## 3. BUGS DE CODIGO

| # | Archivo | Bug | Severidad |
|---|---|---|---|
| 1 | triadic-head/algebra.py | Renombro `map()`->`encode()` pero src/triadic.py mantiene `map()` — inconsistencia entre paquete y codebase (NO crash, solo divergencia API) | BAJO |
| 2 | triadic_head/algebra.py | `analogy()` usaba LCM solo (no removia factores A-especificos de C). Unificado con src/triadic.py (3-step: remove+add) | **FIXED** |
| 3 | src/triadic.py:107 + triadic_head/algebra.py:103 | Degenerate prime guard asignaba primo 2 a todos-negativos. Fixed en ambos: retorna 1 (identity) | **FIXED** |
| 4 | src/torch_transformer.py:358 | InfoNCE anchor_idx podia = pool_idx (autoreferencia) | **FIXED** |
| 5 | benchmarks/subsumption_benchmark.py | 89 pares en codigo, 87 evaluados (2 OOV), paper P12 usa 57 (set diferente) | NOTA |

---

## 4. CLAIMS DEL PAPER QUE NECESITAN CORRECCION

| Claim | Problema | Correccion |
|---|---|---|
| "emergent semantic ordering" | Es transferencia via alignment loss | "semantic ordering via embedding alignment" |
| "closes 72% of gap" | Calculo real: (0.099-0.011)/(0.136-0.011) = 70.4% | Corregir a "70%" en lineas 65, 472, 728 del paper |
| ~~"8x compression (122%)"~~ | **NO EXISTE en paper** — paper dice "8.3% probe accuracy" correctamente | Eliminar de pendientes (falso positivo del audit) |
| "emergent abilities" | Paper ya califica como "gradual" en E5 | **OK** — no requiere cambio |

---

## 5. D-A5 BOOTSTRAP — STATUS EN VIVO

### Diseno

24 conceptos TRAIN (supervision directa) + 23 HOLDOUT:
- 14 "R3-reachable": tienen camino algebraico via quads
- 9 "CTRL": sin camino algebraico (controles)
- 15 analogy quads: 5 exactos, 2 parciales, 8 aproximados

Prediccion: `D = C + (B - A)` en espacio tanh continuo.

### Progreso XL (40M params, 50K steps)

```
Step     Loss    BitTrain  BitHold  Dead  SupLoss  Phase
2500     2.426   48.2%     47.7%    18    0        Lang only
10000    1.672   46.4%     44.3%    26    0        Lang only
22500    1.309   44.6%     42.3%    23    0        Lang only
25000    1.308   51.5%     50.5%    26    1.255    Tri activado
27500    1.267   100.0%    87.0%    30    0.007    Tri +2.5K steps
30000    1.187   100.0%    86.9%    30    0.007    Tri +5K
32500    1.130   100.0%    87.1%    30    0.007    Tri +7.5K
35000    1.072   100.0%    87.1%    30    0.006    Tri +10K
37500    1.055   100.0%    87.2%    30    0.006    Tri +12.5K
40000    1.062   100.0%    87.4%    30    0.006    Tri +15K
42500    0.985   100.0%    87.3%    30    0.006    Tri +17.5K
45000    1.005   100.0%    87.2%    30    0.006    Tri +20K
47500    0.963   100.0%    87.2%    30    0.006    Tri +22.5K
50000    0.975   100.0%    87.2%    30    0.006    FINAL
```

### Resultados finales (predict phase)

| Grupo | Direct | Algebraic | Delta |
|-------|--------|-----------|-------|
| R3-reachable (14) | 87.5% | **90.7%** | +3.2% |
| CTRL (9) | 85.9% | N/A | N/A |
| **Trivial baseline** | **90.2%** | — | — |

**R3 algebraica (90.7%) SUPERA trivial baseline (90.2%).** Directo (87.5%) no.

Mejores casos algebraicos:
- **reina:** 77.8% -> 100.0% (+22.2%) via man:woman=king:queen
- **silencioso:** 82.5% -> 96.8% (+14.3%) via ensemble 2 quads
- **odio:** 90.5% -> 98.4% (+7.9%) via happy:sad=love:hate
- **liquido:** 88.9% -> 95.2% (+6.3%) via man:woman=solid:liquid

### Criterios de exito

1. [x] Holdout direct > 75% — 87.5% **PASS**
2. [x] Algebraic > 80% — 90.7% **PASS**
3. [ ] Algebraic > direct + 5% — +3.2% **FAIL**
4. [ ] Reachable > control + 10% — +4.8% (87.5% vs 85.9%+10%=95.9%) **FAIL**
5. [x] Algebraic > 90.2% trivial — 90.7% **PASS** (margin: +0.5pp)

### Herramientas de analisis

```bash
# Monitorear en vivo
python playground/monitor_bootstrap.py --watch 30

# Analisis completo post-training
python playground/analyze_bootstrap.py

# Correr prediccion algebraica
python playground/danza_bootstrap.py --phase predict --checkpoint checkpoints/danza_bootstrap_xl/
```

---

## 6. EXPERIMENTOS PRIORITARIOS

### Tier 0: Completado
- ~~**D-A5 Bootstrap XL**~~ — **COMPLETADO.** R3 algebraic 90.7% > trivial 90.2%. 3/5 criterios PASS.

### Tier 1: Impactan paper directamente
- **P15 a escala XL** (49-bit estructurado) — COMPLETADO, resultados en playground (NO en paper)
- **GPT-2 Medium + InfoNCE** — cerrar gap con Engine PCA
- ~~**Reconciliar 51/63/64**~~ — **COMPLETADO** (`PRIMITIVE_RECONCILIATION.md`)
- **sub_weight sweep** — Pareto subsumption/PPL

### Tier 2: Fortalecen la tesis
- **Dato real de onda sinusoidal** — la objecion mas repetida
- **7 estructuras algebraicas** — solo 2 de 9 testeadas
- **Falsabilidad activa** — buscar contraejemplos honestamente
- **Resolver el 93.4%** (Prime Twins Rule)

### Tier 3: Diferenciadores
- Comparacion con NSM de Wierzbicka
- Cross-linguistico
- Tres Reinos como clasificador

---

## 7. PREGUNTAS ANTICIPADAS DE REVISORES

1. Por que 51, 63 y 64 primitivos en tres partes del proyecto?
2. Si la R3 necesita division y la division rompe primos, funciona o no?
3. Random projections logran gap +0.056 > TriadicGPT +0.020. Por que el modelo es mejor?
   - **Respuesta:** Operaciones algebraicas (analogia 100% vs 16.7%, subsumption 92% vs 0%)
4. Los 6 agentes del debate son LLMs. Validacion humana?
5. "Emergent abilities" pero E5 muestra gradualidad.
6. "Algebraically verifiable" pero subsumption = 0% sin loss auxiliar.
7. Diferencia con WordNet, FrameNet, ConceptNet?
8. Pesos privados pero arquitectura publicada. Reproducible?

---

## 8. CRONOGRAMA

| Semana | Tareas |
|--------|--------|
| 1 | Analizar D-A5, reconciliar 51/63/64, fixes bugs #1-2 |
| 2 | P15 XL, GPT-2 Medium, integrar P12/P15 en paper |
| 3 | Sub_weight sweep, dato real de onda, falsabilidad |
| 4 | Estructuras algebraicas, resolver 93.4%, paper v2 final |
| Post | NSM, Tres Reinos, publicar triadic-head, Zenodo |
