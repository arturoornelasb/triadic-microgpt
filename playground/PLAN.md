# Playground — Plan de Accion

> Laboratorio experimental para Triadic MicroGPT.
> Hardware: RTX 5060 Ti 16 GB | CUDA 12.8 | Python 3.10 | Conda `triadic-microgpt`

---

## Estado actual del proyecto

| Metrica | Valor |
|---------|-------|
| Modelo produccion | Run 15 v1.4-strongalign (40M params) |
| Loss | 0.946 |
| Entropy | 0.749 |
| Semantic gap (from-scratch) | +0.020 |
| Semantic gap (GPT-2 InfoNCE) | +0.099 |
| Domain separation | 1.21 (sentence-level) |
| Dead bits | ~15 / 64 |
| Subsumption | **100% held-out @ k=64** (with sub loss, 25K early stop) |
| Paper | 16+ paginas, compilado (Exp 12 added) |
| Experimentos completados | 12 (29+ runs) |

El proyecto esta maduro. Lo que sigue son ideas de frontera, muchas inspiradas
por *La Danza Cosmica de los Opuestos*.

---

## Linea 1 — Tokenizador Conceptual (del libro, Cap. 34)

### Idea central

Reemplazar BPE (corta por frecuencia) con un tokenizador que corta por
**conceptos**. Cada concepto atómico = un primo. Cada concepto compuesto =
producto de primos.

```
BPE:       "infelicidad" -> ["in", "felic", "idad"]   (fragmentos sin sentido)
Conceptual: "infelicidad" -> [negacion(2) × felicidad(3)]  = 6
```

### Experimentos propuestos

#### 1.1 Concept Vocabulary Builder
- **Que**: construir un vocabulario de ~500 conceptos atomicos a partir de
  WordNet + clustering de embeddings de GPT-2.
- **Como**: agrupar tokens BPE por similitud coseno, extraer centroides,
  asignar un primo a cada centroide.
- **Output**: `playground/concept_vocab.json` — mapa concepto -> primo
- **Esfuerzo**: bajo (~2h GPU)
- **Archivo**: `playground/concept_tokenizer.py`

#### 1.2 Hybrid Tokenizer (BPE + Concept Layer)
- **Que**: mantener BPE como capa 1, agregar capa 2 que mapea secuencias de
  subwords a concept-tokens.
- **Como**: entrenar un mini-classifier (MLP 2 capas) que toma N subword
  embeddings y predice el concept-token.
- **Por que hibrido**: un tokenizador 100% conceptual requiere un lexicon
  perfecto; lo hibrido permite iterar rapido.
- **Output**: metricas de accuracy del clasificador
- **Archivo**: `playground/hybrid_tokenizer.py`

#### 1.3 Concept-Token GPT (end-to-end)
- **Que**: entrenar un TriadicGPT pequeno (5M params) donde los tokens son
  conceptos, no subwords.
- **Hipotesis**: si los tokens ya son primos, el triadic head deberia converger
  mas rapido y con menos dead bits.
- **Metrica clave**: semantic gap vs Run 15 con menos pasos de entrenamiento.
- **Archivo**: `playground/concept_gpt_train.py`

---

## Linea 2 — Modelo de Onda (del libro, Cap. 7-9)

### Idea central

El libro propone que todos los opuestos siguen y(t) = A sin(2pift + phi).
La tanh del triadic head ya es una funcion sigmoidal. Que pasa si la
reemplazamos con una activacion sinusoidal?

### Experimentos propuestos

#### 2.1 Sinusoidal Triadic Head
- **Que**: reemplazar `tanh(Wx)` con `sin(Wx + b)` en el triadic head.
- **Hipotesis**: la sinusoide tiene periodicidad — podria capturar oposiciones
  ciclicas (frio/caliente, amor/odio) mejor que tanh.
- **Riesgo**: la periodicidad puede causar colisiones hash no deseadas.
- **Metrica**: entropy, semantic gap, dead bits vs Run 15.
- **Esfuerzo**: bajo (cambiar 1 linea en torch_transformer.py, entrenar 10K steps)
- **Archivo**: `playground/sin_head_experiment.py`

#### 2.2 Phase-Aware Attention
- **Que**: agregar un bias de fase aprendible a las position embeddings.
- **Inspiracion**: el libro define 5 constantes (ciclicidad, equilibrio neutral,
  proporcionalidad dual, hipotesis maestra, subjetividad relativa). La fase
  phi captura la "perspectiva" del observador.
- **Como**: `pos_emb[i] = sin(i/10000^(2k/d) + phi_k)` donde phi_k es
  aprendible per-head.
- **Esfuerzo**: medio (modificar attention, entrenar 20K steps)
- **Archivo**: `playground/phase_attention.py`

---

## Linea 3 — Regla de Tres Algebraica (del libro, Cap. 25)

### Idea central

C4 = (a * C2 * C3) / (b * C1), con K = 1/(a*b) como medida de "verdad".

### Experimentos propuestos

#### 3.1 Rule-of-Three Loss
- **Que**: nueva componente de loss que optimiza K para tripletas conocidas.
- **Como**: dado (rey, hombre, mujer) -> optimizar que Phi(rey)/Phi(hombre)*Phi(mujer) = Phi(reina).
- **Ventaja**: supervision directa para analogias (actualmente 69.2% verificacion).
- **Riesgo**: puede sobre-ajustar a las tripletas de entrenamiento.
- **Metrica**: analogy accuracy top-1 (actualmente 3.8%).
- **Archivo**: `playground/rule_of_three_loss.py`

#### 3.2 K-Constant Analysis
- **Que**: calcular K = 1/(a*b) para todas las relaciones aprendidas y
  verificar si K correlaciona con "calidad semantica" (gap).
- **Pura evaluacion, sin entrenamiento.**
- **Output**: scatter plot K vs semantic gap por dominio.
- **Archivo**: `playground/k_constant_analysis.py`

---

## Linea 4 — Recuperacion de Subsumption (limitacion actual)

### Problema

Subsumption = 0% a k=64. El libro sugiere que conceptos compuestos deberian
*contener* los primos de sus hiperónimos por construccion.

### Experimentos propuestos

#### 4.1 Supervised Bit Inheritance
- **Que**: agregar una loss que fuerza que si "perro" es hiponimo de "animal",
  los bits activos de "animal" sean un subconjunto de los de "perro".
- **Datos**: WordNet hypernym pairs (~2K pares).
- **Metrica**: subsumption recall (actualmente 0%).
- **Riesgo**: puede interferir con diversity loss.
- **Archivo**: `playground/subsumption_loss.py`

#### 4.2 Hierarchical Bit Allocation
- **Que**: reservar los primeros M bits para categorias generales (dominio),
  los siguientes para sub-categorias, etc.
- **Inspiracion**: el libro organiza opuestos en dominios (sensorial, emocional,
  moral, ontologico, binario, biologico).
- **Como**: loss que penaliza si bits de dominio no coinciden para conceptos
  del mismo dominio.
- **Archivo**: `playground/hierarchical_bits.py`

---

## Linea 5 — Dead Bits y Eficiencia de Representacion

### Problema

~15 de 64 bits tienen entropy < 0.3 (desperdicio de capacidad).

### Experimentos propuestos

#### 5.1 L1 Sparsity on Dead Bits
- **Que**: agregar penalizacion L1 sobre los bits con menor varianza para
  forzar que se activen.
- **Esfuerzo**: bajo (modificar triadic_loss, entrenar 10K steps)
- **Archivo**: `playground/dead_bit_regularization.py`

#### 5.2 Adaptive k via Gumbel-Softmax
- **Que**: en lugar de k fijo, dejar que el modelo aprenda cuantos bits
  necesita por concepto via Gumbel-Softmax.
- **Inspiracion**: diferentes conceptos tienen diferente complejidad. "perro"
  necesita menos bits que "democracia".
- **Esfuerzo**: alto (nueva arquitectura)
- **Archivo**: `playground/adaptive_bits.py`

---

## Linea 6 — Superposicion Cuantica (del libro, Cap. 14-16)

### Idea central

El libro propone que conceptos ambiguos existen en superposicion (como un qubit)
hasta que el contexto los "colapsa". Ejemplo: "banco" = superposicion(mueble,
financiero) hasta que aparece "dinero" o "sentarse".

### Experimento propuesto

#### 6.1 Soft Triadic Signatures
- **Que**: en lugar de binarizar con threshold 0, mantener valores continuos
  [0,1] y solo discretizar para operaciones algebraicas.
- **Como**: usar sigmoid en lugar de hard-threshold para el forward pass;
  binarizar solo en eval.
- **Metrica**: semantic gap, analogy accuracy.
- **Hipotesis**: permitir "superposicion" durante training puede mejorar
  gradientes y reducir dead bits.
- **Archivo**: `playground/soft_signatures.py`

---

## Linea 7 — Benchmarks Rapidos (validacion cruzada)

#### 7.1 Perplexity en otros datasets
- **Que**: evaluar Run 15 en WikiText-2, LAMBADA, o HellaSwag (adaptado a
  4096 vocab).
- **Por que**: TinyStories es muy facil; queremos saber si el triadic head
  generaliza.
- **Archivo**: `playground/cross_dataset_eval.py`

#### 7.2 Triadic vs Random Baseline (control riguroso)
- **Que**: entrenar un modelo identico donde el triadic head tiene pesos
  congelados aleatorios. Si las metricas son similares, el head no esta
  aprendiendo nada util.
- **Archivo**: `playground/random_baseline.py`

---

## Priorizacion (effort vs impact)

| # | Experimento | Impacto | Esfuerzo | GPU hrs | Prioridad | Estado |
|---|-------------|---------|----------|---------|-----------|--------|
| 3.2 | K-Constant Analysis | medio | bajo | 0 | **P0** | ✅ DONE |
| 2.1 | Sin Head | medio | bajo | ~1h | **P1** | ✅ DONE |
| 5.1 | L1 Dead Bits | medio | bajo | ~1h | **P1** | ✅ DONE (redundante) |
| 6.1 | Soft Signatures | alto | bajo | ~2h | **P1** | ✅ DONE (BEST) |
| 7.2 | Random Baseline | alto | bajo | ~2h | **P1** | ✅ DONE (critico) |
| XL | Sigmoid+Anneal XL | alto | alto | ~3h | **P1** | ✅ DONE (mixto, PPL +116%) |
| 1.1 | Concept Vocab | alto | medio | ~2h | **P2** | ✅ DONE |
| 3.1 | Rule-of-Three Loss | alto | medio | ~30m | **P2** | ✅ DONE (mecanismo funciona) |
| 4.1 | Subsumption Loss | alto | medio | ~30m | **P2** | ✅ DONE ⭐ BREAKTHROUGH |
| 2.2 | Phase Attention | medio | medio | ~30m | **P3** | ✅ DONE (negativo) |
| R3+S | R3 + Subsumption combo | muy alto | bajo | ~30m | **P2** | ✅ DONE (Sub wins) |
| P9 | Info Hierarchy Analysis | alto | 0 | 0 | **P0** | ✅ DONE (93% reduction) |
| P10 | R3 Entropy Guard | alto | bajo | ~50m | **P2** | ✅ DONE (unfixable) |
| P11 | Curriculum Sub→R3 | alto | bajo | ~30m | **P2** | ✅ DONE (R3 erases Sub) |
| 1.2 | Hybrid Tokenizer | muy alto | alto | ~5h | **P3** | pendiente |
| 4.2 | Hierarchical Bits | medio | alto | ~4h | **P3** | redundante con 4.1 ✅ |
| 5.2 | Adaptive k | alto | alto | ~6h | **P3** | pendiente |
| XL2 | Sigmoid+Anneal (temp=5) | alto | alto | ~3h | **P3** | pendiente |
| XL-Sub | XL Subsumption Loss | **muy alto** | alto | ~9h | **P1** | ✅ DONE ⭐ (100% held-out @25K) |
| P14 | Concept Head (Phase 4) | alto | medio | ~1min | **P2** | ✅ DONE (negativo — embeddings insuficientes) |
| 1.3 | Concept GPT (49-bit e2e) | muy alto | muy alto | ~10h | **P4** | pendiente (next step for 7×7) |
| 7.1 | Cross-Dataset Eval | medio | medio | ~2min | **P4** | ✅ DONE (expected OOD degradation) |

---

## Convenciones del playground

```
playground/
  PLAN.md                    # Este archivo
  results/                   # JSONs y PNGs de resultados
  concept_tokenizer.py       # Linea 1
  hybrid_tokenizer.py        # Linea 1
  concept_gpt_train.py       # Linea 1
  sin_head_experiment.py     # Linea 2
  phase_attention.py         # Linea 2
  rule_of_three_loss.py      # Linea 3
  k_constant_analysis.py     # Linea 3
  subsumption_loss.py        # Linea 4
  hierarchical_bits.py       # Linea 4
  dead_bit_regularization.py # Linea 5
  adaptive_bits.py           # Linea 5
  soft_signatures.py         # Linea 6
  cross_dataset_eval.py      # Linea 7
  random_baseline.py         # Linea 7
```

- Cada script debe ser **autocontenido**: cargar modelo, correr experimento,
  guardar resultados en `playground/results/`.
- Documentar hallazgos en este archivo (seccion Resultados al final).
- Entrenamientos largos: usar `--steps 10000` como default para explorar antes
  de comprometer GPU por horas.

---

## Resultados

### P0: K-Constant Analysis (2026-03-13) — COMPLETADO

**Hallazgos:**
- **K medio = 1.21** (la Regla de Tres se satisface aproximadamente; K=1 seria perfecto)
- K varia de 0.31 (sun:day::moon:night) a 2.78 (king:queen::boy:girl)
- **Offset similarity media = 0.032** (aritmetica vectorial debil en espacio triadico)
- **Exact match algebraico: 0/15** (esperado a k=64)
- **Similitud prediccion-real = 48.8%** (~mitad de factores primos compartidos)
- **Separacion de dominio (token-level) = 1.035** (consistente con Exp 11)
- Colores tienen mejor intra-sim (0.627), emociones la peor (0.464)

**Conclusion**: La constante K valida que las relaciones aprendidas siguen
la Regla de Tres del libro, pero con varianza alta. La mejor K (0.91-1.05)
se da en father:mother::son:daughter, bird:fly::fish:swim, y red:blue::green:yellow.

---

### P1: Sinusoidal Head (2026-03-13) — COMPLETADO

**Config**: 6L/256D/64bits, 10K steps, 5K stories

| Metrica | TANH | SIN | Delta |
|---------|------|-----|-------|
| Loss final | 1.66 | 1.66 | ~0 |
| Semantic gap | -0.005 | **+0.016** | **+0.021** |
| Dead bits | 10 | 14 | +4 |
| Active bits | 54 | 50 | -4 |
| Entropy | 0.606 | 0.618 | +0.012 |

**Hallazgo clave**: sin(freq*Wx + phase) produce **mejor ordenamiento semantico**
que tanh(Wx) (+0.016 vs -0.005), a costa de +4 dead bits. Language loss identica.
La periodicidad sinusoidal parece capturar relaciones ciclicas mejor.

**Siguiente paso**: Probar sin() en modelo XL (40M params) para ver si escala.

---

### P2: Concept Tokenizer (2026-03-13) — COMPLETADO

**Config**: 256 clusters K-Means sobre embeddings de Run 15

**Hallazgos:**
- **Silhouette = -0.059** — BPE embeddings NO forman clusters semanticos naturales
- Distribucion de clusters altamente sesgada (pocos gigantes, muchos minusculos)
- **PERO** clusters semanticos SI emergen en tamanos pequenos:
  - Colores: {yellow, pink, blue, purple, green} (cluster 751)
  - Nombres femeninos: {Amy, Jill, Sue, Sally, Mia} (cluster 79)
  - Comida positiva: {yummy, exciting, delicious, tasty, lovely} (cluster 1487)
  - Familia: {Mama, Mummy, Mum} (cluster 379)
- Consistencia de grupo: colores 67%, royalty 50%, emociones 20% (media 37%)
- Analogia conceptual: dog:puppy::cat:kitten = 50% sim (mejor caso)

**Conclusion**: Un tokenizador conceptual requiere word-level tokens (no BPE subwords)
como punto de partida. El clustering sobre subword embeddings produce ruido
con islas semanticas. Linea 1.2 (Hybrid Tokenizer) debe mapear subwords→palabras
antes de conceptualizar.

---

### P1: Random Baseline (2026-03-13) — COMPLETADO

**Config**: 6L/256D/64bits, 10K steps, 5K stories

| Metrica | Normal | Frozen Random | Lang Only |
|---------|--------|--------------|-----------|
| Semantic gap | -0.013 | **+0.008** | -0.007 |
| Related vs unrelated | -0.029 | **+0.019** | +0.010 |
| Algebraic analogy | 25% | 50% | **75%** |
| Dead bits | 15 | 19 | 18 |
| Language loss | 1.73 | 1.73 | 1.75 |

**Hallazgo critico**: A 5.8M params / 10K steps, el head congelado aleatorio
SUPERA al entrenado. Esto confirma que el ordenamiento semantico es emergente
solo a 40M+ params (Phase 4). La triadic loss a escala pequena **interfiere**
con la estructura natural de los embeddings.

**Implicacion para playground**: Las comparaciones relativas (variante A vs B)
son validas; los valores absolutos de semantic gap no son fiables a esta escala.
Para resultados definitivos, repetir en XL (40M params, ~76 min por modelo).

---

### P1: Soft Signatures (2026-03-13) — COMPLETADO

**Config**: 6L/256D/64bits, 10K steps, 4 variantes

| Variante | Loss | Entropy | Dead | Gap |
|----------|------|---------|------|-----|
| tanh (baseline) | 1.786 | 0.999 | 0 | -0.042 |
| sigmoid (soft) | 1.739 | 0.999 | 0 | -0.039 |
| **sigmoid+anneal** | **1.728** | **1.000** | **0** | **-0.003** |
| **gumbel-softmax** | **1.732** | **1.000** | **0** | **-0.003** |

**Hallazgo clave**: Temperature annealing (soft→hard) mejora el semantic gap
en +0.039 comparado con tanh puro, y elimina dead bits completamente.
La funcion de activacion importa menos que la estrategia de annealing.

Sigmoid+anneal y Gumbel-Softmax son equivalentes en rendimiento. Ambos
superan a sigmoid sola y a tanh. La "superposicion cuantica" del libro
(empezar suave, colapsar gradualmente) funciona.

**Siguiente paso**: Combinar sin + anneal en un solo modelo. Probar a escala XL.

---

### P1: Dead Bit Regularization (2026-03-13) — COMPLETADO

**Config**: 6L/256D/64bits, 10K steps, 4 variantes

| Variante | Loss | Entropy | Dead Bits |
|----------|------|---------|-----------|
| **Baseline (entropy only)** | **1.691** | 0.999 | 0 |
| L1 targeted (dead bits) | 1.797 | 0.999 | 0 |
| L1 global (all bits) | 1.772 | 0.999 | 0 |
| L1 variance | 1.764 | 0.999 | 0 |

**Hallazgo**: La entropy regularization existente ya elimina dead bits a 10K steps.
La L1 adicional solo perjudica language loss (+0.07-0.10). Los ~15 dead bits de
Run 15 pueden ser un fenomeno de entrenamiento prolongado (50K steps), no una
limitacion de la regularizacion. La L1 no aporta valor adicional.

---

## Resumen de Hallazgos Playground (2026-03-13)

### Ideas que FUNCIONAN (probadas):
1. **Sigmoid + temperature annealing** → +0.039 gap, 0 dead bits (vs tanh)
2. **Sinusoidal activation** → +0.021 gap (vs tanh)
3. **Regla de Tres K ≈ 1.21** → confirma que analogias siguen algebra del libro

### Ideas NEUTRALES (no aportan a esta escala):
4. **L1 dead bit regularization** → redundante con entropy reg existente
5. **Concept tokenizer sobre BPE** → subwords no son conceptos (silhouette -0.06)

### Insight META:
6. **Semantic ordering emerge solo a 40M+ params** → confirmado por random baseline
   (frozen random head supera al entrenado a 5.8M params)

### Siguiente fase (prioridad):
- Combinar **sin + sigmoid_anneal** en un solo modelo (potencial sinergico)
- Probar **sigmoid_anneal a escala XL** (40M params) para validar con absolutos
- **Rule-of-Three Loss** (script listo, no ejecutado aun)
- **Hybrid tokenizer** word-level → concept mapping
