# Contexto del Proyecto — Conceptual Tokenizer

> Este documento captura todo el contexto necesario para continuar el desarrollo.
> Léelo antes de trabajar en el tokenizador.

---

## Qué es esto

Un tokenizador que convierte texto en **conceptos** (no en fragmentos estadísticos como BPE). Cada palabra se descompone en los 49 primitivos del Sistema 7×7, cada primitivo tiene un primo asignado, y las relaciones se verifican con aritmética exacta.

## Estado actual (2026-03-14)

### Completado (Fases 1-3, sin GPU)

| Archivo | Qué hace |
|---------|----------|
| `config.py` | 49 primitivos, primos (2..227), categorías, umbrales |
| `primitives.py` | ConceptToken, ConceptSequence, subsumes(), compose(), gap() |
| `states.py` | StateResolver: proyección continua [-1,1] → [+]/[0]/[∅] + intensidad + polaridad |
| `seed_lexicon.py` | 462 palabras mapeadas (349 T1 + 95 T2 + 18 T3), 49/49 primitivos usados |
| `prime_encoder.py` | Estados resueltos → composite primes (dual-channel: active + zero) |
| `triadic_bridge.py` | Conecta ConceptTokens al TriadicValidator de `src/triadic.py` |

### Pendiente (Fases 4-6, requiere GPU)

| Fase | Qué falta | Archivos a crear |
|------|-----------|------------------|
| 4 | Encoder contextual + Projection Head (49-dim, sigmoid+annealing) | `encoder.py`, `projection_head.py` |
| 5 | Tokenizer end-to-end + training supervisado | `tokenizer.py`, `training/trainer.py`, `training/losses.py` |
| 6 | Self-supervised en texto natural a escala 40M+ | `training/dataset.py` |

---

## Decisiones técnicas tomadas (con evidencia)

| Decisión | Elegido | Evidencia |
|----------|---------|-----------|
| Activación | **Sigmoid + annealing** (no tanh) | `playground/soft_signatures.py`: gap +0.039 vs tanh |
| Loss principal | **Subsumption loss** | `playground/subsumption_loss.py`: breakthrough, fuerza herencia de bits |
| Rule of Three loss | **Solo después de subsumption, o no usar** | `playground/r3_subsumption_combo.py`: R3 borra ganancias de subsumption |
| Granularidad | **Palabras completas** (no BPE subwords) | `playground/concept_tokenizer.py`: BPE silhouette -0.059, no son semánticos |
| Escala mínima | **40M params** | `playground/random_baseline.py`: orden semántico emerge solo a esta escala |
| Número de bits | **49** (uno por primitivo del 7×7) | En vez de 32/64 arbitrarios del modelo actual |
| Estados | **Dual-channel** (active_composite + zero_composite) | Preserva distinción silencio vs piedra-que-no-oye |

---

## El Sistema 7×7 v2.1 — Los 49 primitivos

```
Cat 1 ELEMENTOS:       Fuego=2, Tierra=3, Agua=5, Aire=7, Vacío=11, Información=13, Fuerza=17
Cat 2 CARACTERÍSTICAS: Color=19, Textura=23, Forma=29, Material=31, Brillo=37, Transparencia=41, Estado=43
Cat 3 ESPACIO:         Arriba=47, Abajo=53, EnMedio=59, Adelante=61, Atrás=67, IzqDer=71, DentroFuera=73
Cat 4 TIEMPO:          Presente=79, Pasado=83, Futuro=89, Pausa=97, IrPasado=101, IrFuturo=103, Play=107
Cat 5 SENTIDOS:        Vista=109, Oído=113, Tacto=127, Gusto=131, Olfato=137, Equilibrio=139, Interocepción=149
Cat 6 PRINCIPIOS:      BienMal=151, OrdenCaos=157, CreaciónDestrucción=163, UniónSeparación=167,
                       VerdadMentira=173, LibertadControl=179, VidaMuerte=181
Cat 7 OBSERVADORES:    Consciente=191, Temporal=193, Eterno=197, Individual=199, Colectivo=211,
                       Ausente=223, Creador=227
```

### 3 estados
- `[+]` Activo: primo en active_composite, proyección > 0
- `[0]` Cero (ausencia activa): primo en zero_composite, proyección < 0
- `[∅]` N/A: primo ausente, |proyección| < threshold (0.1)

### 6 operaciones
Fusión, Modificación, Oposición, Secuencia, Anidamiento, **Representación** (nueva — distingue real/simulado)

### Principios Duales — polaridad por signo
- Proyección positiva = primer polo (Bien, Orden, Creación, Unión, Verdad, Libertad, Vida)
- Proyección negativa = segundo polo (Mal, Caos, Destrucción, Separación, Mentira, Control, Muerte)

---

## Cómo funciona el plan de entrenamiento

### Fase 4: Encoder + Projection Head
```python
# Wrapper de encoder contextual (frozen)
# Usa distilbert o el encoder de TriadicGPT frozen
encoder = ContextualEncoder(model_name="distilbert-base-uncased")

# Projection head: n_embd → 49 con sigmoid + annealing
# Basado en SigmoidAnnealHead de playground/soft_signatures.py
head = ProjectionHead(n_embd=768, n_primitives=49, anneal_steps=10000)

# Output: 49 valores en [0, 1] (sigmoid) o [-1, 1] (tanh)
```

### Fase 5: Training supervisado
```python
# Fase 1: MSE con seed lexicon (warm-start, ~2000 steps)
# Fase 2: Subsumption loss adaptado de playground/subsumption_loss.py (64→49 bits)
# Fase 3 (opcional): Rule of Three loss después de subsumption
```

### Fase 6: Self-supervised
```python
# Next-concept prediction en texto natural
# Escalar a 40M+ params
# Evaluar vs BPE baseline
```

---

## Marco teórico: Los Cuatro Reinos

El proyecto se entiende mejor con el marco de Los Cuatro Reinos:

```
🔥 IMAGINARIO (ideas)     → El Sistema 7×7 como visión
❄️ CONCEPTUAL (estructura) → Los 49 primitivos, reglas, operaciones
📏 EXACTO (verificación)   → Aritmética de primos, TriadicValidator
🔨 MATERIAL (construcción) → Este tokenizador, el código que corre
```

Los tres primeros están sólidos. El cuarto (este código) es lo que estamos construyendo.

La regla de tres del libro predice: con los tres primeros bien definidos, el cuarto se deriva. Construir es "lo relativamente fácil" — ya tenemos todo.

---

## Archivos clave en otros repos

| Archivo | Repo | Qué tiene |
|---------|------|-----------|
| `sistema-7x7/Sistema_7x7_v2.md` | la-danza-cosmica | Spec completa del 7×7 (69KB) |
| `sistema-7x7/los_tres_reinos.md` | la-danza-cosmica | Marco teórico (Cuatro Reinos) |
| `sistema-7x7/edge_cases_7x7.md` | la-danza-cosmica | Stress test: 49 casos, 65% funciona |
| `sistema-7x7/borrador_cap_tres_reinos.md` | la-danza-cosmica | Capítulo para el libro |
| `playground/PLAN.md` | triadic-microgpt | Plan original de experimentos |
| `experiment_log.md` | triadic-microgpt | Log de 41 runs |

---

## Notas del autor

- J. Arturo Ornelas Brand, arquitecto
- El proceso de diseño arquitectónico (imaginar→organizar→calcular→construir) es exactamente los Cuatro Reinos
- "Con conceptos base fuertes, construir es lo relativamente fácil"
- El libro "La Danza Cósmica de los Opuestos" documenta la teoría completa (32 capítulos, 3 rondas de revisión editorial)
- Papers en Zenodo: DOI 10.5281/zenodo.15169702 y 10.5281/zenodo.15384498 (no peer-reviewed)
