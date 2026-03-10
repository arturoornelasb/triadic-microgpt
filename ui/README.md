# TriadicGPT Explorer — Desktop UI

Interactive desktop application for exploring, auditing, and conversing with TriadicGPT. Built with PySide6 (Qt for Python) and embedded matplotlib.

## Installation

```bash
conda activate triadic-microgpt
pip install PySide6>=6.4
```

## Launch

```bash
cd /path/to/triadic-microgpt
python ui/app.py
```

---

## Interface Overview

The application has a persistent **model panel** at the top and **7 tabs** below. Tabs 1–6 require a loaded model; Tab 7 (Benchmarks) works standalone.

```
┌────────────────────────────────────────────────────────────────────┐
│  TriadicGPT Explorer       Backend: [TriadicGPT nativo]            │
│  Checkpoint: [checkpoints/torch_run15_strongalign/...]  […] [Cargar]│
│  Tokenizer:  [checkpoints/torch/tokenizer.json]          […]       │
│  ████████████████████████████████████████████████ (loading bar)    │
├──────────┬──────────┬──────────┬──────────┬──────────┬────────┬───┤
│ Encoder  │ Comparar │ Explorar │ Analogía │ Validar  │  Chat  │ B │
├──────────┴──────────┴──────────┴──────────┴──────────┴────────┴───┤
│                          [tab content]                             │
└────────────────────────────────────────────────────────────────────┘
│ Modelo cargado: 12L/512D/8H/64bits | CUDA | 40.1M params           │
```

---

## Model Panel

Located at the top of the window. Supports two backends:

| Field | Native `.pt` | HuggingFace |
|-------|-------------|-------------|
| Backend | TriadicGPT nativo (.pt) | HuggingFace (TriadicWrapper) |
| Checkpoint | Path to `.pt` file | HF model name or `.pt` weights path |
| Tokenizer | Path to `tokenizer.json` | Hidden (HF handles it) |
| Bits (n_bits) | Fixed from checkpoint config | 8–128, default 64 |
| Align mode | — | `infonce` / `rank` / `mse` |

**Default paths:**
- Checkpoint: `checkpoints/torch_run15_strongalign/model_L12_D512_B64_best.pt`
- Tokenizer: `checkpoints/torch/tokenizer.json`

After clicking **Cargar Modelo**, a background thread loads the checkpoint (UI stays responsive). The status label turns green on success and shows model info.

---

## Tab 1 — Encoder

**Purpose:** Visualize how any text is encoded into a prime signature.

**Workflow:**
1. Type a concept or phrase in the text box
2. Click **Encode →**
3. View the resulting bit vector, prime composite Φ, and projection bar chart

**Output panels:**
- **BIT VECTOR** — 64 colored squares. Green = active bit, dark = inactive. Hover shows `Bit i — prime pᵢ = N | val = X.XXX`
- **PRIME COMPOSITE** — Φ = 2 × 3 × 7 × 11 × ... (copy to clipboard with "Copiar Φ")
- **PROJECTION** — Bar chart of raw tanh activations per bit. Green bars = active (>0), red = inactive

---

## Tab 2 — Compare

**Purpose:** Compare two concepts algebraically.

**Workflow:**
1. Enter concept A and concept B
2. Click **Compare →**

**Output:**
- **Similitud %** — Jaccard similarity over prime factor sets
- **A ⊇ B / B ⊇ A** — Subsumption (divisibility check)
- **Factor rows** — Shared factors (green), only in A (yellow), only in B (peach)
- **LCM composition** — A∪B = lcm(Φ_A, Φ_B)
- **Bit vectors** — Side-by-side comparison. Color encoding:
  - Green = shared active bit
  - Yellow = only A active
  - Peach = only B active
  - Dark = both inactive

---

## Tab 3 — Explore

**Purpose:** Compute a pairwise similarity matrix for N concepts.

**Workflow:**
1. Add words with the text field + **+ Agregar** (or press Enter)
2. Select and click **Eliminar** to remove one; **Limpiar** to reset
3. Click **Explore →**

**Default word set:** king, queen, dog, cat, doctor, hospital

**Output:**
- **Similarity heatmap** — N×N matrix, RdYlGn colormap (0=red, 1=green), values annotated
- **Ranked pairs table** — All unique pairs sorted by similarity (highest first), with count of shared prime factors

---

## Tab 4 — Analogy

**Purpose:** Solve and verify prime algebra analogies A:B::C:?

**Workflow:**
1. Fill in A, B, C (e.g. `king`, `queen`, `man`)
2. Click **Calcular →**

**Output:**
- **Transformación A→B** — Which prime factors were removed (red) and added (green) from A to B
- **TARGET Φ** — The computed target prime: `Φ_target = (Φ_C / gcd(Φ_A, Φ_C)) × (Φ_B / gcd(Φ_A, Φ_B))`
- **Results table** — Top-10 vocabulary matches ranked by similarity to Φ_target. ✓ = similarity ≥ median pool similarity
- **Verification** — Top-1 similarity vs median; shows VERIFICA / NO verifica

**Note:** The default vocabulary pool is ~100 common English words from `data/core_concepts.txt` (or built-in fallback). For larger search, add a `data/core_concepts.txt` with one word per line.

---

## Tab 5 — Validate

**Purpose:** Run the full semantic quality audit suite.

**Checks:**
| Check | Passes if |
|-------|-----------|
| **Diversity** | ≥ 75% of encoded concepts have unique prime signatures |
| **Active Bits** | Mean active bit fraction is between 15% and 85% |
| **Semantic Ordering** | Mean intra-group similarity > mean inter-group similarity (gap > 0) |

**Workflow:**
1. Edit word groups in the table (add groups with name + comma-separated words)
2. Click **Run Validation →**

**Default groups:** royalty, animals, medicine, technology

**Output:**
- Global **PASS / FAIL** banner
- Three colored check cards (green = pass, red = fail)
- Per-group table: intra similarity, inter similarity, gap, status

---

## Tab 6 — Chat

**Purpose:** Converse with TriadicGPT and inspect the triadic signature of each turn in real time.

> **Note:** Chat is only available with the native `.pt` backend. HuggingFace mode will return an error.

**Layout (horizontal splitter):**

```
┌─────────────────────────────────┬──────────────────────┐
│  [conversation history]         │ ANÁLISIS TRIÁDICO     │
│                                 │ PREGUNTA             │
│  Usuario: What is a king?       │ Φ: 11688...          │
│  IA: A king is a ruler...       │ [bit vector]         │
│                                 │ RESPUESTA            │
│  [write message...] [Enviar]    │ Φ: 25221...          │
│  [Limpiar]                      │ [bit vector]         │
│                                 │ SIMILITUD: 51%       │
│                                 │ RESP ⊇ PREG: ✗ No   │
│                                 │ COMPARTIDOS: 2 3 7   │
│                                 │ RESP tiene de más:   │
│                                 │ 5 17 23              │
└─────────────────────────────────┴──────────────────────┘
```

The right panel updates after each AI response with the prime signature of the **last prompt** and **last response**, plus their algebraic relationship.

---

## Tab 7 — Benchmarks

**Purpose:** Browse all stored benchmark results in `benchmarks/results/`. **Does not require a loaded model.**

**Workflow:**
1. Select a JSON file from the list (filter by typing in the search box, click ↻ to refresh)
2. Results load automatically

**Supported benchmark types:**

| File pattern | Display |
|---|---|
| `*analogy*` | Top-1/Top-5/Verification accuracy cards + details table + bar chart |
| `*bit_entropy*` | Mean entropy + dead bits cards + per-bit bar chart (red = dead bit H<0.3) |
| `*language*` | PPL card + generated text samples |
| `*subsumption*` / others | Key metrics cards + generic table + bar chart |

**Results directory:** `benchmarks/results/` — 27 JSON files from all experimental runs.

---

## Architecture

```
ui/
├── app.py                  # QApplication entry point; loads style.qss
├── main_window.py          # QMainWindow: ModelPanel + QTabWidget(7) + QStatusBar
├── model_panel.py          # Top bar widget; emits model_loaded(ModelInterface)
├── model_interface.py      # Unified encode/compare/explore/analogy/validate/chat API
│
├── workers/
│   └── model_worker.py     # ModelLoadWorker (load .pt or HF) + TaskWorker (generic)
│
├── widgets/
│   ├── bit_vector_widget.py    # Custom QWidget: N colored squares (binary/gradient/compare)
│   ├── prime_display_widget.py # Prime composite label + factors + copy button
│   └── mpl_canvas.py           # FigureCanvasQTAgg wrapper with dark theme
│
├── tabs/
│   ├── encoder_tab.py      # Tab 1
│   ├── compare_tab.py      # Tab 2
│   ├── explore_tab.py      # Tab 3
│   ├── analogy_tab.py      # Tab 4
│   ├── validate_tab.py     # Tab 5
│   ├── chat_tab.py         # Tab 6
│   └── benchmarks_tab.py  # Tab 7 (no model needed)
│
└── resources/
    └── style.qss           # Dark theme (Catppuccin Mocha)
```

### Threading model

All model inference runs in `QThread` workers to keep the UI responsive:

```python
worker = TaskWorker(iface.encode, "king")
worker.result_ready.connect(on_result)   # fires in main thread via Qt signal
worker.error_occurred.connect(on_error)
worker.start()
```

`ModelLoadWorker` is a specialized worker for the slow initial model load (emits `progress` messages during loading).

### ModelInterface API

```python
iface.encode(text)          -> {composite, bits, projection, n_active, factors}
iface.compare(a, b)         -> {similarity, shared_factors, only_a_factors, only_b_factors,
                                a_subsumes_b, b_subsumes_a, composition, enc_a, enc_b}
iface.explore([words])      -> {matrix, words, pairs, signatures}
iface.analogy(a, b, c)      -> {target_prime, transform_added, transform_removed,
                                matches, median_sim}
iface.validate({groups})    -> {checks, overall_pass, group_details, n_concepts,
                                unique_signatures}
iface.chat(prompt)          -> {response, prompt_prime, prompt_bits, response_prime,
                                response_bits, similarity, resp_subsumes_prompt,
                                shared_factors, resp_extra_factors}
```

Both native `.pt` and HuggingFace (`TriadicWrapper`) backends expose the same API. Chat is native-only.

---

## Customization

### Using a different model

Any `.pt` checkpoint trained with `src/torch_train.py` works. Select it via the **Browse (…)** button or type the path directly. The tokenizer must match the training run (all runs ≥ Run 7 use `checkpoints/torch/tokenizer.json`).

### Using a HuggingFace model

Switch Backend to **HuggingFace (TriadicWrapper)** and enter a HF model name (e.g. `gpt2`, `gpt2-medium`) or a path to a saved `.pt` weights file for the triadic head. Configure:
- **Bits (n_bits):** Number of triadic bits (8–128)
- **Align:** Alignment loss mode used during training (`infonce` recommended for pre-trained models)

### Extending the vocabulary pool (Analogy tab)

Create `data/core_concepts.txt` with one concept per line. The Analogy tab will use this file as the search pool instead of the ~100-word built-in fallback.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: PySide6` | `pip install PySide6` |
| `Checkpoint no encontrado` | Use the Browse button to select the `.pt` file |
| `Tokenizer no encontrado` | Use `checkpoints/torch/tokenizer.json` (works for all runs ≥ 7) |
| Chat returns error on HF backend | Chat requires native `.pt` backend |
| Heatmap is blank | Add ≥ 2 words to the Explore list before clicking Explore |
| App launches but tabs are grey | Load a model first via the top bar |
