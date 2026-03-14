"""
Configuration for the Conceptual Tokenizer.

Defines the 49 primitives of the Sistema 7×7 with their prime assignments,
categories, layers, and state thresholds.
"""

from dataclasses import dataclass, field


# ─────────────────────────────────────────────
# Prime assignments: first 49 primes (2..227)
# Ordered by category position in the 7×7
# ─────────────────────────────────────────────

PRIMES_BY_CATEGORY = {
    "ELEMENTOS": {
        "Fuego": 2,
        "Tierra": 3,
        "Agua": 5,
        "Aire": 7,
        "Vacío": 11,
        "Información": 13,
        "Fuerza": 17,
    },
    "CARACTERÍSTICAS": {
        "Color": 19,
        "Textura": 23,
        "Forma": 29,
        "Material": 31,
        "Brillo": 37,
        "Transparencia": 41,
        "Estado_materia": 43,
    },
    "ESPACIO": {
        "Arriba": 47,
        "Abajo": 53,
        "En_medio": 59,
        "Adelante": 61,
        "Atrás": 67,
        "Izquierda_Derecha": 71,
        "Dentro_Fuera": 73,
    },
    "TIEMPO": {
        "Presente": 79,
        "Pasado": 83,
        "Futuro": 89,
        "Pausa": 97,
        "Ir_al_pasado": 101,
        "Ir_al_futuro": 103,
        "Play": 107,
    },
    "SENTIDOS": {
        "Vista": 109,
        "Oído": 113,
        "Tacto": 127,
        "Gusto": 131,
        "Olfato": 137,
        "Equilibrio": 139,
        "Interocepción": 149,
    },
    "PRINCIPIOS_DUALES": {
        "Bien_Mal": 151,
        "Orden_Caos": 157,
        "Creación_Destrucción": 163,
        "Unión_Separación": 167,
        "Verdad_Mentira": 173,
        "Libertad_Control": 179,
        "Vida_Muerte": 181,
    },
    "OBSERVADORES": {
        "Consciente": 191,
        "Temporal": 193,
        "Eterno": 197,
        "Individual": 199,
        "Colectivo": 211,
        "Ausente": 223,
        "Creador": 227,
    },
}

# Flat mapping: primitive_name -> prime
PRIMITIVE_TO_PRIME = {}
for cat_primes in PRIMES_BY_CATEGORY.values():
    PRIMITIVE_TO_PRIME.update(cat_primes)

# Reverse: prime -> primitive_name
PRIME_TO_PRIMITIVE = {v: k for k, v in PRIMITIVE_TO_PRIME.items()}

# Ordered list of all 49 primitive names (determines projection head positions)
PRIMITIVE_NAMES = list(PRIMITIVE_TO_PRIME.keys())

# Ordered list of all 49 primes
PRIMITIVE_PRIMES = [PRIMITIVE_TO_PRIME[name] for name in PRIMITIVE_NAMES]

# Category membership
PRIMITIVE_TO_CATEGORY = {}
for cat_name, cat_primes in PRIMES_BY_CATEGORY.items():
    for prim_name in cat_primes:
        PRIMITIVE_TO_CATEGORY[prim_name] = cat_name

# Category names ordered
CATEGORY_NAMES = list(PRIMES_BY_CATEGORY.keys())

# Number of primitives
N_PRIMITIVES = 49

# ─────────────────────────────────────────────
# Three-layer architecture
# ─────────────────────────────────────────────

LAYERS = {
    "MUNDO": ["ELEMENTOS", "CARACTERÍSTICAS", "ESPACIO", "TIEMPO", "SENTIDOS"],
    "FUERZAS": ["PRINCIPIOS_DUALES"],
    "CONSCIENCIA": ["OBSERVADORES"],
}

CATEGORY_TO_LAYER = {}
for layer_name, categories in LAYERS.items():
    for cat in categories:
        CATEGORY_TO_LAYER[cat] = layer_name

# ─────────────────────────────────────────────
# Dual principles: bipolar axes
# Positive projection = first pole, negative = second pole
# ─────────────────────────────────────────────

DUAL_POLES = {
    "Bien_Mal": ("Bien", "Mal"),
    "Orden_Caos": ("Orden", "Caos"),
    "Creación_Destrucción": ("Creación", "Destrucción"),
    "Unión_Separación": ("Unión", "Separación"),
    "Verdad_Mentira": ("Verdad", "Mentira"),
    "Libertad_Control": ("Libertad", "Control"),
    "Vida_Muerte": ("Vida", "Muerte"),
}

# Indices of dual principles in the projection vector
DUAL_INDICES = {name: PRIMITIVE_NAMES.index(name) for name in DUAL_POLES}

# ─────────────────────────────────────────────
# State resolution thresholds
# ─────────────────────────────────────────────

@dataclass
class StateConfig:
    """Thresholds for resolving continuous projections to discrete states."""
    # Below this absolute value → N/A (not applicable)
    na_threshold: float = 0.1
    # Intensity bands
    low_threshold: float = 0.3
    high_threshold: float = 0.7


@dataclass
class TokenizerConfig:
    """Full configuration for the conceptual tokenizer."""
    n_primitives: int = N_PRIMITIVES
    state: StateConfig = field(default_factory=StateConfig)
    # Sigmoid annealing (from soft_signatures.py experiments)
    anneal_start_temp: float = 5.0
    anneal_end_temp: float = 0.5
    anneal_steps: int = 10000
