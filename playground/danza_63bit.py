"""
63-Bit Danza Cósmica — End-to-End Training with Real Primitives.

Trains TriadicGPT with 63 bits mapped to the VERIFIED primitives from
"La Danza Cósmica de los Opuestos" (Sistema 7×7 v3.4, toolkit).

Supervision source: 50 manually-factorized anchor concepts (anclas.json),
each with written justification per bit. NO AI-generated factorizations.

This bridges the symbolic algebra (toolkit) with neural training (TriadicGPT).

Loss = L_lang + alpha * (L_triadic + sub_weight * L_sub + sup_weight * L_sup)

Usage:
  python playground/danza_63bit.py                       # base, 10K, smoke test
  python playground/danza_63bit.py --scale xl --steps 50000  # full run (~76 min)
"""

import os
import sys
import csv
import json
import math
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.torch_transformer import TriadicGPT, TriadicGPTConfig
try:
    from src.fast_tokenizer import FastBPETokenizer as BPETokenizer
except ImportError:
    from src.tokenizer import BPETokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORY_SEPARATOR = '<' + '|endoftext|' + '>'

# Data: look locally first (playground/danza_data/), then external repo
_LOCAL_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'danza_data')
_EXTERNAL = os.path.join(os.path.dirname(PROJECT_ROOT), 'la-danza-cosmica-de-los-opuestos',
                          'inventario_de_opuestos', 'toolkit')
TOOLKIT_DIR = _LOCAL_DATA if os.path.isdir(_LOCAL_DATA) else _EXTERNAL

N_BITS = 63  # Real primitive count from Sistema 7×7


# ============================================================
# Load primitives and anchors from the Danza toolkit
# ============================================================

def load_primitives():
    """Load the 63 primitives from primitivos.json."""
    path = os.path.join(TOOLKIT_DIR, 'primitivos.json')
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    primitives = data['primitivos']
    name_to_bit = {p['nombre']: p['bit'] for p in primitives}
    name_to_prime = {p['nombre']: p['primo'] for p in primitives}
    bit_to_name = {p['bit']: p['nombre'] for p in primitives}
    deps = {p['nombre']: p['deps'] for p in primitives}
    duals = {}
    for p in primitives:
        if 'dual' in p:
            duals[p['nombre']] = p['dual']

    return {
        'primitives': primitives,
        'name_to_bit': name_to_bit,
        'name_to_prime': name_to_prime,
        'bit_to_name': bit_to_name,
        'deps': deps,
        'duals': duals,
        'dual_axes': data['ejes_duales'],
    }


def expand_bits(frontier_names, deps):
    """Expand frontier primitives with all transitive dependencies."""
    expanded = set(frontier_names)
    queue = list(frontier_names)
    while queue:
        current = queue.pop()
        for dep in deps.get(current, []):
            if dep not in expanded:
                expanded.add(dep)
                queue.append(dep)
    return sorted(expanded)


def make_target_vector(active_names, name_to_bit):
    """Create a 63-element target vector: +1 for active bits, -1 for inactive."""
    target = torch.full((N_BITS,), -1.0)
    for name in active_names:
        bit = name_to_bit[name]
        target[bit] = 1.0
    return target


# Spanish anchor name → English words likely in TinyStories
ANCHOR_TRANSLATIONS = {
    'frío': ['cold'],
    'caliente': ['hot'],
    'amor': ['love'],
    'odio': ['hate'],
    'indiferencia': ['indifference'],
    'hombre': ['man'],
    'mujer': ['woman'],
    'bueno': ['good'],
    'malo': ['bad', 'evil'],
    'sabio': ['wise'],
    'ignorante': ['ignorant'],
    'creativo': ['creative'],
    'lógico': ['logical'],
    'vivo': ['alive'],
    'muerto': ['dead'],
    'feliz': ['happy'],
    'triste': ['sad'],
    'rey': ['king'],
    'reina': ['queen'],
    'sol': ['sun'],
    'luna': ['moon'],
    'oscuridad': ['darkness'],
    'libre': ['free'],
    'preso': ['prisoner'],
    'rápido': ['fast', 'quick'],
    'lento': ['slow'],
    'inmóvil': ['still', 'frozen'],
    'rico': ['rich'],
    'pobre': ['poor'],
    'orgulloso': ['proud'],
    'humilde': ['humble'],
    'dulce': ['sweet'],
    'amargo': ['bitter'],
    'ruidoso': ['loud', 'noisy'],
    'silencioso': ['quiet', 'silent'],
    'brillante': ['bright', 'shiny'],
    'oscuro': ['dark'],
    'sólido': ['solid', 'hard'],
    'líquido': ['liquid'],
    'gaseoso': ['gas'],
    'enseñar': ['teach'],
    'aprender': ['learn'],
    'abrir': ['open'],
    'cerrar': ['close'],
    'orden_concepto': ['order'],
    'caos_concepto': ['chaos'],
    'apatía': ['apathy'],
    # Skip: estasis_absoluta, hombre_vaciado, inercia_mental, amoral
    # (won't appear in TinyStories)
}

# Concepts too abstract for TinyStories — skip
SKIP_ANCHORS = {'estasis_absoluta', 'hombre_vaciado', 'inercia_mental', 'amoral'}


def load_anchors(prim_data):
    """Load 50 anchors, expand dependencies, create gold target vectors."""
    path = os.path.join(TOOLKIT_DIR, 'anclas.json')
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    name_to_bit = prim_data['name_to_bit']
    deps = prim_data['deps']

    anchors = {}  # english_word -> {spanish, target_vector, frontier_bits, expanded_bits}
    skipped = []

    for spanish_name, info in raw.items():
        if spanish_name.startswith('_'):
            continue
        if spanish_name in SKIP_ANCHORS:
            skipped.append(spanish_name)
            continue
        if spanish_name not in ANCHOR_TRANSLATIONS:
            skipped.append(spanish_name)
            continue

        frontier = info['bits']
        expanded = expand_bits(frontier, deps)
        target = make_target_vector(expanded, name_to_bit)

        for eng_word in ANCHOR_TRANSLATIONS[spanish_name]:
            anchors[eng_word] = {
                'spanish': spanish_name,
                'frontier': frontier,
                'expanded': expanded,
                'n_active': len(expanded),
                'target': target,
                'razon': info.get('razon', ''),
            }

    return anchors, skipped


def load_anchors_v2(prim_data):
    """Load ~104 additional anchors from anclas_v2.json.

    v2 format has 'en' field directly (no separate translations dict).
    Returns anchors in the same format as load_anchors().
    Does NOT replace v1 — designed to be merged with it.
    """
    path = os.path.join(TOOLKIT_DIR, 'anclas_v2.json')
    if not os.path.exists(path):
        return {}, []

    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    name_to_bit = prim_data['name_to_bit']
    deps = prim_data['deps']

    anchors = {}
    skipped = []

    for spanish_name, info in raw.items():
        if spanish_name.startswith('_'):
            continue
        if not isinstance(info, dict) or 'bits' not in info:
            continue

        eng_word = info.get('en')
        if not eng_word:
            skipped.append(spanish_name)
            continue

        frontier = info['bits']
        # Validate all bits exist
        if not all(b in name_to_bit for b in frontier):
            bad = [b for b in frontier if b not in name_to_bit]
            print(f"  WARNING: {spanish_name} has invalid bits: {bad}")
            skipped.append(spanish_name)
            continue

        expanded = expand_bits(frontier, deps)
        target = make_target_vector(expanded, name_to_bit)

        anchors[eng_word] = {
            'spanish': spanish_name,
            'frontier': frontier,
            'expanded': expanded,
            'n_active': len(expanded),
            'target': target,
            'razon': info.get('razon', ''),
        }

    return anchors, skipped


def load_all_anchors(prim_data):
    """Load v1 + v2 anchors merged. v1 takes priority on conflicts."""
    v1, skip1 = load_anchors(prim_data)
    v2, skip2 = load_anchors_v2(prim_data)

    # Merge: v1 wins on conflicts
    merged = {**v2, **v1}
    return merged, skip1 + skip2


# ============================================================
# Subsumption pairs from anchors
# ============================================================

def build_subsumption_pairs(anchors, prim_data):
    """
    Build hypernym-hyponym pairs from anchor concepts.

    Logic: if concept A's active bits are a strict subset of concept B's,
    then A subsumes B. We find all such pairs among the 50 anchors.

    Also use the dual-axis structure: for each dual pair where one anchor
    has pole_a and another has pole_b, neither subsumes the other.
    """
    train_pairs = []
    test_pairs = []

    # Group by English word
    items = [(word, data) for word, data in anchors.items()]

    # Find subsumption pairs
    all_pairs = []
    for i, (w_a, d_a) in enumerate(items):
        bits_a = set(d_a['expanded'])
        for j, (w_b, d_b) in enumerate(items):
            if i == j:
                continue
            bits_b = set(d_b['expanded'])
            # A subsumes B if A's bits ⊆ B's bits (A is hypernym)
            if bits_a < bits_b:  # strict subset
                all_pairs.append((w_a, w_b, d_a, d_b))

    random.seed(42)
    random.shuffle(all_pairs)
    n_test = max(1, int(len(all_pairs) * 0.2))
    test_pairs = all_pairs[:n_test]
    train_pairs = all_pairs[n_test:]

    return train_pairs, test_pairs


# ============================================================
# Regla de Tres evaluation
# ============================================================

# Hand-picked regla de tres quads from the anchors
REGLA_DE_TRES_QUADS = [
    # A:B = C:D — the transformation from A→B applied to C should yield D
    ('man', 'woman', 'king', 'queen'),      # tierra→agua
    ('cold', 'hot', 'quiet', 'loud'),        # tierra→fuego, orden→caos, control→libertad
    ('happy', 'sad', 'love', 'hate'),        # placer→dolor, unión→separación
    ('open', 'close', 'free', 'prisoner'),   # libertad→control, separación→unión
    ('bright', 'dark', 'loud', 'quiet'),     # más→menos (approx)
    ('teach', 'learn', 'king', 'queen'),     # creador_obs→receptivo (NOT exact, but testable)
]


@torch.no_grad()
def evaluate_regla_de_tres(model, tokenizer, anchors, device):
    """
    Evaluate regla de tres: A:B = C:D
    Transform = bits_only_in_B - bits_only_in_A, applied to C → should ≈ D.

    We measure cosine similarity between predicted D and actual D projections.
    """
    model.eval()
    results = []

    def get_proj(word):
        ids = tokenizer.encode(word, add_special=False)[:4]
        if not ids:
            return None
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = model(x)
        proj = out[1]
        return proj[0].mean(dim=0)  # (63,)

    for a_word, b_word, c_word, d_word in REGLA_DE_TRES_QUADS:
        if not all(w in anchors for w in [a_word, b_word, c_word, d_word]):
            continue

        pa, pb, pc, pd = get_proj(a_word), get_proj(b_word), get_proj(c_word), get_proj(d_word)
        if any(p is None for p in [pa, pb, pc, pd]):
            continue

        # Neural regla de tres: predicted_d = pc + (pb - pa)
        predicted_d = pc + (pb - pa)

        # Cosine similarity between prediction and actual
        cos = F.cosine_similarity(predicted_d.unsqueeze(0), pd.unsqueeze(0)).item()

        # Also check binary version
        pred_bits = (predicted_d > 0).long()
        actual_bits = (pd > 0).long()
        bit_match = (pred_bits == actual_bits).float().mean().item()

        results.append({
            'quad': f"{a_word}:{b_word}={c_word}:{d_word}",
            'cosine': cos,
            'bit_accuracy': bit_match,
        })

    model.train()
    return results


# ============================================================
# Model (reuse ConceptTriadicGPT pattern)
# ============================================================

class DanzaTriadicGPT(TriadicGPT):
    """TriadicGPT with 63 bits for the Danza primitives."""

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing to reduce VRAM at cost of ~33% speed."""
        self._grad_checkpoint = True

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        assert T <= self.config.block_size

        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        tok_emb = self.wte(input_ids)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            if getattr(self, '_grad_checkpoint', False) and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)
        triadic_proj = torch.tanh(self.triadic_head(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, triadic_proj, loss


# ============================================================
# Loss functions
# ============================================================

def supervised_anchor_loss(model, word_tensors, target_vectors, n_sample=32):
    """MSE between model projections and gold anchor targets (all 63 bits)."""
    N = word_tensors.shape[0]
    if N == 0:
        return torch.tensor(0.0, device=word_tensors.device)

    if N > n_sample:
        idx = torch.randperm(N, device=word_tensors.device)[:n_sample]
        w_batch = word_tensors[idx]
        t_batch = target_vectors[idx]
    else:
        w_batch = word_tensors
        t_batch = target_vectors

    proj = model(w_batch)[1]         # (n, T, 63)
    pred = proj.mean(dim=1)          # (n, 63) mean-pool tokens

    return F.mse_loss(pred, t_batch)


def subsumption_loss(model, hyper_t, hypo_t, n_sample=32):
    """Subsumption: relu(hyper_01 - hypo_01).mean()"""
    N = hyper_t.shape[0]
    if N == 0:
        return torch.tensor(0.0, device=hyper_t.device)

    if N > n_sample:
        idx = torch.randperm(N, device=hyper_t.device)[:n_sample]
        h_batch, y_batch = hyper_t[idx], hypo_t[idx]
    else:
        h_batch, y_batch = hyper_t, hypo_t

    h_proj = model(h_batch)[1]
    y_proj = model(y_batch)[1]

    h_01 = (h_proj.mean(dim=1) + 1) / 2
    y_01 = (y_proj.mean(dim=1) + 1) / 2

    return F.relu(h_01 - y_01).mean()


def triadic_loss(proj, align_weight, wte, input_ids):
    """Standard 4-component triadic loss (diversity + contrastive + entropy + alignment)."""
    B, T, K = proj.shape

    # 1. Diversity: bit means → 0
    bit_means = proj.mean(dim=(0, 1))
    l_div = (bit_means ** 2).mean()

    # 2. Contrastive: push sequences apart
    seq_means = proj.mean(dim=1)  # (B, K)
    seq_norms = F.normalize(seq_means, dim=1)
    sim_matrix = seq_norms @ seq_norms.T
    mask = ~torch.eye(B, dtype=torch.bool, device=proj.device)
    l_ctr = (sim_matrix[mask] ** 2).mean() if mask.sum() > 0 else torch.tensor(0.0, device=proj.device)

    # 3. Entropy: maximize per-bit entropy
    q = (bit_means + 1) / 2  # map [-1,1] → [0,1]
    eps = 1e-7
    bit_ent = -(q * torch.log2(q + eps) + (1 - q) * torch.log2(1 - q + eps))
    l_ent = 1.0 - bit_ent.mean()

    # 4. Embedding alignment (MSE)
    with torch.no_grad():
        emb = wte(input_ids)  # (B, T, D)
        emb_norm = F.normalize(emb, dim=-1)

    proj_norm = F.normalize(proj, dim=-1)

    n_pairs = min(64, T * T)
    idx_i = torch.randint(0, T, (n_pairs,), device=proj.device)
    idx_j = torch.randint(0, T, (n_pairs,), device=proj.device)

    sim_emb = (emb_norm[:, idx_i] * emb_norm[:, idx_j]).sum(dim=-1)
    sim_tri = (proj_norm[:, idx_i] * proj_norm[:, idx_j]).sum(dim=-1)

    l_align = F.mse_loss(sim_tri, sim_emb.detach())

    return l_div + l_ctr + l_ent + align_weight * l_align


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_anchors(model, word_tensors, target_vectors, valid_words):
    """Per-anchor bit accuracy and top-1 primitive match."""
    model.eval()
    N = word_tensors.shape[0]
    if N == 0:
        model.train()
        return {}

    proj = model(word_tensors)[1]
    pred = proj.mean(dim=1)  # (N, 63)

    # Bit accuracy (per concept)
    pred_bits = (pred > 0).float()
    target_bits = (target_vectors > 0).float()
    per_concept_acc = (pred_bits == target_bits).float().mean(dim=1)

    # Overall
    mean_bit_acc = per_concept_acc.mean().item()

    # Per-word details (top 5 worst)
    results_per_word = []
    for i in range(N):
        results_per_word.append({
            'word': valid_words[i],
            'bit_accuracy': per_concept_acc[i].item(),
            'n_correct': int((pred_bits[i] == target_bits[i]).sum().item()),
        })

    results_per_word.sort(key=lambda x: x['bit_accuracy'])

    # Dead bits & entropy
    all_pred = pred.cpu().numpy()
    bit_means = (all_pred > 0).mean(axis=0)
    eps = 1e-7
    ent = -(bit_means * np.log2(bit_means + eps) +
            (1 - bit_means) * np.log2(1 - bit_means + eps))
    dead_bits = int((ent < 0.3).sum())

    model.train()
    return {
        'mean_bit_accuracy': mean_bit_acc,
        'dead_bits': dead_bits,
        'mean_entropy': float(ent.mean()),
        'worst_5': results_per_word[:5],
        'best_5': results_per_word[-5:],
        'n_concepts': N,
    }


@torch.no_grad()
def evaluate_subsumption(model, hyper_t, hypo_t, n_total):
    """Binary subsumption satisfaction rate."""
    model.eval()
    N = hyper_t.shape[0]
    if N == 0:
        model.train()
        return 0.0, 0.0

    out = model(hyper_t)
    h_proj = out[1]
    out = model(hypo_t)
    y_proj = out[1]

    h_bits = (h_proj.mean(dim=1) > 0).float()
    y_bits = (y_proj.mean(dim=1) > 0).float()

    violations = (h_bits * (1 - y_bits)).sum(dim=1)
    satisfied = (violations == 0).float().sum().item()

    model.train()
    return satisfied / N, violations.mean().item()


# ============================================================
# Dataset
# ============================================================

class TextDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.tokens) - self.block_size - 1)

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.block_size + 1]
        return (torch.tensor(chunk[:-1], dtype=torch.long),
                torch.tensor(chunk[1:], dtype=torch.long))


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='63-Bit Danza Cósmica (End-to-End)')
    parser.add_argument('--scale', choices=['base', 'xl', 'xxl', 'huge'], default='base')
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--sub-weight', type=float, default=5.0)
    parser.add_argument('--sup-weight', type=float, default=2.0)
    parser.add_argument('--align-weight', type=float, default=3.0)
    parser.add_argument('--triadic-warmup-pct', type=float, default=0.5)
    parser.add_argument('--stories', type=int, default=50000)
    parser.add_argument('--vocab', type=int, default=4096)
    parser.add_argument('--block', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--grad-checkpoint', action='store_true',
                        help='Use gradient checkpointing (saves VRAM, +33%% time)')
    parser.add_argument('--no-compile', action='store_true',
                        help='Disable torch.compile (for debugging)')
    parser.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'],
                        default='bfloat16', help='Mixed precision dtype (default: bfloat16)')
    parser.add_argument('--print-every', type=int, default=50)
    parser.add_argument('--save-every', type=int, default=2500)
    parser.add_argument('--eval-every', type=int, default=1000)
    parser.add_argument('--v2', action='store_true',
                        help='Use expanded anchors (anclas.json + anclas_v2.json = 158 concepts)')
    args = parser.parse_args()

    SCALES = {
        'base': {'layers': 6,  'dim': 256,  'heads': 8},   # ~5M params
        'xl':   {'layers': 12, 'dim': 512,  'heads': 8},   # ~40M params
        'xxl':  {'layers': 24, 'dim': 1024, 'heads': 16},  # ~307M params
        'huge': {'layers': 26, 'dim': 1280, 'heads': 20},  # ~517M params (needs grad ckpt)
    }
    preset = SCALES[args.scale]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')   # TF32 for residual float32 ops
        torch.backends.cudnn.benchmark = True         # kernel autotuning
    suffix = f'{args.scale}_v2' if args.v2 else args.scale
    ckpt_dir = os.path.join(PROJECT_ROOT, 'checkpoints', f'danza_63bit_{suffix}')
    os.makedirs(ckpt_dir, exist_ok=True)

    # --- 0. Load primitives & anchors ---
    print()
    print("=" * 70)
    print("  63-BIT DANZA CÓSMICA  —  End-to-End with Real Primitives")
    print("=" * 70)
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print(f"\n[0/6] Loading primitives from La Danza toolkit...")
    prim_data = load_primitives()
    print(f"  Primitives: {len(prim_data['primitives'])} (bits 0-62)")
    print(f"  Dual axes: {len(prim_data['dual_axes'])}")
    print(f"  Dependency chains: {sum(len(v) for v in prim_data['deps'].values())} total deps")

    if args.v2:
        anchors, skipped = load_all_anchors(prim_data)
        print(f"  Anchors loaded: {len(anchors)} English words (v1 + v2 merged)")
    else:
        anchors, skipped = load_anchors(prim_data)
        print(f"  Anchors loaded: {len(anchors)} English words from {len(ANCHOR_TRANSLATIONS)} Spanish concepts")
    if skipped:
        print(f"  Skipped (won't appear in TinyStories): {skipped}")

    # Show a few examples
    for word in ['love', 'hate', 'king', 'queen', 'cold']:
        if word in anchors:
            a = anchors[word]
            n = a['n_active']
            print(f"    {word} ({a['spanish']}): {n} bits active — frontier: {a['frontier']}")

    # --- 1. Subsumption pairs ---
    print(f"\n[1/6] Building subsumption pairs from anchors...")
    train_sub, test_sub = build_subsumption_pairs(anchors, prim_data)
    print(f"  Subsumption pairs: train={len(train_sub)}, test={len(test_sub)}")
    for h_w, y_w, _, _ in train_sub[:3]:
        print(f"    {h_w} ⊆ {y_w}")

    # --- 2. Tokenizer ---
    data_path = os.path.join(PROJECT_ROOT, 'data', 'TinyStories-train.txt')
    with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()
    stories = [s.strip() for s in raw.split(STORY_SEPARATOR) if s.strip() and len(s.strip()) > 30]
    if args.stories and len(stories) > args.stories:
        random.seed(42)
        random.shuffle(stories)
        stories = stories[:args.stories]

    tok_path = os.path.join(ckpt_dir, 'tokenizer.json')
    print(f"\n[2/6] Training BPE tokenizer (vocab={args.vocab})...")
    tokenizer = BPETokenizer(vocab_size=args.vocab)
    tokenizer.train(stories, verbose=True)
    tokenizer.save(tok_path)
    print(f"  Vocab: {tokenizer.vocab_size}")

    # Check which anchors tokenize to single tokens (ideal for supervision)
    single_tok = 0
    multi_tok = 0
    for word in anchors:
        ids = tokenizer.encode(word, add_special=False)
        if len(ids) == 1:
            single_tok += 1
        else:
            multi_tok += 1
    print(f"  Anchor tokenization: {single_tok} single-token, {multi_tok} multi-token")

    # --- 3. Tokenize ---
    print(f"\n[3/6] Tokenizing {len(stories)} stories...")
    all_tokens = []
    for i, story in enumerate(stories):
        ids = tokenizer.encode(story, add_special=True)
        all_tokens.extend(ids)
        if (i + 1) % 10000 == 0:
            print(f"  {i+1}/{len(stories)} ({len(all_tokens):,} tokens)")
    print(f"  Total: {len(all_tokens):,} tokens")

    # --- 4. Model ---
    print(f"\n[4/6] Initializing DanzaTriadicGPT ({N_BITS} bits)...")
    config = TriadicGPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block,
        n_layer=preset['layers'],
        n_embd=preset['dim'],
        n_head=preset['heads'],
        n_triadic_bits=N_BITS,
        dropout=args.dropout,
    )
    model = DanzaTriadicGPT(config).to(device)
    total_params = model.num_params()
    print(f"  Scale: {args.scale} ({preset['layers']}L/{preset['dim']}D/{preset['heads']}H)")
    print(f"  Parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    # Gradient checkpointing (saves VRAM for xxl/huge)
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        print(f"  Gradient checkpointing: ON")

    # torch.compile — fuse CUDA kernels (10-30% speedup, requires Triton/Linux)
    if device.type == 'cuda' and not args.no_compile:
        try:
            import triton  # noqa: F401
            model = torch.compile(model)
            print(f"  torch.compile: ON")
        except ImportError:
            print(f"  torch.compile: SKIPPED (triton not available)")
    else:
        if hasattr(args, 'no_compile') and args.no_compile:
            print(f"  torch.compile: DISABLED (--no-compile)")

    # Mixed precision setup
    use_amp = device.type == 'cuda'
    amp_dtype = {'float32': torch.float32, 'float16': torch.float16,
                 'bfloat16': torch.bfloat16}[args.dtype]
    if use_amp and amp_dtype != torch.float32:
        print(f"  Mixed precision: {args.dtype}")
        if amp_dtype == torch.bfloat16:
            print(f"  Blackwell Tensor Cores: bf16 (no loss scaling needed)")
    else:
        print(f"  Precision: float32")

    # VRAM estimate
    if device.type == 'cuda':
        bytes_per_param = 14 if amp_dtype != torch.float32 else 18  # bf16: 14, fp32: 18
        model_vram = total_params * bytes_per_param / 1024**3
        act_per_item = preset['layers'] * args.block * 11 * preset['dim'] * 2 / 1024**3
        total_vram = model_vram + act_per_item * args.batch_size
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  VRAM estimate: {total_vram:.1f} GB / {gpu_mem:.1f} GB "
              f"(model {model_vram:.1f} + activations {act_per_item * args.batch_size:.1f})")

    # Pre-encode anchor supervision tensors
    anchor_words = []
    anchor_ids_list = []
    anchor_targets = []
    for word, data in anchors.items():
        ids = tokenizer.encode(word, add_special=False)[:4]
        if ids:
            anchor_words.append(word)
            anchor_ids_list.append(ids)
            anchor_targets.append(data['target'])

    # Split 80/20
    random.seed(42)
    indices = list(range(len(anchor_words)))
    random.shuffle(indices)
    n_test = max(1, int(len(indices) * 0.2))
    test_idx = set(indices[:n_test])
    train_idx = [i for i in indices if i not in test_idx]

    def _pack(idx_list):
        if not idx_list:
            z = torch.zeros((0, 1), dtype=torch.long, device=device)
            return z, torch.zeros((0, N_BITS), device=device), []
        words = [anchor_words[i] for i in idx_list]
        ids = [anchor_ids_list[i] for i in idx_list]
        tgts = [anchor_targets[i] for i in idx_list]
        mx = max(len(x) for x in ids)
        padded = torch.tensor([x + [0] * (mx - len(x)) for x in ids],
                               dtype=torch.long, device=device)
        target_t = torch.stack(tgts).to(device)
        return padded, target_t, words

    sup_train_t, sup_train_tgt, sup_train_words = _pack(train_idx)
    sup_test_t, sup_test_tgt, sup_test_words = _pack(list(test_idx))
    print(f"  Supervised anchors: train={len(sup_train_words)}, test={len(sup_test_words)}")

    # Pre-encode subsumption pairs
    def _pack_sub(pairs):
        h_ids, y_ids, valid = [], [], []
        for h_w, y_w, h_d, y_d in pairs:
            h = tokenizer.encode(h_w, add_special=False)[:4]
            y = tokenizer.encode(y_w, add_special=False)[:4]
            if h and y:
                h_ids.append(h)
                y_ids.append(y)
                valid.append((h_w, y_w))
        if not valid:
            z = torch.zeros((0, 1), dtype=torch.long, device=device)
            return z, z, valid
        def pad(lst):
            mx = max(len(x) for x in lst)
            return torch.tensor([x + [0] * (mx - len(x)) for x in lst],
                                 dtype=torch.long, device=device)
        return pad(h_ids), pad(y_ids), valid

    sub_train_h, sub_train_y, sub_train_valid = _pack_sub(train_sub)
    sub_test_h, sub_test_y, sub_test_valid = _pack_sub(test_sub)
    print(f"  Subsumption tensors: train={sub_train_h.shape[0]}, test={sub_test_h.shape[0]}")

    # --- 5. Training ---
    print(f"\n[5/6] Training ({args.steps} steps, warmup={args.triadic_warmup_pct:.0%})...")
    dataset = TextDataset(all_tokens, args.block)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, drop_last=True, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   betas=(0.9, 0.999), weight_decay=0.01)
    # bf16 doesn't need loss scaling; float16 does
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler('cuda') if use_scaler else None
    warmup_steps = int(args.steps * 0.05)
    triadic_start = int(args.steps * args.triadic_warmup_pct)

    csv_path = os.path.join(ckpt_dir, 'training_log.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['step', 'loss', 'lang_loss', 'tri_loss', 'sup_loss', 'sub_loss',
                          'bit_acc_train', 'bit_acc_test', 'sub_train', 'sub_test',
                          'dead_bits', 'entropy'])

    data_iter = iter(loader)
    t0 = time.time()
    best_bit_acc = 0.0

    for step in range(1, args.steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        # LR schedule
        if step <= warmup_steps:
            lr = args.lr * step / warmup_steps
        else:
            progress = (step - warmup_steps) / (args.steps - warmup_steps)
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Forward
        if use_amp:
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                logits, proj, lang_loss = model(x, y)
                l_tri = torch.tensor(0.0, device=device)
                l_sup = torch.tensor(0.0, device=device)
                l_sub = torch.tensor(0.0, device=device)

                if step >= triadic_start:
                    l_tri = triadic_loss(proj, args.align_weight, model.wte, x)
                    l_sup = supervised_anchor_loss(model, sup_train_t, sup_train_tgt)
                    l_sub = subsumption_loss(model, sub_train_h, sub_train_y)

                total = lang_loss + args.alpha * (l_tri + args.sup_weight * l_sup + args.sub_weight * l_sub)

            optimizer.zero_grad(set_to_none=True)
            if scaler:  # float16: use loss scaling
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:  # bfloat16 or float32: direct backward
                total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        else:
            logits, proj, lang_loss = model(x, y)
            l_tri = torch.tensor(0.0, device=device)
            l_sup = torch.tensor(0.0, device=device)
            l_sub = torch.tensor(0.0, device=device)

            if step >= triadic_start:
                l_tri = triadic_loss(proj, args.align_weight, model.wte, x)
                l_sup = supervised_anchor_loss(model, sup_train_t, sup_train_tgt)
                l_sub = subsumption_loss(model, sub_train_h, sub_train_y)

            total = lang_loss + args.alpha * (l_tri + args.sup_weight * l_sup + args.sub_weight * l_sub)

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Print
        if step % args.print_every == 0:
            elapsed = time.time() - t0
            tri_str = f"tri={l_tri.item():.4f} sup={l_sup.item():.4f} sub={l_sub.item():.4f}" if step >= triadic_start else "warmup"
            print(f"  [{step:>6d}/{args.steps}] loss={total.item():.4f} lang={lang_loss.item():.4f} "
                  f"{tri_str} lr={lr:.2e} ({elapsed:.0f}s)")

        # Evaluate
        if step % args.eval_every == 0 or step == args.steps:
            eval_train = evaluate_anchors(model, sup_train_t, sup_train_tgt, sup_train_words)
            eval_test = evaluate_anchors(model, sup_test_t, sup_test_tgt, sup_test_words)

            sub_rate_train, _ = evaluate_subsumption(model, sub_train_h, sub_train_y, len(sub_train_valid))
            sub_rate_test, _ = evaluate_subsumption(model, sub_test_h, sub_test_y, len(sub_test_valid))

            print(f"  --- Eval @ step {step} ---")
            print(f"  Bit accuracy:  train={eval_train.get('mean_bit_accuracy', 0):.1%}  "
                  f"test={eval_test.get('mean_bit_accuracy', 0):.1%}")
            print(f"  Subsumption:   train={sub_rate_train:.1%}  test={sub_rate_test:.1%}")
            print(f"  Dead bits: {eval_train.get('dead_bits', N_BITS)}/{N_BITS}  "
                  f"Entropy: {eval_train.get('mean_entropy', 0):.3f}")

            if eval_test.get('worst_5'):
                worst_strs = [w['word'] + '(' + format(w['bit_accuracy'], '.0%') + ')' for w in eval_test['worst_5'][:3]]
                print(f"  Worst test: {', '.join(worst_strs)}")
            if eval_test.get('best_5'):
                best_strs = [w['word'] + '(' + format(w['bit_accuracy'], '.0%') + ')' for w in eval_test['best_5'][-3:]]
                print(f"  Best test:  {', '.join(best_strs)}")

            csv_writer.writerow([
                step,
                total.item(),
                lang_loss.item(),
                l_tri.item() if step >= triadic_start else 0,
                l_sup.item() if step >= triadic_start else 0,
                l_sub.item() if step >= triadic_start else 0,
                eval_train.get('mean_bit_accuracy', 0),
                eval_test.get('mean_bit_accuracy', 0),
                sub_rate_train,
                sub_rate_test,
                eval_train.get('dead_bits', N_BITS),
                eval_train.get('mean_entropy', 0),
            ])
            csv_file.flush()

            # Save best
            test_acc = eval_test.get('mean_bit_accuracy', 0)
            if test_acc > best_bit_acc:
                best_bit_acc = test_acc
                best_path = os.path.join(ckpt_dir, f'model_best.pt')
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'config': {
                        'vocab_size': config.vocab_size, 'block_size': config.block_size,
                        'n_layer': config.n_layer, 'n_embd': config.n_embd,
                        'n_head': config.n_head, 'n_triadic_bits': config.n_triadic_bits,
                    },
                    'bit_accuracy_test': test_acc,
                    'sub_rate_test': sub_rate_test,
                }, best_path)
                print(f"  ** New best: {test_acc:.1%} → saved {best_path}")

        # Save checkpoint
        if step % args.save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f'model_step{step}.pt')
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'config': {
                    'vocab_size': config.vocab_size, 'block_size': config.block_size,
                    'n_layer': config.n_layer, 'n_embd': config.n_embd,
                    'n_head': config.n_head, 'n_triadic_bits': config.n_triadic_bits,
                },
            }, ckpt_path)

    csv_file.close()

    # --- 6. Final evaluation ---
    print(f"\n[6/6] Final evaluation...")

    # Regla de tres
    r3_results = evaluate_regla_de_tres(model, tokenizer, anchors, device)
    print(f"\n  --- Regla de Tres ---")
    for r in r3_results:
        print(f"  {r['quad']:40s}  cos={r['cosine']:+.3f}  bit_acc={r['bit_accuracy']:.1%}")
    if r3_results:
        mean_cos = np.mean([r['cosine'] for r in r3_results])
        mean_bit = np.mean([r['bit_accuracy'] for r in r3_results])
        print(f"  Mean: cosine={mean_cos:+.3f}, bit_accuracy={mean_bit:.1%}")

    # Final anchor eval
    final_train = evaluate_anchors(model, sup_train_t, sup_train_tgt, sup_train_words)
    final_test = evaluate_anchors(model, sup_test_t, sup_test_tgt, sup_test_words)

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  RESULTS — 63-Bit Danza Cósmica ({args.scale})")
    print(f"{'=' * 70}")
    print(f"  Training time: {elapsed/60:.1f} min")
    print(f"  Bit accuracy (train): {final_train.get('mean_bit_accuracy', 0):.1%}")
    print(f"  Bit accuracy (test):  {final_test.get('mean_bit_accuracy', 0):.1%}")
    print(f"  Best test accuracy:   {best_bit_acc:.1%}")
    print(f"  Dead bits: {final_train.get('dead_bits', N_BITS)}/{N_BITS}")
    if r3_results:
        print(f"  Regla de tres: cos={mean_cos:+.3f}, bits={mean_bit:.1%}")
    print(f"  Checkpoint: {ckpt_dir}")
    print(f"{'=' * 70}")

    # Save results
    results = {
        'experiment': 'danza_63bit',
        'scale': args.scale,
        'steps': args.steps,
        'n_bits': N_BITS,
        'n_anchors_train': len(sup_train_words),
        'n_anchors_test': len(sup_test_words),
        'anchors_train': sup_train_words,
        'anchors_test': sup_test_words,
        'bit_accuracy_train': final_train.get('mean_bit_accuracy', 0),
        'bit_accuracy_test': final_test.get('mean_bit_accuracy', 0),
        'best_bit_accuracy_test': best_bit_acc,
        'dead_bits': final_train.get('dead_bits', N_BITS),
        'mean_entropy': final_train.get('mean_entropy', 0),
        'regla_de_tres': r3_results,
        'training_time_min': elapsed / 60,
    }
    results_path = os.path.join(ckpt_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results: {results_path}")


if __name__ == '__main__':
    main()
