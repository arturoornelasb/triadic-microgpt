"""
D-A1: Post-Hoc Analysis of Self-Supervised Bits (0 GPU).

Loads the Run 15 checkpoint (64 self-supervised bits, no primitive supervision)
and tests whether the discovered bit structure mirrors the Sistema 7×7 properties:

  A1.1 — Emergent Dual Axes (anti-correlation between bit pairs)
  A1.2 — Emergent Dependency Hierarchy (conditional activation DAG)
  A1.3 — Abstraction Gradient (activation frequency distribution)
  A1.4 — Semantic Probing (concept → bit category purity)
  A1.5 — Regla de Tres Transfer (algebraic structure on self-supervised bits)

All tests are CPU-only (0 GPU). Reads existing checkpoint + tokenizer.

Usage:
  python playground/danza_posthoc_analysis.py
  python playground/danza_posthoc_analysis.py --checkpoint path/to/model.pt
  python playground/danza_posthoc_analysis.py --test A1.1    # run single test
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.torch_transformer import TriadicGPT, TriadicGPTConfig
try:
    from src.fast_tokenizer import FastBPETokenizer as BPETokenizer
except ImportError:
    from src.tokenizer import BPETokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# Default paths
# ============================================================
DEFAULT_CHECKPOINT = os.path.join(
    PROJECT_ROOT, 'checkpoints', 'torch_run15_strongalign',
    'model_L12_D512_B64_best.pt')
DEFAULT_TOKENIZER = os.path.join(
    PROJECT_ROOT, 'checkpoints', 'torch_run15_strongalign',
    'tokenizer.json')

# ============================================================
# 113 concepts in 12 semantic domains (from geometric_topology)
# ============================================================
SEMANTIC_DOMAINS = {
    "royalty": ["king", "queen", "prince", "princess", "crown", "throne"],
    "animals": ["dog", "cat", "bird", "fish", "horse", "cow", "pig", "sheep"],
    "family": ["mother", "father", "brother", "sister", "son", "daughter"],
    "emotions": ["happy", "sad", "angry", "afraid", "brave", "kind", "love", "hate"],
    "nature": ["sun", "moon", "star", "river", "mountain", "ocean", "tree", "flower"],
    "elements": ["fire", "water", "earth", "air", "rain", "snow", "cloud", "wind"],
    "body": ["hand", "head", "eye", "heart", "foot", "mouth", "ear", "nose"],
    "home": ["house", "door", "window", "table", "chair", "bed", "lamp", "room"],
    "food": ["bread", "milk", "apple", "cake", "egg", "soup", "rice", "meat"],
    "actions": ["run", "walk", "swim", "fly", "jump", "climb", "fall", "sleep"],
    "colors": ["red", "blue", "green", "white", "black", "yellow", "pink", "gold"],
    "professions": ["doctor", "teacher", "nurse", "farmer", "soldier", "artist"],
}

# Sentence templates for context-aware encoding
SENTENCE_TEMPLATES = {
    "royalty": {
        "king": ["The king sat on his golden throne.", "Once upon a time there was a kind king.", "The king ruled the land with wisdom."],
        "queen": ["The queen wore a beautiful crown.", "The queen smiled at her people.", "Once there was a queen who loved music."],
        "prince": ["The young prince rode his horse.", "The prince dreamed of adventure.", "A brave prince saved the kingdom."],
        "princess": ["The princess danced in the garden.", "A little princess loved to read.", "The princess lived in a tall tower."],
        "crown": ["The golden crown sparkled in the sun.", "He placed the crown on her head.", "The crown was made of shining gold."],
        "throne": ["The throne stood in the great hall.", "She sat on the throne and smiled.", "The old throne was carved from wood."],
    },
    "animals": {
        "dog": ["The dog wagged its tail happily.", "A little dog ran in the park.", "The dog barked at the cat."],
        "cat": ["The cat slept on the warm bed.", "A small cat climbed the tree.", "The cat purred softly by the fire."],
        "bird": ["The bird sang a beautiful song.", "A little bird flew over the trees.", "The bird built a nest in the tree."],
        "fish": ["The fish swam in the clear water.", "A colorful fish jumped out of the pond.", "The fish lived in the blue ocean."],
        "horse": ["The horse galloped through the field.", "A white horse ran very fast.", "The horse ate some fresh grass."],
        "cow": ["The cow gave fresh milk every day.", "A brown cow stood in the meadow.", "The cow chewed grass under the tree."],
        "pig": ["The pig rolled in the mud.", "A little pig ate an apple.", "The pig lived on a small farm."],
        "sheep": ["The sheep had soft white wool.", "A little sheep followed its mother.", "The sheep grazed on the green hill."],
    },
    "family": {
        "mother": ["The mother hugged her child.", "A kind mother made breakfast.", "The mother sang a lullaby."],
        "father": ["The father played with his son.", "A tall father carried the baby.", "The father told a bedtime story."],
        "brother": ["The brother shared his toys.", "A big brother helped his sister.", "The brother and sister played together."],
        "sister": ["The sister drew a pretty picture.", "A little sister laughed and danced.", "The sister loved her baby brother."],
        "son": ["The son helped his father.", "A young son learned to ride a bike.", "The son was proud of his family."],
        "daughter": ["The daughter gave flowers to her mom.", "A small daughter smiled brightly.", "The daughter loved playing in the garden."],
    },
    "emotions": {
        "happy": ["She felt so happy that she smiled.", "The happy boy played all day.", "They were happy to see each other."],
        "sad": ["He was very sad and cried.", "The sad girl looked out the window.", "She felt sad because her friend left."],
        "angry": ["The angry man shouted loudly.", "She was angry about the broken toy.", "He felt angry but tried to be calm."],
        "afraid": ["The little boy was afraid of the dark.", "She was afraid of the thunder.", "He felt afraid but kept walking."],
        "brave": ["The brave girl faced her fear.", "He was brave and stood up tall.", "A brave child helped the lost puppy."],
        "kind": ["The kind woman gave him food.", "She was kind to everyone she met.", "A kind old man helped the children."],
        "love": ["She felt love in her heart.", "They showed love by helping others.", "Love made the family strong."],
        "hate": ["He did not want to feel hate.", "She tried not to hate the rain.", "Hate is not a good feeling."],
    },
    "nature": {
        "sun": ["The sun shone brightly in the sky.", "The warm sun made everyone happy.", "The sun set behind the mountains."],
        "moon": ["The moon glowed in the dark sky.", "She looked at the full moon.", "The moon was big and bright."],
        "star": ["A star twinkled in the night.", "She wished upon a star.", "The star shone above the tree."],
        "river": ["The river flowed through the forest.", "They played by the river.", "The river was cold and clear."],
        "mountain": ["The mountain was very tall.", "They climbed the big mountain.", "Snow covered the mountain top."],
        "ocean": ["The ocean was deep and blue.", "Waves crashed in the ocean.", "She loved swimming in the ocean."],
        "tree": ["The tree had green leaves.", "A bird sat in the tall tree.", "The old tree gave nice shade."],
        "flower": ["The flower was red and pretty.", "She picked a flower for her mom.", "Flowers grew in the garden."],
    },
    "elements": {
        "fire": ["The fire burned bright and warm.", "They sat around the fire.", "The fire crackled in the night."],
        "water": ["The water was cool and fresh.", "She drank a glass of water.", "Water flowed from the fountain."],
        "earth": ["The earth was soft after the rain.", "He dug in the dark earth.", "Plants grow from the earth."],
        "air": ["The fresh air felt nice.", "The air was cold that morning.", "Birds flew through the air."],
        "rain": ["The rain fell on the roof.", "She danced in the rain.", "After the rain came a rainbow."],
        "snow": ["The snow covered everything in white.", "Children played in the snow.", "Soft snow fell from the sky."],
        "cloud": ["A white cloud floated in the sky.", "The cloud looked like a bunny.", "Dark clouds meant rain was coming."],
        "wind": ["The wind blew through the trees.", "A strong wind pushed the leaves.", "The wind was cold today."],
    },
    "body": {
        "hand": ["She held his hand gently.", "He raised his hand to wave.", "Her hand was warm and soft."],
        "head": ["He nodded his head and smiled.", "She put the hat on her head.", "The sun warmed his head."],
        "eye": ["Her eye sparkled with joy.", "He closed his eye and slept.", "The bird watched with one eye."],
        "heart": ["Her heart was full of love.", "His heart beat very fast.", "She had a kind heart."],
        "foot": ["She put her foot in the water.", "His foot hurt from walking.", "The cat sat on his foot."],
        "mouth": ["She opened her mouth to speak.", "The baby put food in his mouth.", "Her mouth formed a big smile."],
        "ear": ["The rabbit had a long ear.", "She whispered in his ear.", "He covered his ear from the noise."],
        "nose": ["The dog sniffed with its nose.", "Her nose was cold and red.", "He touched his nose and laughed."],
    },
    "home": {
        "house": ["The house was big and white.", "They lived in a small house.", "The house had a red door."],
        "door": ["She opened the door slowly.", "The door was painted blue.", "He knocked on the door."],
        "window": ["She looked through the window.", "The window let in bright light.", "A bird sat on the window."],
        "table": ["They sat around the table.", "The table was set for dinner.", "She put flowers on the table."],
        "chair": ["The chair was soft and comfy.", "He sat in the big chair.", "The little chair was just right."],
        "bed": ["The bed was warm and cozy.", "She jumped on the bed.", "He made his bed every morning."],
        "lamp": ["The lamp glowed softly.", "She turned on the lamp.", "The old lamp sat on the table."],
        "room": ["The room was bright and clean.", "She played in her room.", "The room had a big window."],
    },
    "food": {
        "bread": ["The bread was fresh and warm.", "She ate bread with butter.", "He baked bread every morning."],
        "milk": ["She drank a glass of cold milk.", "The milk came from the cow.", "He poured milk on his cereal."],
        "apple": ["The apple was red and sweet.", "She picked an apple from the tree.", "He ate the apple happily."],
        "cake": ["The cake was for her birthday.", "She baked a chocolate cake.", "The cake had pink frosting."],
        "egg": ["The hen laid a brown egg.", "She cooked an egg for breakfast.", "The egg cracked open."],
        "soup": ["The soup was hot and yummy.", "Mother made chicken soup.", "He ate soup on a cold day."],
        "rice": ["She cooked rice for dinner.", "The rice was soft and white.", "They ate rice with chicken."],
        "meat": ["The meat cooked over the fire.", "He cut the meat carefully.", "The meat smelled delicious."],
    },
    "actions": {
        "run": ["The boy liked to run fast.", "She would run in the park.", "They run every morning."],
        "walk": ["She liked to walk in the garden.", "They walk to school together.", "He took a walk by the river."],
        "swim": ["The fish can swim very fast.", "She learned to swim last summer.", "They swim in the lake."],
        "fly": ["The bird can fly very high.", "She wished she could fly.", "Butterflies fly in the garden."],
        "jump": ["The frog can jump very far.", "She liked to jump rope.", "He could jump over the puddle."],
        "climb": ["The cat liked to climb trees.", "He tried to climb the hill.", "She could climb very high."],
        "fall": ["The leaves fall from the trees.", "She was careful not to fall.", "The rain started to fall."],
        "sleep": ["The baby liked to sleep all day.", "The cat curled up to sleep.", "She went to sleep early."],
    },
    "colors": {
        "red": ["The red ball bounced high.", "She wore a red dress.", "The apple was bright red."],
        "blue": ["The sky was bright blue.", "He painted the wall blue.", "She had blue eyes."],
        "green": ["The green grass felt soft.", "She picked a green leaf.", "The frog was bright green."],
        "white": ["The white snow was everywhere.", "She had a white cat.", "The cloud was fluffy and white."],
        "black": ["The black cat sat on the fence.", "She wore a black hat.", "The night sky was black."],
        "yellow": ["The yellow sun was warm.", "She picked a yellow flower.", "The duck was bright yellow."],
        "pink": ["The pink flower was pretty.", "She wore a pink ribbon.", "The sunset was pink and orange."],
        "gold": ["The gold ring was shiny.", "She found a gold coin.", "The gold crown sparkled."],
    },
    "professions": {
        "doctor": ["The doctor helped the sick boy.", "She wanted to be a doctor.", "The doctor was kind and gentle."],
        "teacher": ["The teacher read a story.", "She was a good teacher.", "The teacher helped the children learn."],
        "nurse": ["The nurse gave him medicine.", "The kind nurse helped everyone.", "She was a brave nurse."],
        "farmer": ["The farmer grew vegetables.", "The farmer fed the animals.", "A kind farmer lived on a hill."],
        "soldier": ["The brave soldier marched on.", "The soldier protected the village.", "He became a strong soldier."],
        "artist": ["The artist painted a picture.", "She was a talented artist.", "The artist used bright colors."],
    },
}

# Analogy quads for A1.5
ANALOGY_QUADS = [
    ("man", "woman", "king", "queen"),
    ("man", "woman", "boy", "girl"),
    ("father", "mother", "brother", "sister"),
    ("father", "mother", "son", "daughter"),
    ("king", "queen", "prince", "princess"),
    ("happy", "sad", "love", "hate"),
    ("hot", "cold", "fire", "water"),
    ("big", "small", "tall", "short"),
    ("dog", "cat", "horse", "cow"),
    ("sun", "moon", "bright", "dark"),
]

# Extra words needed for analogies not in SEMANTIC_DOMAINS
EXTRA_WORDS = {
    "man": ["The man walked down the road.", "A tall man helped the boy.", "The man was strong and kind."],
    "woman": ["The woman sang a song.", "A kind woman gave him food.", "The woman wore a blue dress."],
    "boy": ["The boy played in the park.", "A little boy found a frog.", "The boy smiled at his friend."],
    "girl": ["The girl drew a picture.", "A happy girl danced in the rain.", "The girl loved to read books."],
    "hot": ["It was very hot today.", "The hot sun made them tired.", "The soup was too hot to eat."],
    "cold": ["The cold wind blew hard.", "It was a cold winter day.", "The water was very cold."],
    "big": ["The dog was very big.", "A big tree stood in the yard.", "The big bear looked friendly."],
    "small": ["The small bird chirped.", "A small bug sat on a leaf.", "The puppy was very small."],
    "tall": ["The tall tree touched the sky.", "A tall man waved at them.", "The building was very tall."],
    "short": ["The short girl jumped high.", "A short walk felt nice.", "The story was short but funny."],
    "bright": ["The bright star shone above.", "The room was bright and warm.", "She had a bright smile."],
    "dark": ["The night was dark and quiet.", "The room was dark inside.", "He was afraid of the dark."],
}


# ============================================================
# Load model and compute bit vectors
# ============================================================

def load_model(checkpoint_path, tokenizer_path, device='cpu'):
    """Load Run 15 checkpoint and tokenizer."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct config from checkpoint
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    vocab_size = state['wte.weight'].shape[0]
    n_embd = state['wte.weight'].shape[1]
    n_bits = state['triadic_head.weight'].shape[0]
    n_layer = sum(1 for k in state if k.endswith('.attn.c_attn.weight'))
    n_head = 8  # standard for all runs
    block_size = state['wpe.weight'].shape[0]

    config = TriadicGPTConfig(
        vocab_size=vocab_size, block_size=block_size,
        n_layer=n_layer, n_embd=n_embd, n_head=n_head,
        n_triadic_bits=n_bits)

    model = TriadicGPT(config)
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    print(f"  Model: {n_layer}L / {n_embd}D / {n_bits} bits / vocab {vocab_size}")

    tokenizer = BPETokenizer.load(tokenizer_path)
    print(f"  Tokenizer: {tokenizer_path}")

    return model, tokenizer, n_bits


def encode_concept_sentence(model, tokenizer, sentence, target_word, device='cpu'):
    """Encode a sentence and extract the triadic projection for the target word's tokens."""
    ids = tokenizer.encode(sentence, add_special=False)
    word_ids = tokenizer.encode(target_word, add_special=False)

    if not ids or not word_ids:
        return None

    # Find where the word tokens appear in the sentence
    word_len = len(word_ids)
    positions = []
    for i in range(len(ids) - word_len + 1):
        if ids[i:i + word_len] == word_ids:
            positions.extend(range(i, i + word_len))
            break

    if not positions:
        # Fallback: use all tokens (mean pool)
        positions = list(range(len(ids)))

    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        _, proj, _ = model(x)
    # proj: (1, T, n_bits) — extract target positions and mean-pool
    proj = proj[0, positions, :].mean(dim=0)  # (n_bits,)
    return proj


def compute_all_bit_vectors(model, tokenizer, device='cpu'):
    """Compute bit vectors for all 113 concepts using sentence-level aggregation."""
    print("\nComputing bit vectors (sentence-level aggregation)...")
    vectors = {}
    word_to_domain = {}

    for domain, words in SEMANTIC_DOMAINS.items():
        for word in words:
            word_to_domain[word] = domain
            sentences = SENTENCE_TEMPLATES[domain][word]
            projs = []
            for sent in sentences:
                p = encode_concept_sentence(model, tokenizer, sent, word, device)
                if p is not None:
                    projs.append(p)
            if projs:
                vectors[word] = torch.stack(projs).mean(dim=0)  # mean of 3 sentences

    # Extra words for analogies
    for word, sentences in EXTRA_WORDS.items():
        if word not in vectors:
            projs = []
            for sent in sentences:
                p = encode_concept_sentence(model, tokenizer, sent, word, device)
                if p is not None:
                    projs.append(p)
            if projs:
                vectors[word] = torch.stack(projs).mean(dim=0)

    print(f"  Encoded {len(vectors)} concepts ({len(word_to_domain)} in domains)")
    return vectors, word_to_domain


def binarize(proj, threshold=0.0):
    """Convert tanh projections to binary: >threshold → 1, else 0."""
    return (proj > threshold).float()


# ============================================================
# A1.1 — Emergent Dual Axes (anti-correlation)
# ============================================================

def test_a1_1(vectors, n_bits):
    """Find anti-correlated bit pairs in self-supervised bits."""
    print("\n" + "=" * 60)
    print("A1.1 — Emergent Dual Axes (Anti-Correlation)")
    print("=" * 60)

    # Build bit matrix: (N_concepts, n_bits) binary
    words = sorted(vectors.keys())
    bit_matrix = torch.stack([binarize(vectors[w]) for w in words])  # (N, n_bits)
    N = len(words)

    # Compute bit-bit correlation matrix
    # Center the columns
    centered = bit_matrix - bit_matrix.mean(dim=0, keepdim=True)
    stds = centered.std(dim=0, keepdim=True).clamp(min=1e-6)
    normalized = centered / stds
    corr = (normalized.T @ normalized) / N  # (n_bits, n_bits)

    # Find anti-correlated pairs (< -0.3)
    anti_pairs = []
    for i in range(n_bits):
        for j in range(i + 1, n_bits):
            c = corr[i, j].item()
            if c < -0.3:
                anti_pairs.append((i, j, c))

    anti_pairs.sort(key=lambda x: x[2])

    print(f"\n  Anti-correlated pairs (r < -0.3): {len(anti_pairs)}")
    print(f"  Expected if real: ~10-15 | Expected if arbitrary: <5")

    # For each pair, show which concepts activate each side
    recognizable = 0
    for rank, (bi, bj, r) in enumerate(anti_pairs[:15]):
        words_i = [w for w in words if bit_matrix[words.index(w), bi] == 1]
        words_j = [w for w in words if bit_matrix[words.index(w), bj] == 1]
        # Only on one side (exclusive)
        only_i = [w for w in words_i if w not in words_j]
        only_j = [w for w in words_j if w not in words_i]

        label = f"  Pair {rank+1}: bit {bi} ↔ bit {bj} (r={r:.3f})"
        print(label)
        print(f"    Bit {bi} exclusive ({len(only_i)}): {', '.join(only_i[:8])}")
        print(f"    Bit {bj} exclusive ({len(only_j)}): {', '.join(only_j[:8])}")

    # Statistics
    all_corrs = []
    for i in range(n_bits):
        for j in range(i + 1, n_bits):
            all_corrs.append(corr[i, j].item())
    all_corrs = np.array(all_corrs)

    print(f"\n  Correlation statistics:")
    print(f"    Mean: {all_corrs.mean():.4f}")
    print(f"    Std:  {all_corrs.std():.4f}")
    print(f"    Min:  {all_corrs.min():.4f}")
    print(f"    Max:  {all_corrs.max():.4f}")
    print(f"    Pairs r < -0.3: {(all_corrs < -0.3).sum()}")
    print(f"    Pairs r < -0.2: {(all_corrs < -0.2).sum()}")
    print(f"    Pairs r > +0.3: {(all_corrs > 0.3).sum()}")

    return {
        'n_anti_pairs': len(anti_pairs),
        'anti_pairs': [(i, j, r) for i, j, r in anti_pairs],
        'corr_mean': float(all_corrs.mean()),
        'corr_std': float(all_corrs.std()),
        'corr_min': float(all_corrs.min()),
        'corr_matrix': corr.numpy(),
    }


# ============================================================
# A1.2 — Emergent Dependency Hierarchy (conditional activation DAG)
# ============================================================

def test_a1_2(vectors, n_bits):
    """Build dependency DAG from conditional activation patterns."""
    print("\n" + "=" * 60)
    print("A1.2 — Emergent Dependency Hierarchy (Conditional Activation)")
    print("=" * 60)

    words = sorted(vectors.keys())
    bit_matrix = torch.stack([binarize(vectors[w]) for w in words])  # (N, n_bits)
    N = len(words)

    # For each bit pair, compute P(j=1|i=1) and P(j=1|i=0)
    edges = []  # (from_bit, to_bit, strength)
    THRESHOLD = 0.3  # Minimum difference P(j|i) - P(j|¬i) to count as dependency

    for i in range(n_bits):
        mask_i = bit_matrix[:, i] == 1
        mask_not_i = bit_matrix[:, i] == 0
        n_i = mask_i.sum().item()
        n_not_i = mask_not_i.sum().item()

        if n_i < 3 or n_not_i < 3:
            continue  # skip bits with too few activations

        for j in range(n_bits):
            if i == j:
                continue

            # P(j=1 | i=1) and P(j=1 | i=0)
            p_j_given_i = bit_matrix[mask_i, j].mean().item()
            p_j_given_not_i = bit_matrix[mask_not_i, j].mean().item()

            # P(i=1 | j=1) and P(i=1 | j=0) — check asymmetry
            mask_j = bit_matrix[:, j] == 1
            mask_not_j = bit_matrix[:, j] == 0
            n_j = mask_j.sum().item()
            n_not_j = mask_not_j.sum().item()

            if n_j < 3 or n_not_j < 3:
                continue

            p_i_given_j = bit_matrix[mask_j, i].mean().item()
            p_i_given_not_j = bit_matrix[mask_not_j, i].mean().item()

            # j depends on i if:
            #   P(j|i) >> P(j|¬i)  AND  P(i|j) ≈ P(i|¬j)
            forward_strength = p_j_given_i - p_j_given_not_i
            backward_strength = p_i_given_j - p_i_given_not_j

            if forward_strength > THRESHOLD and backward_strength < THRESHOLD:
                edges.append((i, j, forward_strength))

    # Build DAG and compute depth
    children = defaultdict(list)
    parents = defaultdict(list)
    for src, dst, s in edges:
        children[src].append((dst, s))
        parents[dst].append((src, s))

    # Compute depth via BFS from roots (bits with no parents)
    all_bits = set(range(n_bits))
    bits_with_parents = set(dst for _, dst, _ in edges)
    roots = all_bits - bits_with_parents

    depth = {}
    for r in roots:
        depth[r] = 0
    queue = list(roots)
    visited = set(roots)
    while queue:
        current = queue.pop(0)
        for child, _ in children.get(current, []):
            if child not in visited:
                depth[child] = depth.get(current, 0) + 1
                visited.add(child)
                queue.append(child)

    # Assign depth 0 to unvisited
    for b in range(n_bits):
        if b not in depth:
            depth[b] = 0

    max_depth = max(depth.values()) if depth else 0

    print(f"\n  Directed edges (P(j|i)-P(j|¬i) > {THRESHOLD}): {len(edges)}")
    print(f"  Root bits (no parents): {len(roots)}")
    print(f"  DAG depth: {max_depth}")
    print(f"  Expected if real: depth 4-6 | Expected if arbitrary: 0-1")

    # Depth distribution
    depth_counts = defaultdict(int)
    for d in depth.values():
        depth_counts[d] += 1
    print(f"\n  Depth distribution:")
    for d in sorted(depth_counts.keys()):
        print(f"    Depth {d}: {depth_counts[d]} bits")

    # Activation frequency vs depth correlation
    freq = bit_matrix.mean(dim=0).numpy()
    depths_arr = np.array([depth[b] for b in range(n_bits)])
    if depths_arr.std() > 0 and freq.std() > 0:
        corr_depth_freq = np.corrcoef(depths_arr, freq)[0, 1]
    else:
        corr_depth_freq = 0.0
    print(f"\n  Correlation(depth, frequency): {corr_depth_freq:.4f}")
    print(f"  Expected if real: negative (deep bits = rare) | If arbitrary: ~0")

    # Show strongest edges
    edges.sort(key=lambda x: -x[2])
    print(f"\n  Top 10 dependency edges:")
    for src, dst, s in edges[:10]:
        print(f"    bit {src} → bit {dst} (strength {s:.3f})")

    return {
        'n_edges': len(edges),
        'n_roots': len(roots),
        'dag_depth': max_depth,
        'depth_distribution': dict(depth_counts),
        'corr_depth_freq': float(corr_depth_freq),
        'top_edges': [(s, d, st) for s, d, st in edges[:20]],
    }


# ============================================================
# A1.3 — Abstraction Gradient (activation frequency)
# ============================================================

def test_a1_3(vectors, n_bits):
    """Analyze activation frequency distribution across bits."""
    print("\n" + "=" * 60)
    print("A1.3 — Abstraction Gradient (Activation Frequency)")
    print("=" * 60)

    words = sorted(vectors.keys())
    bit_matrix = torch.stack([binarize(vectors[w]) for w in words])  # (N, n_bits)

    freq = bit_matrix.mean(dim=0).numpy()  # per-bit activation frequency

    # Distribution statistics
    foundation = (freq > 0.6).sum()
    mid_range = ((freq >= 0.1) & (freq <= 0.6)).sum()
    rare = (freq < 0.1).sum()
    dead = (freq < 0.01).sum()

    print(f"\n  Activation frequency distribution:")
    print(f"    Foundation (>60%): {foundation} bits")
    print(f"    Mid-range (10-60%): {mid_range} bits")
    print(f"    Rare (<10%): {rare} bits")
    print(f"    Dead (<1%): {dead} bits")
    print(f"    Mean frequency: {freq.mean():.4f}")
    print(f"    Std: {freq.std():.4f}")

    from scipy.stats import skew, kurtosis
    sk = skew(freq)
    kurt = kurtosis(freq)
    print(f"    Skewness: {sk:.4f}")
    print(f"      Expected if real: positive (right-skewed) | If arbitrary: ~0")
    print(f"    Kurtosis: {kurt:.4f}")

    # Show the most and least frequent bits
    sorted_idx = np.argsort(freq)[::-1]
    print(f"\n  Most frequent bits (foundation layer):")
    for i in sorted_idx[:10]:
        active_words = [w for w in words if bit_matrix[words.index(w), i] == 1]
        print(f"    Bit {i}: {freq[i]:.1%} ({len(active_words)} concepts) — {', '.join(active_words[:6])}")

    print(f"\n  Least frequent bits (specific layer):")
    for i in sorted_idx[-10:]:
        active_words = [w for w in words if bit_matrix[words.index(w), i] == 1]
        print(f"    Bit {i}: {freq[i]:.1%} ({len(active_words)} concepts) — {', '.join(active_words[:6])}")

    # Compare to uniform expectation (entropy maximization → 50%)
    uniform_freq = np.full(n_bits, 0.5)
    kl_div = np.sum(np.where(
        freq > 0.001,
        freq * np.log(freq / uniform_freq + 1e-10) +
        (1 - freq) * np.log((1 - freq) / (1 - uniform_freq) + 1e-10),
        0))
    print(f"\n  KL divergence from uniform: {kl_div:.4f}")

    return {
        'foundation_bits': int(foundation),
        'mid_range_bits': int(mid_range),
        'rare_bits': int(rare),
        'dead_bits': int(dead),
        'mean_freq': float(freq.mean()),
        'std_freq': float(freq.std()),
        'skewness': float(sk),
        'kurtosis': float(kurt),
        'kl_from_uniform': float(kl_div),
        'frequencies': freq.tolist(),
    }


# ============================================================
# A1.4 — Semantic Probing (concept → bit category purity)
# ============================================================

def test_a1_4(vectors, word_to_domain, n_bits):
    """Check if individual bits correspond to recognizable semantic categories."""
    print("\n" + "=" * 60)
    print("A1.4 — Semantic Probing (Category Purity)")
    print("=" * 60)

    # Only use words that have a domain label
    labeled_words = [w for w in sorted(vectors.keys()) if w in word_to_domain]
    bit_matrix = torch.stack([binarize(vectors[w]) for w in labeled_words])
    domains = [word_to_domain[w] for w in labeled_words]
    domain_set = sorted(set(domains))

    N = len(labeled_words)
    n_domains = len(domain_set)
    domain_idx = {d: i for i, d in enumerate(domain_set)}

    print(f"\n  Labeled concepts: {N}")
    print(f"  Domains: {n_domains} ({', '.join(domain_set)})")

    # For each bit, compute category distribution of activating concepts
    high_purity_bits = []

    for b in range(n_bits):
        active_mask = bit_matrix[:, b] == 1
        n_active = active_mask.sum().item()
        if n_active < 3:
            continue

        # Count domain distribution among active concepts
        domain_counts = defaultdict(int)
        for i, w in enumerate(labeled_words):
            if active_mask[i]:
                domain_counts[domains[i]] += 1

        # Purity: fraction of activations from top 1-2 domains
        sorted_domains = sorted(domain_counts.items(), key=lambda x: -x[1])
        top1_count = sorted_domains[0][1]
        top2_count = sum(c for _, c in sorted_domains[:2])
        purity_1 = top1_count / n_active
        purity_2 = top2_count / n_active

        if purity_1 > 0.5:
            high_purity_bits.append({
                'bit': b,
                'n_active': n_active,
                'purity_1': purity_1,
                'purity_2': purity_2,
                'top_domain': sorted_domains[0][0],
                'distribution': dict(sorted_domains[:4]),
            })

    high_purity_bits.sort(key=lambda x: -x['purity_1'])

    # Count bits at various purity thresholds
    n_70 = sum(1 for x in high_purity_bits if x['purity_1'] > 0.7)
    n_60 = sum(1 for x in high_purity_bits if x['purity_1'] > 0.6)
    n_50 = sum(1 for x in high_purity_bits if x['purity_1'] > 0.5)

    print(f"\n  Bits with purity > 70%: {n_70}")
    print(f"  Bits with purity > 60%: {n_60}")
    print(f"  Bits with purity > 50%: {n_50}")
    print(f"  Expected if real: 10-20 at >50% | If arbitrary: <5")

    # Mean purity across all bits
    all_purities = []
    for b in range(n_bits):
        active_mask = bit_matrix[:, b] == 1
        n_active = active_mask.sum().item()
        if n_active < 1:
            all_purities.append(1.0 / n_domains)  # worst case
            continue
        domain_counts = defaultdict(int)
        for i in range(N):
            if active_mask[i]:
                domain_counts[domains[i]] += 1
        max_count = max(domain_counts.values())
        all_purities.append(max_count / n_active)

    mean_purity = np.mean(all_purities)
    chance_purity = 1.0 / n_domains

    print(f"\n  Mean purity: {mean_purity:.4f}")
    print(f"  Chance purity (1/{n_domains}): {chance_purity:.4f}")
    print(f"  Above chance: {'+' if mean_purity > chance_purity else '-'}{abs(mean_purity - chance_purity):.4f}")

    # Show the most pure bits
    print(f"\n  Top semantic bits:")
    for info in high_purity_bits[:15]:
        dist_str = ', '.join(f"{d}:{c}" for d, c in info['distribution'].items())
        print(f"    Bit {info['bit']}: purity={info['purity_1']:.0%}, "
              f"n={info['n_active']}, top={info['top_domain']} ({dist_str})")

    return {
        'n_purity_70': n_70,
        'n_purity_60': n_60,
        'n_purity_50': n_50,
        'mean_purity': float(mean_purity),
        'chance_purity': float(chance_purity),
        'high_purity_bits': high_purity_bits[:20],
    }


# ============================================================
# A1.5 — Regla de Tres Transfer (algebraic structure)
# ============================================================

def test_a1_5(vectors, n_bits):
    """Test if regla de tres works on self-supervised bit vectors."""
    print("\n" + "=" * 60)
    print("A1.5 — Regla de Tres Transfer (Algebraic Structure)")
    print("=" * 60)

    results = []
    transform_bits = defaultdict(int)  # track which bits flip consistently

    for a_word, b_word, c_word, d_word in ANALOGY_QUADS:
        if not all(w in vectors for w in [a_word, b_word, c_word, d_word]):
            print(f"  SKIP: {a_word}:{b_word} = {c_word}:{d_word} (missing words)")
            continue

        a = binarize(vectors[a_word])
        b = binarize(vectors[b_word])
        c = binarize(vectors[c_word])
        d = binarize(vectors[d_word])

        # Transform: bits that flip from a→b
        transform_ab = b - a  # +1 = gained, -1 = lost, 0 = same

        # Apply same transform to c
        predicted_d = (c + transform_ab).clamp(0, 1)

        # Compare to actual d
        hamming = (predicted_d != d).sum().item()
        bit_accuracy = 1.0 - hamming / n_bits

        # Cosine on raw tanh projections
        raw_a = vectors[a_word]
        raw_b = vectors[b_word]
        raw_c = vectors[c_word]
        raw_d = vectors[d_word]
        raw_pred = raw_b - raw_a + raw_c
        cos = float(torch.nn.functional.cosine_similarity(
            raw_pred.unsqueeze(0), raw_d.unsqueeze(0)).item())

        # Track which bits flip
        flipped = (transform_ab != 0).nonzero(as_tuple=True)[0]
        for fb in flipped.tolist():
            transform_bits[fb] += 1

        result = {
            'quad': f"{a_word}:{b_word} = {c_word}:{d_word}",
            'hamming': int(hamming),
            'bit_accuracy': float(bit_accuracy),
            'cosine': float(cos),
            'n_flipped': int(len(flipped)),
            'flipped_bits': flipped.tolist(),
        }
        results.append(result)

        status = "OK" if bit_accuracy > 0.9 else "WEAK" if bit_accuracy > 0.8 else "FAIL"
        print(f"  [{status}] {result['quad']}: "
              f"bit_acc={bit_accuracy:.1%}, cos={cos:.3f}, "
              f"flip={len(flipped)} bits")

    if not results:
        print("  No analogy quads could be evaluated!")
        return {'results': [], 'mean_bit_accuracy': 0, 'mean_cosine': 0}

    mean_acc = np.mean([r['bit_accuracy'] for r in results])
    mean_cos = np.mean([r['cosine'] for r in results])

    print(f"\n  Mean bit accuracy: {mean_acc:.1%}")
    print(f"  Mean cosine: {mean_cos:.3f}")

    # Consistency: which bits flip across multiple quads?
    if transform_bits:
        print(f"\n  Transform consistency (bits that flip in multiple quads):")
        for bit, count in sorted(transform_bits.items(), key=lambda x: -x[1])[:10]:
            if count >= 2:
                print(f"    Bit {bit}: flips in {count}/{len(results)} quads")

    # Check gender transform consistency
    gender_quads = [r for r in results if 'man' in r['quad'] or 'father' in r['quad']]
    if len(gender_quads) >= 2:
        # Check if the same bits flip across gender analogy quads
        gender_flips = [set(r['flipped_bits']) for r in gender_quads]
        common = gender_flips[0]
        for s in gender_flips[1:]:
            common = common & s
        print(f"\n  Gender transform consistency:")
        print(f"    Quads: {len(gender_quads)}")
        print(f"    Common flipped bits: {sorted(common) if common else 'none'}")

    return {
        'results': results,
        'mean_bit_accuracy': float(mean_acc),
        'mean_cosine': float(mean_cos),
        'transform_consistency': dict(transform_bits),
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='D-A1: Post-Hoc Analysis of Self-Supervised Bits')
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--tokenizer', default=DEFAULT_TOKENIZER)
    parser.add_argument('--test', default='all', help='Run specific test: A1.1, A1.2, A1.3, A1.4, A1.5, or all')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default=None, help='Save results JSON to this path')
    args = parser.parse_args()

    model, tokenizer, n_bits = load_model(args.checkpoint, args.tokenizer, args.device)
    vectors, word_to_domain = compute_all_bit_vectors(model, tokenizer, args.device)

    results = {}
    tests_to_run = args.test.lower()

    if tests_to_run in ('all', 'a1.1'):
        results['A1.1_dual_axes'] = test_a1_1(vectors, n_bits)

    if tests_to_run in ('all', 'a1.2'):
        results['A1.2_dependency_hierarchy'] = test_a1_2(vectors, n_bits)

    if tests_to_run in ('all', 'a1.3'):
        try:
            results['A1.3_abstraction_gradient'] = test_a1_3(vectors, n_bits)
        except ImportError:
            print("\n  SKIP A1.3: scipy not installed (needed for skewness/kurtosis)")

    if tests_to_run in ('all', 'a1.4'):
        results['A1.4_semantic_probing'] = test_a1_4(vectors, word_to_domain, n_bits)

    if tests_to_run in ('all', 'a1.5'):
        results['A1.5_regla_de_tres'] = test_a1_5(vectors, n_bits)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if 'A1.1_dual_axes' in results:
        r = results['A1.1_dual_axes']
        print(f"  A1.1 Dual axes: {r['n_anti_pairs']} anti-correlated pairs (r<-0.3)")

    if 'A1.2_dependency_hierarchy' in results:
        r = results['A1.2_dependency_hierarchy']
        print(f"  A1.2 Hierarchy: DAG depth {r['dag_depth']}, {r['n_edges']} edges, "
              f"depth↔freq r={r['corr_depth_freq']:.3f}")

    if 'A1.3_abstraction_gradient' in results:
        r = results['A1.3_abstraction_gradient']
        print(f"  A1.3 Abstraction: {r['foundation_bits']} foundation, "
              f"{r['rare_bits']} rare, skew={r['skewness']:.3f}")

    if 'A1.4_semantic_probing' in results:
        r = results['A1.4_semantic_probing']
        print(f"  A1.4 Probing: {r['n_purity_50']} bits >50% purity, "
              f"mean={r['mean_purity']:.3f} (chance={r['chance_purity']:.3f})")

    if 'A1.5_regla_de_tres' in results:
        r = results['A1.5_regla_de_tres']
        print(f"  A1.5 Regla de tres: {r['mean_bit_accuracy']:.1%} bit acc, "
              f"{r['mean_cosine']:.3f} cosine")

    # Save results
    output_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'results', 'danza_posthoc_a1.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert numpy arrays for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    with open(output_path, 'w') as f:
        json.dump(make_serializable(results), f, indent=2)
    print(f"\n  Results saved: {output_path}")


if __name__ == '__main__':
    main()
