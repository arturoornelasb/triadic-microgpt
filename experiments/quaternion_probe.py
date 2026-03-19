"""
Quaternion Probe — Can quaternion rotations capture semantic transformations
that prime algebra misses?

Three tests:
  1. Rotation consistency: is the king→queen rotation similar to man→woman?
  2. Magnitude semantics: do concrete/abstract concepts differ in |Q|²?
  3. Analogy accuracy: quaternion vs prime algebra head-to-head

Uses real projections from a trained TriadicGPT checkpoint.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.evaluate import load_model
from src.triadic import PrimeMapper, TriadicValidator, prime_factors


# ============================================================
# Quaternion Operations (pure numpy, no dependencies)
# ============================================================

def quat_multiply(q1, q2):
    """Hamilton product of two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_conjugate(q):
    """Conjugate: [w, -x, -y, -z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_inverse(q):
    """Inverse: conjugate / |q|²."""
    norm_sq = np.sum(q**2)
    if norm_sq < 1e-10:
        return np.zeros(4)
    return quat_conjugate(q) / norm_sq


def quat_rotate(q_transform, q_target):
    """Apply rotation: q_transform * q_target * q_transform_inv."""
    return quat_multiply(quat_multiply(q_transform, q_target), quat_inverse(q_transform))


def quat_norm(q):
    """Magnitude |Q|."""
    return np.sqrt(np.sum(q**2))


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ============================================================
# Get real projections from checkpoint
# ============================================================

def get_projections(model, tokenizer, concepts, device):
    """Get raw continuous projections (tanh values, not binarized)."""
    results = {}
    for concept in concepts:
        ids = tokenizer.encode(concept, add_special=False)
        if not ids:
            continue
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            _, triadic_proj, _ = model(x)
        proj = triadic_proj[0].mean(dim=0).cpu().numpy()
        results[concept] = proj
    return results


# ============================================================
# Test 1: Rotation Consistency
# ============================================================

def test_rotation_consistency(projections):
    """Are king→queen, man→woman, boy→girl the same rotation?"""
    print("\n" + "=" * 60)
    print("  TEST 1: Quaternion Rotation Consistency")
    print("  Do analogous pairs share the same 4D rotation?")
    print("=" * 60)

    analogy_groups = [
        # (A, B, C, D) where A:B :: C:D
        [("king", "queen"), ("man", "woman"), ("boy", "girl"),
         ("father", "mother"), ("brother", "sister")],
        [("happy", "sad"), ("love", "hate"), ("peace", "war")],
        [("dog", "cat"), ("horse", "cow")],
        [("fire", "water"), ("sun", "moon")],
    ]

    for group in analogy_groups:
        print(f"\n  Group: {group[0][0]}:{group[0][1]} pattern")
        rotations = []

        for a, b in group:
            if a not in projections or b not in projections:
                continue
            q_a = projections[a][:4]
            q_b = projections[b][:4]
            # Transformation = B * A^(-1)
            q_transform = quat_multiply(q_b, quat_inverse(q_a))
            q_transform = q_transform / (quat_norm(q_transform) + 1e-10)  # normalize
            rotations.append((a, b, q_transform))
            print(f"    {a:>10s} -> {b:<10s}  rotation=[{q_transform[0]:+.3f}, {q_transform[1]:+.3f}, {q_transform[2]:+.3f}, {q_transform[3]:+.3f}]")

        # Compare all pairs of rotations
        if len(rotations) >= 2:
            sims = []
            for i in range(len(rotations)):
                for j in range(i + 1, len(rotations)):
                    sim = cosine_sim(rotations[i][2], rotations[j][2])
                    sims.append(sim)
                    print(f"    sim({rotations[i][0]}->{rotations[i][1]}, "
                          f"{rotations[j][0]}->{rotations[j][1]}) = {sim:.4f}")
            avg_sim = np.mean(sims)
            print(f"    AVG rotation similarity: {avg_sim:.4f} "
                  f"{'(CONSISTENT)' if avg_sim > 0.7 else '(INCONSISTENT)' if avg_sim < 0.3 else '(MIXED)'}")


# ============================================================
# Test 2: Magnitude Semantics
# ============================================================

def test_magnitude_semantics(projections):
    """Do quaternion magnitudes correlate with semantic properties?"""
    print("\n" + "=" * 60)
    print("  TEST 2: Quaternion Magnitude Semantics")
    print("  Does |Q|^2 encode anything meaningful?")
    print("=" * 60)

    categories = {
        'concrete': ["dog", "cat", "house", "car", "tree", "book", "table", "chair"],
        'abstract': ["love", "hate", "hope", "fear", "joy", "peace", "dream", "magic"],
        'person': ["king", "queen", "man", "woman", "doctor", "nurse", "mother", "father"],
        'action': ["run", "walk", "swim", "fly", "jump", "climb", "fall", "sleep"],
        'nature': ["fire", "water", "sun", "moon", "star", "cloud", "rain", "snow"],
    }

    print(f"\n  {'Category':<12} {'Mean |Q|^2':>10} {'Std':>8} {'N':>4}")
    print(f"  {'-'*12} {'-'*10} {'-'*8} {'-'*4}")

    cat_masses = {}
    for cat, words in categories.items():
        masses = []
        for w in words:
            if w in projections:
                q = projections[w][:4]
                masses.append(np.sum(q**2))
        if masses:
            cat_masses[cat] = masses
            print(f"  {cat:<12} {np.mean(masses):>10.4f} {np.std(masses):>8.4f} {len(masses):>4}")

    # Are differences significant?
    if 'concrete' in cat_masses and 'abstract' in cat_masses:
        diff = np.mean(cat_masses['concrete']) - np.mean(cat_masses['abstract'])
        print(f"\n  concrete - abstract = {diff:+.4f} "
              f"{'(concrete heavier)' if diff > 0.1 else '(abstract heavier)' if diff < -0.1 else '(no difference)'}")


# ============================================================
# Test 3: Analogy Head-to-Head
# ============================================================

def test_analogy_comparison(projections, mapper):
    """Compare quaternion vs prime algebra for analogy solving."""
    print("\n" + "=" * 60)
    print("  TEST 3: Analogy Solving — Quaternion vs Prime Algebra")
    print("=" * 60)

    analogies = [
        ("king", "queen", "man", "woman"),
        ("king", "queen", "boy", "girl"),
        ("king", "queen", "father", "mother"),
        ("man", "woman", "boy", "girl"),
        ("happy", "sad", "love", "hate"),
        ("happy", "sad", "peace", "war"),
        ("big", "small", "fast", "slow"),
        ("fire", "water", "sun", "moon"),
        ("morning", "night", "summer", "winter"),
        ("dog", "cat", "horse", "cow"),
        ("doctor", "hospital", "teacher", "school"),
        ("brother", "sister", "father", "mother"),
    ]

    # Build vocab for search
    all_concepts = list(projections.keys())
    primes = {c: mapper.map(projections[c]) for c in all_concepts}

    quat_correct = 0
    prime_correct = 0
    both_correct = 0
    total = 0

    print(f"\n  {'A':>8} {'B':>8} {'C':>8} {'D_true':>8} | {'Q_pred':>8} {'P_pred':>8} | {'Q':>3} {'P':>3}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8}   {'-'*8} {'-'*8}   {'-'*3} {'-'*3}")

    for a, b, c, d in analogies:
        if any(x not in projections for x in [a, b, c, d]):
            continue
        total += 1

        # --- Quaternion method ---
        q_a = projections[a][:4]
        q_b = projections[b][:4]
        q_c = projections[c][:4]
        # Transform = B * A^(-1), then apply to C
        q_transform = quat_multiply(q_b, quat_inverse(q_a))
        q_predicted = quat_multiply(q_transform, q_c)

        # Find closest concept to q_predicted
        best_q_sim = -1
        best_q_word = "?"
        for w in all_concepts:
            if w in [a, b, c]:
                continue
            q_w = projections[w][:4]
            sim = cosine_sim(q_predicted, q_w)
            if sim > best_q_sim:
                best_q_sim = sim
                best_q_word = w
        q_hit = best_q_word == d

        # --- Full vector method (all dims, not just 4) ---
        # Also test: use ALL dimensions with vector arithmetic
        v_a = projections[a]
        v_b = projections[b]
        v_c = projections[c]
        v_predicted_full = v_b - v_a + v_c

        best_v_sim = -1
        best_v_word = "?"
        for w in all_concepts:
            if w in [a, b, c]:
                continue
            sim = cosine_sim(v_predicted_full, projections[w])
            if sim > best_v_sim:
                best_v_sim = sim
                best_v_word = w

        # --- Prime algebra method ---
        p_a, p_b, p_c = primes[a], primes[b], primes[c]
        import math
        shared_ab = math.gcd(p_a, p_b)
        transform = p_b // shared_ab if shared_ab > 0 else p_b
        target = (p_c * transform) // math.gcd(p_c, transform)

        def factor_sim(x, y):
            fx, fy = set(prime_factors(x)), set(prime_factors(y))
            if not fx and not fy:
                return 1.0
            union = fx | fy
            return len(fx & fy) / len(union) if union else 0.0

        best_p_sim = -1
        best_p_word = "?"
        for w in all_concepts:
            if w in [a, b, c]:
                continue
            sim = factor_sim(target, primes[w])
            if sim > best_p_sim:
                best_p_sim = sim
                best_p_word = w
        p_hit = best_p_word == d

        if q_hit:
            quat_correct += 1
        if p_hit:
            prime_correct += 1
        if q_hit and p_hit:
            both_correct += 1

        q_mark = "Y" if q_hit else "."
        p_mark = "Y" if p_hit else "."
        print(f"  {a:>8} {b:>8} {c:>8} {d:>8} | {best_q_word:>8} {best_p_word:>8} | {q_mark:>3} {p_mark:>3}")

    if total > 0:
        print(f"\n  Quaternion (4D):   {quat_correct}/{total} ({quat_correct/total:.0%})")
        print(f"  Prime algebra:     {prime_correct}/{total} ({prime_correct/total:.0%})")
        print(f"  Both correct:      {both_correct}/{total} ({both_correct/total:.0%})")
        print(f"  Vec arithmetic (all dims): {best_v_word} (shown for last analogy only)")

        if quat_correct > prime_correct:
            print("\n  --> Quaternion WINS: captures transformations primes miss")
        elif prime_correct > quat_correct:
            print("\n  --> Prime algebra WINS: discrete features are sufficient")
        else:
            print("\n  --> TIE: both capture the same information")


# ============================================================
# Test 4: Do first 4 dims carry more transformation signal?
# ============================================================

def test_dimensional_signal(projections):
    """Compare analogy signal in first 4 dims vs random 4 dims vs all dims."""
    print("\n" + "=" * 60)
    print("  TEST 4: Where is the transformation signal?")
    print("  First 4D vs random 4D vs all dims")
    print("=" * 60)

    pairs = [
        ("king", "queen"), ("man", "woman"), ("boy", "girl"),
        ("happy", "sad"), ("love", "hate"),
        ("fire", "water"), ("sun", "moon"),
    ]

    n_bits = len(next(iter(projections.values())))
    np.random.seed(42)
    random_dims = np.random.choice(n_bits, 4, replace=False)

    print(f"\n  Random dims selected: {random_dims.tolist()}")
    print(f"\n  {'Pair':<20} {'First4':>8} {'Rand4':>8} {'AllDims':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")

    for a, b in pairs:
        if a not in projections or b not in projections:
            continue
        p_a, p_b = projections[a], projections[b]

        sim_first4 = cosine_sim(p_a[:4], p_b[:4])
        sim_rand4 = cosine_sim(p_a[random_dims], p_b[random_dims])
        sim_all = cosine_sim(p_a, p_b)

        print(f"  {a+'-'+b:<20} {sim_first4:>8.4f} {sim_rand4:>8.4f} {sim_all:>8.4f}")


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = 'checkpoints/torch_run15_strongalign/model_L12_D512_B64_best.pt'
    tokenizer_path = 'checkpoints/torch_run15_strongalign/tokenizer.json'

    print("Loading model...")
    model, tokenizer, config = load_model(model_path, tokenizer_path, device)
    n_bits = config.n_triadic_bits
    mapper = PrimeMapper(n_bits)

    # Get all concepts we need
    concepts = [
        "king", "queen", "man", "woman", "boy", "girl", "prince", "princess",
        "father", "mother", "brother", "sister", "friend", "enemy",
        "dog", "cat", "bird", "fish", "horse", "cow",
        "doctor", "nurse", "teacher", "student", "hospital", "school",
        "happy", "sad", "angry", "afraid", "brave", "kind",
        "fire", "water", "sun", "moon", "star", "cloud",
        "big", "small", "fast", "slow", "old", "young",
        "love", "hate", "hope", "fear", "joy", "peace", "war",
        "morning", "night", "summer", "winter",
        "run", "walk", "swim", "fly", "jump", "climb", "fall", "sleep",
        "house", "car", "tree", "book", "table", "chair", "dream", "magic",
        "rain", "snow",
    ]

    print(f"Getting projections for {len(concepts)} concepts...")
    projections = get_projections(model, tokenizer, concepts, device)
    print(f"Got {len(projections)} valid projections (dim={n_bits})")

    del model
    torch.cuda.empty_cache()

    # Run tests
    test_rotation_consistency(projections)
    test_magnitude_semantics(projections)
    test_analogy_comparison(projections, mapper)
    test_dimensional_signal(projections)

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()
