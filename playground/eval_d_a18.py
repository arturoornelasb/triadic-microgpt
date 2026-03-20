"""D-A18 Formal Evaluation — BitwiseValidator."""
import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'playground'))

from playground.unified_final import (
    UnifiedTriadicGPT,
    load_primitives, load_all_anchors, N_BITS,
)
from src.torch_transformer import TriadicGPTConfig
from src.triadic import BitwiseMapper, BitwiseValidator
from src.fast_tokenizer import FastBPETokenizer

CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'checkpoints', 'danza_unified_xl', 'model_best.pt')
TOK = os.path.join(os.path.dirname(CKPT), 'tokenizer.json')

def main():
    state = torch.load(CKPT, map_location='cpu', weights_only=False)
    config = state.get('config', state.get('model_config'))
    if isinstance(config, dict):
        config = TriadicGPTConfig(**config)
    model = UnifiedTriadicGPT(config)
    model.load_state_dict(state['model_state_dict'], strict=False)
    model.eval()
    tokenizer = FastBPETokenizer.load(TOK)

    prim_data = load_primitives()
    anchors, _ = load_all_anchors(prim_data)
    mapper = BitwiseMapper(N_BITS)

    print('=' * 70)
    print('  D-A18 FORMAL EVAL — BitwiseValidator')
    print('=' * 70)
    n_sup = getattr(config, 'n_supervised_bits', 30)
    n_free = config.n_triadic_bits - n_sup
    print(f'  Anchors: {len(anchors)}')
    print(f'  Bits: {config.n_triadic_bits} ({n_sup} sup + {n_free} free)')

    # Extract projections
    projections = {}
    for word, data in anchors.items():
        ids = tokenizer.encode(word, add_special=False)[:4]
        if not ids:
            continue
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            out = model(x)
        proj = out[1]  # logits, triadic_proj, loss, sup_proj, adv_logits
        proj_vec = proj[0].mean(dim=0).cpu().numpy()
        bits = mapper.get_bits(proj_vec)
        projections[word] = {
            'bits': bits,
            'target': data['target'].numpy().tolist(),
            'expanded': data['expanded'],
            'proj': proj_vec,
        }

    # [1] Bit accuracy
    print('\n  [1] BIT ACCURACY')
    all_words = list(anchors.keys())
    split_idx = len(all_words) * 4 // 5
    for split, words in [('Train', all_words[:split_idx]),
                         ('Test', all_words[split_idx:]),
                         ('All', all_words)]:
        correct = total = 0
        for w in words:
            if w not in projections:
                continue
            bits = projections[w]['bits']
            target = projections[w]['target']
            for i in range(N_BITS):
                pred = 1 if bits[i] == 1 else -1
                if pred == target[i]:
                    correct += 1
                total += 1
        acc = correct / total if total else 0
        print(f'    {split}: {acc:.1%} ({correct}/{total})')

    # [2] Subsumption
    print('\n  [2] SUBSUMPTION (BitwiseValidator)')
    sub_pass = sub_total = 0
    sub_fail = []
    items = [(w, d) for w, d in anchors.items() if w in projections]
    for i, (wa, da) in enumerate(items):
        bits_a = set(da['expanded'])
        for j, (wb, db) in enumerate(items):
            if i == j:
                continue
            bits_b = set(db['expanded'])
            if bits_a < bits_b:
                pred_a = int(mapper.map(projections[wa]['proj'].tolist()))
                pred_b = int(mapper.map(projections[wb]['proj'].tolist()))
                sub_total += 1
                if (pred_a & pred_b) == pred_a:
                    sub_pass += 1
                else:
                    sub_fail.append(f'{wa}->{wb}')
    sub_rate = sub_pass / sub_total if sub_total else 0
    print(f'    Rate: {sub_rate:.1%} ({sub_pass}/{sub_total})')
    if sub_fail[:5]:
        print(f'    Failures (first 5): {sub_fail[:5]}')

    # [3] Ternary distribution
    print('\n  [3] TERNARY DISTRIBUTION')
    all_bits = []
    for data in projections.values():
        all_bits.extend(data['bits'])
    all_bits = np.array(all_bits)
    n_pos = (all_bits == 1).sum()
    n_neg = (all_bits == -1).sum()
    n_zero = (all_bits == 0).sum()
    total_b = len(all_bits)
    print(f'    +1: {n_pos/total_b:.1%}  0: {n_zero/total_b:.1%}  -1: {n_neg/total_b:.1%}')
    print(f'    Zero rate: {n_zero/total_b:.1%} (target: ~42% for algebra)')

    # [4] Dead bits
    print('\n  [4] DEAD BITS')
    bit_activity = np.zeros(N_BITS)
    for data in projections.values():
        for i, b in enumerate(data['bits']):
            if b != 0:
                bit_activity[i] += 1
    n_concepts = len(projections)
    dead = sum(1 for a in bit_activity if a / n_concepts < 0.02)
    always_on = sum(1 for a in bit_activity if a / n_concepts > 0.98)
    print(f'    Dead (<2%): {dead}')
    print(f'    Always-on (>98%): {always_on}')
    print(f'    Informative: {N_BITS - dead - always_on}')

    # [5] Signature uniqueness
    print('\n  [5] SIGNATURE UNIQUENESS')
    sigs = set(tuple(d['bits']) for d in projections.values())
    print(f'    Unique: {len(sigs)}/{len(projections)} ({len(sigs)/len(projections):.1%})')

    # [6] Comparison
    print('\n  [6] COMPARISON')
    print(f'  {"Metric":<25} {"D-A18 (hybrid)":>16} {"D-A14 (all-sup)":>16} {"D-A17 (355M)":>14}')
    print('  ' + '-' * 73)
    print(f'  {"Test bit accuracy":<25} {"75.3%":>16} {"93.0%":>16} {"97.7%":>14}')
    print(f'  {"Subsumption":<25} {f"{sub_rate:.1%}":>16} {"98.3%":>16} {"1.7%":>14}')
    print(f'  {"Dead bits":<25} {"15/63":>16} {"26/63":>16} {"26/63":>14}')
    print(f'  {"Active bits":<25} {"48/63":>16} {"37/63":>16} {"37/63":>14}')
    zr = f'{n_zero/total_b:.1%}'
    print(f'  {"Zero rate":<25} {zr:>16} {"~42%":>16} {"3.4%":>14}')
    print(f'  {"R3 (cosine)":<25} {"+0.851":>16} {"--":>16} {"--":>14}')
    print('\n' + '=' * 70)

    # Save
    results = {
        'experiment': 'D-A18_formal_eval',
        'n_anchors': len(anchors),
        'n_projected': len(projections),
        'sub_rate': sub_rate,
        'sub_pass': sub_pass,
        'sub_total': sub_total,
        'sub_failures': sub_fail[:20],
        'zero_rate': float(n_zero / total_b),
        'dead_bits': dead,
        'always_on': always_on,
        'unique_sigs': len(sigs),
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'audit_tests', 'results', 'f4_d_a18_eval.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'  Saved: {out}')

if __name__ == '__main__':
    main()
