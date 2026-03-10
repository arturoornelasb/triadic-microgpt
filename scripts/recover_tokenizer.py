"""
Regenera el tokenizer de Run 15.

Run 15 fue entrenado sin --tokenizer ni --stories explícitos, por lo que usó:
  - vocab_size=4096 (default de --vocab)
  - stories=50000 (default de --stories), shuffled con random.seed(42)
  - corpus: data/TinyStories-train.txt

Este script reproduce exactamente esa lógica para obtener el tokenizer
original (o funcionalmente equivalente), y lo guarda en
checkpoints/torch_run15_strongalign/tokenizer.json.

Criterio de éxito: PPL ≈ 7.69 cuando se evalúa con el modelo Run 15.
"""
import sys
import random

sys.path.insert(0, '.')
from src.fast_tokenizer import FastBPETokenizer

DATA = 'data/TinyStories-train.txt'
SAVE_PATH = 'checkpoints/torch_run15_strongalign/tokenizer.json'
VOCAB_SIZE = 4096
MAX_STORIES = 50000
STORY_SEPARATOR = '<|endoftext|>'

print(f"Loading corpus: {DATA}")
raw = open(DATA, encoding='utf-8').read()
stories = raw.split(STORY_SEPARATOR)
stories = [s.strip() for s in stories if s.strip() and len(s.strip()) > 30]
print(f"Total stories after filter: {len(stories):,}")

if len(stories) > MAX_STORIES:
    random.seed(42)
    random.shuffle(stories)
    stories = stories[:MAX_STORIES]
    print(f"Using first {MAX_STORIES:,} stories (seed=42 shuffle, matching Run 15 training)")

print(f"\nTraining BPE tokenizer (vocab_size={VOCAB_SIZE})...")
tok = FastBPETokenizer(vocab_size=VOCAB_SIZE)
tok.train(stories, verbose=True)

tok.save(SAVE_PATH)
print(f"\nSaved → {SAVE_PATH}")
print(f"Vocab size: {tok.vocab_size}")
print("\nNext step:")
print(f"  python src/evaluate.py \\")
print(f"    --model checkpoints/torch_run15_strongalign/model_L12_D512_B64_best.pt \\")
print(f"    --tokenizer {SAVE_PATH}")
print("\nExpected PPL: 7–10 (was ~7.69 at training time)")
