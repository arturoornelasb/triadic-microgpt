"""
BPE Tokenizer — Byte Pair Encoding for subword tokenization.

Replaces the character-level tokenizer with a learned subword vocabulary.
This lets the model process "the" as 1 token instead of 3, dramatically
improving both speed and language understanding.

Includes special tokens for future chat capability:
  <PAD>, <BOS>, <EOS>, <UNK>, <USER>, <ASSISTANT>

Zero external dependencies — BPE training and encoding from scratch.
"""

import re
import json
from collections import Counter, defaultdict


# ============================================================
# Special Tokens
# ============================================================

SPECIAL_TOKENS = {
    '<PAD>': 0,
    '<BOS>': 1,
    '<EOS>': 2,
    '<UNK>': 3,
    '<USER>': 4,
    '<ASSISTANT>': 5,
}

NUM_SPECIAL = len(SPECIAL_TOKENS)


# ============================================================
# BPE Tokenizer
# ============================================================

class BPETokenizer:
    """
    Byte Pair Encoding tokenizer trained on a text corpus.

    BPE starts with individual characters as tokens, then iteratively
    merges the most frequent adjacent pair into a new token. This
    naturally learns common subwords (e.g., "th", "ing", "the").

    Usage:
        tokenizer = BPETokenizer(vocab_size=4096)
        tokenizer.train(corpus_texts)
        ids = tokenizer.encode("Hello world")
        text = tokenizer.decode(ids)
    """

    def __init__(self, vocab_size=4096):
        self.target_vocab_size = vocab_size
        self.merges = []           # list of (token_a, token_b) merge rules
        self.vocab = {}            # id → token string
        self.token_to_id = {}      # token string → id
        self.special_tokens = dict(SPECIAL_TOKENS)

    def _get_word_freqs(self, texts):
        """
        Split texts into words and count frequencies.
        Each word is represented as a tuple of characters + end-of-word marker.
        """
        word_freqs = Counter()
        for text in texts:
            # Simple word splitting: split on whitespace and punctuation boundaries
            words = re.findall(r'[a-zA-Z]+|[0-9]+|[^\s\w]|\s+', text)
            for word in words:
                # Represent word as space-separated characters with end marker
                chars = tuple(word) + ('</w>',)
                word_freqs[chars] += 1
        return word_freqs

    def _get_pair_counts(self, word_freqs):
        """Count all adjacent token pairs across the vocabulary."""
        pairs = Counter()
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] += freq
        return pairs

    def _merge_pair(self, word_freqs, pair):
        """Merge all occurrences of a pair in the vocabulary."""
        new_word_freqs = {}
        merged = pair[0] + pair[1]

        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq

        return new_word_freqs

    def train(self, texts, verbose=True):
        """
        Train the BPE tokenizer on a list of text strings.

        Args:
            texts: list of strings (training corpus)
            verbose: print progress
        """
        if verbose:
            print(f"  Training BPE tokenizer (target vocab: {self.target_vocab_size})...")

        # Step 1: Get word frequencies
        word_freqs = self._get_word_freqs(texts)

        # Step 2: Initialize vocabulary with all unique characters + </w>
        all_chars = set()
        for word in word_freqs:
            for ch in word:
                all_chars.add(ch)

        # Build initial vocab: special tokens + characters
        self.vocab = {}
        self.token_to_id = {}

        for token, tid in self.special_tokens.items():
            self.vocab[tid] = token
            self.token_to_id[token] = tid

        next_id = NUM_SPECIAL
        for ch in sorted(all_chars):
            self.vocab[next_id] = ch
            self.token_to_id[ch] = next_id
            next_id += 1

        initial_vocab = next_id
        if verbose:
            print(f"    Initial vocab: {initial_vocab} tokens ({len(all_chars)} chars + {NUM_SPECIAL} special)")

        # Step 3: Iteratively merge most frequent pairs
        num_merges = self.target_vocab_size - initial_vocab
        self.merges = []

        for merge_idx in range(num_merges):
            pair_counts = self._get_pair_counts(word_freqs)
            if not pair_counts:
                break

            best_pair = pair_counts.most_common(1)[0]
            pair, count = best_pair

            if count < 2:
                break  # No more useful merges

            # Merge the pair
            word_freqs = self._merge_pair(word_freqs, pair)
            merged_token = pair[0] + pair[1]
            self.merges.append(pair)

            # Add to vocab
            self.vocab[next_id] = merged_token
            self.token_to_id[merged_token] = next_id
            next_id += 1

            if verbose and (merge_idx + 1) % 500 == 0:
                print(f"    Merge {merge_idx+1}/{num_merges}: "
                      f"'{pair[0]}' + '{pair[1]}' → '{merged_token}' (freq={count})")

        actual_vocab_size = len(self.vocab)
        if verbose:
            print(f"    Final vocab: {actual_vocab_size} tokens ({len(self.merges)} merges)")

    def encode(self, text, add_special=True):
        """
        Encode a string into a list of token IDs.

        Args:
            text: input string
            add_special: if True, wrap with <BOS> and <EOS>

        Returns:
            list of integer token IDs
        """
        # Split into words
        words = re.findall(r'[a-zA-Z]+|[0-9]+|[^\s\w]|\s+', text)
        token_ids = []

        if add_special:
            token_ids.append(self.special_tokens['<BOS>'])

        for word in words:
            # Start with character-level tokens
            tokens = list(word) + ['</w>']

            # Apply learned merges in order
            for pair in self.merges:
                merged = pair[0] + pair[1]
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                        new_tokens.append(merged)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens

            # Convert tokens to IDs
            for tok in tokens:
                if tok in self.token_to_id:
                    token_ids.append(self.token_to_id[tok])
                else:
                    token_ids.append(self.special_tokens['<UNK>'])

        if add_special:
            token_ids.append(self.special_tokens['<EOS>'])

        return token_ids

    def decode(self, token_ids, skip_special=True):
        """
        Decode a list of token IDs back into a string.

        Args:
            token_ids: list of integer token IDs
            skip_special: if True, skip special tokens in output

        Returns:
            decoded string
        """
        tokens = []
        special_ids = set(self.special_tokens.values())

        for tid in token_ids:
            if skip_special and tid in special_ids:
                continue
            if tid in self.vocab:
                tokens.append(self.vocab[tid])
            else:
                tokens.append('<UNK>')

        text = ''.join(tokens)
        # Remove end-of-word markers and join
        text = text.replace('</w>', '')
        return text

    @property
    def vocab_size(self):
        """Total vocabulary size including special tokens."""
        return len(self.vocab)

    def encode_chat(self, user_msg, assistant_msg=None):
        """
        Encode a chat turn in the format needed for instruction tuning.

        Returns token IDs for: <BOS> <USER> user_text <ASSISTANT> assistant_text <EOS>
        """
        ids = [self.special_tokens['<BOS>'], self.special_tokens['<USER>']]
        ids.extend(self.encode(user_msg, add_special=False))
        ids.append(self.special_tokens['<ASSISTANT>'])

        if assistant_msg is not None:
            ids.extend(self.encode(assistant_msg, add_special=False))
            ids.append(self.special_tokens['<EOS>'])

        return ids

    def save(self, filepath):
        """Save tokenizer to a JSON file."""
        data = {
            'vocab': {str(k): v for k, v in self.vocab.items()},
            'merges': self.merges,
            'special_tokens': self.special_tokens,
            'target_vocab_size': self.target_vocab_size,
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath):
        """Load tokenizer from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        tokenizer = cls(vocab_size=data['target_vocab_size'])
        tokenizer.vocab = {int(k): v for k, v in data['vocab'].items()}
        tokenizer.token_to_id = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer.merges = [tuple(m) for m in data['merges']]
        tokenizer.special_tokens = data['special_tokens']
        return tokenizer


# ============================================================
# Entry point — Train tokenizer on a corpus file
# ============================================================

if __name__ == '__main__':
    import sys
    import os
    import argparse

    parser = argparse.ArgumentParser(description='Train BPE Tokenizer')
    parser.add_argument('--data', type=str, default=None, help='Path to training text')
    parser.add_argument('--vocab-size', type=int, default=4096, help='Target vocabulary size')
    parser.add_argument('--output', type=str, default=None, help='Output path for tokenizer')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    data_path = args.data or os.path.join(project_root, 'data', 'concepts.txt')
    output_path = args.output or os.path.join(project_root, 'tokenizer.json')

    print("=" * 60)
    print("  BPE Tokenizer Training")
    print("=" * 60)

    with open(data_path, 'r') as f:
        texts = [line.strip() for line in f if line.strip()]

    print(f"  Corpus: {len(texts)} lines, {sum(len(t) for t in texts):,} characters")

    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    tokenizer.train(texts)

    tokenizer.save(output_path)
    print(f"  Saved: {output_path}")

    # Demo
    print()
    print("  Demo:")
    test_texts = texts[:3] if texts else ["Hello world"]
    for text in test_texts:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        print(f"    '{text[:40]}...' → {len(ids)} tokens → '{decoded[:40]}...'")

    # Chat format demo
    print()
    print("  Chat format demo:")
    chat_ids = tokenizer.encode_chat("What is AI?", "AI is artificial intelligence.")
    print(f"    Chat tokens: {chat_ids}")
    print(f"    Decoded: {tokenizer.decode(chat_ids)}")

    print()
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print("=" * 60)
