"""
Fast BPE Tokenizer — HuggingFace `tokenizers` (Rust) backend.

Drop-in replacement for src/tokenizer.py but ~1000× faster.
Trains BPE in seconds, encodes millions of tokens per second.

Same API: train(), encode(), decode(), save(), load(), encode_chat()
Same special tokens: <PAD>, <BOS>, <EOS>, <UNK>, <USER>, <ASSISTANT>
"""

import os
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders


# ============================================================
# Special Tokens (must match src/tokenizer.py)
# ============================================================

SPECIAL_TOKENS_LIST = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', '<USER>', '<ASSISTANT>']
SPECIAL_TOKENS = {tok: i for i, tok in enumerate(SPECIAL_TOKENS_LIST)}
NUM_SPECIAL = len(SPECIAL_TOKENS)


# ============================================================
# Fast BPE Tokenizer
# ============================================================

class FastBPETokenizer:
    """
    HuggingFace tokenizers-based BPE tokenizer.

    Drop-in replacement for BPETokenizer in src/tokenizer.py.
    ~1000× faster training and encoding.

    Usage:
        tokenizer = FastBPETokenizer(vocab_size=4096)
        tokenizer.train(corpus_texts)
        ids = tokenizer.encode("Hello world")
        text = tokenizer.decode(ids)
    """

    def __init__(self, vocab_size=4096):
        self.target_vocab_size = vocab_size
        self.special_tokens = dict(SPECIAL_TOKENS)
        self._tokenizer = None
        self._id_to_token = {}
        self._token_to_id = {}

    def train(self, texts, verbose=True):
        """
        Train the BPE tokenizer on a list of text strings.
        Uses Rust backend — trains in seconds, not hours.
        """
        if verbose:
            print(f"  Training Fast BPE tokenizer (target vocab: {self.target_vocab_size})...")

        # Initialize a BPE tokenizer
        tokenizer = Tokenizer(models.BPE(unk_token='<UNK>'))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        # BPE trainer — reserve slots for special tokens
        trainer = trainers.BpeTrainer(
            vocab_size=self.target_vocab_size,
            special_tokens=SPECIAL_TOKENS_LIST,
            min_frequency=2,
            show_progress=verbose,
        )

        # Train from iterator (no temp files needed)
        tokenizer.train_from_iterator(texts, trainer=trainer)

        # Add post-processor for BOS/EOS
        bos_id = tokenizer.token_to_id('<BOS>')
        eos_id = tokenizer.token_to_id('<EOS>')
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"<BOS>:0 $A:0 <EOS>:0",
            pair=f"<BOS>:0 $A:0 <EOS>:0 <BOS>:1 $B:1 <EOS>:1",
            special_tokens=[
                ('<BOS>', bos_id),
                ('<EOS>', eos_id),
            ],
        )

        # Add ByteLevel decoder (converts Ġ back to spaces)
        tokenizer.decoder = decoders.ByteLevel()

        self._tokenizer = tokenizer
        self._build_mappings()

        if verbose:
            print(f"  Vocab size: {self.vocab_size}")
            print(f"  Training complete!")

    def _build_mappings(self):
        """Build id↔token mappings."""
        vocab = self._tokenizer.get_vocab()
        self._token_to_id = vocab
        self._id_to_token = {v: k for k, v in vocab.items()}
        # Update special tokens with actual IDs
        for tok in SPECIAL_TOKENS_LIST:
            tid = self._tokenizer.token_to_id(tok)
            if tid is not None:
                self.special_tokens[tok] = tid

    def encode(self, text, add_special=True):
        """
        Encode a string into a list of token IDs.

        Args:
            text: input string
            add_special: if True, wrap with <BOS> and <EOS>

        Returns:
            list of integer token IDs
        """
        if not text:
            if add_special:
                return [self.special_tokens['<BOS>'], self.special_tokens['<EOS>']]
            return []

        if add_special:
            # Post-processor adds BOS/EOS automatically
            encoded = self._tokenizer.encode(text)
        else:
            # Temporarily disable post-processor
            old_pp = self._tokenizer.post_processor
            self._tokenizer.post_processor = None
            encoded = self._tokenizer.encode(text)
            self._tokenizer.post_processor = old_pp

        return encoded.ids

    def decode(self, token_ids, skip_special=True):
        """
        Decode a list of token IDs back into a string.

        Args:
            token_ids: list of integer token IDs
            skip_special: if True, skip special tokens in output

        Returns:
            decoded string
        """
        if not token_ids:
            return ""
        text = self._tokenizer.decode(token_ids, skip_special_tokens=skip_special)
        # Clean up any lingering ByteLevel artifacts
        text = text.replace('Ġ', ' ').replace('Ä', '').replace('Ċ', '\n')
        return text

    @property
    def vocab_size(self):
        """Total vocabulary size including special tokens."""
        return self._tokenizer.get_vocab_size()

    def encode_chat(self, user_msg, assistant_msg=None):
        """
        Encode a chat turn for instruction tuning.

        Returns token IDs for: <BOS> <USER> user_text <ASSISTANT> assistant_text <EOS>
        """
        ids = [self.special_tokens['<BOS>'], self.special_tokens['<USER>']]
        ids += self.encode(user_msg, add_special=False)

        ids.append(self.special_tokens['<ASSISTANT>'])

        if assistant_msg:
            ids += self.encode(assistant_msg, add_special=False)

        ids.append(self.special_tokens['<EOS>'])
        return ids

    def save(self, filepath):
        """Save tokenizer to JSON."""
        self._tokenizer.save(filepath)
        # Also save special tokens map alongside
        meta_path = filepath + '.meta'
        with open(meta_path, 'w') as f:
            json.dump({'special_tokens': self.special_tokens, 'type': 'fast_bpe'}, f)

    @classmethod
    def load(cls, filepath):
        """Load tokenizer from JSON."""
        obj = cls()
        obj._tokenizer = Tokenizer.from_file(filepath)
        obj._build_mappings()

        # Load meta if exists
        meta_path = filepath + '.meta'
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
                obj.special_tokens = meta.get('special_tokens', dict(SPECIAL_TOKENS))

        return obj
