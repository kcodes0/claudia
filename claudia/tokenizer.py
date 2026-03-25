"""
Custom BPE tokenizer for Claudia.

4096 vocab — tiny footprint, optimized for small-model training.
Trained on TinyStories data using HuggingFace tokenizers (Rust-fast).
"""

import os
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from datasets import load_dataset


VOCAB_SIZE = 4096
SPECIAL_TOKENS = ["<|pad|>", "<|eos|>", "<|unk|>"]
PAD_ID = 0
EOS_ID = 1
UNK_ID = 2


def train_tokenizer(save_path: str = "claudia_tokenizer.json", num_samples: int = 200_000) -> Tokenizer:
    """Train a BPE tokenizer on TinyStories data."""
    if os.path.exists(save_path):
        print(f"Loading existing tokenizer from {save_path}")
        return Tokenizer.from_file(save_path)

    print(f"Training BPE tokenizer (vocab_size={VOCAB_SIZE}) on {num_samples:,} samples...")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    # Collect text samples
    texts = []
    for i, example in enumerate(ds):
        if i >= num_samples:
            break
        texts.append(example["text"])

    # Build tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train_from_iterator(texts, trainer=trainer)

    # Add post-processor for EOS
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path} (vocab_size={tokenizer.get_vocab_size()})")
    return tokenizer


def load_tokenizer(path: str = "claudia_tokenizer.json") -> Tokenizer:
    """Load a trained tokenizer."""
    return Tokenizer.from_file(path)


class ClaudiaTokenizer:
    """Wrapper around the HF tokenizer for convenience."""

    def __init__(self, path: str = "claudia_tokenizer.json"):
        if not os.path.exists(path):
            self.tokenizer = train_tokenizer(path)
        else:
            self.tokenizer = Tokenizer.from_file(path)
        self.eos_id = EOS_ID
        self.pad_id = PAD_ID
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [enc.ids for enc in self.tokenizer.encode_batch(texts)]
