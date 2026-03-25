"""
Data pipeline for Claudia training.

Uses custom 4K BPE tokenizer for memory-efficient training.
Packs sequences with EOS separators for maximum data utilization.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm
from claudia.tokenizer import ClaudiaTokenizer, EOS_ID


class TokenizedDataset(Dataset):
    """Memory-mapped tokenized dataset for efficient training."""

    def __init__(self, tokens: np.ndarray, seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len
        self.num_sequences = (len(tokens) - 1) // seq_len

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len
        x = torch.from_numpy(self.tokens[start:end].astype(np.int64))
        y = torch.from_numpy(self.tokens[start + 1:end + 1].astype(np.int64))
        return x, y


def prepare_tinystories(
    tokenizer: ClaudiaTokenizer,
    cache_dir: str = "data",
    seq_len: int = 512,
) -> tuple[TokenizedDataset, TokenizedDataset]:
    """Download and tokenize TinyStories with our custom tokenizer."""
    os.makedirs(cache_dir, exist_ok=True)
    train_path = os.path.join(cache_dir, f"tinystories_train_v{tokenizer.vocab_size}_{seq_len}.npy")
    val_path = os.path.join(cache_dir, f"tinystories_val_v{tokenizer.vocab_size}_{seq_len}.npy")

    if os.path.exists(train_path) and os.path.exists(val_path):
        print("Loading cached tokenized data...")
        train_tokens = np.load(train_path)
        val_tokens = np.load(val_path)
    else:
        print("Downloading TinyStories dataset...")
        ds = load_dataset("roneneldan/TinyStories")

        def tokenize_split(split_data, desc: str) -> np.ndarray:
            all_tokens = []
            for example in tqdm(split_data, desc=f"Tokenizing {desc}"):
                tokens = tokenizer.encode(example["text"])
                tokens.append(EOS_ID)
                all_tokens.extend(tokens)
            return np.array(all_tokens, dtype=np.uint16)

        train_tokens = tokenize_split(ds["train"], "train")
        print(f"Train: {len(train_tokens):,} tokens")

        val_tokens = tokenize_split(ds["validation"], "validation")
        print(f"Validation: {len(val_tokens):,} tokens")

        np.save(train_path, train_tokens)
        np.save(val_path, val_tokens)
        print(f"Cached to {cache_dir}/")

    train_dataset = TokenizedDataset(train_tokens, seq_len)
    val_dataset = TokenizedDataset(val_tokens, seq_len)
    print(f"Train: {len(train_dataset):,} seqs | Val: {len(val_dataset):,} seqs | Tok/seq: {seq_len}")
    return train_dataset, val_dataset


def create_dataloaders(
    train_dataset: TokenizedDataset,
    val_dataset: TokenizedDataset,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader, val_loader
