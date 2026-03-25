"""
Evaluation script for Claudia.

Usage:
    uv run evaluate.py --checkpoint checkpoints/claudia_best.pt
"""

import argparse
import math
import os
import torch
from collections import Counter
from tqdm import tqdm

from claudia.model import Claudia
from claudia.tokenizer import ClaudiaTokenizer
from claudia.data import prepare_tinystories, create_dataloaders

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.85"
os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.7"


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = Claudia(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


@torch.no_grad()
def compute_perplexity(model, val_loader, device, max_batches=500):
    total_loss = 0.0
    total_tokens = 0
    for i, (x, y) in enumerate(tqdm(val_loader, desc="Perplexity", total=min(max_batches, len(val_loader)))):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            _, loss = model(x, y)
        total_loss += loss.item() * x.numel()
        total_tokens += x.numel()
    return math.exp(min(total_loss / total_tokens, 100))


def distinct_n(texts, n=2):
    all_ngrams = []
    for text in texts:
        words = text.split()
        all_ngrams.extend(tuple(words[i:i+n]) for i in range(len(words) - n + 1))
    return len(set(all_ngrams)) / max(len(all_ngrams), 1)


def repetition_rate(text, n=4):
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    counts = Counter(ngrams)
    return sum(c - 1 for c in counts.values() if c > 1) / len(ngrams)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(args.checkpoint, device)
    tokenizer = ClaudiaTokenizer()
    print(f"Params: {model.param_count():,}")

    # Perplexity
    _, val_dataset = prepare_tinystories(tokenizer, cache_dir="data", seq_len=config.max_seq_len)
    _, val_loader = create_dataloaders(val_dataset, val_dataset, batch_size=args.batch_size)
    ppl = compute_perplexity(model, val_loader, device)
    print(f"\nPerplexity: {ppl:.2f}")

    # Generation quality
    prompts = ["Once upon a time", "The little girl", "One day a boy named",
               "There was a big", "The dog ran to the", "Mom said to",
               "The cat liked to", "It was a sunny day", "The bird flew over", "A little rabbit"]
    texts = []
    for i in range(args.num_samples):
        prompt = prompts[i % len(prompts)]
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            out = model.generate(ids, max_new_tokens=150, temperature=0.8, top_k=40, top_p=0.9)
        texts.append(tokenizer.decode(out[0].tolist()))

    d1, d2, d3 = distinct_n(texts, 1), distinct_n(texts, 2), distinct_n(texts, 3)
    rep = sum(repetition_rate(t) for t in texts) / len(texts)

    print(f"Distinct-1: {d1:.4f}")
    print(f"Distinct-2: {d2:.4f}")
    print(f"Distinct-3: {d3:.4f}")
    print(f"Repetition (4g): {rep:.4f}")

    print("\n--- Samples ---")
    for s in texts[:5]:
        print(f"  {s[:300]}\n")

    print(f"{'='*50}")
    print(f"  PPL={ppl:.2f} | D1={d1:.3f} D2={d2:.3f} | Rep={rep:.3f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
