"""
Training script for Claudia.

Optimized for 10GB VRAM / 30min on Apple Silicon.
Uses custom 4K vocab tokenizer + float16 mixed precision.

Usage:
    uv run train.py --config tiny
    uv run train.py --config small --batch-size 32
"""

import argparse
import json
import math
import os
import time
import torch
from tqdm import tqdm

from claudia.config import ClaudiaConfig, CONFIGS
from claudia.model import Claudia
from claudia.tokenizer import ClaudiaTokenizer
from claudia.data import prepare_tinystories, create_dataloaders

# Hard 10GB VRAM cap for MPS
# Hard ~10GB VRAM cap (0.85 of ~11.8GB recommended pool)
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.85"
os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.7"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine LR schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, val_loader, device, max_batches=50):
    model.eval()
    total_loss = 0.0
    count = 0
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            _, loss = model(x, y)
        total_loss += loss.item()
        count += 1
    model.train()
    return total_loss / max(count, 1)


def generate_sample(model, tokenizer, device, prompt="Once upon a time"):
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    model.eval()
    with torch.autocast(device_type=device.type, dtype=torch.float16):
        output = model.generate(input_ids, max_new_tokens=150, temperature=0.8, top_k=40, top_p=0.9)
    model.train()
    return tokenizer.decode(output[0].tolist())


def train(args):
    device = get_device()
    print(f"Device: {device}")
    print(f"MPS memory cap: ~10GB (PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.85)")

    # Tokenizer — train or load
    tokenizer = ClaudiaTokenizer()
    print(f"Tokenizer vocab: {tokenizer.vocab_size}")

    # Model
    config = CONFIGS[args.config]
    config.vocab_size = tokenizer.vocab_size  # Match tokenizer
    model = Claudia(config).to(device)
    params = model.param_count()
    param_mb = params * 4 / 1e6  # fp32 size
    print(f"\nClaudia [{args.config}]: {params:,} params ({params/1e6:.1f}M) | ~{param_mb:.0f}MB fp32")

    # Data
    train_dataset, val_dataset = prepare_tinystories(
        tokenizer, cache_dir="data", seq_len=config.max_seq_len,
    )
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay,
    )

    # Schedule calibration
    max_seconds = args.max_minutes * 60
    steps_per_epoch = len(train_loader) // args.grad_accum_steps
    total_steps = steps_per_epoch * args.epochs  # Will be recalibrated
    warmup_steps = min(500, total_steps // 10)
    min_lr = args.lr * 0.1

    print(f"Max time: {args.max_minutes}min | Batch: {args.batch_size}x{args.grad_accum_steps} = {args.batch_size * args.grad_accum_steps} effective")
    print(f"Steps/epoch: {steps_per_epoch:,}")

    os.makedirs(args.output_dir, exist_ok=True)
    log = {"config": args.config, "params": params, "steps": []}

    global_step = 0
    best_val_loss = float("inf")
    tokens_processed = 0
    t0 = time.time()
    calibrated = False

    for epoch in range(args.epochs):
        if time.time() - t0 >= max_seconds:
            break
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
        model.train()
        optimizer.zero_grad()
        accum_loss = 0.0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
        for batch_idx, (x, y) in pbar:
            elapsed = time.time() - t0
            if elapsed >= max_seconds:
                print(f"\n  Time limit ({args.max_minutes}min). Stopping.")
                break

            x, y = x.to(device), y.to(device)

            # Mixed precision forward/backward
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                _, loss = model(x, y)
            loss = loss / args.grad_accum_steps
            loss.backward()
            accum_loss += loss.item()
            tokens_processed += x.numel()

            if (batch_idx + 1) % args.grad_accum_steps == 0:
                # Recalibrate LR schedule after 10 steps
                if not calibrated and global_step == 10:
                    secs_per_step = (time.time() - t0) / 10
                    total_steps = int(max_seconds / secs_per_step)
                    warmup_steps = min(500, total_steps // 10)
                    tok_rate = tokens_processed / (time.time() - t0)
                    calibrated = True
                    print(f"\n  ~{secs_per_step:.3f}s/step → ~{total_steps} steps in {args.max_minutes}min | {tok_rate:,.0f} tok/s")

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                lr = get_lr(global_step, warmup_steps, total_steps, args.lr, min_lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                optimizer.step()
                optimizer.zero_grad()

                remaining = max_seconds - (time.time() - t0)
                pbar.set_postfix({
                    "loss": f"{accum_loss:.4f}",
                    "lr": f"{lr:.2e}",
                    "tok/s": f"{tokens_processed/(time.time()-t0):,.0f}",
                    "left": f"{remaining/60:.1f}m",
                })

                # Eval
                if global_step % args.eval_interval == 0 and global_step > 0:
                    val_loss = estimate_loss(model, val_loader, device)
                    val_ppl = math.exp(min(val_loss, 20))
                    print(f"\n  Step {global_step} | val_loss={val_loss:.4f} | PPL={val_ppl:.2f}")

                    log["steps"].append({
                        "step": global_step, "epoch": epoch+1,
                        "train_loss": accum_loss, "val_loss": val_loss,
                        "val_ppl": val_ppl, "elapsed_min": elapsed/60,
                    })

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save({
                            "model_state_dict": model.state_dict(),
                            "config": config, "step": global_step, "val_loss": val_loss,
                        }, os.path.join(args.output_dir, "claudia_best.pt"))
                        print(f"  Saved best (val_loss={val_loss:.4f})")

                # Sample generation
                if global_step % args.sample_interval == 0 and global_step > 0:
                    sample = generate_sample(model, tokenizer, device)
                    print(f"\n  [Sample@{global_step}] {sample[:300]}\n")

                accum_loss = 0.0
                global_step += 1

    # Final eval
    val_loss = estimate_loss(model, val_loader, device, max_batches=200)
    val_ppl = math.exp(min(val_loss, 20))
    elapsed = time.time() - t0

    print(f"\n{'='*50}")
    print(f"TRAINING COMPLETE")
    print(f"  Steps:       {global_step:,}")
    print(f"  Val loss:    {val_loss:.4f}")
    print(f"  Val PPL:     {val_ppl:.2f}")
    print(f"  Best loss:   {best_val_loss:.4f}")
    print(f"  Time:        {elapsed/60:.1f}min")
    print(f"  Tokens/sec:  {tokens_processed/elapsed:,.0f}")
    print(f"  Total tokens:{tokens_processed:,}")
    print(f"{'='*50}")

    # Save final
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config, "step": global_step, "val_loss": val_loss,
    }, os.path.join(args.output_dir, "claudia_final.pt"))

    log.update({
        "final_val_loss": val_loss, "final_val_ppl": val_ppl,
        "best_val_loss": best_val_loss, "total_time_min": elapsed/60,
        "total_tokens": tokens_processed, "total_steps": global_step,
    })
    with open(os.path.join(args.output_dir, "training_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    # Final samples
    print("\n--- Final Samples ---")
    for prompt in ["Once upon a time", "The little dog", "Sarah was happy because", "One day a boy named Tom"]:
        sample = generate_sample(model, tokenizer, device, prompt)
        print(f"  [{prompt}] → {sample[:300]}\n")

    return val_loss, val_ppl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Claudia")
    parser.add_argument("--config", default="tiny", choices=["tiny", "small", "medium"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-minutes", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--sample-interval", type=int, default=1000)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    train(args)
