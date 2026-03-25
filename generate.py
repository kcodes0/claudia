"""
Interactive generation with Claudia.

Usage:
    uv run generate.py --checkpoint checkpoints/claudia_best.pt
    uv run generate.py --checkpoint checkpoints/claudia_best.pt --prompt "The cat sat on"
"""

import argparse
import torch
from claudia.model import Claudia
from claudia.tokenizer import ClaudiaTokenizer


def load_model(checkpoint_path: str, device: torch.device) -> Claudia:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = Claudia(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded step {checkpoint.get('step', '?')} | val_loss={checkpoint.get('val_loss', '?')} | {model.param_count():,} params")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--num-samples", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    tokenizer = ClaudiaTokenizer()

    if args.prompt:
        for i in range(args.num_samples):
            tokens = tokenizer.encode(args.prompt)
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                output = model.generate(input_ids, max_new_tokens=args.max_tokens,
                                        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
            print(tokenizer.decode(output[0].tolist()))
    else:
        print("\nClaudia Interactive — type a prompt, 'quit' to exit\n")
        while True:
            try:
                prompt = input(">>> ")
            except (EOFError, KeyboardInterrupt):
                break
            if prompt.strip().lower() in ("quit", "exit", "q"):
                break
            if not prompt.strip():
                continue
            tokens = tokenizer.encode(prompt)
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                output = model.generate(input_ids, max_new_tokens=args.max_tokens,
                                        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
            print(f"\n{tokenizer.decode(output[0].tolist())}\n")


if __name__ == "__main__":
    main()
