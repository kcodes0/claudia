"""
Interactive generation with Claudia.

Usage:
    uv run generate.py --checkpoint checkpoints/claudia_best.pt
    uv run generate.py --checkpoint checkpoints/claudia_best.pt --prompt "The cat sat on"
    uv run generate.py --checkpoint checkpoints/claudia_best.pt --raw  # disable grammar polish
"""

import argparse
import torch
from claudia.model import Claudia
from claudia.tokenizer import ClaudiaTokenizer
from claudia.grammar import polish_text


def load_model(checkpoint_path: str, device: torch.device) -> Claudia:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = Claudia(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded step {checkpoint.get('step', '?')} | val_loss={checkpoint.get('val_loss', '?')} | {model.param_count():,} params")
    return model


def generate_text(model, tokenizer, device, prompt, max_tokens=200, temperature=0.8,
                  top_k=50, top_p=0.9, use_grammar=True):
    """Generate text with optional grammar polishing."""
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    with torch.autocast(device_type=device.type, dtype=torch.float16):
        output = model.generate(input_ids, max_new_tokens=max_tokens,
                                temperature=temperature, top_k=top_k, top_p=top_p)
    text = tokenizer.decode(output[0].tolist())
    if use_grammar:
        text = polish_text(text)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--raw", action="store_true", help="Disable grammar polishing")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    tokenizer = ClaudiaTokenizer()
    use_grammar = not args.raw

    if use_grammar:
        print("Grammar engine: ON")

    if args.prompt:
        for i in range(args.num_samples):
            text = generate_text(model, tokenizer, device, args.prompt,
                                 args.max_tokens, args.temperature, args.top_k, args.top_p, use_grammar)
            if args.num_samples > 1:
                print(f"\n--- Sample {i+1} ---")
            print(text)
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
            text = generate_text(model, tokenizer, device, prompt,
                                 args.max_tokens, args.temperature, args.top_k, args.top_p, use_grammar)
            print(f"\n{text}\n")


if __name__ == "__main__":
    main()
