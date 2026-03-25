from dataclasses import dataclass


@dataclass
class ClaudiaConfig:
    vocab_size: int = 4096       # Custom BPE — tiny vocab, huge memory savings
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    intermediate_size: int = 1376
    max_seq_len: int = 512
    dropout: float = 0.0
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    def param_count_estimate(self) -> int:
        # Embedding (tied with lm_head, counted once)
        embed = self.vocab_size * self.hidden_size
        # Per transformer layer
        per_layer = (
            3 * self.hidden_size * self.hidden_size    # QKV projections
            + self.hidden_size * self.hidden_size       # output projection
            + 3 * self.hidden_size * self.intermediate_size  # SwiGLU (gate, up, down)
        )
        # Final norm
        norm = self.hidden_size
        return embed + self.num_layers * per_layer + norm


# Configs tuned for 10GB VRAM / 30min training on Apple Silicon
# With 4K vocab, almost all params go to transformer layers (good!)
CONFIGS = {
    # ~11M params — fast to train, proves the pipeline
    "tiny": ClaudiaConfig(
        hidden_size=384,
        num_layers=6,
        num_heads=6,
        intermediate_size=1024,
        max_seq_len=256,
    ),
    # ~22M params — the sweet spot for our constraints
    "small": ClaudiaConfig(
        hidden_size=512,
        num_layers=8,
        num_heads=8,
        intermediate_size=1376,
        max_seq_len=512,
    ),
    # ~48M params — stretch goal, might be tight on memory
    "medium": ClaudiaConfig(
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        intermediate_size=2048,
        max_seq_len=512,
    ),
}
