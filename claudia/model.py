"""
Claudia: A small decoder-only transformer language model.

Architecture: LLaMA-style with RMSNorm, RoPE, SwiGLU, pre-norm.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from claudia.config import ClaudiaConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def precompute_rope_frequencies(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute the complex exponentials for rotary position embeddings."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to input tensor."""
    # x: (batch, seq_len, num_heads, head_dim)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs[:x.shape[1]].unsqueeze(0).unsqueeze(2)  # (1, seq, 1, head_dim//2)
    x_rotated = x_complex * freqs
    return torch.view_as_real(x_rotated).reshape(*x.shape).type_as(x)


class Attention(nn.Module):
    """Multi-head self-attention with rotary position embeddings."""

    def __init__(self, config: ClaudiaConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim)

        # Apply RoPE to Q and K
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        # Transpose for attention: (B, num_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # (B, num_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    """SwiGLU feedforward network (Shazeer, 2020)."""

    def __init__(self, config: ClaudiaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with attention + SwiGLU FFN."""

    def __init__(self, config: ClaudiaConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size)
        self.attn = Attention(config)
        self.ffn_norm = RMSNorm(config.hidden_size)
        self.ffn = SwiGLU(config)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), freqs, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Claudia(nn.Module):
    """
    Claudia language model.

    A decoder-only transformer for autoregressive language modeling.
    """

    def __init__(self, config: ClaudiaConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight tying: share embedding and output weights
        self.lm_head.weight = self.token_embedding.weight

        # Precompute RoPE frequencies (not a parameter, just a buffer)
        freqs = precompute_rope_frequencies(config.head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("freqs", freqs, persistent=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, f"Sequence length {T} exceeds max {self.config.max_seq_len}"

        # Causal mask: upper triangular = 0 (masked), lower triangular + diagonal = 1
        mask = torch.tril(torch.ones(T, T, device=input_ids.device)).unsqueeze(0).unsqueeze(0)

        x = self.token_embedding(input_ids)

        for layer in self.layers:
            x = layer(x, self.freqs, mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Autoregressive generation with top-k and nucleus (top-p) sampling."""
        for _ in range(max_new_tokens):
            # Crop to max sequence length
            idx_cond = input_ids[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                for b in range(logits.size(0)):
                    logits[b, sorted_indices[b][sorted_indices_to_remove[b]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

        return input_ids
