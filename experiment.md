# Claudia: Experiment Log

## Vision

Claudia is a small, open-weight language model designed to be trained on consumer hardware under tight constraints. The goal: a functioning LLM trained from scratch in 30 minutes using 10GB of VRAM on an Apple Silicon MacBook Pro.

Fully open: open data, open weights, open code.

## Constraints

- **VRAM**: 10GB hard cap (PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.85)
- **Training time**: 30 minutes max
- **Hardware**: Apple M2 Pro, 16GB unified memory, MPS backend
- **Data**: Must be open/public datasets only

## Architecture

**Model family**: Decoder-only transformer (autoregressive LM)

**Design choices** (LLaMA-inspired, not GPT-2):
- RMSNorm instead of LayerNorm — more stable, fewer parameters
- Rotary Position Embeddings (RoPE) — better length generalization
- SwiGLU activation in FFN — empirically superior to GELU/ReLU
- No bias terms — fewer params, LLaMA convention
- Pre-norm architecture — more stable training
- Weight tying — embedding and LM head share weights

**Custom BPE Tokenizer** (4096 vocab):
- Trained from scratch on TinyStories using HuggingFace `tokenizers` (Rust)
- 4K vocab vs GPT-2's 50K = **92% reduction in embedding parameters**
- This was the single most impactful optimization for memory

## Key Optimization: Vocabulary Size

The biggest win came from replacing GPT-2's 50257-token vocabulary with a custom 4096-token BPE:

| Metric | GPT-2 (50K) | Custom (4K) | Savings |
|--------|-------------|-------------|---------|
| Embedding params | 19.3M | 1.6M | **92%** |
| Tiny model total | 30M | 12.2M | 59% |
| Max batch size (10GB) | 8 | 96 | 12x |
| Throughput | ~9K tok/s | ~23K tok/s | 2.6x |
| Logits memory per batch | huge | tiny | massive |

With the 50K vocab, the embedding table alone was larger than the entire transformer.
With 4K vocab, nearly all parameters go to the transformer layers where they matter.

## Model Configurations

| Config | Params | Layers | Hidden | Heads | FFN | Seq Len | Vocab |
|--------|--------|--------|--------|-------|-----|---------|-------|
| Tiny   | 12.2M  | 6      | 384    | 6     | 1024| 256     | 4096  |
| Small  | 27.4M  | 8      | 512    | 8     | 1376| 512     | 4096  |
| Medium | 88.1M  | 12     | 768    | 12    | 2048| 512     | 4096  |

## Training Details

### Throughput Benchmarks (10GB cap)

| Config | Max BS | Steps/s | Tok/s | Tokens/30min |
|--------|--------|---------|-------|-------------|
| Tiny   | 96     | 2.8 (bs32) | 23K | ~41M |
| Small  | 32     | 0.6     | 10K   | ~18M        |

Observation: Tokens/sec is constant (~23K) regardless of batch size for tiny.
Smaller batches = more optimizer steps = better learning per token.
Chose bs=32 with no gradient accumulation for maximum update frequency.

### Training Hyperparameters (Phase 1)
- **Model**: Tiny (12.2M params)
- **Data**: TinyStories (roneneldan/TinyStories)
- **Batch size**: 32, no gradient accumulation
- **Learning rate**: 6e-4, cosine decay with warmup
- **Weight decay**: 0.1
- **Optimizer**: AdamW (β1=0.9, β2=0.95)
- **Precision**: float16 mixed precision (torch.autocast)
- **Gradient clipping**: max norm 1.0
- **Context length**: 256 tokens

## Evaluation Framework

1. **Perplexity** — on held-out TinyStories validation split
2. **Generation quality** — manual inspection of story completions
3. **Distinct-n** — vocabulary diversity (distinct unigrams, bigrams, trigrams)
4. **Repetition rate** — fraction of repeated 4-grams (lower = better)

## Experiment Log

### Experiment 001: Tiny Claudia on TinyStories
- **Date**: 2026-03-24
- **Status**: COMPLETE
- **Config**: Tiny (12.2M params), 4K vocab, 256 context, bs=32
- **Budget**: 30 min, 10GB VRAM

**Results**:
| Metric | Value |
|--------|-------|
| Steps | 4,695 |
| Final val loss | 2.14 |
| Final val PPL | 8.50 |
| Best val loss | 2.17 (step 4500) |
| Throughput | 21,042 tok/s |
| Total tokens | 38.5M |
| Training time | 30.5 min |
| Data coverage | ~8% of 1 epoch |

**Loss trajectory**:
- Step 0: 8.36 (random)
- Step 500: 3.22 (PPL 25.0)
- Step 1000: 2.73 (PPL 15.3)
- Step 2000: 2.39 (PPL 10.9)
- Step 3000: 2.24 (PPL 9.4)
- Step 4500: 2.17 (PPL 8.7)
- Final: 2.14 (PPL 8.5)

**Generation quality** (at step 4000):
> Once upon a time, there was a little girl named Lily. She loved to play outside and run around in the mud. One day, she saw a beautiful butterfly and wanted to catch it.
> But then, her friend Max came over to play. "I want to catch the butterfly, but I want to catch it," said Lily.
> Max smiled and...

**Observations**:
1. Loss was still decreasing at cutoff — model is undertrained but functional
2. Coherent multi-sentence generation with dialogue, names, and narrative structure
3. Some repetitive patterns ("Once upon a time", "little girl named Lily")
4. Only saw 8% of the training data — huge room for improvement with more time
5. Custom 4K tokenizer + fp16 made this possible within 10GB

**Evaluation results** (full validation set):
| Metric | Value | Notes |
|--------|-------|-------|
| Perplexity | 8.30 | On full validation set |
| Distinct-1 | 0.159 | Unigram diversity |
| Distinct-2 | 0.514 | Bigram diversity (decent) |
| Distinct-3 | 0.758 | Trigram diversity (good) |
| Repetition (4-gram) | 0.010 | Very low repetition |

**Key takeaway**: A 12.2M parameter model, trained in 30 minutes on 10GB VRAM, achieves PPL 8.3 on TinyStories with coherent multi-sentence generation, proper dialogue, and minimal repetition. The 4K custom tokenizer was the critical optimization that made this possible.

**Next steps**:
- Consider training longer or on more data subsets
- Try the "small" (27M) config if memory allows
- Instruction tuning as a stretch goal
