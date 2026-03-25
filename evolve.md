# Claudia Evolution Plan

This document is my autonomous roadmap for making Claudia the best small language model she can be within our constraints (10GB VRAM, 30min training cycles, Apple M2 Pro).

## Current State (v0.1)
- 12.2M params, 4K vocab, trained on 8% of TinyStories
- PPL 8.30, Distinct-2 0.514, near-zero repetition
- Generates coherent children's stories but limited domain

## Evolution Priorities

### Evolution 1: Grammar Heuristics Engine
**Why**: Even at PPL 8.3, the model occasionally produces grammatical errors — unbalanced quotes, broken sentences at the generation boundary, repeated phrases, and capitalization issues. A post-generation heuristics engine can catch and repair these without retraining.

**What to build**:
- Sentence boundary repair (incomplete final sentence removal)
- Quote balancing (open/close matching)
- Capitalization normalization (sentence starts, proper nouns)
- Repetition detection and pruning (n-gram and sentence-level)
- Punctuation repair (double periods, missing spaces after punctuation)
- EOS handling (clean cutoff at natural boundaries)

### Evolution 2: Extended Training
**Why**: We only trained on 8% of TinyStories in 30 minutes. The loss was still dropping. Training on more data will directly improve perplexity and generation quality.

**What to do**:
- Resume training from the best checkpoint for another 30min cycle
- Use the cached tokenized data (no re-tokenization overhead)
- Adjust LR schedule for continued training (lower starting LR, warm restart)

### Evolution 3: Architecture Tuning
**Why**: Small models benefit disproportionately from architecture tweaks.

**What to explore**:
- Increase context window (256 → 512) if memory allows
- Try the "small" config (27M params) with reduced batch size
- Experiment with different LR schedules and warmup strategies

### Evolution 4: Data Diversity
**Why**: TinyStories produces coherent but narrow output. Mixing in diverse open data expands Claudia's knowledge.

**What to do**:
- Add a subset of FineWeb-Edu (educational web text)
- Train a new tokenizer on the mixed corpus
- Evaluate knowledge breadth beyond children's stories

### Evolution 5: Instruction Tuning (Stretch)
**Why**: A model that can follow instructions is genuinely useful.

**What to do**:
- Curate a small instruction dataset (Alpaca format)
- Fine-tune with a chat template
- Evaluate on simple instruction-following tasks

## Execution Order
1. Grammar heuristics engine (immediate — no training needed)
2. Extended training cycle (next 30min run)
3. Evaluate and log results
4. Decide between Architecture Tuning vs Data Diversity based on results
5. Iterate

## Success Metrics
- PPL < 6 on TinyStories validation
- Clean, grammatically correct output in >95% of generations
- Coherent multi-paragraph generation
- Some ability to handle prompts beyond children's stories
