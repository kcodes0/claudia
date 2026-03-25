"""
Grammar heuristics engine for Claudia.

Post-generation text cleanup that fixes common issues from autoregressive
generation without requiring model retraining. Operates as a pipeline of
small, composable repair functions.
"""

import re
from dataclasses import dataclass


@dataclass
class GrammarStats:
    """Tracks what the grammar engine fixed."""
    sentences_trimmed: int = 0
    quotes_balanced: int = 0
    caps_fixed: int = 0
    repetitions_removed: int = 0
    punctuation_fixed: int = 0
    whitespace_fixed: int = 0


def fix_whitespace(text: str) -> str:
    """Normalize whitespace: collapse runs, strip edges, fix spacing around punctuation."""
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    # Collapse multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Space before punctuation (remove)
    text = re.sub(r" ([.,!?;:])", r"\1", text)
    # Missing space after punctuation (add)
    text = re.sub(r"([.,!?;:])([A-Za-z])", r"\1 \2", text)
    # Space after opening quote
    text = re.sub(r'(" )([A-Z])', r'"\2', text)
    return text.strip()


def fix_punctuation(text: str) -> str:
    """Fix common punctuation errors."""
    # Double periods
    text = re.sub(r"\.{2,}", ".", text)
    # Multiple exclamation/question marks → single
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)
    # Period after exclamation/question
    text = re.sub(r"([!?])\.", r"\1", text)
    # Comma before period
    text = re.sub(r",\.", ".", text)
    return text


def fix_capitalization(text: str) -> str:
    """Capitalize sentence starts."""
    # After sentence-ending punctuation + space
    def cap_match(m):
        return m.group(1) + " " + m.group(2).upper()

    text = re.sub(r"([.!?])\s+([a-z])", cap_match, text)

    # Capitalize first character of text
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    # Capitalize after newline
    lines = text.split("\n")
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped and stripped[0].islower():
            indent = line[:len(line) - len(stripped)]
            lines[i] = indent + stripped[0].upper() + stripped[1:]
    text = "\n".join(lines)

    # Capitalize "i" when standalone
    text = re.sub(r"\bi\b", "I", text)
    # But fix over-capitalization in common words
    text = re.sub(r"\bIn\b", lambda m: "in" if m.start() > 0 and text[m.start()-1] not in ".!?\n" else "In", text)
    text = re.sub(r"\bIt\b", lambda m: "it" if m.start() > 0 and text[m.start()-1] not in ".!?\n " else "It", text)
    text = re.sub(r"\bIs\b", lambda m: "is" if m.start() > 0 and text[m.start()-1] not in ".!?\n " else "Is", text)
    text = re.sub(r"\bIf\b", lambda m: "if" if m.start() > 0 and text[m.start()-1] not in ".!?\n " else "If", text)

    return text


def balance_quotes(text: str) -> str:
    """Balance unmatched quotation marks."""
    # Count double quotes
    count = text.count('"')
    if count % 2 != 0:
        # Find the last quote — if it's an opener (preceded by space/newline/start), close it
        last_quote_idx = text.rfind('"')
        # Check if this is likely an unclosed quote (opener)
        if last_quote_idx > 0 and text[last_quote_idx - 1] in " \n(":
            # It's an opener without a closer — try to close at sentence end
            rest = text[last_quote_idx + 1:]
            # Find next sentence end
            end_match = re.search(r"[.!?]", rest)
            if end_match:
                insert_pos = last_quote_idx + 1 + end_match.end()
                text = text[:insert_pos] + '" ' + text[insert_pos:].lstrip()
            else:
                # Just add closing quote at end
                text = text.rstrip()
                if text[-1] in ".!?":
                    text = text[:-1] + text[-1] + '"'
                else:
                    text += '"'
        else:
            # It's a dangling closer or misplaced — remove trailing quote
            text = text[:last_quote_idx] + text[last_quote_idx + 1:]

    return text


def trim_incomplete_sentence(text: str) -> str:
    """Remove the final incomplete sentence if the text was cut off mid-sentence."""
    text = text.rstrip()
    if not text:
        return text

    # If text ends with sentence-ending punctuation or closing quote, it's complete
    if text[-1] in '.!?"':
        return text

    # Find the last sentence boundary
    # Look for the last .!? that's followed by a space (not inside abbreviation)
    last_boundary = -1
    for m in re.finditer(r'[.!?][""]?\s', text):
        last_boundary = m.end() - 1  # position of the space

    if last_boundary > len(text) * 0.3:  # Only trim if we keep at least 30% of text
        return text[:last_boundary].rstrip()

    # If no good boundary, just return as-is with ellipsis indicator removed
    return text


def remove_repetitions(text: str, window: int = 3) -> str:
    """Remove repeated sentences and phrases."""
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= 1:
        return text

    # Remove exact duplicate consecutive sentences
    deduped = [sentences[0]]
    for s in sentences[1:]:
        if s.strip().lower() != deduped[-1].strip().lower():
            deduped.append(s)

    # Remove near-duplicate sentences (same sentence appearing within window)
    seen = {}
    result = []
    for i, s in enumerate(deduped):
        key = s.strip().lower()
        if key in seen and (i - seen[key]) < window:
            continue  # Skip duplicate within window
        seen[key] = i
        result.append(s)

    # Also detect repeated phrases within sentences
    final_text = " ".join(result)

    # Remove immediately repeated phrases (3+ words)
    final_text = re.sub(r'\b(\w+(?:\s+\w+){2,})\s+\1\b', r'\1', final_text)

    return final_text


def clean_eos_artifacts(text: str, eos_token: str = "<|eos|>") -> str:
    """Remove any EOS token artifacts that leaked into text."""
    text = text.replace(eos_token, "")
    text = text.replace("<|pad|>", "")
    text = text.replace("<|unk|>", "")
    return text.strip()


def polish(text: str) -> tuple[str, GrammarStats]:
    """
    Full grammar polishing pipeline.

    Returns the cleaned text and stats about what was fixed.
    """
    stats = GrammarStats()
    original = text

    # Step 1: Clean artifacts
    text = clean_eos_artifacts(text)

    # Step 2: Fix whitespace
    cleaned = fix_whitespace(text)
    if cleaned != text:
        stats.whitespace_fixed += 1
    text = cleaned

    # Step 3: Fix punctuation
    cleaned = fix_punctuation(text)
    if cleaned != text:
        stats.punctuation_fixed += 1
    text = cleaned

    # Step 4: Remove repetitions
    cleaned = remove_repetitions(text)
    if cleaned != text:
        stats.repetitions_removed += 1
    text = cleaned

    # Step 5: Balance quotes
    cleaned = balance_quotes(text)
    if cleaned != text:
        stats.quotes_balanced += 1
    text = cleaned

    # Step 6: Fix capitalization
    cleaned = fix_capitalization(text)
    if cleaned != text:
        stats.caps_fixed += 1
    text = cleaned

    # Step 7: Trim incomplete final sentence
    cleaned = trim_incomplete_sentence(text)
    if cleaned != text:
        stats.sentences_trimmed += 1
    text = cleaned

    # Final whitespace cleanup
    text = fix_whitespace(text)

    return text, stats


def polish_text(text: str) -> str:
    """Convenience function — just return the polished text."""
    result, _ = polish(text)
    return result
