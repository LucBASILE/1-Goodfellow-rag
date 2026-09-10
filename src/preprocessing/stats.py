"""Distribution statistics over a list of chunks.

Used to decide the embedding strategy: target chunk size, choice of a model
based on its context window, detection of chunks too large (truncated)
or too small (poor embedding).

Includes a body / front-back matter SEGMENTATION: in a book, the index, the
bibliography, the notation... generate a mass of tiny chunks (noise)
that skew the global distribution. The segmented view reveals this bias.

Usage:
    from preprocessing.stats import print_chunk_stats, print_chunk_stats_by_segment

    chunks = chunk_document("extracted/...jsonl")   # list of Chunk (or of str)
    print_chunk_stats(chunks)                        # global distribution
    print_chunk_stats_by_segment(chunks)             # body vs front/back matter

    # EXACT token count with the embedding model's tokenizer:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    print_chunk_stats(chunks, count_tokens=lambda t: len(tok.encode(t)))
"""

from statistics import mean, median
from typing import Callable, Sequence

# Chapters considered as body text (the rest = front/back matter).
CONTENT_PREFIXES = ("Chapter", "Appendix")


def _estimate_tokens(text: str) -> int:
    """Quick estimate without a model: ~1.33 token per word (English).
    For an exact count, pass the model's tokenizer via `count_tokens`."""
    return round(len(text.split()) / 0.75)


def _percentile(sorted_values: list[int], q: float) -> int:
    """q-th percentile (q between 0 and 1) on an already-sorted list."""
    idx = min(len(sorted_values) - 1, int(len(sorted_values) * q))
    return sorted_values[idx]


def _distribution(values: Sequence[int]) -> dict:
    s = sorted(values)
    return {
        "min": s[0],
        "max": s[-1],
        "mean": round(mean(s), 1),
        "median": median(s),
        "p90": _percentile(s, 0.90),
        "p95": _percentile(s, 0.95),
    }


def content_segment(chunk) -> str:
    """Classify a chunk: 'body' (chapters/appendices) or 'front/back matter'
    (index, bibliography, notation, table of contents...). Serves as the default key
    for the segmented view. Requires a chunk with a `.chapter` attribute."""
    chapter = getattr(chunk, "chapter", "") or ""
    return "body" if chapter.startswith(CONTENT_PREFIXES) else "front/back matter"


def chunk_stats(
    chunks: Sequence,
    count_tokens: Callable[[str], int] = _estimate_tokens,
    thresholds: Sequence[int] = (256, 512),
    small_token_threshold: int = 20,
) -> dict:
    """Compute the distribution of chunk sizes (words, characters, tokens).

    Args:
        chunks: list of chunks. Each element can be a str, or an object
            with a `.text` attribute (e.g. the Chunk dataclass).
        count_tokens: function text -> number of tokens. By default a
            per-word estimate; pass the model's tokenizer for an exact count.
        thresholds: high token thresholds to watch (model windows).
        small_token_threshold: low threshold; fraction of "tiny" chunks below it
            (reveals the noise: index entries, fragments...).

    Returns:
        dict with the number of chunks, the distribution for words/characters/tokens,
        the fraction of chunks over each high threshold, and the fraction of tiny ones.
    """
    texts = [c if isinstance(c, str) else c.text for c in chunks]
    if not texts:
        return {"count": 0}

    words = [len(t.split()) for t in texts]
    chars = [len(t) for t in texts]
    tokens = [count_tokens(t) for t in texts]
    n = len(texts)
    n_small = sum(tok < small_token_threshold for tok in tokens)

    return {
        "count": n,
        "words": _distribution(words),
        "chars": _distribution(chars),
        "tokens": _distribution(tokens),
        "over_threshold": {
            t: {
                "count": sum(tok > t for tok in tokens),
                "pct": round(100 * sum(tok > t for tok in tokens) / n, 1),
            }
            for t in thresholds
        },
        "small": {
            "under": small_token_threshold,
            "count": n_small,
            "pct": round(100 * n_small / n, 1),
        },
    }


def chunk_stats_by_segment(
    chunks: Sequence, key: Callable[[object], str] = content_segment, **kwargs
) -> dict:
    """Like chunk_stats but broken down by segment (default: body vs front/back
    matter). Returns {'ALL': ..., '<segment>': ...} for each group."""
    groups: dict[str, list] = {}
    for chunk in chunks:
        groups.setdefault(key(chunk), []).append(chunk)

    result = {"ALL": chunk_stats(chunks, **kwargs)}
    for label in sorted(groups):
        result[label] = chunk_stats(groups[label], **kwargs)
    return result


def format_stats(stats: dict) -> str:
    """Render the stats as a text table (handy for a README/CLI)."""
    if stats.get("count", 0) == 0:
        return "No chunk."

    lines = [f"{stats['count']} chunks", ""]
    lines.append(f"{'':8} {'min':>6} {'median':>7} {'mean':>7} {'p90':>6} {'p95':>6} {'max':>7}")
    for key in ("words", "chars", "tokens"):
        d = stats[key]
        lines.append(
            f"{key:8} {d['min']:>6} {d['median']:>7} {d['mean']:>7} "
            f"{d['p90']:>6} {d['p95']:>6} {d['max']:>7}"
        )
    lines.append("")
    small = stats["small"]
    lines.append(f"< {small['under']} tokens (tiny) : {small['count']} ({small['pct']}%)")
    for threshold, info in stats["over_threshold"].items():
        lines.append(f"> {threshold} tokens : {info['count']} ({info['pct']}%)")
    return "\n".join(lines)


def print_chunk_stats(chunks: Sequence, **kwargs) -> dict:
    """Compute and print the global stats. Also returns the dict."""
    stats = chunk_stats(chunks, **kwargs)
    print(format_stats(stats))
    return stats


def print_chunk_stats_by_segment(chunks: Sequence, **kwargs) -> dict:
    """Compute and print the stats broken down by segment (reveals the
    front/back matter bias). Also returns the dict {segment: stats}."""
    by_segment = chunk_stats_by_segment(chunks, **kwargs)
    for label, stats in by_segment.items():
        print(f"\n===== {label} =====")
        print(format_stats(stats))
    return by_segment
