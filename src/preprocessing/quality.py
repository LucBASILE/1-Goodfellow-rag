"""Chunk quality: detection (hallucinations, malformations, duplicates) and
remediation (pymupdf fallback on the pages Nougat hallucinated).

Detection inspired by the Gopher / RefinedWeb filters. Remediation = mini-router:
Nougat strategy by default, pymupdf fallback on failure.
"""

import hashlib
import re
import zlib
from collections import Counter
from dataclasses import dataclass, field, replace

import pymupdf

from preprocessing.chunking import Chunk

# Thresholds calibrated on Goodfellow. The PRIMARY signal is the repeated n-gram:
# robust. Compression alone wrongly flags mathematical derivations
# (repeated LaTeX notation that compresses well) -> very low threshold, backup only.
MAX_REPEATED_NGRAM = 0.40    # PRIMARY: above = hallucination (loop)
MIN_COMPRESSION = 0.12       # backup: below = extreme degeneration
MAX_SYMBOL_RATIO = 0.60      # indicative: above = little text (noisy on LaTeX)
NGRAM_N = 3


# --- Detection -----------------------------------------------------------------


def compression_ratio(text: str) -> float:
    """compressed size / raw size. Low = very repetitive (a sentence in a
    loop compresses to ~2%). The best signal for hallucinations."""
    data = text.encode("utf-8")
    if not data:
        return 1.0
    return len(zlib.compress(data, 9)) / len(data)


def repeated_ngram_fraction(text: str, n: int = NGRAM_N) -> float:
    """fraction of n-grams that are repetitions. High = redundant text.
    More robust than the simple unique-word ratio."""
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return (len(ngrams) - len(set(ngrams))) / len(ngrams)


def symbol_ratio(text: str) -> float:
    """fraction of non-alphanumeric characters (excluding spaces). High = table,
    bare equations, gibberish. NB: noisy on LaTeX -> indicative, not a hard cut."""
    stripped = [c for c in text if not c.isspace()]
    if not stripped:
        return 0.0
    non_alnum = sum(not c.isalnum() for c in stripped)
    return non_alnum / len(stripped)


def normalized_hash(text: str) -> str:
    """hash of a normalized version (lowercase, collapsed spaces) -> exact duplicates."""
    norm = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


@dataclass
class QualityReport:
    compression: float
    repeated_ngram: float
    symbol: float
    reasons: list[str] = field(default_factory=list)

    @property
    def is_suspect(self) -> bool:
        return bool(self.reasons)


def assess(
    chunk,
    min_compression: float = MIN_COMPRESSION,
    max_repeated_ngram: float = MAX_REPEATED_NGRAM,
    max_symbol_ratio: float = MAX_SYMBOL_RATIO,
) -> QualityReport:
    """Assess a chunk (str or object with .text) and list the reasons for suspicion."""
    text = chunk if isinstance(chunk, str) else chunk.text
    comp = compression_ratio(text)
    rep = repeated_ngram_fraction(text)
    sym = symbol_ratio(text)

    reasons = []
    if comp < min_compression:
        reasons.append(f"repetitive (compression {comp:.2f})")
    if rep > max_repeated_ngram:
        reasons.append(f"repeated n-grams ({rep:.2f})")
    if sym > max_symbol_ratio:
        reasons.append(f"too many symbols ({sym:.2f})")

    return QualityReport(comp, rep, sym, reasons)


def find_duplicates(chunks) -> list[list]:
    """Return the groups of chunks with identical text (after normalization)."""
    groups: dict[str, list] = {}
    for chunk in chunks:
        text = chunk if isinstance(chunk, str) else chunk.text
        groups.setdefault(normalized_hash(text), []).append(chunk)
    return [g for g in groups.values() if len(g) > 1]


def filter_chunks(chunks, **thresholds) -> tuple[list, list]:
    """Split the chunks into (clean, suspect) according to the quality thresholds."""
    clean, suspect = [], []
    for chunk in chunks:
        (suspect if assess(chunk, **thresholds).is_suspect else clean).append(chunk)
    return clean, suspect


# --- Remediation: pymupdf fallback ---------------------------------------------

# Capture from "Figure X.Y:" to the end (the caption + the text that follows).
FIGURE_RE = re.compile(r"Figure\s+\d+\.\d+:.*", re.S)


def _recover_page(doc: pymupdf.Document, page: int) -> str:
    """pymupdf text of a page: the caption if found, else all the text."""
    text = doc.load_page(page).get_text().strip()
    match = FIGURE_RE.search(text)
    if match:
        text = match.group(0)
    return re.sub(r"\s+", " ", text).strip()


def clean_with_fallback(
    chunks: list[Chunk], pdf_path: str, min_words: int = 5, **thresholds
) -> list[Chunk]:
    """Replace the hallucinated chunks with the recovered pymupdf content.

    A hallucinated page = as soon as ONE of its chunks is suspect (the hallucination
    is page-level: Nougat botches the rendering of a whole page) -> we remove ALL
    the chunks of the page and keep ONE pymupdf chunk (the Figure X.Y: caption).
    A page from which pymupdf extracts almost nothing (< min_words) is dropped.
    """
    doc = pymupdf.open(pdf_path)

    suspect_pages: dict[int, Chunk] = {}
    for chunk in chunks:
        if assess(chunk, **thresholds).is_suspect:
            suspect_pages.setdefault(chunk.page, chunk)  # 1 representative/page

    kept = [c for c in chunks if c.page not in suspect_pages]

    recovered = 0
    for page, representative in suspect_pages.items():
        text = _recover_page(doc, page)
        if len(text.split()) >= min_words:
            kept.append(replace(representative, text=text))
            recovered += 1

    kept.sort(key=lambda c: c.id)  # keep the reading order
    for new_id, chunk in enumerate(kept):
        chunk.id = new_id

    print(f"fallback: {len(suspect_pages)} suspect pages -> "
          f"{recovered} recovered, {len(suspect_pages) - recovered} dropped")
    return kept


if __name__ == "__main__":
    from preprocessing.chunking import chunk_document, pack_chunks

    jsonl = "extracted/Deeplearning - Ian Goodfellow.jsonl"
    pdf = "data/Deeplearning - Ian Goodfellow.pdf"
    chunks = pack_chunks(chunk_document(jsonl))

    clean, suspect = filter_chunks(chunks)
    print(f"{len(chunks)} chunks -> {len(clean)} clean | {len(suspect)} suspect")
    for c in suspect[:20]:
        print(f"  p{c.page:3} | {', '.join(assess(c).reasons):40} | {c.text[:50]}")
    print(f"\n{len(find_duplicates(chunks))} duplicate groups\n")

    cleaned = clean_with_fallback(chunks, pdf)
    print(f"{len(chunks)} -> {len(cleaned)} chunks after fallback")
