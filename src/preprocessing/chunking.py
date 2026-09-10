"""Chunking: Nougat JSONL -> paragraph-chunks attached to chapter/section,
then packing (merging the tiny ones). Two close steps, kept together.
"""

import json
import re
from dataclasses import dataclass, replace


@dataclass
class Chunk:
    id: int
    text: str
    source: str
    chapter: str
    section: str
    page: int
    embedding: list[float] | None = None  # filled in later by the embedding step


HEADING_RE = re.compile(r"^(#{2,4})\s+(.+)$")

# A chapter is "body" if it starts with one of these prefixes. The rest
# (Index, Bibliography, References, Notation, table of contents, front matter...) is
# noise for a RAG: a mass of tiny chunks that pollute retrieval.
CONTENT_PREFIXES = ("Chapter", "Appendix")


def is_content_chapter(chapter: str) -> bool:
    return chapter.startswith(CONTENT_PREFIXES)


CHAPTER_RE = re.compile(r"^Chapter\s+(\d+)", re.IGNORECASE)


def _normalize(title: str) -> str:
    """lowercase + collapsed spaces -> compare titles up to case/whitespace."""
    return re.sub(r"\s+", " ", title).strip().lower()


def _chapter_number(title: str) -> str | None:
    """Number of a 'Chapter N ...' title (else None)."""
    m = CHAPTER_RE.match(title)
    return m.group(1) if m else None


def load_blocks(path: str) -> list[tuple[str, int, str]]:
    """Load the JSONL and return the blocks IN ORDER, keeping their page.
    -> we do NOT concatenate everything: each block keeps its page and source."""
    blocks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            page, source = rec["page"], rec["source"]
            for raw in re.split(r"\n{2,}", rec.get("text", "")):
                block = raw.strip()
                if block:
                    blocks.append((block, page, source))
    return blocks


def is_equation(block: str) -> bool:
    """Nougat equation block: starts with \\[ (display) or $$."""
    b = block.lstrip()
    return b.startswith(r"\[") or b.startswith("$$")


def is_annotation(block: str) -> bool:
    """Small derivation note, e.g. '(by equation 2.49)'. We also glue it
    backwards so as not to break the cascade of derivations."""
    b = block.strip()
    return len(b) < 200 and b.startswith("(") and b.endswith(")")


def chunk_document(path: str, content_only: bool = True) -> list[Chunk]:
    """Split the document into paragraph-chunks attached to their chapter/section.

    Args:
        content_only: if True (default), ignore the front/back matter (index,
            bibliography, notation...). Set False to keep everything (e.g. to
            compare distributions in the stats).
    """
    chunks: list[Chunk] = []
    chapter = "(front matter)"
    chapter_num = None
    section = "(introduction)"
    next_id = 0

    for block, page, source in load_blocks(path):
        m = HEADING_RE.match(block)

        if m:
            level, title = m.group(1), m.group(2).strip()

            # Fix 1: repeated header (same title up to case/whitespace) -> ignore.
            # Catches e.g. "Numerical Computation" vs "Numerical computation".
            if _normalize(title) in (_normalize(chapter), _normalize(section)):
                continue

            if level == "##":                 # chapter / part / appendix
                num = _chapter_number(title)
                # Fix 2: same chapter number already open -> it's a repeated
                # header whose text Nougat glitched (e.g. "Deep Feedforward" vs
                # "Deep Feedback Network Networks"). Keep the 1st title (the good one).
                # NB: assumes chapters numbered "Chapter N" (documented).
                if num is not None and num == chapter_num:
                    continue
                chapter, chapter_num, section = title, num, "(introduction)"
            else:                             # ### / #### : (sub-)section
                section = title
            continue                          # a title is not content

        # --- content ---
        # filter: skip the front/back matter (first of all, so that even the
        # equations/annotations of an index/biblio don't glue onto the body)
        if content_only and not is_content_chapter(chapter):
            continue

        # equation or annotation -> glued to the previous block (cascade)
        if (is_equation(block) or is_annotation(block)) and chunks:
            chunks[-1].text += "\n\n" + block
            continue

        chunks.append(Chunk(next_id, block, source, chapter, section, page))
        next_id += 1

    return chunks


# --- Packing: merging the tiny chunks ------------------------------------------


def _words(text: str) -> int:
    return len(text.split())


def pack_chunks(
    chunks: list[Chunk], min_words: int = 30, max_words: int = 250
) -> list[Chunk]:
    """Merge a chunk with the previous one if:
      - same (chapter, section)   ← never crosses a title
      - one of the two < min_words ← we don't touch already-healthy chunks
      - total <= max_words        ← upper bound (embedding model window)

    max_words=250 ~= 330 tokens: OK for a 512-token model (bge/gte),
    too much for MiniLM (256) -> drop to ~180 in that case.
    """
    packed: list[Chunk] = []
    for chunk in chunks:
        if packed:
            prev = packed[-1]
            same_section = (chunk.chapter, chunk.section) == (prev.chapter, prev.section)
            one_is_small = _words(prev.text) < min_words or _words(chunk.text) < min_words
            fits = _words(prev.text) + _words(chunk.text) <= max_words
            if same_section and one_is_small and fits:
                prev.text += "\n\n" + chunk.text  # page/id = those of the 1st (order)
                continue
        packed.append(replace(chunk))  # copy: we don't mutate the input

    for new_id, chunk in enumerate(packed):  # contiguous ids after merging
        chunk.id = new_id
    return packed


if __name__ == "__main__":
    path = "extracted/Deeplearning - Ian Goodfellow.jsonl"

    all_chunks = chunk_document(path, content_only=False)
    kept = chunk_document(path, content_only=True)
    packed = pack_chunks(kept)
    print(f"raw {len(all_chunks)} -> filtered {len(kept)} -> packed {len(packed)} chunks\n")

    for c in packed[:20]:
        preview = c.text[:90].replace("\n", " ")
        print(f"[{c.id:4}] p{c.page:3} | {c.chapter[:20]:20} > {c.section[:22]:22} | {preview}")
