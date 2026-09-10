"""Local Chroma inspector / tester (version-proof: same client that wrote the data).

Reads the chroma/ folder directly (PersistentClient) -> no server needed.
/!\\ Do NOT run while a `chroma run` server holds the folder (SQLite lock).

`search` delegates the retrieval to retrieval.retrieve() and just prints it:
  1. bi-encoder (bge-small) -> cosine -> n_candidates
  2. if rerank=True: cross-encoder (bge-reranker) re-orders -> keep top k

`search` is positional; CLI args arrive as strings (converted inside retrieve):
    search(query, k=5, n_candidates=30, chapter=None, rerank=False)

Usage:
    uv run python src/inspect_db.py count
    uv run python src/inspect_db.py peek 10
    uv run python src/inspect_db.py chapters
    uv run python src/inspect_db.py search "what is backpropagation"
    # order: query  k  n_candidates  chapter  rerank
    uv run python src/inspect_db.py search "activation function" 5 30 "Chapter 6 Deep Feedforward Networks" True
"""

import sys
import textwrap
from collections import Counter

import chromadb

from retrieval import retrieve

CHROMA_DIR = "chroma"
COLLECTION = "goodfellow"


def _collection():
    return chromadb.PersistentClient(path=CHROMA_DIR).get_collection(COLLECTION)


def _show(doc, meta, dist=None, score=None):
    head = f"{meta.get('chapter', '?')[:26]} > {meta.get('section', '?')[:24]} (p{meta.get('page')})"
    if dist is not None:
        head = f"[dist {dist:.3f}] " + head
    if score is not None:
        head = f"[score {score:.3f}] " + head
    print(head)
    print(textwrap.indent(textwrap.fill(doc[:200], 90), "    "), "\n")


def count():
    print(_collection().count(), "chunks")


def peek(n=10):
    col = _collection()
    print(f"{col.count()} chunks total. Preview of {n}:\n")
    res = col.get(limit=int(n), include=["documents", "metadatas"])
    for doc, meta in zip(res["documents"], res["metadatas"]):
        _show(doc, meta)


def chapters():
    """List the distinct chapters (to get the exact string to filter on)."""
    metas = _collection().get(include=["metadatas"])["metadatas"]
    for chap, n in sorted(Counter(m["chapter"] for m in metas).items()):
        print(f"{n:5}  {chap}")


def search(query, k=5, n_candidates=30, chapter=None, rerank=False):
    """Semantic search via retrieval.retrieve() + display.
    - chapter: optional filter on a chapter.
    - rerank : re-ranks the n_candidates via the cross-encoder, keeps top k."""
    filt = f"  (chapter filter = {chapter})" if chapter else ""
    print(f"Q: {query}{filt}\n")
    for doc, meta, dist, score in retrieve(query, k, n_candidates, chapter, rerank):
        _show(doc, meta, dist, score)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    cmd, args = sys.argv[1], sys.argv[2:]
    {"count": count, "peek": peek, "chapters": chapters, "search": search}[cmd](*args)
