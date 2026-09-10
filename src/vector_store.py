"""Chroma layer: collection creation, indexing, search.

Persistent Chroma, cosine metric (consistent with the normalized embeddings).
get_or_create_collection is idempotent -> no "if exists" to write.

This module does ONLY the storage/search layer. The full orchestration
(build -> embed -> index) lives in main.py.
"""

import chromadb

from preprocessing.chunking import Chunk

CHROMA_DIR = "chroma"
COLLECTION = "goodfellow"
BATCH = 2000  # Chroma caps the size of an upsert -> we split it

# BGE: the instruction goes ONLY on the question, not on the passages.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def get_collection(path: str = CHROMA_DIR, name: str = COLLECTION):
    """Open (or create) the persistent collection with cosine metric."""
    client = chromadb.PersistentClient(path=path)
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(chunks: list[Chunk], collection=None):
    """Insert the (already embedded) chunks into Chroma. upsert = replayable without error."""
    if collection is None:
        collection = get_collection()

    missing = [c.id for c in chunks if c.embedding is None]
    if missing:
        raise ValueError(f"{len(missing)} chunks without embedding (e.g. id {missing[0]})")

    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        collection.upsert(
            ids=[str(c.id) for c in batch],
            embeddings=[c.embedding for c in batch],
            documents=[c.text for c in batch],
            metadatas=[
                {
                    "source": c.source,
                    "chapter": c.chapter,
                    "section": c.section,
                    "page": c.page,
                }
                for c in batch
            ],
        )
    return collection


def search(query_embedding: list[float], k: int = 5, collection=None, where: dict | None = None):
    """Retrieve the k chunks closest to a query vector.

    query_embedding: embedding of the QUESTION (BGE instruction prepended, normalized).
    where: optional filter on the metadata, e.g. {"chapter": "Chapter 6 ..."}
           or {"page": {"$gte": 188, "$lte": 240}}.
    """
    if collection is None:
        collection = get_collection()
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )