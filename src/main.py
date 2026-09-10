import torch

from sentence_transformers import SentenceTransformer
from pathlib import Path
from preprocessing.chunking import chunk_document, pack_chunks
from preprocessing.quality import clean_with_fallback
from vector_store import get_collection, index_chunks

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DATA_DIR = Path("data")
EXTRACTED_DIR = Path("extracted")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def _build_source_chunks(pdf_path: Path, jsonl_path: Path) -> list:
    """One source's JSONL -> clean chunks (chunk + pack + quality/fallback)."""
    chunks = chunk_document(str(jsonl_path))              # paragraphs + chap/section + filter
    chunks = pack_chunks(chunks)                           # merge the tiny ones
    chunks = clean_with_fallback(chunks, str(pdf_path))    # anti-hallucination + fallback
    return chunks


def build_all_chunks(show_stats: bool = False) -> list:
    """Build and EMBED all the clean chunks. Returns the embedded chunks."""
    # extract_all()   # <- uncomment to re-extract with Nougat (slow)

    all_chunks = []
    for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
        jsonl_path = EXTRACTED_DIR / f"{pdf_path.stem}.jsonl"
        if not jsonl_path.exists():
            print(f"!! {jsonl_path} missing — run the extraction first")
            continue
        print(f"\n=== {pdf_path.name} ===")
        all_chunks.extend(_build_source_chunks(pdf_path, jsonl_path))
    print(f"\nTOTAL: {len(all_chunks)} clean chunks")

    if show_stats:
        from preprocessing.quality import assess, find_duplicates
        from preprocessing.stats import print_chunk_stats

        print("\n--- distribution ---")
        print_chunk_stats(all_chunks)
        print(f"remaining suspects: {sum(assess(c).is_suspect for c in all_chunks)} | "
              f"duplicates: {len(find_duplicates(all_chunks))}")

    # embedding (question <-> chunks in the same space, normalized for cosine)
    model = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    vectors = model.encode(
        [c.text for c in all_chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    for chunk, vec in zip(all_chunks, vectors):
        chunk.embedding = vec.tolist()
    return all_chunks


def main(show_stats: bool = False):
    """Build + embed + index into Chroma. Run ONCE (or when the data changes).
    To ASK questions afterwards: uv run python src/generation.py"""
    chunks = build_all_chunks(show_stats=show_stats)
    collection = get_collection()
    index_chunks(chunks, collection)
    print(f"\nindexed: {collection.count()} chunks in '{collection.name}'")
    return collection


if __name__ == "__main__":
    main()
