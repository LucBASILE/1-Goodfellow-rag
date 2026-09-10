from vector_store import QUERY_INSTRUCTION, search

_embedder = None
_reranker = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _embedder


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        # CPU: the reranker (xlm-roberta) quickly saturates the MPS GPU under
        # memory pressure; on CPU it is a bit slower but robust.
        _reranker = CrossEncoder("BAAI/bge-reranker-base", device="cpu")
    return _reranker


def retrieve(question, k=5, n_candidates=30, chapter=None, rerank=False):
    """Question -> top-k relevant chunks.

    Returns a list of (doc, meta, dist, score); score = None without rerank.
    Two stages: bi-encoder (cosine, brings back n_candidates) then, if rerank,
    cross-encoder (re-ranks the candidates, keeps the top k).

    Robust to CLI calls: k/n_candidates/rerank may arrive as strings.
    """
    k, n_candidates = int(k), int(n_candidates)
    rerank = str(rerank).lower() == "true"
    if rerank and n_candidates < k:
        n_candidates = k

    # --- Stage 1: bi-encoder (BGE instruction on the question) -> cosine ---
    q_emb = _get_embedder().encode(
        [QUERY_INSTRUCTION + question], normalize_embeddings=True
    )[0].tolist()
    where = {"chapter": chapter} if chapter else None
    res = search(q_emb, k=n_candidates, where=where)

    data = list(zip(res["documents"][0], res["metadatas"][0], res["distances"][0]))

    # --- Stage 2: optional rerank (cross-encoder), else top-k of the bi-encoder ---
    if rerank:
        pairs = [(question, doc) for doc, _, _ in data]  # NO instruction here
        scores = _get_reranker().predict(pairs, batch_size=16)
        ranked = sorted(zip(data, scores), key=lambda x: x[1], reverse=True)[:k]
        return [(doc, meta, dist, score) for (doc, meta, dist), score in ranked]

    return [(doc, meta, dist, None) for doc, meta, dist in data[:k]]
