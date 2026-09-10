import sys
import ollama

from retrieval import retrieve
from tqdm import tqdm

MODEL = "qwen2.5:7b"
NUM_CTX = 8192

SYSTEM_PROMPT = (
    'You are an assistant answering questions about the book "Deep Learning" '
    "(Goodfellow, Bengio, Courville). Answer ONLY from the provided context. "
    "Cite your sources as (chapter, section, page). If the answer is not in the "
    'context, reply exactly: "I can\'t find this in the text." '
    "Never use outside knowledge."
)


def _format_context(results) -> str:
    """Concatenate the retrieved chunks + their source into a context block."""
    blocks = []
    for doc, meta, _dist, _score in results:
        source = f"[{meta['chapter']} > {meta['section']} (p{meta['page']})]"
        blocks.append(f"{source}\n{doc}")
    return "\n\n---\n\n".join(blocks)



def _check_and_pull_models(model):
    try:
        ollama.show(model)
        print(f"✅ Model '{model}' available.")
        return
    except ollama.ResponseError as e:
        if e.status_code != 404:
            raise
    except ConnectionError:
        print(f"❌ Unable to contact Ollama. Is the application launched?")
        return

    print(f"Downloading '{model}'...")
    bars = {} 
    for chunk in ollama.pull(model, stream=True):
        digest = chunk.get("digest")
        total = chunk.get("total") or 0
        completed = chunk.get("completed") or 0
        status = chunk.get("status", "")

        if not digest or not total:
            print(status)
            continue

        if digest not in bars:
            bars[digest] = tqdm(
                total=total, unit="B", unit_scale=True, desc=digest[7:19]
            )
        bars[digest].update(completed - bars[digest].n)

    for bar in bars.values():
        bar.close()
    print(f"✅ '{model}' downloaded.")


def answer(question, k=5, n_candidates=30, chapter=None, rerank=True, model=MODEL):
    """Question -> answer grounded in the book. Returns (answer, sources)."""

    _check_and_pull_models(model)

    results = retrieve(
        question, k=k, n_candidates=n_candidates, chapter=chapter, rerank=rerank
    )
    if not results:
        return "I can't find this in the text.", []

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{_format_context(results)}\n\nQuestion: {question}",
            },
        ],
        options={"num_ctx": NUM_CTX},
    )
    return response.message.content, results


def _print_answer(question):
    reply, sources = answer(question)
    print("\n" + reply + "\n")
    if sources:
        print("Sources:")
        for _doc, meta, _dist, _score in sources:
            print(f"  - {meta['chapter']} > {meta['section']} (p{meta['page']})")
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:  # one-shot : python generation.py "question"
        _print_answer(" ".join(sys.argv[1:]))
    else:  # interactive REPL
        print(f"RAG Goodfellow ({MODEL}). Ctrl+C to quit.\n")
        while True:
            try:
                q = input("Question > ").strip()
                if q:
                    _print_answer(q)
            except (KeyboardInterrupt, EOFError):
                print("\nBye.")
                break
