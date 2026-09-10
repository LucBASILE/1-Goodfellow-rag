"""Streamlit UI for the Goodfellow RAG — two pages:
  - Search:     ask a question, compare retrieval configs and the generated answer.
  - Evaluation: shows the PRE-COMPUTED retrieval eval (chart + table). Visitors
                see the comparison without re-running anything.

Run from the project root (streamlit adds src/ to the path by itself):
    uv run streamlit run src/app.py

Prerequisites: the Chroma index built (uv run python src/main.py) + Ollama running
(only needed for the Search page's "Generate" toggle).
"""

import json
import subprocess
import sys
from pathlib import Path

import streamlit as st

from generation import answer
from retrieval import retrieve

MODELS = ["qwen2.5:7b", "llama3.1:8b-instruct-q4_K_M"]
EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"


def _to_katex(text: str) -> str:
    """Nougat's LaTeX delimiters -> $ / $$ so Streamlit renders the equations."""
    text = text.replace("\\[", "\n$$\n").replace("\\]", "\n$$\n")
    return text.replace("\\(", "$").replace("\\)", "$")


def search_page():
    with st.sidebar:
        st.header("Settings")
        model = st.selectbox(
            "LLM model", MODELS,
            help="The LLM (via Ollama) that writes the answer from the retrieved chunks. "
            "Does NOT affect retrieval, only generation. qwen2.5:7b = stronger on "
            "math/technical content; llama3.1:8b = very good instruction-following.",
        )
        generate = st.toggle(
            "Generate an answer (Ollama)", value=True,
            help="ON: retrieval + LLM call -> a cited answer. "
            "OFF: retrieval only (fast, no LLM) -> to tune k/rerank without waiting.",
        )
        st.divider()
        rerank = st.toggle(
            "Reranking (cross-encoder)", value=True,
            help="2nd retrieval stage. ON: a cross-encoder (bge-reranker) re-reads each "
            "(question, chunk) pair and re-ranks the n_candidates -> higher PRECISION, but "
            "slower. OFF: keep the cosine (bi-encoder) order -> faster, coarser.",
        )
        k = st.slider(
            "k — chunks kept", 1, 15, 5,
            help="Number of chunks finally kept (shown + sent to the LLM). "
            "Larger = more context but more noise and more tokens.",
        )
        n_candidates = st.slider(
            "n_candidates — bi-encoder", 5, 60, 30,
            help="How many chunks the bi-encoder (cosine) brings back BEFORE reranking. "
            "Only matters when reranking is ON: larger = wider net but slower. Must be >= k.",
        )
        chapter = st.text_input(
            "Chapter filter (optional)", "",
            help="Restricts the search to ONE chapter (exact metadata filter). "
            "Empty = search the whole book. Title must be exact.",
        )

    question = st.text_input("Question (in English)", "What is dropout and why is it used?")
    if not (st.button("Search", type="primary") and question):
        return

    chapter_filter = chapter or None
    kwargs = dict(k=k, n_candidates=n_candidates, chapter=chapter_filter, rerank=rerank)

    if generate:
        with st.spinner(f"Retrieval + generation ({model})..."):
            reply, results = answer(question, model=model, **kwargs)
        st.subheader("Answer")
        st.markdown(_to_katex(reply))
        st.divider()
    else:
        with st.spinner("Retrieving..."):
            results = retrieve(question, **kwargs)

    st.subheader(f"{len(results)} chunks retrieved")
    st.caption(
        f"rerank={rerank} · k={k} · n_candidates={n_candidates}"
        + (f" · chapter={chapter_filter}" if chapter_filter else "")
    )
    for doc, meta, dist, score in results:
        metric = f"score {score:.3f}" if score is not None else f"dist {dist:.3f}"
        header = f"[{metric}]  {meta['chapter']} > {meta['section']} (p{meta['page']})"
        with st.expander(header):
            st.markdown(_to_katex(doc))


def _refusal_section():
    """Anti-hallucination showcase: does the RAG refuse out-of-scope questions?"""
    refusal_json = EVAL_DIR / "refusal.json"
    if not refusal_json.exists():
        return

    import pandas as pd

    st.divider()
    st.subheader("🛡️ Anti-hallucination — out-of-scope refusal")
    st.markdown(
        "A second, **judge-free** test: for **12 out-of-scope** questions (post-2016 topics "
        "like Transformers/BERT/RLHF, or unrelated ones) the RAG *should refuse* — reply "
        "*\"I can't find this in the text.\"* — and for **6 genuine in-scope** questions it "
        "*should still answer*. No LLM-judge: we just check whether the reply is a refusal. "
        "This deliberately avoids grading an open-source LLM with a smarter proprietary one."
    )

    data = json.loads(refusal_json.read_text())
    df = pd.DataFrame(
        {
            "Out-of-scope refusal (↑ better)": {m: v["out_of_scope_refusal_rate"] for m, v in data.items()},
            "In-scope refusal (↓ better)": {m: v["in_scope_refusal_rate"] for m, v in data.items()},
        }
    )
    st.dataframe(df.style.format("{:.0%}"), use_container_width=True)
    st.caption(
        "Both models refuse **100%** of out-of-scope questions and answer **100%** of in-scope "
        "ones — the anti-hallucination system prompt holds. (A lesson from building this: I first "
        "mislabelled *diffusion models* as out-of-scope — they are in Goodfellow, Ch. 20 — and the "
        "test caught my own error.)"
    )


def _citation_section():
    """Citation-validity showcase: are the pages the LLM cites real?"""
    citations_json = EVAL_DIR / "citations.json"
    if not citations_json.exists():
        return

    import pandas as pd

    st.divider()
    st.subheader("📌 Citation validity — are the cited pages real?")
    st.markdown(
        "The LLM must cite its sources as *(chapter, section, page)*. Third **judge-free** "
        "test: the context handed to the model tags every chunk with its real page, so a "
        "faithful answer can only cite pages that were **actually retrieved**. We check each "
        "cited page against the retrieved chunks — a page that isn't there is a *fabricated "
        "citation* (a subtle hallucination). Measured on 25 sampled questions."
    )

    data = json.loads(citations_json.read_text())
    df = pd.DataFrame(
        {
            "Grounded citations (↑ better)": {m: v["grounded_citation_rate"] for m, v in data.items()},
            "Fully-grounded answers (↑)": {m: v["fully_grounded_answers"] for m, v in data.items()},
            "Answers without any citation (↓)": {m: v["answers_without_citation"] for m, v in data.items()},
        }
    )
    st.dataframe(df.style.format("{:.0%}"), use_container_width=True)
    st.caption(
        "**No fabricated citations** on either model — every cited page maps to a retrieved "
        "chunk. The revealing difference is *discipline*: **llama always cites its source**, "
        "while **qwen answers without a citation ~25% of the time** (mostly on broad *concept* "
        "and *why* questions). So citation *correctness* is solved here; citation *coverage* is "
        "where qwen is weaker — a measurable trade-off, not a guess."
    )

    pdf_check = EVAL_DIR / "citations_pdf_check.json"
    if pdf_check.exists():
        c = json.loads(pdf_check.read_text())
        st.markdown(
            f"**Independent PDF cross-check.** The test above only proves the model cites pages "
            f"it was *given*. So, separately (no RAG, no LLM), I checked all "
            f"**{c['cited_pages_checked']} cited pages against the raw book PDF**: does each page "
            f"actually discuss the question's topic? **{c['topic_within_1_page']}/"
            f"{c['cited_pages_checked']}** match on a distinctive keyword; the handful that don't "
            f"are morphology artifacts (*\"overfit\"* vs *\"overfitting\"*, *\"bias and variance\"* "
            f"vs *\"bias-variance\"*), **all manually confirmed on-topic**. Two independent "
            f"verifications agree — and neither uses an LLM judge."
        )


def evaluation_page():
    st.header("📊 Retrieval evaluation")
    st.markdown(
        "Retrieval is scored on **100 questions** (8 types, from simple definitions to "
        "tricky multi-source ones) with **page-level ground truth**: a retrieved chunk is "
        "a *hit* if its page falls in the expected range. Metrics: **Hit@k**, **MRR**, "
        "**Coverage@k**. We compare the **bi-encoder alone** vs **+ cross-encoder reranking**."
    )

    results_json = EVAL_DIR / "results.json"
    results_png = EVAL_DIR / "results.png"

    if results_json.exists():
        import pandas as pd

        data = json.loads(results_json.read_text())
        df = pd.DataFrame(data).T  # configs as rows, metrics as columns
        st.dataframe(df.style.format("{:.2f}"), use_container_width=True)
    else:
        st.info("No results yet — run `uv run python eval/evaluate.py` first.")

    if results_png.exists():
        st.image(str(results_png), use_container_width=True)

    sweep_png = EVAL_DIR / "sweep.png"
    if sweep_png.exists():
        st.subheader("n_candidates sweep (rerank ON)")
        st.caption(
            "Does a larger reranked candidate pool help? The metrics are flat-to-declining "
            "as n_candidates grows -> a bigger pool adds distractors, it does not help."
        )
        st.image(str(sweep_png), use_container_width=True)

    st.subheader("Takeaway")
    st.markdown(
        "On this corpus, **reranking does not improve retrieval** (bi-encoder ≈ reranker on "
        "every metric). Why: the corpus is small and topically distinctive, the bi-encoder "
        "already reaches **Hit@10 ≈ 0.95** (the right chunk is almost always retrieved), and "
        "the page-level ground truth is lenient. The bottleneck is *ranking*, not *recall* — "
        "and the reranker doesn't fix it here. A measured decision: **skip the reranker** "
        "(simpler and faster) rather than adding it because it's expected to help."
    )

    _refusal_section()
    _citation_section()

    with st.expander("⚙️ Re-run the evaluation (slow — needs the Chroma index)"):
        st.caption("Not needed to view the results above; they are pre-computed and committed.")
        if st.button("Run evaluation now"):
            with st.spinner("Running eval/evaluate.py on 100 questions (a few minutes)..."):
                try:
                    subprocess.run(
                        [sys.executable, str(EVAL_DIR / "evaluate.py")],
                        cwd=str(EVAL_DIR.parent), check=True,
                    )
                    st.success("Done — reloading.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}\nRun it in a terminal: uv run python eval/evaluate.py")


st.set_page_config(page_title="RAG Goodfellow", layout="wide")
st.title("📖 RAG — Deep Learning (Goodfellow)")

page = st.sidebar.radio("Page", ["🔍 Search", "📊 Evaluation"])
st.sidebar.divider()

if page == "🔍 Search":
    search_page()
else:
    evaluation_page()
