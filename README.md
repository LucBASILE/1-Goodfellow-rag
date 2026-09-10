# RAG on Goodfellow's *Deep Learning*

A **from-scratch, fully local** Retrieval-Augmented Generation system over Ian Goodfellow's
*Deep Learning* (Goodfellow, Bengio & Courville, 2016) — an 800-page, equation-heavy technical
book.

Built as a **learning + portfolio project**: no LangChain, no hosted API. Every stage —
equation-aware PDF extraction, structure-aware chunking, quality filtering, embedding, vector
search, cross-encoder reranking, and generation — is implemented and, crucially, **measured**.
The emphasis is less on plumbing and more on *evaluating* the pipeline with **objective,
judge-free tests** and being explicit about what does and doesn't work.

> **Stack:** Python 3.13 · [Nougat](https://github.com/facebookresearch/nougat) (equation-aware OCR) ·
> `sentence-transformers` (bge-small / bge-reranker) · Chroma (cosine) · Ollama (qwen2.5:7b,
> llama3.1:8b) · Streamlit. **Runs entirely offline on an Apple-Silicon laptop.**

> **Note on AI assistance.** This README was written with the help of an AI assistant. The core
> pipeline and every design decision are my own; AI was used as a pair-programming and writing
> tool. Specific AI-assisted parts are flagged inline where relevant (the Streamlit UI, the
> evaluation question set).

---



## Why from scratch (and not LangChain)?

The goal is to *understand* a RAG, not to assemble one. Writing each stage by hand surfaces the
decisions a framework hides — how to keep equations intact through extraction, why semantic
chunking hurts a well-structured book, when a reranker earns its cost, and how to know any of it
actually works. A production system would sensibly reach for a framework; a learning project
should not.

---



## Pipeline

```
PDF ──► extraction ──► chunking ──► quality ──► embedding ──► Chroma
       (Nougat,       (structure-   (Gopher/    (bge-small,    (cosine)
        eq-aware)      aware)        RefinedWeb  384-dim)          │
                                     filters)                      ▼
                                                       retrieval ──► (rerank?) ──► generation
                                                       (bi-encoder)  (cross-enc)   (Ollama, cited)
```

**~2,600 chunks** indexed from the book's content chapters (front matter, bibliography and index
are filtered out).

### Design decisions


| Stage          | Choice                                                                    | Why                                                                                                                                                                                                                                                                                               |
| -------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Extraction** | Nougat (`facebook/nougat-base`), pymupdf fallback                         | The book is equation-heavy; Nougat emits LaTeX so formulas survive as `\( … \)` instead of turning into garbage. A page-level quality check falls back to pymupdf when Nougat hallucinates.                                                                                                       |
| **Chunking**   | Structure-aware (Markdown headings), **not semantic**                     | *Semantic chunking ∝ 1 / structure available.* The book already has a clean chapter/section hierarchy — splitting on it beats guessing boundaries by embedding similarity. Equations are glued to the preceding block; tiny chunks are packed (min ~30 / max ~250 words, never across a heading). |
| **Quality**    | Compression ratio, repeated-n-gram fraction, symbol ratio                 | Gopher / RefinedWeb-style heuristics drop extraction noise (repeated headers, symbol soup) before it pollutes the index.                                                                                                                                                                          |
| **Embeddings** | `BAAI/bge-small-en-v1.5` (384-dim, cosine, normalized)                    | Small, strong, MPS-friendly; the BGE query instruction is prepended at search time.                                                                                                                                                                                                               |
| **Reranking**  | `BAAI/bge-reranker-base` cross-encoder — **available but off by default** | See the evaluation: on this corpus it doesn't help. Kept as a toggle so the *measurement* is reproducible.                                                                                                                                                                                        |
| **Generation** | Ollama, `qwen2.5:7b` / `llama3.1:8b`, `num_ctx=8192`                      | Local, open-source. A strict system prompt forces *answer only from context, cite (chapter, section, page), else refuse.*                                                                                                                                                                         |


---



## Evaluation — the heart of the project

RAG demos are easy; **knowing whether a RAG works is the hard part.** This project evaluates
both halves — retrieval *and* generation — with **four objective tests, none of which use an
LLM judge.**

### Why no LLM-as-judge?

The generator is an **open-source** model. Grading it with a *smarter, proprietary* judge (GPT-4
class) would quietly reintroduce the dependency the whole project avoids, and make the results
un-reproducible offline. So every metric here is a **deterministic check** — page membership,
string refusal, citation grounding — that anyone can re-run locally and audit. This is a
deliberate trade-off, not an oversight.

### 1. Retrieval quality — `eval/evaluate.py`

Scored against `eval/golden.json` **(102 questions, 8 types**, from simple definitions to tricky
multi-source ones) with **page-level ground truth**: a retrieved chunk is a *hit* if its page
falls in the expected range. (Page-level, not section-level, because section metadata is noisy —
e.g. dropout splits across "7.1 Introduction" and "7.12 Dropout".)

*The question set was drafted with AI assistance (to generate and phrase the 102 questions across
the 8 types), then reviewed and corrected by hand — including fixing the ground-truth page ranges,
which is where the real work was.*


| Config              | Hit@1 | Hit@3 | Hit@5 | Hit@10   | MRR  | Cov  |
| ------------------- | ----- | ----- | ----- | -------- | ---- | ---- |
| **bi-encoder only** | 0.63  | 0.80  | 0.88  | **0.95** | 0.74 | 0.94 |
| **+ rerank**        | 0.61  | 0.80  | 0.87  | 0.94     | 0.72 | 0.94 |


**Finding: the cross-encoder reranker does *not* improve retrieval here** — it's flat-to-slightly-worse
on every metric. An `n_candidates` sweep (10 → 30 → 60) confirms it: a bigger reranked pool only
adds distractors. Why: the corpus is small and topically distinctive, the bi-encoder already
reaches **Hit@10 ≈ 0.95** (the right chunk is almost always retrieved), so the bottleneck is
*ranking precision*, not recall — and a general-domain reranker on technical text doesn't fix it.
**The measured decision is to ship without the reranker** (simpler, faster) rather than add it
because it's "supposed to help".

retrieval metrics

### 2. Out-of-scope refusal (anti-hallucination) — `eval/refusal.py`

Does the RAG **refuse** questions whose answer isn't in the book (Transformers, BERT, RLHF — all
post-2016 — plus unrelated trivia), while still **answering** genuine in-scope questions? Pure
string check on the refusal sentence — no judge.


| Model       | Out-of-scope refusal (↑ better) | In-scope refusal (↓ better) |
| ----------- | ------------------------------- | --------------------------- |
| qwen2.5:7b  | **100%** (12/12)                | 0% (0/6)                    |
| llama3.1:8b | **100%** (12/12)                | 0% (0/6)                    |




### 3. Citation validity — `eval/citations.py`

The prompt makes the model cite `(chapter, section, page)`. Since the context is tagged with each
chunk's real page, a faithful answer can only cite pages that were **actually retrieved** — a page
that isn't there is a *fabricated citation*. Measured on 25 sampled questions.


| Model       | Grounded citations (↑) | Fully-grounded answers | Answers without any citation (↓) |
| ----------- | ---------------------- | ---------------------- | -------------------------------- |
| qwen2.5:7b  | **100%**               | **100%**               | 25%                              |
| llama3.1:8b | **100%**               | **100%**               | 0%                               |


**No fabricated citations on either model.** The revealing difference is *discipline*: llama always
cites its source; qwen answers without a citation ~25% of the time (mostly on broad *concept* / *why*
questions). Citation **correctness** is solved; citation **coverage** is where qwen is weaker — a
measurable trade-off, not a guess.

### 4. Independent PDF cross-check — `eval/verify_citations_pdf.py`

Test 3 only proves the model cites pages it was *given*. A **second, independent** verification
(no RAG, no LLM) checks all **71 cited pages against the raw book PDF**: does each page actually
discuss the question's topic? **64/71** match on a distinctive keyword on the exact page; the rest
are morphology artifacts (*"overfit"* vs *"overfitting"*, *"bias and variance"* vs *"bias-variance"*),
**all manually confirmed on-topic**. Two independent checks agree — grounding (RAG side) and real
content (PDF side) — and neither uses an LLM judge.

---



## App

A two-page Streamlit UI:

- **🔍 Search** — ask a question; compare bi-encoder vs reranking live, toggle generation, filter by
chapter, inspect the retrieved chunks (equations rendered via KaTeX).
- **📊 Evaluation** — the four tests above, rendered from **pre-computed, committed artifacts**. A
visitor sees the whole comparison **without running any model, Ollama, or Chroma.**

*The Streamlit UI was built with AI assistance — it's the presentation layer, not the focus of the
project. The pipeline, evaluation methodology, and results it displays are my own work.*

---



## Run it

```bash
# 1. install (uv)
uv sync

# 2. build the index (extract → chunk → quality → embed → Chroma). Runs once.
uv run python src/main.py

# 3. launch the app (Ollama must be running for the Search page's "Generate")
uv run streamlit run src/app.py
```

Re-running the evaluations (needs the built index + Ollama):

```bash
uv run python eval/evaluate.py          # retrieval (+ `sweep` for the n_candidates sweep)
uv run python eval/refusal.py           # out-of-scope refusal
uv run python eval/citations.py         # citation grounding
uv run python eval/verify_citations_pdf.py   # independent PDF cross-check (needs local PDF)
```

The evaluation artifacts (`eval/*.json`, `eval/*.png`) are committed, so the Evaluation page and
all the tables above render without re-running anything.

---



## Project structure

```
src/
  ingestion/
    extractors.py        # 5-extractor comparison + Nougat dispatcher
    extract_content.py   # PDF → JSONL (resumable)
  preprocessing/
    chunking.py          # structure-aware chunking + equation cascade + packing
    quality.py           # Gopher/RefinedWeb-style quality filters + fallback
    stats.py             # chunk-size distribution stats
  vector_store.py        # Chroma collection, indexing, cosine search
  retrieval.py           # bi-encoder + optional cross-encoder rerank
  generation.py          # Ollama call, cited-answer system prompt
  main.py                # build_all_chunks() + index into Chroma
  app.py                 # Streamlit UI (Search + Evaluation)
eval/
  golden.json            # 102 questions, page-level ground truth
  evaluate.py            # retrieval metrics (Hit@k, MRR, Coverage) + sweep
  refusal.py             # out-of-scope refusal test
  citations.py           # citation-grounding test
  verify_citations_pdf.py# independent PDF cross-check
  *.json / *.png         # committed results
```

---



## Limitations & next steps

- **General-domain models on a technical corpus.** The embedder and reranker aren't math-specialized;
a domain-adapted embedder could lift Hit@1.
- **Page-level ground truth is lenient** by design (section metadata is noisy). It measures recall
well but is forgiving on fine ranking.
- **Small-to-big retrieval** (return the whole parent section to the generator) is designed but not
yet implemented. This is also the most promising lever on the *quality* axis the metrics above can't
see: retrieval matches on small, precise chunks (good for the bi-encoder), but the generator would
then receive the **full parent section** rather than a possibly truncated fragment — more complete,
self-contained context to ground the answer in. It targets exactly the gap noted above (a chunk can
be "on the right page" yet too partial to answer well) by decoupling *what we match on* from *what we
feed the LLM*.
- **Latency** isn't yet measured as a formal metric.

### What the retrieval metrics do and don't measure

It's worth being explicit about the ceiling of this evaluation. The retrieval metrics answer one
question — *is the right information available to the LLM, and well-ranked?* — and nothing beyond it.
They deliberately don't measure two things:

- **Chunk content quality.** `_relevant()` only checks a page *number* against the expected ranges;
  it never reads the chunk. A chunk can be "on the right page" yet be badly split or truncated. That
  concern is handled **upstream** by the quality filters (`quality.py`) at ingestion time, not scored
  here — a design choice: clean at the source rather than grade after the fact.
- **Answer satisfaction.** Whether the final answer is actually *good* is a separate axis this harness
  doesn't touch directly. Measuring it objectively would require an LLM judge (rejected — it would
  need a model smarter than the one being judged) or large-scale human annotation. Instead it's
  approached through objective **proxies**: correct refusal (no hallucination) and grounded, PDF-verified
  citations (the answer really leans on the provided text).

**On precision@k (and why it's absent).** A natural fourth retrieval metric would be **precision@k** —
the *fraction* of the top-k that is relevant (`sum(flags[:k]) / k`), i.e. the *purity* of the retrieved
set rather than just "is at least one relevant chunk present". It measures **noise**: how many
distractors get sent to the LLM alongside the good chunk. It is **not** included here, on purpose:
this corpus is small and distinctive, the bi-encoder already reaches **Hit@10 ≈ 0.95** (recall — the
real goal — is essentially solved), and the lenient page-level ground truth (wide ranges) would make a
page-level precision noisy and hard to read. You'd reach for precision@k when the aim is to **tune `k`
to minimise the noise fed to the LLM**, or on a large corpus with many false positives — neither of
which is the point of this portfolio project, where the RAG isn't headed to production. Trivial to add
if that changes; left out because it wouldn't change the conclusion here.

---



## Getting the book (and a note on copyright)

The book is **copyright Goodfellow, Bengio & Courville / MIT Press**. The authors offer it **free
to read as HTML** at [deeplearningbook.org](https://www.deeplearningbook.org/); by their own terms
(their MIT Press contract), **no PDF is authorized for redistribution**.

So this repository **ships no book content**: `data/` (the source PDF), `extracted/` (the derived
Nougat text) and `chroma/` (the vector store, which holds the chunk text) are all git-ignored. The
committed evaluation artifacts contain only metrics and questions — never the book's text.

The results here were produced **locally, from my own purchased copy** of the book; nothing from it
is redistributed. To run the full pipeline yourself, obtain your own copy of the book, place it at
`data/Deeplearning - Ian Goodfellow.pdf`, then:

```bash
uv run python src/main.py          # extract → chunk → quality → embed → index
uv run streamlit run src/app.py    # explore + see the evaluation
```

The **Evaluation** page needs none of this — it renders the committed results with no book, no
models, no Ollama. If you'd like a **live technical walkthrough** of the retrieval/generation side,
feel free to reach out and I'll gladly demo it end to end.

Citation, as provided by the authors:

```bibtex
@book{Goodfellow-et-al-2016,
    title={Deep Learning},
    author={Ian Goodfellow and Yoshua Bengio and Aaron Courville},
    publisher={MIT Press},
    note={\url{http://www.deeplearningbook.org}},
    year={2016}
}
```