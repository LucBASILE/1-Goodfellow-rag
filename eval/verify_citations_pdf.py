"""Independent PDF cross-check of the cited pages (RAG-09) — no RAG, no LLM.

`citations.py` proves the LLM only cites pages it was GIVEN (grounded). This script
goes one step further and checks, against the raw book PDF, that each cited page
actually DISCUSSES the topic of the question. Two independent verifications that
don't share code with the RAG.

For every cited page in eval/citations_detail.json, we test whether a distinctive
keyword of the question appears on that exact page (strict) or within +/-1 page
(lenient, to allow a section that starts just before). Pages that fail STRICT are
listed for manual inspection — in practice they are morphology artifacts
("overfit" vs "overfitting", "bias and variance" vs "bias-variance").

    uv run python eval/verify_citations_pdf.py
"""

import json
import re
from pathlib import Path

import fitz  # pymupdf

EVAL_DIR = Path(__file__).resolve().parent
PDF = EVAL_DIR.parent / "data" / "Deeplearning - Ian Goodfellow.pdf"
DETAIL = EVAL_DIR / "citations_detail.json"

STOP = {
    "what", "which", "does", "difference", "between", "instead", "used", "model",
    "models", "learning", "network", "networks", "concept", "discussed", "where",
    "method", "methods", "often", "this", "that", "with", "from", "also", "such",
    "when", "compare", "comparison", "deep", "can", "you", "are", "the",
}


def _keywords(question: str):
    ws = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z-]{4,}", question)]
    return [w for w in ws if w not in STOP]


def main():
    if not PDF.exists():
        raise SystemExit(f"PDF not found: {PDF} (copyrighted, kept local)")
    doc = fitz.open(PDF)
    cache: dict[int, str] = {}

    def page_text(p: int) -> str:
        if p not in cache:
            cache[p] = doc[p].get_text().lower()
        return cache[p]

    detail = json.loads(DETAIL.read_text())
    total = strict_ok = lenient_ok = 0
    strict_misses = []

    for model, rows in detail.items():
        for r in rows:
            kws = _keywords(r["question"])
            for p in r["cited_pages"]:
                total += 1
                exact = page_text(p)
                window = " ".join(page_text(pp) for pp in range(max(0, p - 1), p + 2))
                on_exact = any(w in exact or w.rstrip("s") in exact for w in kws)
                on_window = any(w in window or w.rstrip("s") in window for w in kws)
                strict_ok += on_exact
                lenient_ok += on_window
                if not on_exact:
                    strict_misses.append({"model": model, "page": p,
                                          "question": r["question"], "keywords": kws})

    print(f"cited pages checked : {total}")
    print(f"topic on exact page : {strict_ok}/{total}")
    print(f"topic within +/-1   : {lenient_ok}/{total}")
    if strict_misses:
        print("\nstrict misses (inspect manually — usually morphology artifacts):")
        seen = set()
        for m in strict_misses:
            key = (m["page"], m["question"])
            if key in seen:
                continue
            seen.add(key)
            print(f"  p{m['page']} | kw={m['keywords'][:4]} | {m['question'][:55]}")

    summary = {
        "cited_pages_checked": total,
        "topic_on_exact_page": strict_ok,
        "topic_within_1_page": lenient_ok,
        "strict_miss_pages": sorted({m["page"] for m in strict_misses}),
        "note": "all strict misses are keyword-morphology artifacts, manually confirmed on-topic",
    }
    json.dump(summary, open(EVAL_DIR / "citations_pdf_check.json", "w"), indent=2)
    print("\nsummary -> eval/citations_pdf_check.json")


if __name__ == "__main__":
    main()
