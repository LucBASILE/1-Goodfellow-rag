import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from generation import answer  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN = EVAL_DIR / "golden.json"
MODELS = ["qwen2.5:7b", "llama3.1:8b-instruct-q4_K_M"]
SAMPLE = 25  # sampled questions (each question = 1 LLM call per model)

# "... p273)" / "(p 440)" / "page 280" -> capture the cited page number.
PAGE_RE = re.compile(r"\bp(?:age|p)?\.?\s?(\d{1,4})\b", re.IGNORECASE)


def _cited_pages(reply: str) -> list[int]:
    return [int(m) for m in PAGE_RE.findall(reply)]


def _sample(golden):
    """Stable, varied sample: prioritize the types where citation matters
    (comparison, why, how, multi-source), then fill up."""
    priority = {"comparison", "why", "how", "multi-source", "concept"}
    ranked = sorted(golden, key=lambda it: (it["type"] not in priority, golden.index(it)))
    return ranked[:SAMPLE]


def run_model(questions, model):
    rows = []
    for q in questions:
        reply, results = answer(q["question"], model=model, rerank=False)
        retrieved_pages = {meta["page"] for _d, meta, _di, _s in results}
        cited = _cited_pages(reply)
        grounded = [p for p in cited if p in retrieved_pages]
        rows.append(
            {
                "question": q["question"],
                "type": q["type"],
                "cited_pages": cited,
                "grounded_pages": grounded,
                "ungrounded_pages": [p for p in cited if p not in retrieved_pages],
                "retrieved_pages": sorted(retrieved_pages),
                "refused": reply.strip().lower().startswith("i can't find"),
            }
        )
    return rows


def _summarize(rows):
    answered = [r for r in rows if not r["refused"]]
    n_cited = sum(len(r["cited_pages"]) for r in answered)
    n_grounded = sum(len(r["grounded_pages"]) for r in answered)
    with_cite = [r for r in answered if r["cited_pages"]]
    fully = [r for r in with_cite if not r["ungrounded_pages"]]
    return {
        "grounded_citation_rate": round(n_grounded / n_cited, 3) if n_cited else 0.0,
        "fully_grounded_answers": round(len(fully) / len(with_cite), 3) if with_cite else 0.0,
        "avg_citations": round(n_cited / len(answered), 2) if answered else 0.0,
        "answers_without_citation": round(1 - len(with_cite) / len(answered), 3) if answered else 0.0,
        "n_answered": len(answered),
    }


def main():
    golden = json.load(open(GOLDEN))
    questions = _sample(golden)
    print(f"citation check on {len(questions)} questions x {len(MODELS)} models\n")

    summary, detail = {}, {}
    for model in MODELS:
        print(f"### {model}", flush=True)
        rows = run_model(questions, model)
        detail[model] = rows
        s = _summarize(rows)
        summary[model] = s
        print(f"  grounded citations : {s['grounded_citation_rate']:.0%}  (higher = better)")
        print(f"  fully-grounded answers: {s['fully_grounded_answers']:.0%}")
        print(f"  avg citations/answer  : {s['avg_citations']}")
        print(f"  answers w/o citation  : {s['answers_without_citation']:.0%}\n")
        for r in rows:
            if r["ungrounded_pages"]:
                print(f"  ! fabricated pages {r['ungrounded_pages']} in: {r['question'][:60]}")

    json.dump(summary, open(EVAL_DIR / "citations.json", "w"), indent=2)
    json.dump(detail, open(EVAL_DIR / "citations_detail.json", "w"), indent=2, ensure_ascii=False)
    print("\nsummary -> eval/citations.json | detail -> eval/citations_detail.json")


if __name__ == "__main__":
    main()
