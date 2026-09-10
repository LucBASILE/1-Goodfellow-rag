"""Out-of-scope refusal test (RAG-09, generation robustness) — JUDGE-FREE.

Objective test: for questions whose answer is NOT in the book (post-2016 topics,
unrelated topics), does the RAG correctly REFUSE ("I can't find this in the text")
instead of hallucinating? And does it still ANSWER genuine in-scope questions
(control)? Compares qwen vs llama. No LLM-judge -> just a string check on refusal.

    uv run python eval/refusal.py     (needs Ollama running)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from generation import answer  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
MODELS = ["qwen2.5:7b", "llama3.1:8b-instruct-q4_K_M"]

# Questions whose answer is NOT in Goodfellow (2016) -> the RAG must REFUSE.
OUT_OF_SCOPE = [
    "What is a Transformer architecture with self-attention?",   # 2017
    "What is BERT?",                                              # 2018
    "What is GPT (generative pretrained transformer)?",
    "What is reinforcement learning from human feedback (RLHF)?",
    "What is LoRA (low-rank adaptation) fine-tuning?",
    "What is a vision transformer (ViT)?",
    "What is retrieval-augmented generation (RAG)?",
    "What is prompt engineering for large language models?",     # post-2020
    "What is the capital of France?",                            # off-topic
    "How do I bake a chocolate cake?",
    "Who wrote Romeo and Juliet?",
    "What is the offside rule in football?",
]

# Control: genuine questions from the book -> the RAG must NOT refuse.
IN_SCOPE = [
    "What is backpropagation?",
    "What is dropout?",
    "What is maximum likelihood estimation?",
    "What is a convolutional network?",
    "What is early stopping?",
    # Niche topic BUT present in the book (Sohl-Dickstein 2015, Ch20) ->
    # the RAG must answer, not refuse. (lesson: this was NOT out-of-scope.)
    "What is a diffusion model for image generation?",
]

REFUSAL_MARKERS = [
    "can't find this in the text", "cannot find this in the text",
    "couldn't find", "could not find", "not in the text",
    "not in the provided", "not covered", "no information",
    "isn't in the text", "can't find", "cannot find",
]


def is_refusal(text: str) -> bool:
    """True refusal = the answer IS the marker, not a real answer that hedges.

    The system prompt asks for exactly "I can't find this in the text." (~30 chars).
    Some models answer correctly THEN append the marker at the end: that is
    NOT a refusal. So we require the marker present AND a short answer (the refusal
    dominates, there is no substantial content alongside it).
    """
    t = text.lower().strip()
    if not any(m in t for m in REFUSAL_MARKERS):
        return False
    return len(t) < 120


def run_set(questions, model):
    out = []
    for q in questions:
        reply, _ = answer(q, model=model, rerank=False)
        out.append({"question": q, "refused": is_refusal(reply), "answer": reply.strip()})
    return out


def main():
    summary, detail = {}, {}
    for model in MODELS:
        print(f"\n### {model}", flush=True)
        oos = run_set(OUT_OF_SCOPE, model)
        ins = run_set(IN_SCOPE, model)
        detail[model] = {"out_of_scope": oos, "in_scope": ins}

        oos_rate = sum(r["refused"] for r in oos) / len(oos)   # HIGH = good
        ins_rate = sum(r["refused"] for r in ins) / len(ins)   # LOW = good
        summary[model] = {
            "out_of_scope_refusal_rate": round(oos_rate, 2),
            "in_scope_refusal_rate": round(ins_rate, 2),
        }
        print(f"  out-of-scope refusal : {oos_rate:.0%}  (higher = better)")
        print(f"  in-scope refusal     : {ins_rate:.0%}  (lower = better)")
        leaked = [r["question"] for r in oos if not r["refused"]]
        if leaked:
            print("  ! answered instead of refusing (hallucination risk):")
            for q in leaked:
                print(f"      - {q}")

    json.dump(summary, open(EVAL_DIR / "refusal.json", "w"), indent=2)
    json.dump(detail, open(EVAL_DIR / "refusal_detail.json", "w"), indent=2, ensure_ascii=False)
    print(f"\nsummary -> eval/refusal.json | full answers -> eval/refusal_detail.json")


if __name__ == "__main__":
    main()
