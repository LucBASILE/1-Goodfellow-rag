"""
Score retrieve() against eval/golden.json. A retrieved chunk is "relevant" if
its page falls into one of the expected page_ranges.

Metrics (averaged over the questions):
  - Hit@k       : is a relevant chunk in the top-k? (0/1)
  - MRR         : 1 / rank of the 1st relevant chunk
  - Coverage@k  : fraction of the distinct page_ranges covered in the top-k

Aggregated PER config (rerank on/off) AND PER question type. Saves a
chart (eval/results.png) and the numbers (eval/results.json).

Run from the project root:
    uv run python eval/evaluate.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from retrieval import retrieve  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN = EVAL_DIR / "golden.json"
CUTOFFS = [1, 3, 5, 10]
MAX_K = 10
METRIC_KEYS = [f"hit@{k}" for k in CUTOFFS] + ["mrr", "cov"]
METRIC_LABELS = [f"Hit@{k}" for k in CUTOFFS] + ["MRR", "Cov"]

CONFIGS = [
    {"name": "bi-encoder only", "n_candidates": 30, "rerank": False},
    {"name": "+ rerank", "n_candidates": 30, "rerank": True},
]


def _free_mps():
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def _relevant(page, ranges):
    return any(lo <= page <= hi for lo, hi in ranges) # returns True if any of the page ranges contain the page


def _coverage(pages, ranges):
    covered = sum(any(lo <= p <= hi for p in pages) for lo, hi in ranges)
    return covered / len(ranges)


def evaluate_config(golden, n_candidates, rerank):
    rows = []
    for i, item in enumerate(golden):
        results = retrieve(item["question"], k=MAX_K, n_candidates=n_candidates, rerank=rerank)
        pages = [meta["page"] for _doc, meta, _d, _s in results]
        ranges = item["page_ranges"]
        flags = [_relevant(p, ranges) for p in pages]

        first = next((i for i, f in enumerate(flags) if f), None)
        row = {
            "type": item["type"],
            "mrr": 1.0 / (first + 1) if first is not None else 0.0,
            "cov": _coverage(pages, ranges),
        }
        for k in CUTOFFS:
            row[f"hit@{k}"] = 1.0 if any(flags[:k]) else 0.0
        rows.append(row)
        if i % 10 == 0:
            _free_mps()  # avoids GPU memory accumulation
    return rows


SWEEP_NCAND = [10, 30, 60]


def run_sweep():
    """Vary n_candidates (rerank ON) to test whether a larger pool helps."""
    golden = json.load(open(GOLDEN))
    print(f"sweep n_candidates={SWEEP_NCAND} (rerank ON) | {len(golden)} questions\n")

    rows_by_n = {}
    for n in SWEEP_NCAND:
        print(f"  n_candidates={n} ...", flush=True)
        rows_by_n[n] = evaluate_config(golden, n_candidates=n, rerank=True)
        _free_mps()

    lines = [("hit@1", "Hit@1"), ("hit@5", "Hit@5"), ("mrr", "MRR")]
    print(f"\n{'n_cand':>7} " + " ".join(f"{lbl:>7}" for _, lbl in lines))
    for n in SWEEP_NCAND:
        print(f"{n:>7} " + " ".join(f"{_mean(rows_by_n[n], key):7.2f}" for key, _ in lines))

    summary = {str(n): {k: round(_mean(rows_by_n[n], k), 3) for k in METRIC_KEYS} for n in SWEEP_NCAND}
    json.dump(summary, open(EVAL_DIR / "sweep.json", "w"), indent=2)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for key, lbl in lines:
        ax.plot(SWEEP_NCAND, [_mean(rows_by_n[n], key) for n in SWEEP_NCAND], marker="o", label=lbl)
    ax.set_xlabel("n_candidates (reranked pool size)")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Metrics (rerank ON) vs n_candidates")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(EVAL_DIR / "sweep.png", dpi=120)
    print("\nchart -> eval/sweep.png")


def _mean(rows, key):
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def plot(results, types, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(results)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    # -- Panel A: grouped bars of the global metrics --
    x = np.arange(len(METRIC_LABELS))
    w = 0.8 / len(names)
    for i, name in enumerate(names):
        vals = [_mean(results[name], k) for k in METRIC_KEYS]
        bars = ax1.bar(x + i * w, vals, w, label=name)
        for b, v in zip(bars, vals):
            ax1.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(x + w * (len(names) - 1) / 2)
    ax1.set_xticklabels(METRIC_LABELS)
    ax1.set_ylim(0, 1.08)
    ax1.set_title("Global metrics per config")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # -- Panel B: Hit@5 heatmap by type x config --
    M = np.array([[_mean([r for r in results[n] if r["type"] == t], "hit@5")
                   for n in names] for t in types])
    im = ax2.imshow(M, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=15, ha="right")
    ax2.set_yticks(range(len(types)))
    ax2.set_yticklabels(types)
    for i in range(len(types)):
        for j in range(len(names)):
            ax2.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax2.set_title("Hit@5 by question type")
    fig.colorbar(im, ax=ax2, fraction=0.046)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\nchart -> {path}")


def main():
    golden = json.load(open(GOLDEN))
    types = sorted({item["type"] for item in golden})
    print(f"{len(golden)} questions\n")

    results = {}
    for cfg in CONFIGS:
        print(f"  running: {cfg['name']} ...", flush=True)
        results[cfg["name"]] = evaluate_config(golden, cfg["n_candidates"], cfg["rerank"])
        _free_mps()

    # -- text table --
    print("\n=== global ===")
    print(f"{'config':18} " + " ".join(f"{c:>6}" for c in METRIC_LABELS))
    for name, rows in results.items():
        vals = [_mean(rows, k) for k in METRIC_KEYS]
        print(f"{name:18} " + " ".join(f"{v:6.2f}" for v in vals))

    # -- saves --
    summary = {
        name: {k: round(_mean(rows, k), 3) for k in METRIC_KEYS}
        for name, rows in results.items()
    }
    json.dump(summary, open(EVAL_DIR / "results.json", "w"), indent=2)
    plot(results, types, EVAL_DIR / "results.png")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        # run the sweep of n_candidates (rerank ON) to test whether a larger pool helps
        run_sweep()
    else:
        main()
