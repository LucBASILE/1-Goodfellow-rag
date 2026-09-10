import json
from pathlib import Path

import pymupdf

from ingestion.extractors import extract

DATA_DIR = Path("data")
EXTRACTED_DIR = Path("extracted")


def extract_source(pdf_path: Path, jsonl_path: Path) -> None:
    done = set()
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            done = {json.loads(line)["page"] for line in f}

    n_pages = pymupdf.open(pdf_path).page_count
    todo = [p for p in range(n_pages) if p not in done]
    if not todo:
        print(f"  extraction already complete ({n_pages} pages)")
        return

    with open(jsonl_path, "a") as f:
        for i, page in enumerate(todo, 1):
            text = extract(str(pdf_path), "nougat", pages=[page])[page]
            record = {"page": page, "source": pdf_path.name, "text": text}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  page {page + 1}/{n_pages} ({i}/{len(todo)} remaining)")


def extract_all() -> None:
    EXTRACTED_DIR.mkdir(exist_ok=True)
    for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
        print(f"\n=== {pdf_path.name} ===")
        extract_source(pdf_path, EXTRACTED_DIR / f"{pdf_path.stem}.jsonl")


if __name__ == "__main__":
    extract_all()
