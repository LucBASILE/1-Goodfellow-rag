import io
import os

import pymupdf
import pymupdf4llm

from docling.document_converter import DocumentConverter

_NOUGAT = None


def extract(loading_path: str, extractor: str, pages=None) -> dict[int, str]:
    if extractor == "pymupdf":
        return _extract_pymupdf(loading_path, pages)
    if extractor == "pymupdf4llm":
        return _extract_pymupdf4llm(loading_path, pages)
    if extractor == "docling":
        return _extract_docling(loading_path, pages)
    if extractor == "nougat":
        return _extract_nougat(loading_path, pages)
    raise ValueError(f"Unknown extractor: {extractor}")


def _extract_pymupdf(loading_path: str, pages) -> dict[int, str]:
    document = pymupdf.open(loading_path)
    print(f"Loading : {loading_path} | {document.page_count} pages founded")
    if pages is None:
        pages = range(document.page_count)
    return {p: document.load_page(p).get_text() for p in pages}


def _extract_pymupdf4llm(loading_path: str, pages) -> dict[int, str]:
    page_list = list(pages) if pages is not None else None
    chunks = pymupdf4llm.to_markdown(loading_path, pages=page_list, page_chunks=True)
    texts = [chunk["text"] for chunk in chunks]
    keys = page_list if page_list is not None else range(len(texts))
    return dict(zip(keys, texts))


def _extract_docling(loading_path: str, pages) -> dict[int, str]:
    converter = DocumentConverter()
    if pages is None:
        result = converter.convert(loading_path)
        return {0: result.document.export_to_markdown()}
    out = {}
    for p in pages:
        result = converter.convert(loading_path, page_range=(p + 1, p + 1))
        out[p] = result.document.export_to_markdown()
    return out


def _get_nougat():
    global _NOUGAT
    if _NOUGAT is None:
        import torch
        from transformers import NougatProcessor, VisionEncoderDecoderModel

        processor = NougatProcessor.from_pretrained("facebook/nougat-base")
        model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        model.to(device)
        _NOUGAT = (processor, model, device)
    return _NOUGAT


_NOUGAT_PREPROCESS = {
    "do_crop_margin": True,
    "do_resize": True,
    "size": {"height": 896, "width": 672},
    "resample": 2,
    "do_thumbnail": True,
    "do_align_long_axis": False,
    "do_pad": True,
    "do_rescale": True,
    "rescale_factor": 1 / 255,
    "do_normalize": True,
    "image_mean": [0.485, 0.456, 0.406],
    "image_std": [0.229, 0.224, 0.225],
}


def _extract_nougat(loading_path: str, pages) -> dict[int, str]:
    from PIL import Image

    processor, model, device = _get_nougat()
    document = pymupdf.open(loading_path)
    if pages is None:
        pages = range(document.page_count)

    out = {}
    for p in pages:
        pix = document.load_page(p).get_pixmap(dpi=200)
        image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        pixel_values = processor(
            image, return_tensors="pt", **_NOUGAT_PREPROCESS
        ).pixel_values.to(device)
        outputs = model.generate(
            pixel_values,
            min_length=1,
            max_new_tokens=3584,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
        )
        text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        out[p] = processor.post_process_generation(text, fix_markdown=False)
    return out


if __name__ == "__main__":
    data_path = "data"
    extractor = "nougat"
    page_numbers_to_save = (123, 283, 254, 66, 114, 285, 161, 106, 107, 99)

    os.makedirs(f"output/{extractor}", exist_ok=True)

    for filename in os.listdir(data_path):
        loading_path = f"{data_path}/{filename}"
        stem = filename.rsplit(".", 1)[0]

        pages = extract(loading_path, extractor, pages=page_numbers_to_save)

        for page_number, text in pages.items():
            out_path = f"output/{extractor}/{stem}_{page_number}.txt"
            with open(out_path, "w") as f:
                f.write(text)

        print(f"{filename} : {len(pages)} pages -> output/{extractor}/")
        print("--------------------------------")
