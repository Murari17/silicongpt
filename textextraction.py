from __future__ import annotations

from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF
from pdfminer.high_level import extract_text as pdfminer_extract_text
from pdfminer.pdfparser import PDFSyntaxError

try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:  # pragma: no cover - optional dependency
    RapidOCR = None

BASE_DIR = Path(__file__).resolve().parent
raw_dir = BASE_DIR / "data" / "raw"
extracted_dir = BASE_DIR / "data" / "extracted"


def is_git_lfs_pointer(pdf_path: Path) -> bool:
    try:
        with pdf_path.open("r", encoding="utf-8") as handle:
            first_line = handle.readline().strip()
        return first_line.startswith("version https://git-lfs.github.com/spec/v1")
    except UnicodeDecodeError:
        return False
    except OSError:
        return False


def extract_text_from_page(page: fitz.Page) -> str:
    try:
        return (page.get_text("text") or "").strip()
    except Exception:
        # Generic fallback: try blocks/words, then give up.
        try:
            blocks = page.get_text("blocks") or []
            if blocks:
                texts = [str(b[4]).strip() for b in blocks if len(b) > 4 and b[4]]
                return " ".join(texts).strip()
        except Exception:
            pass
        try:
            words = page.get_text("words") or []
            if words:
                return " ".join(w[4] for w in words if len(w) > 4 and w[4]).strip()
        except Exception:
            pass
        return ""


def ocr_page(page: fitz.Page, ocr_engine: RapidOCR) -> str:
    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    image_bytes = pixmap.tobytes("png")
    result, _ = ocr_engine(image_bytes)
    if not result:
        return ""

    lines: list[str] = []
    for item in result:
        if not item or len(item) < 2:
            continue
        text = str(item[1]).strip()
        if text:
            lines.append(text)
    return " ".join(lines).strip()


def extract_with_pymupdf(pdf_path: Path, ocr_engine: RapidOCR | None) -> str:
    document = fitz.open(pdf_path)
    try:
        page_texts: list[str] = []
        total_pages = document.page_count
        for i, page in enumerate(document):
            if i % 10 == 0:
                print(f"  [PyMuPDF] {pdf_path.name}: page {i + 1}/{total_pages} ...")
            text = extract_text_from_page(page)
            # For scanned pages, text layer is usually empty or tiny. OCR only when needed.
            if len(text) < 30 and ocr_engine is not None:
                try:
                    ocr_text = ocr_page(page, ocr_engine)
                    if ocr_text:
                        text = ocr_text
                except Exception:
                    # If OCR fails on a page, keep the text layer result and continue.
                    pass
            if text:
                page_texts.append(text)
        return "\n\n".join(page_texts).strip()
    finally:
        document.close()


def extract_with_pdfminer(pdf_path: Path) -> str:
    return (pdfminer_extract_text(str(pdf_path)) or "").strip()


def extract_pdf_text(pdf_path: Path, ocr_engine: RapidOCR | None) -> tuple[str, str]:
    if is_git_lfs_pointer(pdf_path):
        raise ValueError(
            f"{pdf_path.name} is a Git LFS pointer, not the actual PDF. Run 'git lfs pull' or 'git lfs checkout data/raw'."
        )

    # Prefer PyMuPDF because it is more tolerant across many PDFs.
    try:
        text = extract_with_pymupdf(pdf_path, ocr_engine)
        if text:
            return text, "pymupdf"
    except Exception:
        pass

    # Fallback to pdfminer for PDFs PyMuPDF cannot parse.
    try:
        text = extract_with_pdfminer(pdf_path)
        if text:
            return text, "pdfminer"
    except PDFSyntaxError as exc:
        raise PDFSyntaxError(f"{pdf_path.name}: {exc}")
    except Exception:
        pass

    return "", "unreadable"


def main() -> None:
    pdf_files = sorted(raw_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {raw_dir}")

    extracted_dir.mkdir(parents=True, exist_ok=True)

    for old_txt in extracted_dir.glob("*.txt"):
        old_txt.unlink()

    ocr_engine = RapidOCR() if RapidOCR is not None else None
    if ocr_engine is None:
        print("OCR backend not available; scanned PDFs will only use embedded text if present.")

    processed = 0
    skipped = 0
    for pdf_path in pdf_files:
        output_txt = extracted_dir / f"{pdf_path.stem}.txt"
        print(f"Starting extraction for: {pdf_path.name} ...")
        try:
            text, method = extract_pdf_text(pdf_path, ocr_engine)
        except ValueError as exc:
            print(f"Skipped pointer file: {pdf_path.name} ({exc})")
            skipped += 1
            continue
        except PDFSyntaxError as exc:
            print(f"Skipped malformed PDF: {pdf_path.name} ({exc})")
            skipped += 1
            continue
        except Exception as exc:
            print(f"Skipped unreadable PDF: {pdf_path.name} ({exc})")
            skipped += 1
            continue

        if not text:
            print(f"Skipped empty PDF: {pdf_path.name} (no extractable text)")
            skipped += 1
            continue

        with output_txt.open("w", encoding="utf-8") as f:
            f.write(text)
        print(f"Extracted via {method}: {pdf_path.name} -> {output_txt.name}")
        processed += 1

    print(f"Extraction complete: {processed} PDF file(s) extracted, {skipped} skipped.")


if __name__ == "__main__":
    main()
