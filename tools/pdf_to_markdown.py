from __future__ import annotations

import argparse
import re
from pathlib import Path

import fitz
import numpy as np
from rapidocr_onnxruntime import RapidOCR


def clean_line(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_inline_text(text: str) -> str:
    text = clean_line(text)
    text = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", text)
    text = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", text)
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"([,.;:])(?=[^\s])", r"\1 ", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return clean_line(text)


def is_upper_like(text: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return False
    uppercase_ratio = sum(1 for char in letters if char.isupper()) / len(letters)
    return uppercase_ratio > 0.85


def format_markdown_text(text: str) -> str:
    formatted_lines: list[str] = []

    for raw_line in text.splitlines():
        line = normalize_inline_text(raw_line)
        if not line:
            if formatted_lines and formatted_lines[-1] != "":
                formatted_lines.append("")
            continue

        if re.fullmatch(r"\d+(?:/\d+)?", line):
            continue

        if re.match(r"^Chuong\b", line, flags=re.IGNORECASE):
            line = f"### {line}"
        elif re.match(r"^Dieu\s+\d+\.", line, flags=re.IGNORECASE):
            line = f"#### {line}"
        elif re.match(r"^[a-z]\)", line):
            line = f"- {line}"
        elif is_upper_like(line) and 4 <= len(line) <= 80:
            line = f"### {line}"

        formatted_lines.append(line)

    while formatted_lines and formatted_lines[-1] == "":
        formatted_lines.pop()

    return "\n".join(formatted_lines)


def extract_text_layer(page: fitz.Page) -> str:
    raw_text = page.get_text("text", sort=True)
    lines = [clean_line(line) for line in raw_text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def render_page(page: fitz.Page, scale: float) -> np.ndarray:
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    image = np.frombuffer(pix.samples, dtype=np.uint8)
    return image.reshape(pix.height, pix.width, pix.n)


def group_ocr_lines(result: list[list]) -> str:
    entries = []
    for box, text, score in result:
        text = clean_line(text)
        if not text or score < 0.3:
            continue

        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        top = min(ys)
        bottom = max(ys)
        left = min(xs)
        entries.append(
            {
                "text": text,
                "top": top,
                "bottom": bottom,
                "left": left,
                "height": max(bottom - top, 1.0),
            }
        )

    if not entries:
        return ""

    entries.sort(key=lambda item: (item["top"], item["left"]))

    grouped_lines: list[dict[str, float | str]] = []
    current_group: list[dict[str, float | str]] = []
    current_top = 0.0
    current_height = 0.0

    for entry in entries:
        if not current_group:
            current_group = [entry]
            current_top = float(entry["top"])
            current_height = float(entry["height"])
            continue

        threshold = max(10.0, current_height * 0.6)
        if abs(float(entry["top"]) - current_top) <= threshold:
            current_group.append(entry)
            current_top = (current_top + float(entry["top"])) / 2
            current_height = max(current_height, float(entry["height"]))
            continue

        current_group.sort(key=lambda item: float(item["left"]))
        grouped_lines.append(
            {
                "text": " ".join(str(item["text"]) for item in current_group),
                "top": min(float(item["top"]) for item in current_group),
                "bottom": max(float(item["bottom"]) for item in current_group),
                "height": max(float(item["height"]) for item in current_group),
            }
        )
        current_group = [entry]
        current_top = float(entry["top"])
        current_height = float(entry["height"])

    if current_group:
        current_group.sort(key=lambda item: float(item["left"]))
        grouped_lines.append(
            {
                "text": " ".join(str(item["text"]) for item in current_group),
                "top": min(float(item["top"]) for item in current_group),
                "bottom": max(float(item["bottom"]) for item in current_group),
                "height": max(float(item["height"]) for item in current_group),
            }
        )

    markdown_lines: list[str] = []
    previous_bottom = 0.0

    for index, line in enumerate(grouped_lines):
        if index > 0:
            gap = float(line["top"]) - previous_bottom
            if gap > float(line["height"]) * 1.4:
                markdown_lines.append("")

        markdown_lines.append(clean_line(str(line["text"])))
        previous_bottom = float(line["bottom"])

    return "\n".join(markdown_lines)


def extract_ocr(page: fitz.Page, ocr_engine: RapidOCR, scale: float) -> str:
    image = render_page(page, scale)
    result, _ = ocr_engine(image)
    if not result:
        return ""
    return group_ocr_lines(result)


def convert_pdf(pdf_path: Path, output_path: Path, min_text_chars: int, ocr_scale: float) -> None:
    doc = fitz.open(pdf_path)
    ocr_engine = RapidOCR()

    parts = [f"# {pdf_path.stem}", "", f"- Source: `{pdf_path.name}`", f"- Pages: {doc.page_count}", ""]

    for page_number, page in enumerate(doc, start=1):
        text_layer = extract_text_layer(page)
        use_ocr = len(text_layer) < min_text_chars
        raw_page_text = extract_ocr(page, ocr_engine, ocr_scale) if use_ocr else text_layer
        page_text = format_markdown_text(raw_page_text)

        parts.append(f"## Page {page_number}")
        if page_text:
            parts.append(page_text)
        else:
            parts.append("_No text extracted from this page._")
        parts.append("")

    output_path.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PDF files to Markdown.")
    parser.add_argument("pdfs", nargs="+", help="One or more PDF file paths.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for Markdown output. Defaults to each PDF's parent directory.",
    )
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=100,
        help="Use OCR when extracted text is shorter than this threshold.",
    )
    parser.add_argument(
        "--ocr-scale",
        type=float,
        default=1.5,
        help="Render scale for OCR fallback.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else None

    for pdf_arg in args.pdfs:
        pdf_path = Path(pdf_arg).resolve()
        target_dir = output_dir.resolve() if output_dir else pdf_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        output_path = target_dir / f"{pdf_path.stem}.md"
        convert_pdf(pdf_path, output_path, args.min_text_chars, args.ocr_scale)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
