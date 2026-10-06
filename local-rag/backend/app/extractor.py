"""Layout-aware PDF extraction using PyMuPDF.

The extractor intentionally preserves page, bounding-box and style information.
That metadata gives the generic sectionizer stronger structural signals than text
regex alone, while still working when the PDF extractor returns mixed blocks.
"""

from pathlib import Path
from typing import Any

import pymupdf

from .utils import normalize_whitespace, sanitize_text


def _span_style(spans: list[dict[str, Any]]) -> tuple[float | None, bool]:
    sizes = [float(s.get("size", 0)) for s in spans if s.get("size") is not None]
    max_size = max(sizes) if sizes else None
    bold = any("bold" in str(s.get("font", "")).lower() for s in spans)
    return max_size, bold


def extract_pdf(pdf_path: str) -> dict[str, Any]:
    source = Path(pdf_path)
    elements: list[dict[str, Any]] = []
    visual_count = 0

    with pymupdf.open(pdf_path) as doc:
        page_count = len(doc)
        for page_number, page in enumerate(doc, start=1):
            blocks = page.get_text("dict").get("blocks", [])
            page_images = page.get_images(full=True)
            image_xrefs = {item[0] for item in page_images}

            for block_index, block in enumerate(blocks, start=1):
                block_type = block.get("type")
                bbox = block.get("bbox")

                if block_type == 0:  # text
                    spans = [
                        span
                        for line in block.get("lines", [])
                        for span in line.get("spans", [])
                    ]
                    text = " ".join(
                        sanitize_text(span.get("text", ""))
                        for span in spans
                        if sanitize_text(span.get("text", ""))
                    )
                    text = normalize_whitespace(text)
                    if not text:
                        continue
                    font_size, bold = _span_style(spans)
                    elements.append(
                        {
                            "id": f"text-{page_number}-{block_index}",
                            "type": "text",
                            "content": text,
                            "page": page_number,
                            "bbox": bbox,
                            "font_size": font_size,
                            "bold": bold,
                            "xrefs": [],
                        }
                    )

                elif block_type == 1:  # image block
                    xref = block.get("xref")
                    if xref is None and image_xrefs:
                        xref = next(iter(image_xrefs))
                    if xref is None:
                        continue
                    base = Path("/tmp/rag-extraction") / source.stem / "images"
                    base.mkdir(parents=True, exist_ok=True)
                    image_path = base / f"page-{page_number}-image-{visual_count + 1}.png"
                    try:
                        pix = pymupdf.Pixmap(doc, xref)
                        if pix.alpha:
                            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                        pix.save(str(image_path))
                        visual_count += 1
                    finally:
                        try:
                            pix = None
                        except UnboundLocalError:
                            pass

                    elements.append(
                        {
                            "id": f"image-{page_number}-{block_index}",
                            "type": "image",
                            "content": "",
                            "page": page_number,
                            "bbox": bbox,
                            "image_path": str(image_path),
                            "xref": xref,
                        }
                    )

            # Native table extraction when available. Tables remain first-class evidence
            # and are not forced through the image/VLM path.
            try:
                tables = page.find_tables()
                for table_index, table in enumerate(tables.tables, start=1):
                    markdown = table.to_markdown()
                    if markdown and markdown.strip():
                        elements.append(
                            {
                                "id": f"table-{page_number}-{table_index}",
                                "type": "table",
                                "content": normalize_whitespace(markdown),
                                "page": page_number,
                                "bbox": list(table.bbox),
                                "font_size": None,
                                "bold": False,
                                "xrefs": [],
                            }
                        )
            except (AttributeError, TypeError, ValueError):
                # Some PyMuPDF builds do not include table support.
                pass

            # Preserve page-level text even if PyMuPDF did not expose useful blocks.
            if not any(e.get("page") == page_number for e in elements):
                fallback = normalize_whitespace(page.get_text())
                if fallback:
                    elements.append(
                        {
                            "id": f"page-text-{page_number}",
                            "type": "text",
                            "content": fallback,
                            "page": page_number,
                            "bbox": None,
                            "font_size": None,
                            "bold": False,
                            "xrefs": [],
                        }
                    )

    return {
        "schema_version": 1,
        "source": {
            "filename": source.name,
            "path": str(source),
            "page_count": page_count,
        },
        "elements": elements,
        "stats": {
            "pages": page_count,
            "elements": len(elements),
            "visuals": visual_count,
        },
    }
