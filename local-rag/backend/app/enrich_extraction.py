"""Element-level OCR/Vision enrichment.

The module is intentionally independent of chunking. It enriches atomic visual
or table elements after document structure has been assigned.
"""

from typing import Any

from .utils import save_json
from .vision import enrich_visual


def enrich_document(document: dict[str, Any], limit: int | None = None) -> dict[str, Any]:
    count = 0
    for element in document.get("elements", []):
        element_type = str(element.get("type", "")).lower()
        image_path = element.get("image_path")
        if element_type not in {"image", "chart", "figure"} or not image_path:
            continue
        if limit is not None and count >= limit:
            break

        result = enrich_visual(str(image_path))
        element["ocr_text"] = result.get("ocr_text", "")
        element["vision_description"] = result.get("vision_description", "")
        element["evidence"] = {
            "type": "multimodal_enrichment",
            "ocr": result.get("ocr_text", ""),
            "vision": result.get("vision_description", ""),
            "models": result.get("models", {}),
            "errors": result.get("errors", []),
        }
        count += 1

    document.setdefault("stats", {})["enriched_visuals"] = count
    return document


def save_enriched_document(document: dict[str, Any], output_path: str) -> None:
    save_json(output_path, document)
