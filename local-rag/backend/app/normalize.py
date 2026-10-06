"""Normalize extracted elements without destroying source evidence."""

from typing import Any

from .utils import get_first, normalize_whitespace, sanitize_text


def _normalize_element(element: dict[str, Any], index: int) -> dict[str, Any]:
    out = dict(element)
    out["id"] = str(get_first(element, "id", "element_id", default=f"element-{index}"))
    out["type"] = str(get_first(element, "type", "element_type", default="text")).lower()
    out["page"] = int(get_first(element, "page", "page_number", default=1) or 1)
    out["content"] = normalize_whitespace(
        str(get_first(element, "content", "text", "value", default=""))
    )
    out.setdefault("bbox", None)
    out.setdefault("font_size", None)
    out.setdefault("bold", False)
    out.setdefault("image_path", None)
    out.setdefault("ocr_text", "")
    out.setdefault("vision_description", "")
    out.setdefault("evidence", {})
    return out


def normalize_document(document: dict[str, Any]) -> dict[str, Any]:
    elements = [
        _normalize_element(element, i)
        for i, element in enumerate(document.get("elements", []), start=1)
    ]

    pages = sorted({int(e.get("page", 1)) for e in elements})
    return {
        "schema_version": 2,
        "source": document.get("source", {}),
        "elements": elements,
        "stats": {
            **document.get("stats", {}),
            "normalized_elements": len(elements),
            "normalized_pages": len(pages),
        },
    }
