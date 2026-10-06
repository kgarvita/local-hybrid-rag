"""Section-aware Parent/Child chunk construction.

Parents are larger context units; children are the retrieval/embedding units.
Original element IDs, section paths, pages and modality evidence are preserved.
"""

from __future__ import annotations

from typing import Any

from .config import CHILD_CHUNK_OVERLAP, CHILD_CHUNK_SIZE, PARENT_CHUNK_SIZE
from .utils import normalize_whitespace


def _element_text(e: dict[str, Any]) -> str:
    parts: list[str] = []
    content = normalize_whitespace(str(e.get("content", "")))
    ocr = normalize_whitespace(str(e.get("ocr_text", "")))
    vision = normalize_whitespace(str(e.get("vision_description", "")))
    if content:
        parts.append(content)
    if ocr:
        parts.append(f"[OCR] {ocr}")
    if vision:
        parts.append(f"[VISUAL] {vision}")
    return "\n".join(parts)


def _word_windows(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    if size <= 0:
        raise ValueError("Chunk size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("Overlap must be >= 0 and smaller than chunk size")
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step)]


def _modality(e: dict[str, Any]) -> str:
    etype = str(e.get("type", "text"))
    if e.get("ocr_text") or e.get("vision_description"):
        return "multimodal"
    if etype in {"image", "chart", "table"}:
        return "multimodal"
    return "text"


def _make_unit(elements: list[dict[str, Any]], index: int, chunk_type: str) -> dict[str, Any]:
    contents = [_element_text(e) for e in elements]
    content = normalize_whitespace(" ".join(c for c in contents if c))
    pages = sorted({int(e.get("page", 1)) for e in elements})
    ids = [str(e.get("id", "")) for e in elements if e.get("id")]
    section_path = list(elements[0].get("section_path") or []) if elements else []
    modalities = sorted({_modality(e) for e in elements})
    return {
        "chunk_index": index,
        "chunk_type": chunk_type,
        "content": content,
        "page": pages[0] if pages else None,
        "section_path": section_path,
        "element_ids": ids,
        "evidence": {
            "element_count": len(ids),
            "pages": pages,
            "modalities": modalities,
        },
    }


def _section_groups(document: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """Group document elements by section while preserving document order."""
    groups: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for e in document.get("elements", []):
        sid = str(e.get("section_id") or "root")
        if sid not in groups:
            groups[sid] = []
            order.append(sid)
        groups[sid].append(e)
    return [groups[sid] for sid in order]


def build_parent_child_chunks(
    document: dict[str, Any],
    *,
    parent_size: int = PARENT_CHUNK_SIZE,
    child_size: int = CHILD_CHUNK_SIZE,
    child_overlap: int = CHILD_CHUNK_OVERLAP,
) -> dict[str, list[dict[str, Any]]]:
    """Build larger section-aware parents and smaller retrieval children.

    Parents are formed from original elements and never cross section boundaries.
    Children are formed inside each parent. Large individual elements are split by
    words while retaining the originating element id and evidence metadata.
    """
    if parent_size <= 0:
        raise ValueError("PARENT_CHUNK_SIZE must be positive")
    if child_size <= 0:
        raise ValueError("CHILD_CHUNK_SIZE must be positive")
    if child_overlap < 0 or child_overlap >= child_size:
        raise ValueError("CHILD_CHUNK_OVERLAP must be >= 0 and smaller than CHILD_CHUNK_SIZE")

    parents: list[dict[str, Any]] = []
    children: list[dict[str, Any]] = []
    parent_counter = 0
    child_counter = 0
    global_chunk_index = 0

    for section_elements in _section_groups(document):
        current: list[dict[str, Any]] = []
        current_words = 0

        def flush_parent() -> None:
            nonlocal current, current_words, parent_counter, global_chunk_index
            if not current:
                return
            text = " ".join(_element_text(e) for e in current if _element_text(e))
            if not text.strip():
                current = []
                current_words = 0
                return
            parent_counter += 1
            global_chunk_index += 1
            parent_type = "multimodal" if any(_modality(e) == "multimodal" for e in current) else "text"
            parent = _make_unit(current, global_chunk_index, parent_type)
            parent.update({
                "parent_id": f"parent-{parent_counter}",
                "chunk_level": "parent",
            })
            parents.append(parent)

            # Children are generated only from elements in this parent.
            child_atomic: list[dict[str, Any]] = []
            child_word_count = 0

            def flush_child() -> None:
                nonlocal child_atomic, child_word_count, child_counter, global_chunk_index
                if not child_atomic:
                    return
                pieces: list[dict[str, Any]] = []
                for element in child_atomic:
                    txt = _element_text(element)
                    words = txt.split()
                    if len(words) <= child_size:
                        pieces.append({"text": txt, "element": element})
                    else:
                        for piece in _word_windows(txt, child_size, 0):
                            pieces.append({"text": piece, "element": element})
                for piece in pieces:
                    global_chunk_index += 1
                    child_counter += 1
                    parent_id = parent["parent_id"]
                    element = piece["element"]
                    child_type = "multimodal" if _modality(element) == "multimodal" else "text"
                    child = {
                        "chunk_index": global_chunk_index,
                        "chunk_type": child_type,
                        "chunk_level": "child",
                        "parent_id": parent_id,
                        "child_id": f"child-{child_counter}",
                        "content": normalize_whitespace(piece["text"]),
                        "page": int(element.get("page", 1)),
                        "section_path": list(element.get("section_path") or []),
                        "element_ids": [str(element.get("id", ""))] if element.get("id") else [],
                        "evidence": {
                            "element_count": 1,
                            "pages": [int(element.get("page", 1))],
                            "modalities": [_modality(element)],
                        },
                    }
                    children.append(child)
                child_atomic = []
                child_word_count = 0

            # Group elements into child-sized retrieval units.
            for element in current:
                txt = _element_text(element)
                if not txt:
                    continue
                words = len(txt.split())
                if child_atomic and child_word_count + words > child_size:
                    flush_child()
                child_atomic.append(element)
                child_word_count += words
            flush_child()

            current = []
            current_words = 0

        for element in section_elements:
            text = _element_text(element)
            if not text:
                continue
            words = len(text.split())
            if current and current_words + words > parent_size:
                flush_parent()
            current.append(element)
            current_words += words
        flush_parent()

    return {"parents": parents, "children": children}


# Backward-compatible name used only by older callers.
def build_baseline_chunks(document: dict[str, Any], **_: Any) -> list[dict[str, Any]]:
    result = build_parent_child_chunks(document)
    return result["children"]
