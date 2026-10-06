"""Generic, conservative section hierarchy detection.

The detector deliberately avoids document-specific headings. It combines:
1. numbered/lettered heading syntax,
2. typography (larger/bold text), and
3. structural words such as ABSTRACT/REFERENCES.

Formulae, chart labels, equations and metadata lines are explicitly rejected.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

from .utils import normalize_whitespace

NUMBERED_RE = re.compile(r"^(\d+(?:\.\d+){0,5})[\.)]?\s+(.+)$")
LETTERED_RE = re.compile(r"^([A-Z])[\.)]?\s+(.+)$")
STRUCTURAL = {
    "ABSTRACT",
    "INTRODUCTION",
    "CONCLUSION",
    "CONCLUSIONS",
    "REFERENCES",
    "ACKNOWLEDGEMENTS",
    "ACKNOWLEDGMENTS",
    "APPENDIX",
}
META_RE = re.compile(r"^(?:arxiv|doi:|submitted|accepted|published|copyright)\b", re.I)
DATE_RE = re.compile(r"\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{4}\b", re.I)
MATH_CHARS = set("=<>≤≥∑∫√∞→←↔∂∇±×÷^_{}[]")


def _text(e: dict[str, Any]) -> str:
    return normalize_whitespace(str(e.get("content") or e.get("text") or ""))


def _word_count(text: str) -> int:
    return len(text.split())


def _alpha_count(text: str) -> int:
    return sum(ch.isalpha() for ch in text)


def _looks_formula_or_label(text: str) -> bool:
    if not text:
        return True
    if any(ch in text for ch in MATH_CHARS):
        return True
    if DATE_RE.search(text):
        return True
    if re.search(r"\bT\s*=\s*\d+\b", text):
        return True
    if re.fullmatch(r"[\d\s.,%()\-/]+", text):
        return True
    if len(text) < 3 or len(text) > 150:
        return True
    if _alpha_count(text) < 3:
        return True
    if re.search(r"\b(?:accuracy|error|mce|fr|t5d)\s*\(?%?\)?$", text, re.I) and _word_count(text) <= 4:
        return True
    digit_ratio = sum(ch.isdigit() for ch in text) / max(1, len(text))
    if digit_ratio > 0.35:
        return True
    return False


def numbered_level(text: str) -> int | None:
    match = NUMBERED_RE.match(text)
    if not match:
        return None
    number, title = match.groups()
    if _looks_formula_or_label(title):
        return None
    if not title[0].isalpha():
        return None
    if _word_count(title) < 1:
        return None
    return number.count(".") + 1


def lettered_level(text: str) -> int | None:
    match = LETTERED_RE.match(text)
    if not match:
        return None
    title = match.group(2)
    if _looks_formula_or_label(title):
        return None
    if len(title.split()) < 2:
        return None
    return 1


def _is_structural(text: str) -> bool:
    return text.upper().rstrip(".:") in STRUCTURAL


def _page_font_baseline(elements: list[dict[str, Any]]) -> dict[int, float]:
    by_page: dict[int, list[float]] = defaultdict(list)
    for e in elements:
        size = e.get("font_size")
        if e.get("type") == "text" and isinstance(size, (int, float)) and size > 0:
            by_page[int(e.get("page", 1))].append(float(size))
    baseline = {}
    for page, sizes in by_page.items():
        sizes.sort()
        baseline[page] = sizes[len(sizes) // 2]
    return baseline


def heading_candidate_score(e: dict[str, Any], page_baseline: float | None) -> tuple[float, int]:
    text = _text(e)
    if e.get("type") not in {"text", "heading"} or _looks_formula_or_label(text):
        return 0.0, 0
    if META_RE.search(text):
        return 0.0, 0

    score = 0.0
    level = 1

    nlevel = numbered_level(text)
    if nlevel is not None:
        first_word = text.split(maxsplit=1)[1] if len(text.split(maxsplit=1)) == 2 else ""
        score += 5.0 if (not first_word or first_word[0].isupper()) else 3.0
        level = nlevel

    llevel = lettered_level(text)
    if llevel is not None and nlevel is None:
        score += 3.0
        level = llevel

    if _is_structural(text):
        score += 4.0

    font_size = e.get("font_size")
    bold = bool(e.get("bold"))
    if page_baseline and isinstance(font_size, (int, float)):
        if float(font_size) >= page_baseline * 1.10:
            score += 2.5
        elif float(font_size) >= page_baseline * 1.03:
            score += 1.0
    if bold:
        score += 1.0

    words = text.split()
    if len(words) <= 12 and text.isupper():
        score += 1.0

    # A heading should be compact. Very long prose is not a heading.
    if len(words) > 18:
        score -= 3.0

    return score, level


def build_section_hierarchy(document: dict[str, Any]) -> dict[str, Any]:
    elements = document.get("elements", [])
    baselines = _page_font_baseline(elements)
    stack: list[tuple[int, str]] = []
    sections: list[dict[str, Any]] = []
    section_lookup: dict[str, dict[str, Any]] = {}
    root_id = "root"

    for e in elements:
        e["section_id"] = root_id
        e["section_path"] = []

    for idx, e in enumerate(elements):
        text = _text(e)
        score, level = heading_candidate_score(e, baselines.get(int(e.get("page", 1))))
        is_heading = score >= 4.5
        if not is_heading:
            continue

        section_id = f"section-{len(sections) + 1}"
        while stack and stack[-1][0] >= level:
            stack.pop()
        parent_id = stack[-1][1] if stack else root_id
        path = [section_lookup[s[1]]["title"] for s in stack]
        section = {
            "id": section_id,
            "title": text,
            "level": level,
            "parent_id": parent_id,
            "element_index": idx,
            "page": int(e.get("page", 1)),
            "path": path + [text],
            "score": round(score, 2),
        }
        sections.append(section)
        section_lookup[section_id] = section
        stack.append((level, section_id))

        e["section_id"] = section_id
        e["section_path"] = section["path"]

    # Attach each non-heading element to the most recent open section.
    active_stack: list[dict[str, Any]] = []
    section_by_element = {s["element_index"]: s for s in sections}
    for idx, e in enumerate(elements):
        if idx in section_by_element:
            active_stack = [s for s in active_stack if s["level"] < section_by_element[idx]["level"]]
            active_stack.append(section_by_element[idx])
        if active_stack:
            e["section_id"] = active_stack[-1]["id"]
            e["section_path"] = active_stack[-1]["path"]

    return {
        "schema_version": 2,
        "source": document.get("source", {}),
        "elements": elements,
        "sections": sections,
        "stats": {
            **document.get("stats", {}),
            "sections": len(sections),
        },
    }
