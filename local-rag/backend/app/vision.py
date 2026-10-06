"""Local multimodal enrichment through Ollama.

OCR and visual interpretation are stored as secondary evidence. The original
extracted content is never overwritten or treated as generated truth.
"""

import base64
from pathlib import Path
from typing import Any

import requests

from .config import OLLAMA_BASE_URL, OLLAMA_TIMEOUT, OCR_MODEL, VISION_MODEL
from .utils import normalize_whitespace


def _image_b64(path: str) -> str:
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode("ascii")


def ollama_image_call(model: str, prompt: str, image_path: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [_image_b64(image_path)],
            }
        ],
        "stream": False,
        "options": {"temperature": 0},
    }
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=OLLAMA_TIMEOUT,
    )
    response.raise_for_status()
    body: dict[str, Any] = response.json()
    return normalize_whitespace(
        str(body.get("message", {}).get("content", ""))
    )


def enrich_visual(image_path: str, run_ocr: bool = True, run_vision: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {
        "image_path": image_path,
        "ocr_text": "",
        "vision_description": "",
        "models": {},
        "errors": [],
    }

    if run_ocr:
        try:
            result["ocr_text"] = ollama_image_call(
                OCR_MODEL,
                "Extract all readable text from this image. Preserve labels, numbers, table cells, axis labels, legends and headings. Return only the extracted text.",
                image_path,
            )
            result["models"]["ocr"] = OCR_MODEL
        except Exception as exc:
            result["errors"].append({"stage": "ocr", "error": f"{type(exc).__name__}: {exc}"})

    if run_vision:
        try:
            result["vision_description"] = ollama_image_call(
                VISION_MODEL,
                "Describe this figure, chart, diagram, or visual for retrieval. Include what it represents, important entities, trends, relationships, labels and any values that are visually evident. Do not invent values that cannot be read.",
                image_path,
            )
            result["models"]["vision"] = VISION_MODEL
        except Exception as exc:
            result["errors"].append({"stage": "vision", "error": f"{type(exc).__name__}: {exc}"})

    return result
