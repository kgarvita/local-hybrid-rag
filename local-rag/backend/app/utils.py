import hashlib
import json
import re
from pathlib import Path
from typing import Any


def log(message: str = "") -> None:
    print(message, flush=True)


def banner(title: str) -> None:
    log()
    log("=" * 72)
    log(title)
    log("=" * 72)


def sanitize_text(text: str) -> str:
    return (text or "").replace("\x00", "").strip()


def normalize_whitespace(text: str) -> str:
    text = sanitize_text(text)
    return re.sub(r"\s+", " ", text)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_first(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default
