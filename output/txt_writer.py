from __future__ import annotations

from pathlib import Path
from typing import Iterable


def write_txt(segments: Iterable[dict], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for seg in segments:
            text = (seg.get("text") or "").strip()
            if text:
                f.write(text + "\n")
    return output_path
