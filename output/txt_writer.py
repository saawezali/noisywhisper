from __future__ import annotations

from pathlib import Path
from typing import Iterable


def _segment_text(seg: object) -> str:
    if isinstance(seg, dict):
        return str(seg.get("text") or "").strip()
    return str(getattr(seg, "text", "") or "").strip()


def write_txt(segments: Iterable[object], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for seg in segments:
            text = _segment_text(seg)
            if text:
                f.write(text + "\n")
    return output_path
