from __future__ import annotations

from pathlib import Path
from typing import Iterable

from utils.timefmt import format_srt_timestamp


def write_srt(segments: Iterable[dict], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            text = (seg.get("text") or "").strip()

            f.write(f"{i}\n")
            f.write(f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n")
            f.write(text + "\n\n")
    return output_path
