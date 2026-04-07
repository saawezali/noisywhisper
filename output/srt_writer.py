from __future__ import annotations

from pathlib import Path
from typing import Iterable

from utils.timefmt import format_srt_timestamp


def _segment_text(seg: object) -> str:
    if isinstance(seg, dict):
        return str(seg.get("text") or "").strip()
    return str(getattr(seg, "text", "") or "").strip()


def _segment_start_seconds(seg: object) -> float:
    if isinstance(seg, dict):
        if "start" in seg:
            return float(seg.get("start", 0.0))
        return float(seg.get("start_ms", 0)) / 1000.0
    if hasattr(seg, "start_ms"):
        return float(getattr(seg, "start_ms", 0)) / 1000.0
    return float(getattr(seg, "start", 0.0))


def _segment_end_seconds(seg: object) -> float:
    if isinstance(seg, dict):
        if "end" in seg:
            return float(seg.get("end", 0.0))
        return float(seg.get("end_ms", 0)) / 1000.0
    if hasattr(seg, "end_ms"):
        return float(getattr(seg, "end_ms", 0)) / 1000.0
    return float(getattr(seg, "end", 0.0))


def write_srt(segments: Iterable[object], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = _segment_start_seconds(seg)
            end = _segment_end_seconds(seg)
            text = _segment_text(seg)

            f.write(f"{i}\n")
            f.write(f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n")
            f.write(text + "\n\n")
    return output_path
