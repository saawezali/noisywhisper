from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Iterable


def _normalize_segment(seg: object) -> dict:
    if isinstance(seg, dict):
        return seg
    if is_dataclass(seg):
        return asdict(seg)
    return {
        "text": str(getattr(seg, "text", "") or ""),
        "start_ms": int(getattr(seg, "start_ms", 0) or 0),
        "end_ms": int(getattr(seg, "end_ms", 0) or 0),
    }


def write_json(segments: Iterable[object], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "segments": [_normalize_segment(seg) for seg in segments],
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return output_path
