from __future__ import annotations

import json
from pathlib import Path

from core.transcribe import TranscriptSegment


def write_json(
    segments: list[TranscriptSegment],
    output_path: str,
    source_file: str | None = None,
) -> Path:
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "source_file": source_file,
        "segment_count": len(segments),
        "segments": [
            {
                "text": seg.text,
                "start_ms": seg.start_ms,
                "end_ms": seg.end_ms,
                "confidence": seg.confidence,
                "words": [
                    {
                        "word": w.word,
                        "start_ms": w.start_ms,
                        "end_ms": w.end_ms,
                        "probability": w.probability,
                    }
                    for w in seg.words
                ],
            }
            for seg in segments
        ],
    }

    with target.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return target
