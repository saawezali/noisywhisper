from __future__ import annotations

from pathlib import Path

from core.transcribe import TranscriptSegment


def write_txt(segments: list[TranscriptSegment], output_path: str) -> Path:
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    lines = [seg.text.strip() for seg in segments if seg.text.strip()]
    text = "\n".join(lines).strip() + "\n"

    with target.open("w", encoding="utf-8") as f:
        f.write(text)

    return target
