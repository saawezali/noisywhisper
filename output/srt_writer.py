from __future__ import annotations

from pathlib import Path

from core.transcribe import TranscriptSegment


def _ms_to_srt_time(ms: int) -> str:
    ms = max(0, ms)
    hours = ms // 3_600_000
    ms -= hours * 3_600_000
    minutes = ms // 60_000
    ms -= minutes * 60_000
    seconds = ms // 1_000
    millis = ms - seconds * 1_000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _wrap_text(text: str, max_chars: int = 42) -> str:
    words = text.strip().split()
    if not words:
        return ""

    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return "\n".join(lines)


def write_srt(segments: list[TranscriptSegment], output_path: str) -> Path:
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    blocks = []
    index = 1
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue

        start_ts = _ms_to_srt_time(seg.start_ms)
        end_ts = _ms_to_srt_time(seg.end_ms)
        wrapped = _wrap_text(text)
        blocks.append(f"{index}\n{start_ts} --> {end_ts}\n{wrapped}\n")
        index += 1

    with target.open("w", encoding="utf-8") as f:
        f.write("\n".join(blocks).strip() + "\n")

    return target
