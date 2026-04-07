from dataclasses import dataclass
from pathlib import Path

from output.json_writer import write_json
from output.srt_writer import write_srt
from output.txt_writer import write_txt


@dataclass
class Segment:
    start_ms: int
    end_ms: int
    text: str


def test_text_srt_json_writers(tmp_path: Path) -> None:
    segments = [
        Segment(start_ms=1000, end_ms=2300, text="Merhaba dunya"),
        Segment(start_ms=3000, end_ms=5100, text="Nasılsın"),
    ]

    txt_path = write_txt(segments, tmp_path / "out.txt")
    srt_path = write_srt(segments, tmp_path / "out.srt")
    json_path = write_json(segments, tmp_path / "out.json")

    txt = txt_path.read_text(encoding="utf-8")
    srt = srt_path.read_text(encoding="utf-8")
    payload = json_path.read_text(encoding="utf-8")

    assert "Merhaba dunya" in txt
    assert "Nasılsın" in txt
    assert "00:00:01,000 --> 00:00:02,300" in srt
    assert "00:00:03,000 --> 00:00:05,100" in srt
    assert '"start_ms": 1000' in payload
    assert '"text": "Nasılsın"' in payload
