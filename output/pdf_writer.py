from __future__ import annotations

from datetime import datetime
from pathlib import Path

from core.transcribe import TranscriptSegment


def write_pdf(
    segments: list[TranscriptSegment],
    output_path: str,
    title: str = "NoisyWhisper Transcript",
) -> Path:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas
    except Exception as exc:
        raise RuntimeError(
            "reportlab is required for PDF export. Install dependencies first."
        ) from exc

    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    pdf = canvas.Canvas(str(target), pagesize=A4)
    width, height = A4

    left = 18 * mm
    right = width - 18 * mm
    y = height - 20 * mm

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(left, y, title)
    y -= 8 * mm

    pdf.setFont("Helvetica", 9)
    pdf.drawString(left, y, datetime.now().strftime("Generated: %Y-%m-%d %H:%M:%S"))
    y -= 10 * mm

    pdf.setFont("Helvetica", 10)
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue

        start_s = seg.start_ms // 1000
        hh = start_s // 3600
        mm = (start_s % 3600) // 60
        ss = start_s % 60
        line = f"[{hh:02d}:{mm:02d}:{ss:02d}] {text}"

        max_chars = 105
        chunks = [line[i : i + max_chars] for i in range(0, len(line), max_chars)]
        for chunk in chunks:
            if y < 20 * mm:
                pdf.showPage()
                y = height - 20 * mm
                pdf.setFont("Helvetica", 10)
            pdf.drawString(left, y, chunk)
            y -= 6 * mm

    pdf.save()
    return target
