from utils.timefmt import format_srt_timestamp


def test_format_srt_timestamp_zero() -> None:
    assert format_srt_timestamp(0.0) == "00:00:00,000"


def test_format_srt_timestamp_rounding() -> None:
    assert format_srt_timestamp(3661.789) == "01:01:01,789"
