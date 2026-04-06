from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logger(
    log_file: str = "noisywhisper.log",
    level: str = "INFO",
    max_mb: int = 5,
    backup_count: int = 3,
) -> logging.Logger:
    logger = logging.getLogger("noisywhisper")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    if logger.handlers:
        return logger

    path = Path(log_file).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    rotating_handler = RotatingFileHandler(
        path,
        maxBytes=max_mb * 1024 * 1024,
        backupCount=backup_count,
        encoding="utf-8",
    )
    rotating_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(rotating_handler)
    logger.addHandler(stream_handler)

    return logger
