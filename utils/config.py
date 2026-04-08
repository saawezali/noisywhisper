from __future__ import annotations

import configparser
from pathlib import Path


DEFAULT_CONFIG = {
    "transcription": {
        "noise_reduction": "true",
        "beam_size": "5",
        "compute_type": "int8",
        "vad_threshold": "0.5",
        "model_path": "models/turkish-noisy-v1",
        "model_repo_id": "Cosmobillian/turkish_whisper_for_noisy_datas",
        "fallback_model_repo_id": "Systran/faster-whisper-large-v3",
        "auto_download_model": "false",
    },
    "export": {
        "default_formats": "txt",
        "export_dir": "outputs",
    },
    "logging": {
        "level": "INFO",
        "file": "noisywhisper.log",
        "max_mb": "5",
        "backup_count": "3",
    },
}


def load_config(config_path: str = "config.ini") -> configparser.ConfigParser:
    path = Path(config_path).expanduser().resolve()
    parser = configparser.ConfigParser()
    parser.read_dict(DEFAULT_CONFIG)

    if path.exists():
        parser.read(path, encoding="utf-8")
    else:
        save_config(parser, str(path))

    return parser


def save_config(config: configparser.ConfigParser, config_path: str = "config.ini") -> Path:
    path = Path(config_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        config.write(f)
    return path
