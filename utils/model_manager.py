from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path

REQUIRED_MODEL_FILES = (
    "model.bin",
    "config.json",
    "tokenizer.json",
)

HF_CONFIG_CANDIDATES = (
    "config.json",
    "config.hf.original.json",
)

TRANSFORMERS_WEIGHT_FILES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)

OPTIONAL_COPY_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocabulary.json",
    "vocab.json",
    "merges.txt",
    "normalizer.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "preprocessor_config.json",
)

TRANSFORMERS_SHARD_PATTERNS = (
    "model-*-of-*.safetensors",
    "pytorch_model-*-of-*.bin",
)


def is_valid_ct2_model_dir(model_dir: str | Path) -> bool:
    path = Path(model_dir).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        return False

    return all(path.joinpath(name).exists() for name in REQUIRED_MODEL_FILES)


def has_transformers_checkpoint(model_dir: str | Path) -> bool:
    path = Path(model_dir).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        return False
    return any(path.joinpath(name).exists() for name in TRANSFORMERS_WEIGHT_FILES)


def _read_json_dict(path: Path) -> dict | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return raw if isinstance(raw, dict) else None


def _resolve_hf_config(path: Path) -> tuple[Path | None, dict | None]:
    for name in HF_CONFIG_CANDIDATES:
        candidate = path.joinpath(name)
        parsed = _read_json_dict(candidate)
        if parsed is None:
            continue
        if parsed.get("model_type") == "whisper":
            return candidate, parsed
    return None, None


def is_valid_hf_whisper_model_dir(model_dir: str | Path) -> bool:
    path = Path(model_dir).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        return False

    has_weights = has_transformers_checkpoint(path)
    if not has_weights:
        return False

    _cfg_path, cfg = _resolve_hf_config(path)
    return cfg is not None


def _download_snapshot(repo_id: str, target_dir: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(
            "huggingface-hub is required for auto model download. "
            "Install with: pip install huggingface-hub"
        ) from exc

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
    )


def _copy_optional_files(source_dir: Path, destination_dir: Path) -> None:
    for name in OPTIONAL_COPY_FILES:
        src = source_dir.joinpath(name)
        if not src.exists() or not src.is_file():
            continue
        shutil.copy2(src, destination_dir.joinpath(name))


def _convert_transformers_to_ct2(
    model_dir: Path,
    logger: logging.Logger,
    quantization: str = "int8",
) -> None:
    try:
        from ctranslate2.converters import TransformersConverter
    except Exception as exc:
        raise RuntimeError(
            "ctranslate2 converter is unavailable. Install/verify ctranslate2 and transformers."
        ) from exc

    converter = TransformersConverter(str(model_dir))

    with tempfile.TemporaryDirectory(prefix="ct2_convert_", dir=str(model_dir.parent)) as tmp:
        conversion_attempts = [
            {
                "quantization": quantization,
                "copy_files": [
                    name
                    for name in OPTIONAL_COPY_FILES
                    if model_dir.joinpath(name).exists()
                ],
                "force": True,
            },
            {
                "quantization": quantization,
                "copy_files": [
                    name
                    for name in OPTIONAL_COPY_FILES
                    if model_dir.joinpath(name).exists()
                ],
            },
            {
                "quantization": quantization,
            },
        ]

        last_error: Exception | None = None
        converted_output_dir: Path | None = None
        for attempt_idx, kwargs in enumerate(conversion_attempts, start=1):
            output_dir = Path(tmp).joinpath(f"ct2_model_{attempt_idx}")
            if output_dir.exists():
                shutil.rmtree(output_dir)

            try:
                converter.convert(str(output_dir), **kwargs)
                converted_output_dir = output_dir
                break
            except TypeError as exc:
                last_error = exc
                continue
            except Exception as exc:
                last_error = exc
                break

        if converted_output_dir is None:
            raise RuntimeError(
                f"CTranslate2 conversion failed: {last_error or 'unknown error'}"
            )

        _copy_optional_files(model_dir, converted_output_dir)

        for item in converted_output_dir.iterdir():
            dst = model_dir.joinpath(item.name)
            if item.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(item, dst)
            else:
                shutil.copy2(item, dst)

    logger.info("Model conversion finished at %s", model_dir)


def _prune_large_transformers_artifacts(model_dir: Path, logger: logging.Logger) -> None:
    removed: list[str] = []

    for name in TRANSFORMERS_WEIGHT_FILES:
        file_path = model_dir.joinpath(name)
        if file_path.exists() and file_path.is_file():
            file_path.unlink(missing_ok=True)
            removed.append(file_path.name)

    for pattern in TRANSFORMERS_SHARD_PATTERNS:
        for shard in model_dir.glob(pattern):
            if shard.is_file():
                shard.unlink(missing_ok=True)
                removed.append(shard.name)

    # Remove stale runtime clones produced by previous transformer backend runs.
    for pattern in ("hf_runtime_*", "tmp_hf_source*", "tmp_ct2_test*"):
        for sibling in model_dir.parent.glob(pattern):
            if sibling == model_dir:
                continue
            if sibling.exists() and sibling.is_dir():
                shutil.rmtree(sibling, ignore_errors=True)
                removed.append(sibling.name)

    if removed:
        logger.info("Pruned large model artifacts: %s", sorted(set(removed)))


def ensure_model_available(
    model_dir: str,
    primary_repo_id: str | None,
    auto_download: bool,
    logger: logging.Logger | None = None,
    fallback_repo_id: str | None = None,
    quantization: str = "int8",
) -> Path:
    """
    Ensure a local CTranslate2 model directory exists and is complete.

    Returns:
        Resolved model directory path.
    """
    log = logger or logging.getLogger(__name__)
    path = Path(model_dir).expanduser().resolve()

    if is_valid_ct2_model_dir(path):
        return path

    if is_valid_hf_whisper_model_dir(path):
        try:
            log.info("Converting local Hugging Face Whisper model to CTranslate2 at %s", path)
            _convert_transformers_to_ct2(path, log, quantization=quantization)
            if is_valid_ct2_model_dir(path):
                _prune_large_transformers_artifacts(path, log)
                return path
        except Exception as exc:
            log.warning(
                "Local Hugging Face model conversion failed; using transformers checkpoint directly: %s",
                exc,
            )
        log.info("Using local Hugging Face Whisper model at %s", path)
        return path

    if has_transformers_checkpoint(path):
        try:
            log.info("Converting existing Transformers checkpoint to CTranslate2 at %s", path)
            _convert_transformers_to_ct2(path, log, quantization=quantization)
            if is_valid_ct2_model_dir(path):
                _prune_large_transformers_artifacts(path, log)
                return path
        except Exception as exc:
            log.warning("Local model conversion attempt failed: %s", exc)

    if path.exists() and path.is_dir():
        present = [p.name for p in path.iterdir() if p.is_file()]
        log.warning(
            "Model directory exists but looks incomplete: %s (files=%s)",
            path,
            present,
        )

    if not auto_download:
        raise FileNotFoundError(
            "Model directory not found or incomplete. "
            f"Expected either a CTranslate2 model or a Hugging Face Whisper checkpoint at: {path}"
        )

    path.mkdir(parents=True, exist_ok=True)

    attempted_repos: list[str] = []
    if primary_repo_id:
        attempted_repos.append(primary_repo_id)
    if fallback_repo_id and fallback_repo_id not in attempted_repos:
        attempted_repos.append(fallback_repo_id)

    if not attempted_repos:
        raise FileNotFoundError(
            "No model repository configured for auto-download and local model is missing."
        )

    download_errors: list[str] = []
    for repo in attempted_repos:
        try:
            log.info("Downloading model snapshot from %s into %s", repo, path)
            _download_snapshot(repo, path)
            if is_valid_ct2_model_dir(path):
                log.info("Model ready at %s", path)
                return path

            if is_valid_hf_whisper_model_dir(path):
                log.info("Hugging Face Whisper model ready at %s", path)
                return path

            if has_transformers_checkpoint(path):
                log.info(
                    "Downloaded Transformers checkpoint from %s. Converting to CTranslate2...",
                    repo,
                )
                _convert_transformers_to_ct2(path, log, quantization=quantization)
                if is_valid_ct2_model_dir(path):
                    _prune_large_transformers_artifacts(path, log)
                    log.info("Model converted and ready at %s", path)
                    return path

            download_errors.append(
                f"{repo}: download/conversion completed but required files missing"
            )
            log.warning(
                "Downloaded %s but required CTranslate2 files are still missing in %s",
                repo,
                path,
            )
        except Exception as exc:
            download_errors.append(f"{repo}: {exc}")
            log.warning("Model download failed for %s: %s", repo, exc)

    detail = " | ".join(download_errors) if download_errors else "unknown error"
    raise RuntimeError(
        "Unable to prepare model automatically. "
        f"Target={path}. Attempts={detail}"
    )
