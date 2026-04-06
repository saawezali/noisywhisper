from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

REQUIRED_MODEL_FILES = (
    "model.bin",
    "config.json",
    "tokenizer.json",
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
        output_dir = Path(tmp).joinpath("ct2_model")
        output_dir.mkdir(parents=True, exist_ok=True)

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
        converted = False
        for kwargs in conversion_attempts:
            try:
                converter.convert(str(output_dir), **kwargs)
                converted = True
                break
            except TypeError as exc:
                last_error = exc
                continue
            except Exception as exc:
                last_error = exc
                break

        if not converted:
            raise RuntimeError(
                f"CTranslate2 conversion failed: {last_error or 'unknown error'}"
            )

        _copy_optional_files(model_dir, output_dir)

        for item in output_dir.iterdir():
            dst = model_dir.joinpath(item.name)
            if item.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(item, dst)
            else:
                shutil.copy2(item, dst)

    logger.info("Model conversion finished at %s", model_dir)


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

    if has_transformers_checkpoint(path):
        try:
            log.info("Converting existing Transformers checkpoint to CTranslate2 at %s", path)
            _convert_transformers_to_ct2(path, log, quantization=quantization)
            if is_valid_ct2_model_dir(path):
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
            f"Expected CTranslate2 files in: {path}"
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

            if has_transformers_checkpoint(path):
                log.info(
                    "Downloaded Transformers checkpoint from %s. Converting to CTranslate2...",
                    repo,
                )
                _convert_transformers_to_ct2(path, log, quantization=quantization)
                if is_valid_ct2_model_dir(path):
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
