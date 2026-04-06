from __future__ import annotations

import logging
from importlib import metadata

import numpy as np


def _build_dependency_hint(import_error: Exception) -> str:
    message = str(import_error)
    if "torchaudio.backend" in message:
        try:
            torch_ver = metadata.version("torch")
        except Exception:
            torch_ver = "not-installed"

        try:
            ta_ver = metadata.version("torchaudio")
        except Exception:
            ta_ver = "not-installed"

        return (
            "DeepFilterNet import failed because torchaudio backend APIs are missing. "
            f"Detected torch={torch_ver}, torchaudio={ta_ver}. "
            "Use matching versions, for example: torch>=2.2,<2.6 and "
            "torchaudio>=2.2,<2.6"
        )

    return message


def denoise_audio(
    audio: np.ndarray,
    sample_rate: int,
    enabled: bool = True,
    logger: logging.Logger | None = None,
) -> np.ndarray:
    """
    Apply DeepFilterNet denoising when available.

    If DeepFilterNet is not installed or fails at runtime, input audio is returned
    unchanged and a warning is logged.
    """
    if not enabled:
        return audio

    if audio.size == 0:
        return audio

    log = logger or logging.getLogger(__name__)

    try:
        from df.enhance import enhance, init_df
    except Exception as exc:
        hint = _build_dependency_hint(exc)
        log.warning("DeepFilterNet not available, skipping denoise: %s", hint)
        return audio

    try:
        model, df_state, _ = init_df()

        # DeepFilterNet APIs differ across versions. Try ndarray first.
        enhanced = enhance(model, df_state, audio)

        if hasattr(enhanced, "cpu"):
            enhanced = enhanced.cpu().numpy()

        cleaned = np.asarray(enhanced, dtype=np.float32).reshape(-1)
        if cleaned.size == 0:
            log.warning("DeepFilterNet returned empty audio, falling back to source")
            return audio

        return cleaned
    except Exception as first_exc:
        # Fallback path for versions that require torch tensors.
        try:
            import torch

            model, df_state, _ = init_df()
            # Create a writable copy to avoid torch warnings on read-only buffers.
            writable_audio = np.array(audio, dtype=np.float32, copy=True)
            tensor_audio = torch.from_numpy(writable_audio).float().unsqueeze(0)
            enhanced = enhance(model, df_state, tensor_audio)
            if hasattr(enhanced, "cpu"):
                enhanced = enhanced.cpu().numpy()
            cleaned = np.asarray(enhanced, dtype=np.float32).reshape(-1)
            if cleaned.size == 0:
                return audio
            return cleaned
        except Exception as second_exc:
            log.warning(
                "DeepFilterNet failed; skipping denoise. first=%s second=%s",
                first_exc,
                second_exc,
            )
            return audio
