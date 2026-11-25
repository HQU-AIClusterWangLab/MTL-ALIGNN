from __future__ import annotations

from pathlib import Path

import torch


def _extract_state(payload):
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        if payload and all(isinstance(key, str) for key in payload):
            return payload
    raise ValueError("Unsupported pretrained checkpoint format.")


def load_alignn_backbone(model, checkpoint_path: str | Path) -> dict:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if not hasattr(model.encoder, "backbone"):
        raise ValueError("Pretrained ALIGNN weights require an ALIGNN encoder.")
    state = _extract_state(torch.load(path, map_location="cpu"))
    cleaned = {}
    prefixes = (
        "module.encoder.backbone.",
        "encoder.backbone.",
        "module.backbone.",
        "backbone.",
        "module.",
    )
    target_keys = set(model.encoder.backbone.state_dict())
    for key, value in state.items():
        candidate = key
        for prefix in prefixes:
            if candidate.startswith(prefix):
                candidate = candidate[len(prefix) :]
                break
        if (
            candidate in target_keys
            and hasattr(value, "shape")
            and tuple(value.shape)
            == tuple(model.encoder.backbone.state_dict()[candidate].shape)
        ):
            cleaned[candidate] = value
    if not cleaned:
        raise RuntimeError(
            "No compatible ALIGNN backbone parameters were found in the checkpoint."
        )
    result = model.encoder.backbone.load_state_dict(cleaned, strict=False)
    return {
        "checkpoint": str(path),
        "loaded_parameter_tensors": len(cleaned),
        "target_parameter_tensors": len(target_keys),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }
