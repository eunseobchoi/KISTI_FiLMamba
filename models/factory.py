from __future__ import annotations

from dataclasses import fields
from typing import Any, Dict, Type

from filmamba.models.filmamba import FiLMamba, ModelConfig
from filmamba.models.hifi_mamba import HiFiMamba, HiFiModelConfig
from filmamba.models.unrolled3d import StrongUnrolled3D, StrongUnrolled3DConfig


def _filter_for_dataclass(cfg: Dict[str, Any], cls: Type) -> Dict[str, Any]:
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in cfg.items() if k in valid}


def normalize_model_family(cfg: Dict[str, Any]) -> str:
    return str(cfg.get("model_family", "filmamba")).strip().lower()


def build_model_from_config(cfg: Dict[str, Any]):
    family = normalize_model_family(cfg)
    if family in {"filmamba", "ours"}:
        return FiLMamba(ModelConfig(**_filter_for_dataclass(cfg, ModelConfig)))
    if family in {"hifi", "hifimamba", "hifi_mamba"}:
        return HiFiMamba(HiFiModelConfig(**_filter_for_dataclass(cfg, HiFiModelConfig)))
    if family in {"unrolled3d", "unrolled3d_strong", "strong3d", "3d_unrolled_strong"}:
        return StrongUnrolled3D(StrongUnrolled3DConfig(**_filter_for_dataclass(cfg, StrongUnrolled3DConfig)))
    raise ValueError(
        f"Unsupported model_family={cfg.get('model_family')}; use one of "
        f"['filmamba', 'hifi_mamba', 'unrolled3d_strong']"
    )
