from __future__ import annotations

"""
HuggingFace weight loading with automatic key mapping.

This module provides utilities to download (via huggingface_hub) and load
weights (via safetensors) into Gemma3ForCausalLM with automatic key mapping
between HuggingFace and local naming conventions. It supports:

- Single file: model.safetensors
- Sharded: model.safetensors.index.json + model-00001-of-000NN.safetensors
- Automatic key mapping from HF patterns to local model structure
- Shape validation with automatic transpose detection
- Comprehensive loading reports with mapping statistics

Usage::
    model = Gemma3ForCausalLM(config)
    report = load_weights_into(model, "google/gemma-3-270m")
    print(report.loaded_tensors)
"""

import json
import logging
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Pattern, Tuple, Union
from .model import Gemma3ForCausalLM
import torch

try:
    from safetensors.torch import load_file
except Exception as e:  # pragma: no cover
    load_file = None  # type: ignore

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except Exception as e:  # pragma: no cover
    hf_hub_download = None  # type: ignore
    snapshot_download = None  # type: ignore

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# HF key regex patterns for mapping hf weight keys to local weight keys
HF_TO_LOCAL_PATTERNS: List[Tuple[Pattern[str], str]] = [
    # Embeddings
    (re.compile(r"^model\.embed_tokens\.(.+)$"), r"embed_tokens.\1"),
    # Attention projections
    (
        re.compile(r"^model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.(.+)$"),
        r"layers.\1.attention.\2.\3",
    ),
    # MLP projections
    (
        re.compile(r"^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.(.+)$"),
        r"layers.\1.mlp.\2.\3",
    ),
    # Layer norms
    (re.compile(r"^model\.layers\.(\d+)\.input_layernorm\.(.+)$"), r"layers.\1.input_norm.\2"),
    (
        re.compile(r"^model\.layers\.(\d+)\.post_attention_layernorm\.(.+)$"),
        r"layers.\1.post_norm.\2",
    ),
    # Final norm
    (re.compile(r"^model\.norm\.(.+)$"), r"norm.\1"),
    # LM head (tied with embeddings locally)
    (re.compile(r"^lm_head\.(.+)$"), r"lm_head.\1"),
    # Fallback: drop leading "model." if present
    (re.compile(r"^model\.(.+)$"), r"\1"),
]

TRANSPOSE_PATTERNS: List[Pattern[str]] = []

@dataclass
class LoadReport:
    repo_or_path: str
    files: List[str]
    loaded_tensors: int
    missing_keys: List[str]
    unexpected_keys: List[str]
    total_params: int
    shape_mismatches: List[str]
    mapping_stats: Dict[str, Union[int, float]]
    transformations: Dict[str, List[str]]

def _map_hf_key_to_local(hf_key: str) -> Tuple[str, bool]:
    """Return the local key for an HF weight key along with mapping flag."""
    for pattern, replacement in HF_TO_LOCAL_PATTERNS:
        if pattern.match(hf_key):
            local_key = pattern.sub(replacement, hf_key)
            logger.debug("Mapped %s -> %s", hf_key, local_key)
            return local_key, True

    logger.warning("No mapping rule for key %s", hf_key)
    return hf_key, False


def _needs_transpose(key: str, hf_shape: torch.Size, local_shape: torch.Size) -> bool:
    """Detect whether a tensor should be transposed to match the local shape."""

    if tuple(hf_shape) == tuple(local_shape):
        return False

    if len(hf_shape) == 2 and len(local_shape) == 2:
        reversed_match = tuple(hf_shape) == tuple(reversed(local_shape))
        if reversed_match:
            return True
        for pattern in TRANSPOSE_PATTERNS:
            if pattern.match(key) and reversed_match:
                return True

    return False


def _apply_weight_transforms(
    key: str,
    tensor: torch.Tensor,
    target_shape: torch.Size,
) -> Tuple[torch.Tensor, List[str]]:
    """Apply necessary transforms (currently transpose) to match target shape."""

    applied: List[str] = []
    updated = tensor

    if _needs_transpose(key, updated.shape, target_shape):
        updated = updated.transpose(0, 1).contiguous()
        applied.append("transpose")

    return updated, applied


def _map_state_dict(
    model_state: Mapping[str, torch.Tensor],
    hf_state: Mapping[str, torch.Tensor],
) -> Tuple[
    Dict[str, torch.Tensor],
    List[str],
    List[str],
    Counter,
    Dict[str, List[str]],
    List[str],
]:
    """
    Map HF keys to local keys, applying transforms where necessary.
    Returns mapped weights, missing keys, unexpected HF keys, mapping stats,
    transformation log, and shape mismatch descriptions.
    """
    mapped: Dict[str, torch.Tensor] = {}
    unexpected: List[str] = []
    shape_mismatches: List[str] = []
    transformations: Dict[str, List[str]] = defaultdict(list)
    stats: Counter = Counter()

    for hf_key, tensor in hf_state.items():
        stats["total_hf_keys"] += 1
        local_key, was_mapped = _map_hf_key_to_local(hf_key)
        if local_key == hf_key:
            stats["identity"] += 1
        if was_mapped and local_key != hf_key:
            stats["remapped"] += 1

        if local_key not in model_state:
            unexpected.append(hf_key)
            continue

        target_tensor = model_state[local_key]
        transformed, applied = _apply_weight_transforms(local_key, tensor, target_tensor.shape)

        if tuple(transformed.shape) != tuple(target_tensor.shape):
            shape_mismatches.append(
                f"{local_key}: expected {tuple(target_tensor.shape)}, got {tuple(transformed.shape)} (from {hf_key})"
            )
            continue

        if applied:
            transformations[local_key].extend(applied)

        mapped[local_key] = transformed
        stats["successfully_mapped"] += 1

    missing = [k for k in model_state.keys() if k not in mapped]

    return mapped, missing, unexpected, stats, dict(transformations), shape_mismatches


def _format_strict_error(
    missing: List[str], unexpected: List[str], shape_mismatches: List[str]
) -> str:
    """Human-friendly strict-mode error description."""

    parts: List[str] = []
    if missing:
        parts.append("Missing keys: " + ", ".join(sorted(missing)))
    if unexpected:
        parts.append("Unexpected HF keys: " + ", ".join(sorted(unexpected)))
    if shape_mismatches:
        parts.append("Shape mismatches: " + "; ".join(shape_mismatches))
    return "\n".join(parts)


def _is_local_repo(path: str | Path) -> bool:
    p = Path(path)
    return p.exists() and p.is_dir()


def _detect_files(repo_or_path: str | Path) -> Tuple[Path, List[Path]]:
    """
    Return (root_dir, list_of_safetensor_files). If remote, download snapshot.
    Raises RuntimeError if required files are not present or download tools missing.
    """
    if _is_local_repo(repo_or_path):
        root = Path(repo_or_path)
    else:
        if snapshot_download is None:
            raise RuntimeError("huggingface_hub is required for remote repositories")
        # mirror to local cache; allow offline envs to skip
        root = Path(
            snapshot_download(repo_id=str(repo_or_path), allow_patterns=["*.safetensors", "*.json"])
        )

    # Prefer sharded files if index exists
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        try:
            idx = json.loads(index_path.read_text())
            # file_map is dict param_name -> filename
            shard_files = sorted({f for f in idx.get("weight_map", {}).values()})
            files = [root / f for f in shard_files]
        except Exception as e:
            raise RuntimeError(f"Invalid safetensors index at {index_path}: {e}")
    else:
        single = root / "model.safetensors"
        if not single.exists():
            # fallback: any .safetensors in root
            files = list(root.glob("*.safetensors"))
            if not files:
                raise RuntimeError(f"No .safetensors files found under {root}")
            # if multiple, keep deterministic order
            files.sort()
        else:
            files = [single]

    return root, files


def _load_safetensors(files: Iterable[Path]) -> Dict[str, torch.Tensor]:
    if load_file is None:
        raise RuntimeError("safetensors is required to load weights")
    state: Dict[str, torch.Tensor] = {}
    for f in files:
        # Loading the safetensors into torch format...
        shard = load_file(str(f))
        # merge; later shards can overwrite if duplicate (unlikely in proper index)
        for k, v in shard.items():
            state[k] = v
    return state


def load_weights_into(
    model: Gemma3ForCausalLM,
    repo_or_path: str | Path,
    *,
    strict: bool = False,
    device: Optional[torch.device] = None,
) -> LoadReport:
    """Load HF safetensors into the given model.
    - repo_or_path: local dir with safetensors or HF repo id (e.g., 'google/gemma-3-270m')
    - strict: if True, raise on any missing/unexpected keys or shape mismatches
    - device: optional device to move loaded tensors before assignment
    """
    root, files = _detect_files(repo_or_path)
    weights = _load_safetensors(files)

    # Optional device placement
    if device is not None:
        weights = {k: v.to(device) for k, v in weights.items()}

    model_state = model.state_dict()
    mapped, missing, unexpected, stats, transformations, shape_mismatches = _map_state_dict(
        model_state, weights
    )

    missing_sorted = sorted(set(missing))
    unexpected_sorted = sorted(set(unexpected))

    if strict and (missing_sorted or unexpected_sorted or shape_mismatches):
        raise RuntimeError(_format_strict_error(missing_sorted, unexpected_sorted, shape_mismatches))

    # Load intersecting tensors (non-strict mode ignores reported missing)
    model.load_state_dict(mapped, strict=False)

    total = int(stats.get("total_hf_keys", 0))
    loaded = int(stats.get("successfully_mapped", 0))
    remapped = int(stats.get("remapped", 0))
    identity = int(stats.get("identity", 0))
    unmapped = max(total - loaded, 0)
    transposed = sum(1 for ops in transformations.values() for op in ops if op == "transpose")
    mapping_stats: Dict[str, Union[int, float]] = {
        "total_hf_keys": total,
        "successfully_mapped": loaded,
        "remapped": remapped,
        "identity": identity,
        "unmapped": unmapped,
        "transposed": transposed,
        "mapping_rate": float(loaded / total) if total else 0.0,
    }

    return LoadReport(
        repo_or_path=str(repo_or_path),
        files=[str(f) for f in files],
        loaded_tensors=loaded,
        missing_keys=missing_sorted,
        unexpected_keys=unexpected_sorted,
        total_params=sum(p.numel() for p in model.parameters()),
        shape_mismatches=shape_mismatches,
        mapping_stats=mapping_stats,
        transformations=transformations,
    )


__all__ = ["load_weights_into", "LoadReport"]
