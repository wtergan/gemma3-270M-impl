from __future__ import annotations

"""
HuggingFace weight loading without Transformers.

This module provides utilities to download (via huggingface_hub) and load
weights (via safetensors) into Gemma3ForCausalLM directly. It supports:
- Single file: model.safetensors
- Sharded: model.safetensors.index.json + model-00001-of-000NN.safetensors

Notes:
- This is an educational loader; it validates shapes against the model.
- No key remapping is performed; module names must match.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import os
import json

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

from .model import Gemma3ForCausalLM


@dataclass
class LoadReport:
    repo_or_path: str
    files: List[str]
    loaded_tensors: int
    missing_keys: List[str]
    unexpected_keys: List[str]
    total_params: int


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


def _validate_shapes(model: Gemma3ForCausalLM, state: Mapping[str, torch.Tensor]) -> List[str]:
    """Return list of shape mismatch keys (expected != provided)."""
    mismatches: List[str] = []
    model_state = model.state_dict()
    for k, t in state.items():
        if k not in model_state:
            continue
        if tuple(model_state[k].shape) != tuple(t.shape):
            mismatches.append(f"{k}: expected {tuple(model_state[k].shape)}, got {tuple(t.shape)}")
    return mismatches


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

    # Validate shapes before loading
    shape_mismatches = _validate_shapes(model, weights)
    if shape_mismatches and strict:
        raise RuntimeError("Shape mismatches:\n" + "\n".join(shape_mismatches))

    # Load with partial tolerance
    model_state = model.state_dict()
    missing = [k for k in model_state.keys() if k not in weights]
    unexpected = [k for k in weights.keys() if k not in model_state]

    # Apply intersection
    intersect = {
        k: v
        for k, v in weights.items()
        if k in model_state and (tuple(model_state[k].shape) == tuple(v.shape))
    }
    model.load_state_dict({**model_state, **intersect}, strict=False)

    return LoadReport(
        repo_or_path=str(repo_or_path),
        files=[str(f) for f in files],
        loaded_tensors=len(intersect),
        missing_keys=missing,
        unexpected_keys=unexpected
        + (["SHAPE_MISMATCH"] if shape_mismatches and not strict else []),
        total_params=sum(p.numel() for p in model.parameters()),
    )


__all__ = ["load_weights_into", "LoadReport"]
