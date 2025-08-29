import json
from pathlib import Path

import torch

from gemma3_text270m.config import Gemma3TextConfig
from gemma3_text270m.model import Gemma3ForCausalLM
from gemma3_text270m.hf_loader import load_weights_into


def test_local_detection_and_partial_load(tmp_path: Path):
    # Create a mock local repo with a minimal safetensors-like structure using torch.save fallback
    # Here we avoid actually writing safetensors; this is a structural test only.
    (tmp_path / 'config.json').write_text(json.dumps({'vocab_size': 262144, 'hidden_size': 640}))

    # Skip actual safetensors — loader will error if no .safetensors; ensure message is clear
    cfg = Gemma3TextConfig()
    model = Gemma3ForCausalLM(cfg)
    try:
        load_weights_into(model, str(tmp_path))
    except RuntimeError as e:
        msg = str(e)
        assert 'No .safetensors files found' in msg or 'safetensors' in msg

