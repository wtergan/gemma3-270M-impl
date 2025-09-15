# Gemma-3-270M Implementation

An educational, from-scratch implementation scaffold of the Gemma-3-270M text model.
The goal is clarity over peak performance: a compact codebase that demonstrates
modern LLM architecture patterns without depending on the Transformers library.

## Features (incremental)
- Modular components: config, tokenizer, attention (MQA), MLP, KV cache, blocks, model
- Direct HuggingFace weight loading via `safetensors` + `huggingface_hub` (no Transformers)
- Clean generation utilities (prefill + decode) with sampling
- Tests designed for readability and learning

### Recent updates (2025-09-15)
- Added `context_length=32,768` and generator now enforces it by windowing inputs.
- Default `sliding_window` set to `512` (local attention span).
- Split RoPE bases: `rope_local_theta=10_000.0` (sliding/local) and `rope_theta=1_000_000.0` (global).
  - Global/full attention layers use `rope_theta`.
  - Sliding/local attention layers use `rope_local_theta`.

## Install (uv recommended)
```bash
# Create virtual environment
uv venv
# Activate
source .venv/bin/activate  # PowerShell: .venv\\Scripts\\Activate.ps1
# Add core deps
uv add torch safetensors huggingface_hub sentencepiece tokenizers
```

Alternatively with pip:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick check
```bash
python -c "import gemma3_text270m as g; print(g.__version__)"
```

## Configuration

`Gemma3TextConfig` is locked to the 270M variant for core architecture fields, but exposes runtime knobs:

- `context_length` (int, default `32768`): maximum tokens processed; generator crops inputs to last `context_length` tokens.
- `sliding_window` (int, default `512`): local attention span for sliding layers and default ring‑buffer KV capacity.
- `rope_theta` (float, default `1e6`): RoPE base used by global/full attention layers.
- `rope_local_theta` (float, default `1e4`): RoPE base used by sliding/local attention layers.

Override examples:

```python
from gemma3_text270m import Gemma3TextConfig

# Programmatic override
cfg = Gemma3TextConfig(context_length=16384, sliding_window=256, rope_theta=1e6, rope_local_theta=1e4)

# Or via HF-style dict (ignored keys are safe)
hf_cfg = {
    "vocab_size": 262_144,
    "hidden_size": 640,
    "intermediate_size": 2_048,
    "num_hidden_layers": 18,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "head_dim": 256,
    "context_length": 32768,
    "sliding_window": 512,
    "rope_theta": 1_000_000.0,
    "rope_local_theta": 10_000.0,
}
cfg = Gemma3TextConfig.from_hf_dict(hf_cfg)
```

## Weight Loading & Generation

Below is a minimal, end‑to‑end example that instantiates the model, loads
weights (local directory or HuggingFace repo), and generates text.

Prereqs in your env: `torch`, `safetensors`, `huggingface_hub`, and a tokenizer
backend (`sentencepiece` or `tokenizers`). Some HF repos require authentication.

```python
import torch
from gemma3_text270m import (
    Gemma3TextConfig,
    Gemma3ForCausalLM,
    Gemma3Tokenizer,
    Gemma3Generator,
    load_weights_into,
)

# 1) Build config + model (defaults shown under “Configuration”)
cfg = Gemma3TextConfig()
model = Gemma3ForCausalLM(cfg).eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 2a) Load weights from a local directory containing *.safetensors
# repo_or_path could be something like "/path/to/checkpoint_dir"
# load_weights_into(model, repo_or_path="/path/to/checkpoint_dir", device=device)

# 2b) Or load from a HuggingFace repo (requires huggingface_hub + access)
# Example repo id placeholder; replace with your actual 270M checkpoint repo
# load_weights_into(model, repo_or_path="google/gemma-3-270m", device=device)

# 3) Prepare tokenizer (from local path or HF)
# tok = Gemma3Tokenizer.from_local_path("/path/to/tokenizer.json")
tok = Gemma3Tokenizer()  # simple fallback for quick smoke; prefer real tokenizer

# 4) Generate text (generator respects cfg.context_length)
gen = Gemma3Generator(model, tok, device=device)
text = gen.generate(
    "Hello, I'm a small LLM",
    max_new_tokens=32,
    temperature=0.0,  # greedy; >0 enables sampling
)
print(text)
```

Tips:
- For HF loading behind a gated license, authenticate via `huggingface-cli login`.
- Generation is intentionally simple and recomputes attention each step (educational baseline). KV‑cache integration can be added later for speed.

## Project structure
```
 gemma3_text270m/
   __init__.py
   attention.py
   block.py
   config.py
   generate.py
   hf_loader.py
   kvcache.py
   mlp.py
   model.py
   tokenizer.py
 tests/
   ...
 .vault/
 .kanónes/
 pyproject.toml
 uv.lock
 README.md
 LICENSE
```

### Attention pattern
- 18 layers with a 5:1 sliding:global pattern repeated 3×.
- Sliding layers use `sliding_window` and `rope_local_theta`; global layers use full causal attention and `rope_theta`.

## Development
- Format: `uv run black .`
- Type check: `uv run mypy gemma3_text270m/`
- Test: `uv run pytest -q`

## License
MIT
