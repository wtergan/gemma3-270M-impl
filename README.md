# Gemma-3-270M Implementation

An educational, from-scratch implementation scaffold of the Gemma-3-270M text model.
The goal is clarity over peak performance: a compact codebase that demonstrates
modern LLM architecture patterns without depending on the Transformers library.

## Features (incremental)
- Modular components: config, tokenizer, attention (MQA), MLP, KV cache, blocks, model
- Direct HuggingFace weight loading via `safetensors` + `huggingface_hub` (no Transformers)
- Clean generation utilities (prefill + decode) with sampling
- Tests designed for readability and learning

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

# 1) Build config + model
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

# 4) Generate text
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
 tests/
   __init__.py
 pyproject.toml
 setup.py
 requirements.txt
 README.md
 LICENSE
```

## Development
- Format: `uv run black .`
- Type check: `uv run mypy gemma3_text270m/`
- Test: `uv run pytest -q`

## License
MIT
