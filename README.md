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
