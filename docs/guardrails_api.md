# Safety Guardrails API

This module provides three safety checks: hallucination detection, dosage validation, and interaction checking.

Usage:

from src.guardrails import get_guardrails

guardrails = get_guardrails()

# Run checks
response_data = {
  'confidence': 0.92,
  'citations': [{'text': 'Source: FDA Label'}],
  'response': 'Metformin is used to treat type 2 diabetes.'
}
chunks = [{'text': 'Metformin indicated for type 2 diabetes.'}] * 8
result = guardrails.run_all_safety_checks(response_data, 'What is metformin used for?', chunks)

See code docstrings for details on inputs/outputs.


### LLM adapter

A small adapter `src/llm_adapter.py` is provided to centralize calls to Gemini or OpenAI. It exposes `generate_text(prompt)` and `embed_text(text)`.

By default the adapter will attempt to use the provider set in `src/config.py` via `GEMINI_API_KEY` or `OPENAI_API_KEY`. If the provider SDK is not installed, the adapter falls back to a deterministic local stub suitable for tests.

Example usage:

from src.llm_adapter import generate_text, embed_text
text = generate_text('Summarize X')
vec = embed_text('some text')

## Gemini (Google) SDK setup

If you want the adapter to use Google's Gemini models (recommended for production), install the official SDK and set your key in the environment.

1. Install the SDK (example):

```powershell
pip install --upgrade google-generative-ai
```

2. Set your API key in the environment (Windows PowerShell example):

```powershell
$env:GEMINI_API_KEY = 'your_gemini_api_key_here'
```

3. Configure optional models via environment variables:

- `LLM_MODEL` — the text generation model name (default: `models/text-bison-001`)
- `EMBEDDING_MODEL` — the embeddings model name (default: `models/embed-text-embedding-3-small`)

4. How the adapter uses the SDK:

- The adapter will attempt to `import google.generativeai as genai` and call `genai.generate()` for text and `genai.embeddings.create()` for embeddings. If the SDK is not available or raises an error the adapter falls back to a local deterministic stub (useful for unit tests and CI).

5. Example (PowerShell) run after install and env var set:

```powershell
$env:GEMINI_API_KEY = 'YOUR_KEY'
python -c "from src.llm_adapter import generate_text; print(generate_text('Say hi'))"
```

Notes and troubleshooting:

- If the SDK changes its API surface, update `src/llm_adapter.py` accordingly. The adapter uses a few tolerant access patterns (checking `resp.text`, `resp['candidates']`, and `resp['output']`) to adapt to different SDK return formats.
- For secure deployments, prefer setting the key in a secrets manager or system-wide environment variable rather than committing it in code.

### Auto-suggestion (safe rephrase)

The `run_all_safety_checks()` function accepts an optional `generate_suggestion` boolean flag. When set to `True` and the overall status is not `SAFE`, the guardrails will call the configured LLM adapter to generate a concise safe rephrasing or user-facing warning.

Example:

from src.guardrails import get_guardrails
g = get_guardrails(retriever)
result = g.run_all_safety_checks(response_data, query, chunks, generate_suggestion=True)
if result['suggestion']:
    print(result['suggestion'])

Note: Generating suggestions requires the LLM adapter to be configured with a provider key (Gemini/OpenAI). The adapter will fall back to a deterministic stub if no provider SDK/key is available.

### CI skeleton

Create `.github/workflows/ci.yml` with a minimal job that runs unit tests (example below). This file is a template and may require customization for your CI environment.

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Run tests
        run: |
          pytest -q evaluation
```

