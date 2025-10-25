"""
LLM adapter layer for Gemini/OpenAI with local stubs.

API:
 - generate_text(prompt, max_tokens=200) -> str
 - embed_text(text) -> List[float]

This adapter tries to import official SDKs if available. If not, it uses a deterministic stub (safe for unit tests).
"""
from typing import List
import os
import time
import random
from src.config import API_PROVIDER, API_KEY
from src.utils import log_event


def _with_retries(fn, attempts: int = 3, initial_delay: float = 0.5):
    delay = initial_delay
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            log_event(f'Attempt {i+1}/{attempts} failed: {e}', 'DEBUG')
            if i < attempts - 1:
                # jittered exponential backoff
                time.sleep(delay + random.random() * 0.1)
                delay *= 2
    raise last_exc


def generate_text(prompt: str, max_tokens: int = 200, temperature: float = 0.0) -> str:
    provider = API_PROVIDER
    if provider == 'gemini':
        try:
            def _call():
                import google.generativeai as genai
                genai.configure(api_key=API_KEY)
                model = os.getenv('LLM_MODEL', 'models/text-bison-001')
                return genai.generate(model=model, input=prompt, max_output_tokens=max_tokens)

            resp = _with_retries(_call, attempts=int(os.getenv('GEMINI_RETRIES', '3')))
            # resp may be a proto or dict depending on SDK; try multiple access patterns
            if hasattr(resp, 'text'):
                return resp.text
            if isinstance(resp, dict) and 'candidates' in resp and resp['candidates']:
                return resp['candidates'][0].get('content', '')
            if isinstance(resp, dict) and 'output' in resp:
                return resp['output']
            return str(resp)
        except Exception as e:
            log_event(f'Gemini generation fallback after retries: {e}', 'WARN')
    elif provider == 'openai':
        try:
            import openai
            openai.api_key = API_KEY
            resp = openai.ChatCompletion.create(
                model=os.getenv('LLM_MODEL', 'gpt-3.5-turbo'),
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp['choices'][0]['message']['content']
        except Exception as e:
            log_event(f'OpenAI generation fallback: {e}', 'WARN')

    # Fallback deterministic stub
    log_event('Using local stub for generate_text', 'INFO')
    return f"[STUB GENERATED] {prompt[:200]}"


def embed_text(text: str) -> List[float]:
    provider = API_PROVIDER
    if provider == 'gemini':
        try:
            def _call_embed():
                import google.generativeai as genai
                genai.configure(api_key=API_KEY)
                model = os.getenv('EMBEDDING_MODEL', 'models/embed-text-embedding-3-small')
                return genai.embeddings.create(model=model, input=[text])

            resp = _with_retries(_call_embed, attempts=int(os.getenv('GEMINI_RETRIES', '3')))
            # Try to extract embedding vector
            if isinstance(resp, dict) and 'data' in resp and resp['data']:
                return resp['data'][0].get('embedding')
            if hasattr(resp, 'embeddings') and resp.embeddings:
                return resp.embeddings[0].embedding
            return list(resp)
        except Exception as e:
            log_event(f'Gemini embed fallback after retries: {e}', 'WARN')
    elif provider == 'openai':
        try:
            import openai
            openai.api_key = API_KEY
            resp = openai.Embedding.create(model=os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'), input=text)
            return resp['data'][0]['embedding']
        except Exception as e:
            log_event(f'OpenAI embedding fallback: {e}', 'WARN')

    # deterministic stub: simple hash-based vector
    log_event('Using local stub for embed_text', 'INFO')
    h = abs(hash(text))
    # produce small fixed-dim vector
    vec = [float((h >> (i * 8)) & 0xFF) / 255.0 for i in range(16)]
    return vec
