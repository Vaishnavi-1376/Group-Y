import os
from importlib import reload

import pytest


def _set_provider(env):
    # Ensure the src package reads updated env vars on import
    os.environ.pop('GEMINI_API_KEY', None)
    os.environ.pop('OPENAI_API_KEY', None)
    if env == 'gemini':
        os.environ['GEMINI_API_KEY'] = 'test'
    elif env == 'openai':
        os.environ['OPENAI_API_KEY'] = 'test'


def test_generate_text_and_embed_stub_gemini(monkeypatch):
    _set_provider('gemini')
    # reload config and adapter to pick up env change
    import src.config as config
    reload(config)
    import src.llm_adapter as adapter
    reload(adapter)

    txt = adapter.generate_text('hello world')
    assert isinstance(txt, str)
    assert txt.startswith('[STUB') or len(txt) > 0

    vec = adapter.embed_text('hello world')
    assert isinstance(vec, list)
    assert len(vec) >= 8


def test_generate_text_and_embed_stub_openai(monkeypatch):
    _set_provider('openai')
    import src.config as config
    reload(config)
    import src.llm_adapter as adapter
    reload(adapter)

    txt = adapter.generate_text('another prompt')
    assert isinstance(txt, str)
    assert txt.startswith('[STUB') or len(txt) > 0

    vec = adapter.embed_text('another prompt')
    assert isinstance(vec, list)
    assert len(vec) >= 8
