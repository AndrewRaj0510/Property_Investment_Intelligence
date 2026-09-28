"""LLM factory: provider selection and the reasoning-model knobs (F40).

The project moved from Ollama to OpenAI. Both paths are kept, so the factory is
the one place that has to know which knobs are safe for which backend - and two
of them are not cosmetic:

* a completion ceiling sized for the visible answer STARVES a reasoning model,
  because thinking tokens are spent first and count against the same ceiling.
  That is exactly how listing classification was silently disabled on Ollama
  (F39), and it reproduces on gpt-5: a 200-token ceiling returned an empty
  string.
* `format` (Ollama's JSON grammar) and `reasoning_effort` (OpenAI reasoning
  models only) are each valid on exactly one backend.

No test here makes a network call - the model object is inspected, not invoked.
"""

from __future__ import annotations

import pytest

import config
from llm.provider import _is_reasoning_model, get_chat_model


@pytest.fixture
def openai_settings(monkeypatch):
    s = config.get_settings()
    monkeypatch.setattr(s, "llm_provider", "openai", raising=False)
    monkeypatch.setattr(s, "openai_api_key", "sk-test", raising=False)
    monkeypatch.setattr(s, "openai_model", "gpt-5", raising=False)
    return s


@pytest.fixture
def ollama_settings(monkeypatch):
    s = config.get_settings()
    monkeypatch.setattr(s, "llm_provider", "ollama", raising=False)
    monkeypatch.setattr(s, "ollama_base_url", "http://localhost:11434", raising=False)
    monkeypatch.setattr(s, "ollama_model", "gpt-oss:120b-cloud", raising=False)
    return s


def test_openai_provider_builds_chatopenai(openai_settings):
    assert type(get_chat_model()).__name__ == "ChatOpenAI"


def test_ollama_provider_builds_chatollama(ollama_settings):
    assert type(get_chat_model()).__name__ == "ChatOllama"


def test_provider_match_is_case_and_space_insensitive(monkeypatch, openai_settings):
    monkeypatch.setattr(openai_settings, "llm_provider", "  OpenAI ", raising=False)
    assert config.get_settings().uses_openai


def test_reasoning_headroom_is_added_to_the_requested_ceiling(
    monkeypatch, openai_settings
):
    """The call site's `num_predict` budgets the ANSWER; thinking tokens need
    their own room on top or the response comes back empty."""
    monkeypatch.setattr(
        openai_settings, "openai_reasoning_headroom_tokens", 2000, raising=False
    )
    m = get_chat_model(num_predict=800)
    assert m.max_tokens == 2800


def test_headroom_applies_to_the_default_ceiling_too(monkeypatch, openai_settings):
    monkeypatch.setattr(
        openai_settings, "openai_reasoning_headroom_tokens", 500, raising=False
    )
    assert get_chat_model().max_tokens == 1300


def test_ollama_ceiling_is_left_alone(ollama_settings):
    """Ollama's num_predict bounds VISIBLE output only, so no headroom applies."""
    assert get_chat_model(num_predict=800).num_predict == 800


def test_ollama_format_grammar_is_not_sent_to_openai(openai_settings):
    """`format` is applied by Ollama's local inference engine. Forwarding it
    would be an unknown parameter to the OpenAI API."""
    m = get_chat_model(format={"type": "object"})
    assert "format" not in m.model_kwargs


@pytest.mark.parametrize("name", ["gpt-5", "GPT-5.2", "o1-preview", "o3-mini", "o4"])
def test_reasoning_families_are_recognised(name):
    assert _is_reasoning_model(name)


@pytest.mark.parametrize("name", ["gpt-4o", "gpt-4.1-mini", "gpt-3.5-turbo", "", None])
def test_non_reasoning_models_are_not(name):
    assert not _is_reasoning_model(name)


def test_reasoning_effort_sent_for_a_reasoning_model(monkeypatch, openai_settings):
    monkeypatch.setattr(openai_settings, "openai_reasoning_effort", "low", raising=False)
    assert get_chat_model().reasoning_effort == "low"


def test_reasoning_effort_withheld_from_a_non_reasoning_model(
    monkeypatch, openai_settings
):
    """gpt-4o rejects the parameter, so sending it would break the chain."""
    monkeypatch.setattr(openai_settings, "openai_model", "gpt-4o", raising=False)
    monkeypatch.setattr(openai_settings, "openai_reasoning_effort", "low", raising=False)
    assert get_chat_model().reasoning_effort is None


def test_reasoning_effort_can_be_disabled(monkeypatch, openai_settings):
    monkeypatch.setattr(openai_settings, "openai_reasoning_effort", "", raising=False)
    assert get_chat_model().reasoning_effort is None


def test_enrich_timeout_is_passed_through(openai_settings):
    assert get_chat_model(timeout_s=20).request_timeout == 20


def test_default_timeout_is_the_shared_request_budget(openai_settings):
    assert get_chat_model().request_timeout == config.get_settings().llm_request_timeout_s


def test_client_retries_are_disabled(openai_settings):
    """A silent retry would double an enrichment call's wall time against a
    budget the caller already chose."""
    assert get_chat_model().max_retries == 0


def test_openai_needs_a_key(monkeypatch, openai_settings):
    assert config.get_settings().llm_configured
    monkeypatch.setattr(openai_settings, "openai_api_key", "", raising=False)
    assert not config.get_settings().llm_configured


def test_ollama_needs_no_key(monkeypatch, ollama_settings):
    monkeypatch.setattr(ollama_settings, "openai_api_key", "", raising=False)
    assert config.get_settings().llm_configured
