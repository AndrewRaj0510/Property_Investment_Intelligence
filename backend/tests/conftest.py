"""Shared pytest fixtures (OG-03).

**Tests must declare the configuration they exercise.** Settings are loaded from
the developer's real ``.env``, so a test that asserts "fails closed when
unconfigured" silently inverts once those credentials exist - which is exactly
what happened to the security and DLD suites after the JWT and DDA creds went
live (F20). The ``unconfigured_*`` fixtures below force the state under test
instead of inheriting it, so the suite means the same thing on every machine.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import config
from main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


def _blank(monkeypatch, *names: str) -> None:
    """Clear settings fields on the cached singleton (monkeypatch reverts them)."""
    s = config.get_settings()
    for n in names:
        monkeypatch.setattr(s, n, "", raising=False)


@pytest.fixture
def unconfigured_auth(monkeypatch):
    """Force the fail-closed auth path regardless of the developer's .env.

    ``jwt_configured`` requires all three of these, so clearing them is what the
    "no auth configured" deployment actually looks like.
    """
    _blank(monkeypatch, "jwt_secret", "admin_username", "admin_password_hash")
    assert not config.get_settings().jwt_configured
    return config.get_settings()


@pytest.fixture
def unconfigured_dld(monkeypatch):
    """Force the live-only DLD refusal path regardless of the developer's .env."""
    _blank(
        monkeypatch,
        "dld_client_id",
        "dld_client_secret",
        "dld_security_identifier",
    )
    assert not config.get_settings().dld_configured
    return config.get_settings()


@pytest.fixture
def unconfigured_llm(monkeypatch):
    """Force the "no LLM" path regardless of provider or the developer's .env.

    Blanks BOTH providers' credentials: `llm_configured` reads the OpenAI key
    when LLM_PROVIDER=openai and the Ollama URL+model otherwise, so clearing
    only one leaves the assertion depending on which provider happens to be
    selected on this machine.
    """
    _blank(monkeypatch, "openai_api_key", "ollama_base_url", "ollama_model")
    assert not config.get_settings().llm_configured
    return config.get_settings()


@pytest.fixture
def configured_llm(monkeypatch):
    """Make `llm_configured` true WITHOUT depending on real credentials.

    Pins the provider to Ollama at a local URL, so a test that forgets to stub
    its chain fails by connection refusal rather than by silently spending money
    against the developer's real OpenAI key.
    """
    s = config.get_settings()
    monkeypatch.setattr(s, "llm_provider", "ollama", raising=False)
    monkeypatch.setattr(s, "ollama_base_url", "http://localhost:11434", raising=False)
    monkeypatch.setattr(s, "ollama_model", "test-model", raising=False)
    assert config.get_settings().llm_configured
    return s
