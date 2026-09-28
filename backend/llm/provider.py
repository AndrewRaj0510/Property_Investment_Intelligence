"""Shared LLM factory - OpenAI or Ollama (F19, F40).

Every chain builds its model here so the provider, model name, budget and
timeout are configured in exactly one place. Switching provider is an env
change (``LLM_PROVIDER``), never a code change.

Two providers are supported on purpose:

* **openai** (default) - a hosted model, selected with ``OPENAI_MODEL``. This is
  what the app runs on now.
* **ollama** - self-hosted, no API key. It is what the project was originally
  built and measured on (F18/F19) and remains the zero-cost offline path.

**Configuration gates differ.** OpenAI needs a secret, so "configured" means
"has a key". Ollama is reachable-or-not instead, and that can only be known by
calling it. ``Settings.llm_configured`` handles both; each chain keeps its
existing failure policy either way (the /score narrative degrades to None,
/explain raises - F18).
"""

from __future__ import annotations

import structlog

from config import get_settings

log = structlog.get_logger(__name__)


def get_chat_model(temperature: float = 0.0, timeout_s: int | None = None, **kwargs):
    """Build a chat model bound to the configured provider.

    ``temperature`` is passed per-call because the chains differ deliberately:
    narration uses a little warmth, extraction and Q&A use 0 for
    reproducibility.

    ``timeout_s`` overrides the default request timeout. The optional /score
    enrichment passes a much shorter budget than a user-initiated call, because
    it must never hold a thread on work the caller has already abandoned.

    ``num_predict`` is the call site's ceiling on generated tokens. The name is
    Ollama's; the OpenAI branch maps it onto ``max_completion_tokens`` so no
    chain has to know which provider is active.
    """
    s = get_settings()
    if s.uses_openai:
        return _openai_model(s, temperature, timeout_s, **kwargs)
    return _ollama_model(s, temperature, timeout_s, **kwargs)


def _openai_model(s, temperature: float, timeout_s: int | None, **kwargs):
    from langchain_openai import ChatOpenAI

    kwargs.pop("format", None)

    requested = kwargs.pop("num_predict", 800)
    opts = {
        "model": s.openai_model,
        "api_key": s.openai_api_key,
        "temperature": temperature,
        "max_completion_tokens": requested + s.openai_reasoning_headroom_tokens,
        "timeout": timeout_s or s.llm_request_timeout_s,
        "max_retries": 0,
    }
    effort = (s.openai_reasoning_effort or "").strip()
    if effort and _is_reasoning_model(s.openai_model):
        opts["reasoning_effort"] = effort
    if s.openai_base_url:
        opts["base_url"] = s.openai_base_url
    opts.update(kwargs)
    return ChatOpenAI(**opts)


def _is_reasoning_model(name: str) -> bool:
    """True for OpenAI model families that think before answering.

    These are the families that accept `reasoning_effort` and that spend
    completion tokens on reasoning - both facts the factory has to act on.
    """
    return (name or "").strip().lower().startswith(("gpt-5", "o1", "o3", "o4"))


def _ollama_model(s, temperature: float, timeout_s: int | None, **kwargs):
    from langchain_ollama import ChatOllama

    opts = {
        "model": s.ollama_model,
        "base_url": s.ollama_base_url,
        "temperature": temperature,
        "num_predict": kwargs.pop("num_predict", 800),
        "client_kwargs": {"timeout": timeout_s or s.llm_request_timeout_s},
    }
    if s.ollama_api_key:
        opts["client_kwargs"]["headers"] = {
            "Authorization": f"Bearer {s.ollama_api_key}"
        }
    opts.update(kwargs)
    return ChatOllama(**opts)


def clamp_for_prompt(value):
    """Bound a value's size before it is placed into a prompt.

    ``/explain`` accepts a CLIENT-SUPPLIED analysis, so the field whitelist in
    ``answer_question.prune_context`` controls *which* fields are sent but not
    how large they are - an unbounded ``title`` measured ~176k tokens in a single
    request during review. Rate limiting caps requests per minute, not tokens per
    request, so the bound has to be here.

    Truncation is marked in-band so the model is never silently handed a
    half-sentence and left to infer the rest.
    """
    s = get_settings()
    if isinstance(value, str):
        if len(value) > s.llm_max_field_chars:
            return value[: s.llm_max_field_chars] + " …[truncated]"
        return value
    if isinstance(value, list):
        clamped = [clamp_for_prompt(v) for v in value[: s.llm_max_list_items]]
        if len(value) > s.llm_max_list_items:
            clamped.append(f"…[{len(value) - s.llm_max_list_items} more omitted]")
        return clamped
    if isinstance(value, dict):
        return {k: clamp_for_prompt(v) for k, v in value.items()}
    return value


def strip_reasoning(text: str) -> str:
    """Remove chain-of-thought markup some models emit around their answer.

    Reasoning models (gpt-oss, deepseek-r1, qwen3) wrap internal deliberation in
    ``<think>`` tags. Ollama usually splits that into a separate `thinking`
    field, and the OpenAI API never exposes it at all, but when the tags survive
    into the content they must not reach the user - a narrative that opens with
    the model debating itself is worse than no narrative at all.
    """
    import re

    if not text:
        return ""
    cleaned = re.sub(
        r"<(think|thinking|reasoning)>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE
    )
    cleaned = re.sub(
        r"<(think|thinking|reasoning)>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE
    )
    return cleaned.strip()


_EM_DASHES = "\u2014\u2015"
_EN_DASHES = "\u2013\u2012\u2212"
_ALL_DASHES = _EM_DASHES + _EN_DASHES


def strip_dashes(text: str) -> str:
    """Rewrite em and en dashes out of model prose, preserving the sentence.

    The prompts also forbid them, but a prompt is a request and this is the
    guarantee: models reach for an em dash constantly in explanatory prose, and
    one that slips through lands in the investor-facing narrative on the results
    screen, where the rest of the copy has none.
    """
    import re

    if not text:
        return ""
    cleaned = re.sub(rf"^[ 	]*[{_ALL_DASHES}][ 	]+", "", text, flags=re.MULTILINE)
    cleaned = re.sub(rf"[ 	]*[{_EM_DASHES}][ 	]*", " - ", cleaned)
    cleaned = re.sub(rf"[{_EN_DASHES}]", "-", cleaned)
    return cleaned.strip()


def clean_output(raw: str) -> str:
    """The full cleanup every user-facing model response passes through.

    Reasoning markup first (it can contain dashes of its own that would
    otherwise be rewritten before being deleted), then dash normalisation.
    """
    return strip_dashes(strip_reasoning(raw))
