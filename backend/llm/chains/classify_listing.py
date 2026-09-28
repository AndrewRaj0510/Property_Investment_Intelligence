"""Listing classification chain (M2-04) - live-only, Ollama.

Extracts finish_tier, view_type and amenity_score from a listing description.

**Why this is not a plain ``with_structured_output`` call (F19).** Ollama applies
its JSON-schema ``format`` constraint in the LOCAL inference engine. A server
that proxies to a hosted backend (Ollama's ``*-cloud`` model tags) silently
ignores it: measured against ``gpt-oss:20b-cloud``, even
``{"capital": {"type":"string"}}`` with ``required`` came back as the prose
"The capital of France is **Paris**". Observed failure modes were:

  * marketing copy instead of JSON (LangChain ``with_structured_output`` raised
    ``OUTPUT_PARSING_FAILURE``),
  * valid JSON with INVENTED keys (``fit_out``/``marble`` instead of the schema's),
  * markdown bullets (``- **finish_tier**: Luxury``) with an out-of-range score
    (``9`` where the schema says 0-5).

So the schema is requested (it costs nothing and a local model honours it) but
never trusted: the response goes through a tolerant parser that accepts JSON,
fenced JSON, or labelled prose, then every field is validated against the
allowed vocabulary. Anything unrecognised yields ``None`` for that field rather
than a guess - a wrong finish tier shown to an investor is worse than a blank.
"""

from __future__ import annotations

import json
import re

import structlog
from pydantic import BaseModel, Field

from config import get_settings
from llm.prompts.amenity_score import AMENITY_SCORE_INSTRUCTIONS
from llm.prompts.finish_tier import FINISH_TIER_INSTRUCTIONS
from llm.prompts.view_type import VIEW_TYPE_INSTRUCTIONS

log = structlog.get_logger(__name__)

FINISH_TIERS = ("Basic", "Mid", "High", "Luxury")
VIEW_TYPES = ("Sea", "Skyline", "Community", "Pool", "None")


class ListingFeatures(BaseModel):
    """All fields optional: an unparseable answer must degrade per-field."""

    finish_tier: str | None = Field(default=None, description="Basic | Mid | High | Luxury")
    view_type: str | None = Field(default=None, description="Sea | Skyline | Community | Pool | None")
    amenity_score: int | None = Field(default=None, description="0-5", ge=0, le=5)


_SYSTEM = (
    "You extract structured features from a Dubai property listing.\n\n"
    f"FINISH TIER:\n{FINISH_TIER_INSTRUCTIONS}\n\n"
    f"VIEW TYPE:\n{VIEW_TYPE_INSTRUCTIONS}\n\n"
    f"AMENITY SCORE:\n{AMENITY_SCORE_INSTRUCTIONS}\n\n"
    "Respond with ONLY a JSON object, no prose, no markdown fence, no "
    "explanation, in exactly this form:\n"
    '{{"finish_tier": "Basic|Mid|High|Luxury", '
    '"view_type": "Sea|Skyline|Community|Pool|None", '
    '"amenity_score": <integer 0 to 5>}}\n'
    "Use those exact key names. amenity_score is on a 0-5 scale, never 0-10."
)

_FORMAT_SCHEMA = {
    "type": "object",
    "properties": {
        "finish_tier": {"type": "string", "enum": list(FINISH_TIERS)},
        "view_type": {"type": "string", "enum": list(VIEW_TYPES)},
        "amenity_score": {"type": "integer", "minimum": 0, "maximum": 5},
    },
    "required": ["finish_tier", "view_type", "amenity_score"],
}

_chain = None


def _get_chain():
    """Build (and cache) the classification chain."""
    global _chain
    if _chain is not None:
        return _chain
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    from llm.provider import get_chat_model

    settings = get_settings()
    extra = {"format": _FORMAT_SCHEMA} if settings.ollama_enforces_schema else {}
    llm = get_chat_model(
        temperature=0,
        num_predict=800,
        timeout_s=settings.llm_enrich_timeout_s,
        **extra,
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", _SYSTEM), ("human", "Listing description:\n{description}")]
    )
    _chain = prompt | llm | StrOutputParser()
    return _chain


def _extract_json(text: str) -> dict | None:
    """Best-effort JSON object out of a model response."""
    if not text:
        return None
    candidates = [text]
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        candidates.append(fence.group(1))
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        candidates.append(brace.group(0))
    for c in candidates:
        try:
            parsed = json.loads(c.strip())
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, TypeError):
            continue
    return None


def _extract_labelled(text: str, field: str) -> str | None:
    """Recover a field from labelled prose, e.g. ``- **finish_tier**: Luxury``.

    Covers the markdown-bullet mode a cloud-proxied model falls into when the
    schema is ignored.
    """
    m = re.search(
        rf"{field}\W{{0,4}}\s*:?\s*\**\s*([A-Za-z0-9]+)", text, re.IGNORECASE
    )
    return m.group(1) if m else None


def _norm_choice(value, allowed: tuple[str, ...]) -> str | None:
    """Match a value to the allowed vocabulary case-insensitively, else None."""
    if value is None:
        return None
    v = str(value).strip().strip("*.,").lower()
    for a in allowed:
        if v == a.lower():
            return a
    return None


def _norm_score(value) -> int | None:
    """Coerce an amenity score into 0-5, else None.

    A model that ignores the schema may answer on its own scale ("9 out of 10").
    A 0-10 answer is rescaled rather than clamped - clamping 9 to 5 would assert
    a top-of-range score the model never gave.
    """
    if value is None:
        return None
    m = re.search(r"\d+", str(value))
    if not m:
        return None
    n = int(m.group(0))
    if 0 <= n <= 5:
        return n
    if 6 <= n <= 10:
        import math

        return math.floor(n / 2 + 0.5)
    return None


def parse_features(text: str) -> ListingFeatures:
    """Parse a raw model response into validated features (never raises)."""
    from llm.provider import strip_reasoning

    text = strip_reasoning(text or "")
    data = _extract_json(text) or {}

    finish = _norm_choice(data.get("finish_tier"), FINISH_TIERS)
    view = _norm_choice(data.get("view_type"), VIEW_TYPES)
    score = _norm_score(data.get("amenity_score"))

    if finish is None:
        finish = _norm_choice(_extract_labelled(text, "finish_tier"), FINISH_TIERS)
    if view is None:
        view = _norm_choice(_extract_labelled(text, "view_type"), VIEW_TYPES)
    if score is None:
        score = _norm_score(_extract_labelled(text, "amenity_score"))

    return ListingFeatures(finish_tier=finish, view_type=view, amenity_score=score)


def classify(description: str) -> ListingFeatures:
    """Classify one listing. Requires the LLM to be configured."""
    settings = get_settings()
    if not settings.llm_configured:
        raise RuntimeError(
            "The LLM is not configured; listing classification requires "
            "OLLAMA_BASE_URL and OLLAMA_MODEL (no keyword fallback)."
        )
    from llm.provider import clamp_for_prompt

    raw = _get_chain().invoke({"description": clamp_for_prompt(description or "")})
    feats = parse_features(raw)
    if feats.finish_tier is None and feats.view_type is None:
        log.warning("listing_classify_unparsed", sample=(raw or "")[:200])
    return feats
