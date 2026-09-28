"""SHAP → plain-language explanation chain (M3D-03) - live-only.

Uses GPT-4o-mini to compose a 2-sentence narrative from the top SHAP drivers.
Requires OpenAI to be configured; there is no template fallback.
"""

from __future__ import annotations

import structlog

from config import get_settings
from llm.prompts.shap_narrative import SHAP_NARRATIVE_INSTRUCTIONS

log = structlog.get_logger(__name__)

_chain = None


def _get_chain():
    global _chain
    if _chain is not None:
        return _chain
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    from llm.provider import get_chat_model

    llm = get_chat_model(temperature=0.3)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SHAP_NARRATIVE_INSTRUCTIONS),
            (
                "human",
                "Community: {community}\n"
                "Predicted yield: {predicted_yield}%\n"
                "Community median yield: {community_median}%\n"
                "Top drivers: {top_drivers}",
            ),
        ]
    )
    _chain = prompt | llm | StrOutputParser()
    return _chain


def explain(
    community: str,
    predicted_yield: float,
    community_median: float | None,
    top_drivers: list[dict],
) -> str:
    settings = get_settings()
    if not settings.llm_configured:
        raise RuntimeError(
            "The LLM is not configured; SHAP narrative generation requires "
            "OLLAMA_BASE_URL and OLLAMA_MODEL (no template fallback)."
        )
    from llm.provider import clean_output

    drivers_text = "; ".join(d["label"] for d in top_drivers[:3])
    raw = _get_chain().invoke(
        {
            "community": community,
            "predicted_yield": predicted_yield,
            "community_median": community_median if community_median is not None else "n/a",
            "top_drivers": drivers_text,
        }
    )
    return clean_output(raw)
