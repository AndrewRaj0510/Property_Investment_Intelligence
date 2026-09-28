"""POST /api/v1/explain - grounded Q&A over an existing score result (M3D-05).

The "ask about your analysis" endpoint. The user supplies a question plus the
``/score`` result they are looking at; the LLM answers strictly from that JSON
and refuses when the answer is not in it (see llm/prompts/qa_grounded.py).

Deliberately NOT a chat endpoint: there is no conversation history and no tool
access, so each answer is a pure function of one analysis and one question. That
keeps the hallucination surface to what the prompt can be audited against.
"""

from __future__ import annotations

import asyncio

import structlog
from fastapi import APIRouter, HTTPException, Request

from api.rate_limit import limiter
from api.schemas.explain import ExplainRequest, ExplainResponse
from llm.chains.answer_question import answer

log = structlog.get_logger(__name__)
router = APIRouter()


@router.post("/explain", response_model=ExplainResponse, tags=["explainability"])
@limiter.limit("20/minute")
async def explain_analysis(request: Request, req: ExplainRequest) -> ExplainResponse:
    context = req.context.model_dump()
    if req.price_history is not None:
        context["price_history"] = req.price_history.model_dump()
    try:
        text = await asyncio.to_thread(answer, context, req.question)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        log.warning("explain_unconfigured", error=str(exc))
        raise HTTPException(
            status_code=503,
            detail="Q&A requires an LLM; set OLLAMA_BASE_URL and OLLAMA_MODEL.",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        log.error("explain_failed", error=str(exc))
        raise HTTPException(
            status_code=502, detail="Could not generate an answer. Please retry."
        ) from exc

    log.info("explain_answered", question_chars=len(req.question))
    return ExplainResponse(answer=text, question=req.question)
