"""Request/response schemas for POST /api/v1/explain (grounded Q&A)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from api.schemas.market import PriceHistory
from api.schemas.score import ScoreResponse

MAX_QUESTION_CHARS = 500


class ExplainRequest(BaseModel):
    """A question about a score result the client already holds.

    The analysis is echoed back rather than re-scored server-side: re-fetching
    the listing would mean another paid Apify call per question, and the user
    should be asking about the exact result they are looking at. It is validated
    against ``ScoreResponse``, so a malformed or hand-crafted context is
    rejected by Pydantic before it can reach the prompt.
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUESTION_CHARS,
        description="A question about this analysis, e.g. 'why is the confidence low?'",
    )
    context: ScoreResponse = Field(
        ..., description="The score result being asked about (from POST /score)."
    )
    price_history: PriceHistory | None = Field(
        None,
        description=(
            "The community price history shown beside this analysis "
            "(from GET /market/price-history), if the client has it."
        ),
    )


class ExplainResponse(BaseModel):
    answer: str
    question: str
