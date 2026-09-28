"""Request/response schemas for POST /api/v1/score (live listing scoring)."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

MAX_TEXT = 8_000
MAX_SHORT_TEXT = 500
MAX_URL_CHARS = 2_048
MAX_AMENITIES = 100

_ALLOWED_SCHEMES = ("http://", "https://")
_FIXTURE_SCHEME = "fixture://"


class ScoreRequest(BaseModel):
    url: str = Field(
        ...,
        min_length=1,
        max_length=MAX_URL_CHARS,
        description="A Bayut or Dubizzle listing detail URL.",
    )

    @field_validator("url")
    @classmethod
    def _check_scheme(cls, v: str) -> str:
        """Reject non-web schemes before the URL reaches the fetcher.

        ``detect_site`` bounds the HOST but not the SCHEME, so this is what stops
        a ``file://``/``gopher://``-style URL being handed onward. ``fixture://``
        stays valid at this layer because whether it is ACCEPTED is a deployment
        decision (``settings.fixtures_allowed``), enforced in the fetcher - a
        schema-level rejection here would make that setting unreachable.
        """
        v = v.strip()
        low = v.lower()
        if low.startswith(_FIXTURE_SCHEME) or low.startswith(_ALLOWED_SCHEMES):
            return v
        raise ValueError("URL must start with http:// or https://")


class ScoredListing(BaseModel):
    """The normalized listing as fetched from the portal (also carries the
    amenities / coordinates used for UI context, not for the yield model)."""

    source: str | None = Field(None, max_length=MAX_SHORT_TEXT)
    listing_url: str | None = Field(None, max_length=MAX_URL_CHARS)
    listing_id: str | None = Field(None, max_length=MAX_SHORT_TEXT)
    title: str | None = Field(None, max_length=MAX_TEXT)
    description: str | None = Field(None, max_length=MAX_TEXT)
    property_type: str | None = Field(None, max_length=MAX_SHORT_TEXT)
    bedrooms: int | None = Field(None, ge=0, le=100)
    bathrooms: int | None = Field(None, ge=0, le=100)
    area_sqft: float | None = Field(None, ge=0, le=10_000_000)
    price: float | None = Field(None, ge=0)
    purpose: str | None = Field(None, max_length=MAX_SHORT_TEXT)
    rent_frequency: str | None = Field(None, max_length=MAX_SHORT_TEXT)
    annual_rent: float | None = Field(None, ge=0)
    community: str | None = Field(None, max_length=MAX_SHORT_TEXT)
    latitude: float | None = Field(None, ge=-90, le=90)
    longitude: float | None = Field(None, ge=-180, le=180)
    amenities: list[str] | None = Field(None, max_length=MAX_AMENITIES)
    detail_gated: bool = False


class ModelBlock(BaseModel):
    expected_gross_yield: float | None = None
    segment_matched: bool = False
    segment_fallback: bool = False
    shap: dict[str, float] | None = None


class ScoringBlock(BaseModel):
    actual_gross_yield: float
    expected_gross_yield: float
    expected_basis: str = Field(max_length=MAX_SHORT_TEXT)
    annual_rent: float
    sale_price: float
    rent_basis: str = Field(max_length=MAX_SHORT_TEXT)
    price_basis: str = Field(max_length=MAX_SHORT_TEXT)
    yield_vs_expected_pp: float


class Verdict(BaseModel):
    recommend: bool
    label: str = Field(max_length=MAX_SHORT_TEXT)
    reason: str = Field(max_length=MAX_TEXT)
    confidence: str = Field(max_length=MAX_SHORT_TEXT)


class BaselineBlock(BaseModel):
    """Recorded DLD/Ejari figures for this (community, property type) bucket.

    The medians are what the verdict is computed against. The percentiles and
    maxima below describe the SPREAD of the same bucket: they are not used in
    scoring at all, and exist so the grounded Q&A can answer questions about
    range ("what is the highest rent achieved here?") from records instead of
    refusing. All are optional, because a snapshot refreshed before they were
    added simply will not carry them.
    """

    area: str = Field(max_length=MAX_SHORT_TEXT)
    property_type: str = Field(max_length=MAX_SHORT_TEXT)
    median_gross_yield: float
    median_sale_price: float
    median_annual_rent: float
    n_sales: int
    n_rents: int
    low_sample: bool

    p25_sale_price: float | None = None
    p75_sale_price: float | None = None
    p90_sale_price: float | None = None
    max_sale_price: float | None = None
    p25_annual_rent: float | None = None
    p75_annual_rent: float | None = None
    p90_annual_rent: float | None = None
    max_annual_rent: float | None = None


class ListingCharacter(BaseModel):
    """LLM-extracted qualitative attributes of the pasted listing.

    CONTEXT ONLY - these are not features of the yield model (the DLD/Ejari
    training data has no finish/view columns). They describe how the unit is
    presented, which can explain an asking price that sits away from the segment
    median, but they never enter the yield computation.
    """

    finish_tier: str | None = Field(None, max_length=MAX_SHORT_TEXT)
    view_type: str | None = Field(None, max_length=MAX_SHORT_TEXT)
    amenity_score: int | None = Field(None, ge=0, le=5)


class NarrativePoint(BaseModel):
    """One specific finding in the structured narrative."""

    label: str = Field("", max_length=MAX_SHORT_TEXT)
    detail: str = Field(max_length=MAX_TEXT)


class NarrativeDetail(BaseModel):
    """The investor-readable explanation, in three parts.

    A single paragraph carrying five or six figures is the reading problem this
    layer exists to solve, so the model is asked for an intro, a list of
    specific findings, and a closing section on what to do next, and the UI is
    given that structure rather than left to guess at paragraph breaks.

    ``action`` is concrete due diligence grounded in this analysis. It is not a
    buy/sell instruction and never a price forecast: there is no price model
    behind this platform, so a forecast would be a claim with no method behind
    it (see the prompt in llm/prompts/score_narrative.py).
    """

    intro: str = Field("", max_length=MAX_TEXT)
    points: list[NarrativePoint] = Field(default_factory=list, max_length=10)
    action: str = Field("", max_length=MAX_TEXT)


class ScoreResponse(BaseModel):
    listing: ScoredListing
    scoring: ScoringBlock
    verdict: Verdict
    model: ModelBlock | None = None
    baseline: BaselineBlock | None = None
    character: ListingCharacter | None = None
    source_mode: str = Field("live", max_length=MAX_SHORT_TEXT)
    narrative: str | None = Field(None, max_length=MAX_TEXT)
    narrative_detail: NarrativeDetail | None = None
