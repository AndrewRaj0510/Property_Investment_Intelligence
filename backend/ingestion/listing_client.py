"""Live listing fetcher - score ONE pasted listing URL (live-only).

The scoring flow is URL-driven: a user pastes a single property URL, we fetch
just that listing's details, and the model scores it against the area+type
yield medians in ``community_yield_medians``. This module detects the portal
from the URL, runs the right Apify actor for exactly that one listing, and
returns ONE normalized dict regardless of source.

Why Apify actors for both portals - the constraint that drove every choice here
is that the source must accept an **individual listing detail URL**:

* ``memo23/apify-bayut-scraper``  - accepts ``bayut.com/property/details-<id>.html``
* ``memo23/dubizzle-search-listing-s-scraper`` - accepts ``dubizzle.com/property/details-<id>``

Rejected alternatives (all search-only, so they cannot score a pasted URL):
RapidAPI Bayut (``uae-real-estate3``: /search-property etc.), RapidAPI Dubizzle
(search/category only, no detail endpoint), ``azzouzana`` Property Finder actor
(search-results pages only), ``abotapi/bayut-com-scraper`` (richer amenities but
search URLs only). Direct scraping of Bayut is not viable either - the site is
behind DataDome + hCaptcha.

Live-only: no synthetic fallback. Missing credentials raise
``ListingSourceNotConfigured``; an unreadable/blocked listing raises
``ListingFetchError``, so the caller tells the user plainly rather than
fabricating a score.
"""

from __future__ import annotations

import html
import re
from urllib.parse import urlparse

import httpx
import structlog

from config import get_settings

log = structlog.get_logger(__name__)

APIFY_BASE = "https://api.apify.com/v2"


class ListingSourceNotConfigured(RuntimeError):
    """Raised when APIFY_TOKEN is absent (live-only contract)."""


class ListingFetchError(RuntimeError):
    """Raised when a listing could not be read (blocked, 404, bad URL, ...)."""


class UnsupportedListingSite(ValueError):
    """Raised when the URL isn't a supported portal."""


_SITE_DOMAINS = {
    "bayut": ("bayut.com",),
    "dubizzle": ("dubizzle.com",),
}


_ACTOR_INPUT: dict[str, dict] = {
    "bayut": {"fullDetails": True},
}


def _host_matches(host: str, domain: str) -> bool:
    """True when ``host`` is exactly ``domain`` or a subdomain of it."""
    return host == domain or host.endswith(f".{domain}")


def detect_site(url: str) -> str:
    """Return 'bayut' | 'dubizzle' for a listing URL, else raise."""
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    for site, domains in _SITE_DOMAINS.items():
        if any(_host_matches(host, d) for d in domains):
            return site
    raise UnsupportedListingSite(
        f"Unsupported listing site: {host or url!r}. Supported: bayut, dubizzle."
    )


async def fetch_listing(url: str) -> dict:
    """Fetch ONE listing by URL and return a normalized property dict.

    Keys: ``source, listing_url, listing_id, title, description, property_type,
    bedrooms, bathrooms, area_sqft, price, purpose, rent_frequency,
    annual_rent, community, community_candidates, latitude, longitude,
    amenities, detail_gated``.

    Fixture short-circuit (F19): a ``fixture://<name>`` URL, or replay mode,
    returns a previously-captured listing instead of calling Apify. This is the
    ONLY bypass - everything downstream (model, baseline, verdict, LLM) runs
    unchanged, so an offline run exercises the real scoring path.
    """
    from ingestion import listing_fixtures as fx

    settings_ = get_settings()
    if fx.is_fixture_url(url):
        if not settings_.fixtures_allowed:
            raise UnsupportedListingSite(
                "Fixture URLs are not accepted by this deployment. Paste a real "
                "Bayut or Dubizzle listing URL."
            )
        return fx.load(fx.fixture_name(url))
    if settings_.replay_listings:
        log.info("listing_replay_mode", url=url)
        return fx.any_fixture()

    site = detect_site(url)
    settings = get_settings()
    actor = (
        settings.bayut_apify_actor if site == "bayut"
        else settings.dubizzle_apify_actor
    )
    item = await _run_actor_for_url(actor, url, _ACTOR_INPUT.get(site))
    normalized = (
        _normalize_bayut(item, url) if site == "bayut"
        else _normalize_dubizzle(item, url)
    )
    log.info(
        "listing_fetched",
        site=site,
        listing_id=normalized.get("listing_id"),
        community=normalized.get("community"),
    )
    return normalized


async def _run_actor_for_url(
    actor: str, url: str, extra_input: dict | None = None
) -> dict:
    """Run an Apify actor for exactly one URL; return its first dataset item.

    Uses ``run-sync-get-dataset-items`` so a single HTTP call starts the run and
    returns the dataset - no polling. ``maxItems: 1`` keeps cost to one result
    (these actors bill per result). ``extra_input`` carries per-site actor
    options (see ``_ACTOR_INPUT``).
    """
    s = get_settings()
    if not s.listings_configured:
        raise ListingSourceNotConfigured(
            "APIFY_TOKEN is not set; cannot fetch live listings."
        )
    endpoint = f"{APIFY_BASE}/acts/{actor}/run-sync-get-dataset-items"
    payload: dict = {"startUrls": [{"url": url}], "maxItems": 1}
    if extra_input:
        payload.update(extra_input)
    try:
        async with httpx.AsyncClient(timeout=s.listing_request_timeout_s) as http:
            resp = await http.post(
                endpoint,
                headers={"Authorization": f"Bearer {s.apify_token}"},
                json=payload,
            )
    except httpx.HTTPError as exc:
        raise ListingFetchError(f"Apify request failed for {url!r}: {exc}") from exc

    if resp.status_code == 401:
        raise ListingSourceNotConfigured("Apify token rejected (401).")
    if resp.status_code == 404:
        raise ListingFetchError(f"Apify actor {actor!r} not found (404).")
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise ListingFetchError(
            f"Apify returned {resp.status_code} for {url!r}: the listing may be "
            "blocked or the service is unavailable."
        ) from exc

    try:
        items = resp.json()
    except ValueError as exc:
        raise ListingFetchError(
            f"Apify returned a non-JSON response for {url!r}."
        ) from exc
    if not items:
        raise ListingFetchError(
            f"No data returned for {url!r} - the listing may be removed, "
            "private, or blocked. Try another listing URL."
        )
    return items[0]


def _normalize_bayut(d: dict, url: str) -> dict:
    detail_gated = "fallback" in str(d.get("source") or "").lower()
    if detail_gated:
        log.warning(
            "listing_detail_gated",
            url=url,
            actor_source=d.get("source"),
            detail="portal detail page unavailable; served from search index "
                   "(no description text)",
        )
    price = _to_float(d.get("price"))
    freq = (d.get("rentFrequency") or "").lower() or None
    purpose = d.get("purpose")
    geo = d.get("geography") or {}
    return {
        "source": "bayut",
        "listing_url": url,
        "listing_id": str(d.get("externalID") or d.get("id") or ""),
        "title": d.get("title"),
        "description": clean_description(d.get("description")),
        "property_type": _category_name(d.get("category")),
        "bedrooms": _to_int(d.get("rooms")),
        "bathrooms": _to_int(d.get("baths")),
        "area_sqft": _area_to_sqft(_to_float(d.get("area"))),
        "price": price,
        "purpose": purpose,
        "rent_frequency": freq,
        "annual_rent": None if purpose == "for-sale" else _annualise(price, freq),
        "community": _bayut_community(d.get("location")),
        "community_candidates": _broader_locations(
            _bayut_location_names(d.get("location")), 2
        ),
        "latitude": geo.get("lat") if isinstance(geo, dict) else None,
        "longitude": geo.get("lng") if isinstance(geo, dict) else None,
        "amenities": _bayut_amenities(d.get("amenities")),
        "detail_gated": detail_gated,
    }


def _bayut_location_names(location) -> list[str]:
    """Ordered breadcrumb names from Bayut's hierarchical location."""
    if isinstance(location, list) and location:
        return [
            lv.get("name") for lv in location
            if isinstance(lv, dict) and lv.get("name")
        ]
    return []


def _bayut_community(location) -> str | None:
    """Community from Bayut's hierarchical location breadcrumb.

    Breadcrumb runs Country > City > Community > Tower; index 2 is the
    community when present, else the deepest level available.
    """
    if isinstance(location, list) and location:
        names = [lv.get("name") for lv in location if isinstance(lv, dict)]
        names = [n for n in names if n]
        if len(names) >= 3:
            return names[2]
        if names:
            return names[-1]
    return None


def _broader_locations(names: list[str], chosen_index: int) -> list[str]:
    """Breadcrumb levels BROADER than the one the portal named, deepest first.

    Index 0 is the country and 1 the city, so community granularity starts at 2;
    anything above that is too coarse to be an area and is excluded. Only
    broader levels are offered - generalising "Costa Brava" to "DAMAC Lagoons"
    is sound, guessing a specific tower from a community is not.
    """
    return list(reversed(names[2:chosen_index]))


def _category_name(cat) -> str | None:
    if isinstance(cat, list) and cat:
        last = cat[-1]
        if isinstance(last, dict):
            return last.get("name")
    if isinstance(cat, dict):
        return cat.get("name")
    return None


def _bayut_amenities(amenities) -> list[str]:
    """Flatten Bayut amenities to a list of names.

    Handles both shapes seen in the wild: grouped
    (``[{"name": grp, "amenities": [{"text": ...}]}]``) and already-flat lists
    of strings/dicts.
    """
    out: list[str] = []
    if not isinstance(amenities, list):
        return out
    for entry in amenities:
        if isinstance(entry, str):
            out.append(entry)
        elif isinstance(entry, dict):
            inner = entry.get("amenities")
            if isinstance(inner, list):
                for a in inner:
                    if isinstance(a, dict) and (a.get("text") or a.get("name")):
                        out.append(a.get("text") or a.get("name"))
                    elif isinstance(a, str):
                        out.append(a)
            elif entry.get("text") or entry.get("name"):
                out.append(entry.get("text") or entry.get("name"))
    return out


def _normalize_dubizzle(d: dict, url: str) -> dict:
    price = _to_float(d.get("price"))
    coords = d.get("coordinates") or {}
    purpose = _dubizzle_purpose(d)
    return {
        "source": "dubizzle",
        "listing_url": _multilingual(d.get("absolute_url")) or url,
        "listing_id": str(d.get("listing_id") or ""),
        "title": _multilingual(d.get("name")),
        "description": clean_description(_multilingual(d.get("description"))),
        "property_type": _dubizzle_type(d, url),
        "bedrooms": _to_int(d.get("bedrooms")),
        "bathrooms": _to_int(d.get("bathrooms")),
        "area_sqft": _to_float(d.get("size")),
        "price": price,
        "purpose": purpose,
        "rent_frequency": None,
        "annual_rent": None if purpose == "for-sale" else price,
        "community": _dubizzle_community(d.get("location_path")),
        "community_candidates": _broader_locations(
            _dubizzle_location_names(d.get("location_path")),
            max(len(_dubizzle_location_names(d.get("location_path"))) - 1, 0),
        ),
        "latitude": coords.get("lat") if isinstance(coords, dict) else None,
        "longitude": coords.get("lng") if isinstance(coords, dict) else None,
        "amenities": _dubizzle_amenities(d.get("amenities")),
        "detail_gated": False,
    }


def _multilingual(v):
    """Dubizzle name/description can be a {'en': .., 'ar': ..} dict."""
    if isinstance(v, dict):
        return v.get("en") or next(iter(v.values()), None)
    return v


_DUBIZZLE_TYPE_TOKENS = (
    "apartmentflat", "apartment/flat", "apartment", "flat", "studio",
    "penthouse", "hotel apartment", "duplex",
    "villa", "townhouse", "compound", "residential building", "whole building",
)


def _looks_like_type(text: str | None) -> bool:
    """True when a breadcrumb names a property type rather than a category.

    Dubizzle's breadcrumb ends with the *listing category* on some pages
    ("Property for Rent"), not the property type - so taking the last element
    blindly yields an unmappable string and the listing is rejected as
    'could not determine type' even though every other field is present.
    """
    if not text:
        return False
    t = text.strip().lower()
    if "property for" in t:
        return False
    return any(tok in t for tok in _DUBIZZLE_TYPE_TOKENS)


def _dubizzle_type(d: dict, url: str = "") -> str | None:
    """Property type from the breadcrumb, falling back to the URL path.

    Scans breadcrumbs deepest-first for one that actually names a type, rather
    than assuming the last element is it. If none qualifies, the URL path is
    used - a Dubizzle listing URL carries the type as a segment
    (`/residential/apartmentflat/...`).
    """
    cats = d.get("categories")
    names: list[str] = []
    if isinstance(cats, list):
        for c in cats:
            n = _multilingual(c.get("name")) if isinstance(c, dict) else c
            if n:
                names.append(str(n))

    for name in reversed(names):
        if _looks_like_type(name):
            return name

    for seg in urlparse(url).path.lower().split("/"):
        if _looks_like_type(seg):
            return seg

    return names[-1] if names else None


def _dubizzle_purpose(d: dict) -> str | None:
    parts = []
    for c in d.get("categories") or []:
        parts.append(str(_multilingual(c.get("name")) if isinstance(c, dict) else c))
    parts.append(str(d.get("absolute_url") or ""))
    blob = " ".join(parts).lower()
    if "rent" in blob:
        return "for-rent"
    if "sale" in blob or "sell" in blob:
        return "for-sale"
    for key in ("purpose", "listing_type", "offering_type", "type"):
        v = str(d.get(key) or "").lower()
        if "rent" in v:
            return "for-rent"
        if "sale" in v or "sell" in v or "buy" in v:
            return "for-sale"
    return None


def _dubizzle_location_names(location_path) -> list[str]:
    """Ordered breadcrumb names from Dubizzle's location_path."""
    if isinstance(location_path, list) and location_path:
        names = [
            _multilingual(lv.get("name") if isinstance(lv, dict) else lv)
            for lv in location_path
        ]
        return [n for n in names if n]
    return []


def _dubizzle_community(location_path) -> str | None:
    """Deepest meaningful level of Dubizzle's location breadcrumb.

    Kept as the primary name because it is the most specific thing the portal
    knows, but it is frequently a CLUSTER rather than a DLD area - measured:
    `UAE > Dubai > DAMAC Lagoons > Costa Brava`. See `community_candidates`.
    """
    names = _dubizzle_location_names(location_path)
    return names[-1] if names else None


def _dubizzle_amenities(amenities) -> list[str]:
    out: list[str] = []
    if isinstance(amenities, list):
        for a in amenities:
            if isinstance(a, dict):
                name = _multilingual(a.get("name") or a.get("text"))
                if name:
                    out.append(name)
            elif isinstance(a, str):
                out.append(a)
    return out


_TAG_RE = re.compile(r"<[^>]+>")
_BREAK_RE = re.compile(
    r"</?\s*(?:br|p|div|li|tr|h[1-6])\b[^>]*>", re.IGNORECASE
)


def clean_description(raw: str | None) -> str | None:
    """Portal descriptions are HTML; the rest of the system wants plain text.

    Dubizzle returns the advertiser's copy with real markup in it - measured on
    a live listing: ``"Presenting a Brand New 4-Bedroom townhouse...<br><br>
    <strong>Property Features:</strong><br>-Back to Back<br>"``. Two things
    went wrong with that string:

    * the UI renders the description as text, so a reader saw the literal
      characters ``<br><br><strong>`` running through the advertiser's copy;
    * it is also the ONLY input to the LLM `character` classifier, which spent
      its (clamped) prompt budget on markup and returned nothing parseable.

    Structural tags become newlines so the copy keeps its shape, inline tags are
    dropped, HTML entities are decoded, and runs of blank lines are collapsed so
    a clamped preview is not wasted on whitespace. Sanitising here rather than
    in the UI means every consumer - UI, LLM, Q&A - gets the same clean text,
    and nothing downstream is ever tempted to render portal HTML as markup.
    """
    if not raw:
        return raw
    text = _BREAK_RE.sub("\n", raw)
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)
    text = text.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in text.split("\n")]
    out: list[str] = []
    for ln in lines:
        if not ln and (not out or not out[-1]):
            continue
        out.append(ln)
    return "\n".join(out).strip() or None


_SQM_TO_SQFT = 10.7639


def _area_to_sqft(area_m2: float | None) -> float | None:
    """The Bayut actor reports floor area in SQUARE METRES (field ``area``), but
    the normalized listing exposes ``area_sqft`` and the model expects sqft. Convert
    m² → sqft so downstream price/rent-per-sqft and the model's sqft→m² step are
    correct (a raw 128.5 m² was otherwise stored as "128.5 sqft" → a nonsensical
    ~12 m² fed to the model)."""
    if area_m2 is None:
        return None
    return round(area_m2 * _SQM_TO_SQFT, 2)


_ANNUALISE_FACTOR = {"yearly": 1, "monthly": 12, "weekly": 52, "daily": 365}


def _annualise(price: float | None, freq: str | None) -> float | None:
    """Convert a rent price at its stated frequency into an annual figure.

    When the frequency is missing/unrecognised we must NOT silently assume yearly:
    Bayut rentals are frequently monthly, so treating a 10k/month rent as 10k/year
    understates the yield ~12x (review F5). Instead we infer the most likely
    frequency from magnitude - a Dubai residential ANNUAL rent is realistically
    ≳ AED 20k, so a small figure is almost certainly monthly - and fall back to
    monthly (the common portal default) when still ambiguous, rather than yearly.
    """
    if price is None:
        return None
    factor = _ANNUALISE_FACTOR.get((freq or "").lower())
    if factor is not None:
        return price * factor
    if price >= 20_000:
        return price
    return price * 12


def _to_int(v) -> int | None:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _to_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
