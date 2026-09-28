"""Bayut rental-listing scraper (M1B-01) - live-only.

Bayut is a Next.js application: every search page embeds a ``__NEXT_DATA__``
``<script>`` tag holding the fully-structured listing payload (price, beds,
baths, area, type, location, coordinates, description) that hydrates the page.
Parsing that JSON is far more robust than scraping rendered HTML cards - the
visual markup/class names churn constantly, but the data contract behind it is
stable and clean.

Strategy:
    1. Fetch the search page over ``httpx`` with a realistic browser
       User-Agent (no JS execution needed - the data is in the initial HTML).
    2. Extract and parse the ``__NEXT_DATA__`` JSON.
    3. Walk it for the listings array (Algolia ``hits``) and map each hit to
       our rental-listing row schema, including the description text for the
       downstream LLM classification stage.

This is **live-only**: there is no synthetic fallback. If a page can't be
fetched or contains no listings, that page is skipped (and logged); the caller
gets only genuine listings.

The data.dubai DLD source is UAE-only; Bayut is public, but it does rate-limit
and bot-detect, so we keep a polite inter-page delay and a single shared client.
"""

from __future__ import annotations

import asyncio
import json
import re

import structlog

log = structlog.get_logger(__name__)

SOURCE = "bayut"

_BASE = "https://www.bayut.com"
_SEARCH_PATH = "/to-rent/property/dubai/{slug}/page-{page}/"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

_PAGE_DELAY_S = 2.0
_REQUEST_TIMEOUT_S = 30.0

_NEXT_DATA_RE = re.compile(
    r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>',
    re.DOTALL,
)


def _community_slug(community: str) -> str:
    """Bayut community URL slug, e.g. 'Dubai Marina' -> 'dubai-marina'."""
    return community.strip().lower().replace(" ", "-")


async def scrape_community(community: str, max_pages: int = 10) -> list[dict]:
    """Scrape live Bayut rental listings for one community.

    Pages through the community's to-rent search up to ``max_pages``, parsing the
    embedded ``__NEXT_DATA__`` JSON on each page. Returns mapped listing rows;
    stops early when a page yields no listings (reached the end / blocked).
    """
    import httpx

    slug = _community_slug(community)
    results: list[dict] = []

    async with httpx.AsyncClient(
        headers=_HEADERS, timeout=_REQUEST_TIMEOUT_S, follow_redirects=True
    ) as http:
        for page_no in range(1, max_pages + 1):
            url = _BASE + _SEARCH_PATH.format(slug=slug, page=page_no)
            try:
                resp = await http.get(url)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                log.warning("bayut_page_failed", url=url, error=str(exc))
                break

            hits = _extract_hits(resp.text)
            if not hits:
                log.info("bayut_page_empty", url=url, page=page_no)
                break

            rows = [_map_listing(h, community) for h in hits]
            results.extend(r for r in rows if r is not None)

            if page_no < max_pages:
                await asyncio.sleep(_PAGE_DELAY_S)

    log.info("bayut_scraped", community=community, rows=len(results))
    return results


def _extract_hits(html: str) -> list[dict]:
    """Pull the listings array out of a page's ``__NEXT_DATA__`` payload.

    Bayut's search results are Algolia ``hits``; their exact nesting under
    ``props.pageProps`` has shifted over time, so we search the parsed tree for
    the first list of listing-shaped dicts rather than hard-coding a path.
    """
    m = _NEXT_DATA_RE.search(html)
    if not m:
        return []
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError as exc:
        log.warning("bayut_next_data_parse_failed", error=str(exc))
        return []

    page_props = (
        data.get("props", {}).get("pageProps", {}) if isinstance(data, dict) else {}
    )
    for key in ("hits", "results", "listings", "properties"):
        val = page_props.get(key)
        if _is_listing_list(val):
            return val
    search_result = page_props.get("searchResult") or page_props.get("algoliaResult")
    if isinstance(search_result, dict) and _is_listing_list(search_result.get("hits")):
        return search_result["hits"]

    found = _find_listing_list(page_props)
    return found or []


def _is_listing_list(val) -> bool:
    """A non-empty list whose first element looks like a Bayut listing dict."""
    if not isinstance(val, list) or not val:
        return False
    first = val[0]
    if not isinstance(first, dict):
        return False
    return "price" in first and ("externalID" in first or "id" in first)


def _find_listing_list(node, depth: int = 0) -> list[dict] | None:
    """Depth-first search for the first listing-shaped list in a nested tree."""
    if depth > 8:
        return None
    if _is_listing_list(node):
        return node
    if isinstance(node, dict):
        for v in node.values():
            found = _find_listing_list(v, depth + 1)
            if found:
                return found
    elif isinstance(node, list):
        for v in node:
            found = _find_listing_list(v, depth + 1)
            if found:
                return found
    return None


def _map_listing(hit: dict, community: str) -> dict | None:
    """Map one Bayut Algolia hit to our rental-listing row schema.

    Only ``rent`` purpose listings are kept (the search is to-rent, but guard
    against mixed payloads). Missing numeric fields stay ``None`` rather than
    being fabricated.
    """
    if hit.get("purpose") not in (None, "for-rent", "rent"):
        return None

    listing_id = hit.get("externalID") or hit.get("id")
    slug = hit.get("slug") or hit.get("permalink")
    listing_url = f"{_BASE}/property/details-{slug}.html" if slug else None

    return {
        "source": SOURCE,
        "listing_id": str(listing_id) if listing_id is not None else None,
        "listing_url": listing_url,
        "community": _hit_community(hit) or community,
        "property_type": _category_name(hit),
        "bedrooms": _to_int(hit.get("rooms")),
        "bathrooms": _to_int(hit.get("baths")),
        "floor_area_sqft": _to_float(hit.get("area")),
        "annual_rent_aed": _annual_rent(hit),
        "rent_frequency": hit.get("rentFrequency"),
        "furnished": hit.get("furnishingStatus"),
        "latitude": _geo(hit, "lat"),
        "longitude": _geo(hit, "lng"),
        "title": hit.get("title"),
        "description": hit.get("description"),
        "amenities_raw": _amenities(hit),
        "llm_processed": False,
    }


def _hit_community(hit: dict) -> str | None:
    """Best community name from the hit's location breadcrumb hierarchy."""
    loc = hit.get("location")
    if isinstance(loc, list) and loc:
        names = [lvl.get("name") for lvl in loc if isinstance(lvl, dict)]
        if len(names) >= 3 and names[2]:
            return names[2]
        if names:
            return names[-1]
    return None


def _category_name(hit: dict) -> str | None:
    cat = hit.get("category")
    if isinstance(cat, list) and cat:
        last = cat[-1]
        if isinstance(last, dict):
            return last.get("name")
    if isinstance(cat, dict):
        return cat.get("name")
    return None


def _annual_rent(hit: dict) -> float | None:
    """Annualise the rent. Bayut ``price`` is per ``rentFrequency`` period."""
    price = _to_float(hit.get("price"))
    if price is None:
        return None
    freq = (hit.get("rentFrequency") or "yearly").lower()
    multiplier = {
        "yearly": 1,
        "monthly": 12,
        "weekly": 52,
        "daily": 365,
    }.get(freq, 1)
    return price * multiplier


def _amenities(hit: dict) -> list[str]:
    """Flatten Bayut amenity groups into a flat list of amenity names."""
    out: list[str] = []
    amenities = hit.get("amenities")
    if isinstance(amenities, list):
        for group in amenities:
            if isinstance(group, dict):
                for a in group.get("amenities", []) or []:
                    if isinstance(a, dict) and a.get("text"):
                        out.append(a["text"])
    return out


def _geo(hit: dict, key: str):
    geo = hit.get("geography") or hit.get("_geoloc")
    if isinstance(geo, dict):
        return geo.get(key)
    return None


def _to_int(value) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
