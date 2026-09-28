"""Dubizzle rental-listing scraper (M1B-02) - live-only.

Same interface as the Bayut scraper. The live Playwright path mirrors
``bayut_scraper`` and is not yet implemented; calling it raises so the caller
records the gap rather than persisting fabricated data.
"""

from __future__ import annotations

import structlog

log = structlog.get_logger(__name__)

SOURCE = "dubizzle"


async def scrape_community(community: str, max_pages: int = 10) -> list[dict]:
    raise NotImplementedError(
        "Dubizzle live scraping is not implemented yet; mirror bayut_scraper."
    )
