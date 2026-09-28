"""Saved listing fixtures - exercise the scoring flow without a live fetch (F19).

Every `/score` call normally costs a paid Apify run and depends on a live portal
URL that may be delisted tomorrow. That makes iterating on the scorer or the LLM
layer slow, expensive and irreproducible.

This module supplies **already-normalized listings** so the rest of the flow -
model inference, DLD baseline lookup, ratio completion, verdict, LLM narrative -
runs exactly as in production. Only the network fetch is replaced.

Two ways in:

  * ``fixture://<name>`` as the scored URL - always resolves here, no env change.
  * ``LISTING_FIXTURE_MODE=replay`` - any URL resolves to a saved capture,
    letting the real frontend run end-to-end offline.

**These are recordings, not inventions.** ``capture_listing.py`` writes them from
real Apify responses, so the fields are the portals' own. A hand-written fixture
would test the scorer against fiction; that is why ``score_listing_url`` labels
fixture-backed results with ``source_mode="fixture"`` - a fixture score is never
silently presented as a live one.
"""

from __future__ import annotations

import json
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "data" / "listing_fixtures"
FIXTURE_SCHEME = "fixture://"

REQUIRED_KEYS = ("source", "listing_url", "property_type", "community")


class FixtureNotFound(FileNotFoundError):
    """Raised when a named fixture doesn't exist on disk."""


def is_fixture_url(url: str) -> bool:
    return (url or "").strip().lower().startswith(FIXTURE_SCHEME)


def fixture_name(url: str) -> str:
    return (url or "").strip()[len(FIXTURE_SCHEME):].strip("/")


def available() -> list[str]:
    """Names of all saved fixtures."""
    if not FIXTURE_DIR.is_dir():
        return []
    return sorted(p.stem for p in FIXTURE_DIR.glob("*.json"))


def load(name: str) -> dict:
    """Load one fixture by name. Raises ``FixtureNotFound`` if absent."""
    safe = Path(name).name
    path = FIXTURE_DIR / f"{safe}.json"
    if not path.is_file():
        raise FixtureNotFound(
            f"No listing fixture named {safe!r}. Available: {', '.join(available()) or '(none)'}. "
            f"Capture one with: python -m scripts.capture_listing <url> --name {safe}"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    listing = data.get("listing", data)
    missing = [k for k in REQUIRED_KEYS if k not in listing]
    if missing:
        raise ValueError(f"Fixture {safe!r} is missing required keys: {missing}")
    log.info("listing_fixture_loaded", name=safe, community=listing.get("community"))
    return listing


def save(name: str, listing: dict, meta: dict | None = None) -> Path:
    """Write a captured listing to disk."""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIXTURE_DIR / f"{Path(name).name}.json"
    payload = {"meta": meta or {}, "listing": listing}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def any_fixture() -> dict:
    """A fixture for replay mode when the URL names none in particular."""
    names = available()
    if not names:
        raise FixtureNotFound(
            "LISTING_FIXTURE_MODE=replay but no fixtures are saved. Capture one "
            "with: python -m scripts.capture_listing <url> --name <name>"
        )
    return load(names[0])
