"""Capture a real listing to a reusable fixture (F19).

Run ONCE per listing you want to test against; every later run is free and
offline. This is what makes `fixture://<name>` scoring honest - the saved
listing is a real Apify response, not an invention.

    # capture (costs one Apify call)
    python -m scripts.capture_listing "https://www.bayut.com/property/details-1234.html" --name marina_2br

    # then score it forever, offline and free
    curl -X POST localhost:8000/api/v1/score -H "Content-Type: application/json" \
         -d '{"url": "fixture://marina_2br"}'

    python -m scripts.capture_listing --list
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone

from ingestion import listing_fixtures as fx
from ingestion.listing_client import fetch_listing


async def _capture(url: str, name: str) -> int:
    print(f"Fetching {url} ...")
    try:
        listing = await fetch_listing(url)
    except Exception as exc:  # noqa: BLE001 - CLI: report and exit non-zero
        print(f"ERROR: could not fetch the listing: {exc}", file=sys.stderr)
        return 1

    path = fx.save(
        name,
        listing,
        meta={
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "source_url": url,
        },
    )
    print(f"Saved fixture -> {path}")
    print(
        json.dumps(
            {
                k: listing.get(k)
                for k in ("source", "community", "property_type", "bedrooms",
                          "area_sqft", "price", "purpose", "annual_rent")
            },
            indent=2,
        )
    )
    print(f'\nScore it offline with:  {{"url": "fixture://{name}"}}')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("url", nargs="?", help="A Bayut or Dubizzle listing detail URL.")
    ap.add_argument("--name", help="Fixture name (defaults to the listing id).")
    ap.add_argument("--list", action="store_true", help="List saved fixtures.")
    args = ap.parse_args()

    if args.list:
        names = fx.available()
        print("\n".join(f"  fixture://{n}" for n in names) if names else "(no fixtures saved)")
        return 0

    if not args.url:
        ap.error("a URL is required (or use --list)")

    name = args.name or args.url.rstrip("/").split("/")[-1].split(".")[0]
    return asyncio.run(_capture(args.url, name))


if __name__ == "__main__":
    raise SystemExit(main())
