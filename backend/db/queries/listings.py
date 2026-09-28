"""Rental listing queries."""

from __future__ import annotations

from db.client import get_db
from db.models import RENTAL_LISTINGS


def upsert_listings(rows: list[dict]) -> int:
    if not rows:
        return 0
    db = get_db()
    db.table(RENTAL_LISTINGS).upsert(rows, on_conflict="listing_url").execute()
    return len(rows)


def all_listings() -> list[dict]:
    return get_db().table(RENTAL_LISTINGS).select("*").execute().data


def unprocessed_listings() -> list[dict]:
    return (
        get_db()
        .table(RENTAL_LISTINGS)
        .select("*")
        .eq("llm_processed", False)
        .execute()
        .data
    )


def update_listing_features(listing_url: str, features: dict) -> None:
    payload = dict(features)
    payload["llm_processed"] = True
    (
        get_db()
        .table(RENTAL_LISTINGS)
        .update(payload)
        .eq("listing_url", listing_url)
        .execute()
    )
