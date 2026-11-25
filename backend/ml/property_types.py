"""Portal property type → the model's canonical type vocabulary.

The trained model knows exactly two types, ``apartment`` and ``villa`` (verified
against the live ``feature_matrix`` table). A studio is an apartment with
``bedrooms=0``, and penthouses and hotel apartments train in the apartment
buckets, so none of them is a type of its own.

One map for every caller: the listing scorer, ``/predict`` and the price-history
route previously kept separate maps, and the ``/predict`` one sent "studio" to a
type the model has never seen.
"""

from __future__ import annotations

TYPE_ALIASES = {
    "apartment": "apartment",
    "apartments": "apartment",
    "flat": "apartment",
    "flats": "apartment",
    "unit": "apartment",
    "residential": "apartment",
    "apartmentflat": "apartment",
    "apartment/flat": "apartment",
    "apartment flat": "apartment",
    "duplex": "apartment",
    "studio": "apartment",
    "studios": "apartment",
    "penthouse": "apartment",
    "penthouses": "apartment",
    "hotel apartment": "apartment",
    "hotel apartments": "apartment",
    "villa": "villa",
    "villas": "villa",
    "townhouse": "villa",
    "townhouses": "villa",
    "compound": "villa",
    "complex villas": "villa",
    "arabian house": "villa",
}


def canonical_type(raw_type: str | None) -> str | None:
    """Map a portal's property type to the model's canonical type, or None."""
    if not raw_type:
        return None
    return TYPE_ALIASES.get(raw_type.strip().lower())


def canonical_type_or_raw(raw_type: str | None) -> str:
    """Canonical type when known, else the normalised input.

    For callers that must still pass a type through: an unknown type then matches
    no segment and is reported as such, rather than being silently relabelled.
    """
    return canonical_type(raw_type) or (raw_type or "").strip().lower()
