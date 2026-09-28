"""Resolve a portal's community label to a DLD ``area_name_en`` (F21).

Shared by the model (``ml.predict``) and the baseline lookup
(``services.listing_scorer``) so both always agree on which DLD area a listing
belongs to - a mismatch there silently produces a model prediction for one area
scored against another area's medians.

Resolution runs cheapest-first and stops at the first hit:

1. **Exact** DLD area name (the listing already speaks DLD).
2. **Data-derived alias** - ``master_project_en`` → ``area_name_en`` mined from
   the raw transactions (see ``db.queries.community_aliases``).
3. **Abbreviation expansion** - JVC/JLT/DIFC etc., then retry (2).
4. **Suffix-stripped** retry - portals append sub-locality noise
   ("JVC District 14", "Dubai Marina - Marina Promenade").
5. **Unique prefix/containment** against known names, only when exactly one
   candidate matches - an ambiguous match resolves to nothing rather than
   guessing an area, because a wrong area yields a confidently wrong verdict.

Unresolved names are returned unchanged: the caller then finds no baseline and
reports "Insufficient data", which is the honest outcome.
"""

from __future__ import annotations

import re
from functools import lru_cache

import structlog

log = structlog.get_logger(__name__)

_ABBREVIATIONS = {
    "jvc": "jumeirah village circle",
    "jvt": "jumeirah village triangle",
    "jlt": "jumeirah lake towers",
    "jbr": "jumeirah beach residence",
    "difc": "dubai international financial centre",
    "dso": "dubai silicon oasis",
    "ibn battuta": "ibn battuta",
    "mbr city": "mohammed bin rashid city",
    "mbr": "mohammed bin rashid city",
    "the greens": "greens",
    "the views": "views",
    "the springs": "springs",
    "the meadows": "meadows",
    "the lakes": "lakes",
    "downtown": "downtown dubai",
    "marina": "dubai marina",
    "palm": "palm jumeirah",
    "the palm": "palm jumeirah",
}

_SUFFIX_NOISE = re.compile(
    r"\b(district|cluster|phase|zone|tower|towers|building|residence|residences|"
    r"community|sub[- ]?community|area|plot|street|road)\b.*$",
    re.IGNORECASE,
)


def _norm(text: str) -> str:
    """Lowercase, strip punctuation/extra whitespace for comparison."""
    t = (text or "").strip().lower()
    t = t.replace("&", "and")
    t = re.sub(r"[_/,()\[\]]+", " ", t)
    t = re.sub("\\s*[-\u2013\u2014]\\s*", " - ", t)
    return re.sub(r"\s+", " ", t).strip()


def _strip_trailing_acronym(text: str) -> str:
    """Drop a trailing acronym that merely repeats the words before it.

    Bayut and Dubizzle label communities as "Jumeirah Village Triangle (JVT)".
    ``_norm`` flattens the parentheses into a separator, leaving
    ``"jumeirah village triangle jvt"`` - which matches no alias key, because the
    mined alias is ``"jumeirah village triangle"``. Measured: a real JVT listing
    resolved to nothing, found no baseline, and was scored by model
    extrapolation at 13.77% and labelled "Insufficient data", when the area is in
    fact present in the DLD data (as ``Al Barsha South Fifth``, 1,179 apartment
    sales) with a real 7.71% median yield.

    The acronym is dropped ONLY when it is genuinely redundant - its letters are
    the initials of the preceding words, or it is a known abbreviation expanding
    to exactly those words. So "al barsha south fifth" keeps "fifth" (initials
    are "abs"), and "palm jumeirah" keeps "jumeirah" (too long to be an
    acronym). A real name is never truncated to make a match.
    """
    parts = text.split()
    if len(parts) < 2:
        return text
    last, head = parts[-1], parts[:-1]
    if not last.isalpha() or not (2 <= len(last) <= 5):
        return text
    head_text = " ".join(head)
    if last == "".join(w[0] for w in head if w):
        return head_text
    if _ABBREVIATIONS.get(last) == head_text:
        return head_text
    return text


def _strip_noise(text: str) -> str:
    """Drop trailing sub-locality noise and any '- suffix' tail."""
    t = _SUFFIX_NOISE.sub("", text).strip(" -")
    t = t.split(" - ")[0].strip()
    return re.sub(r"\s+", " ", t).strip()


@lru_cache(maxsize=1)
def _alias_map() -> dict[str, str]:
    """Data-derived aliases, cached per process (refreshed offline)."""
    from db.queries.community_aliases import load_aliases

    aliases = load_aliases()
    return {_norm(k): v for k, v in aliases.items()}


@lru_cache(maxsize=1)
def _known_areas() -> dict[str, str]:
    """``{normalized DLD area: exact DLD area}`` from the medians table."""
    try:
        from config import get_settings
        from db.queries.yield_agg import MEDIANS_TABLE
        from db.pg import connect as _connect
        from db.pg import quote_ident as _quote_ident

        with _connect(get_settings()) as conn, conn.cursor() as cur:
            cur.execute(f"SELECT DISTINCT area FROM {_quote_ident(MEDIANS_TABLE)}")
            return {_norm(r[0]): r[0] for r in cur.fetchall() if r[0]}
    except Exception as exc:  # noqa: BLE001 - DB unreachable / table missing
        log.warning("known_areas_unavailable", error=str(exc))
        return {}


def clear_cache() -> None:
    """Drop cached lookups (after refreshing the alias table, or in tests).

    Tolerates the lookups having been replaced (tests inject plain callables
    with no ``cache_clear``) - a cache-invalidation helper must never be the
    thing that raises.
    """
    for fn in (_alias_map, _known_areas):
        clear = getattr(fn, "cache_clear", None)
        if clear:
            clear()


def _unique_partial(norm_name: str, areas: dict[str, str]) -> str | None:
    """The single area whose name contains / is contained by ``norm_name``.

    Returns None when zero or MORE THAN ONE candidate matches - "Al Barsha
    South" alone must not silently resolve to one of Fourth/Fifth/First.
    """
    if len(norm_name) < 5:
        return None
    hits = {
        exact
        for norm_area, exact in areas.items()
        if norm_name == norm_area
        or norm_area.startswith(f"{norm_name} ")
        or norm_name.startswith(f"{norm_area} ")
    }
    return hits.pop() if len(hits) == 1 else None


def resolve_community(community: str) -> str:
    """Best-effort DLD ``area_name_en`` for a portal community label.

    Returns the input unchanged when nothing resolves confidently.
    """
    raw = (community or "").strip()
    if not raw:
        return raw

    areas = _known_areas()
    aliases = _alias_map()
    name = _norm(raw)

    if name in areas:
        return areas[name]
    if name in aliases:
        return aliases[name]
    expanded = _ABBREVIATIONS.get(name)
    if expanded:
        if expanded in aliases:
            return aliases[expanded]
        if expanded in areas:
            return areas[expanded]
    deacronymed = _strip_trailing_acronym(name)
    if deacronymed != name:
        if deacronymed in areas:
            return areas[deacronymed]
        if deacronymed in aliases:
            return aliases[deacronymed]

    stripped = _strip_noise(deacronymed)
    if stripped and stripped != name:
        if stripped in areas:
            return areas[stripped]
        if stripped in aliases:
            return aliases[stripped]
        exp2 = _ABBREVIATIONS.get(stripped)
        if exp2:
            if exp2 in aliases:
                return aliases[exp2]
            if exp2 in areas:
                return areas[exp2]
    for candidate in (name, deacronymed, stripped):
        if candidate:
            hit = _unique_partial(candidate, areas)
            if hit:
                return hit

    log.info("community_unresolved", community=raw)
    return raw
