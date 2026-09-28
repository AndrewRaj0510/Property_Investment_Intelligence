"""Portal descriptions are HTML; everything downstream wants plain text (F39).

Found by scoring ONE real Dubizzle listing (Costa Brava townhouse, listing
17873981). Its description came back as the advertiser's copy with real markup
in it, and two things consumed it raw: the results screen, which renders the
description as TEXT (so a reader saw the literal characters ``<br><br>
<strong>``), and the LLM `character` classifier, for which it is the only input.
"""

from __future__ import annotations

from ingestion.listing_client import (
    _normalize_bayut,
    _normalize_dubizzle,
    clean_description,
)

REAL = (
    "Presenting a Brand New 4-Bedroom townhouse unit in Costa Brava at DAMAC "
    "Lagoons, Park &amp; Lagoon Access.<br><br><br><strong>Property Features:"
    "</strong><br>-Back to Back<br>-Bua 2,280 sq ft<br>"
)


def test_real_listing_description_has_no_markup_left():
    out = clean_description(REAL)
    assert "<" not in out and ">" not in out
    assert "<br>" not in out and "strong" not in out


def test_entities_are_decoded():
    """`&amp;` reached the screen verbatim in the advertiser's own sentence."""
    assert clean_description("Park &amp; Lagoon") == "Park & Lagoon"
    assert clean_description("a &lt;b&gt; c &quot;d&quot;") == 'a <b> c "d"'


def test_structural_tags_become_line_breaks():
    """The copy's shape is meaningful - a heading then a bulleted run."""
    out = clean_description(REAL)
    assert "Property Features:" in out
    lines = [ln for ln in out.split("\n") if ln]
    assert "-Back to Back" in lines
    assert "-Bua 2,280 sq ft" in lines


def test_inline_tags_are_dropped_without_joining_words():
    assert clean_description("<p>A</p><p>B</p>") == "A\n\nB"
    assert clean_description("<div>x</div>   <div>y</div>") == "x\n\ny"


def test_blank_line_runs_are_collapsed():
    """The real listing had `<br><br><br>` runs. Left as-is they would spend a
    clamped preview's whole line budget on whitespace."""
    assert clean_description("<br><br><br>x<br><br>y") == "x\n\ny"


def test_non_breaking_spaces_become_ordinary_spaces():
    assert clean_description("a&nbsp;b") == "a b"


def test_plain_text_is_untouched():
    assert clean_description("A quiet studio.") == "A quiet studio."


def test_empty_and_none_are_preserved():
    """`None` must stay `None` - `_classify_character` distinguishes 'no
    description' from 'empty description', and the UI's gated notice keys off
    a falsy description."""
    assert clean_description(None) is None
    assert clean_description("") == ""
    assert clean_description("<br><br>") is None


def test_both_normalizers_clean_their_description():
    dz = _normalize_dubizzle({"listing_id": "1", "description": {"en": REAL}}, "u")
    assert "<br>" not in dz["description"]
    by = _normalize_bayut({"externalID": "1", "description": REAL}, "u")
    assert "<br>" not in by["description"]
