"""Every user-facing model response leaves this codebase without a dash in it.

The prompts forbid em and en dashes, but a prompt is a request: models reach for
an em dash constantly in explanatory prose, and one that slips through lands in
the investor-facing narrative on the results screen, where the rest of the copy
has none. ``clean_output`` is the guarantee behind that request.

These tests exist because the guarantee has already been broken once, silently.
A project-wide pass that removed every literal em dash from the source rewrote
the sanitiser's OWN character-class constants into spaces and hyphens
(``_EM_DASHES = " -  - "``), so the function still ran, still returned a string,
and stripped nothing at all. Nothing failed. The constants are escape sequences
now, and these tests would have caught it either way.
"""

from __future__ import annotations

import pytest

from llm.provider import clean_output, strip_dashes

EM = chr(0x2014)
HORIZONTAL_BAR = chr(0x2015)
EN = chr(0x2013)
FIGURE_DASH = chr(0x2012)
MINUS = chr(0x2212)
ALL_DASHES = (EM, HORIZONTAL_BAR, EN, FIGURE_DASH, MINUS)


@pytest.mark.parametrize("dash", ALL_DASHES)
def test_no_dash_character_survives(dash: str):
    """The contract, stated once: none of them reach the page."""
    assert dash not in strip_dashes(f"The yield {dash} 6.2% {dash} is high.")
    assert dash not in strip_dashes(f"a{dash}b")
    assert dash not in clean_output(f"Answer {dash} final.")


def test_spaced_em_dash_becomes_a_spaced_hyphen():
    assert strip_dashes(f"The yield {EM} 6.2% {EM} is above the median.") == (
        "The yield - 6.2% - is above the median."
    )


def test_unspaced_em_dash_gains_the_spaces_it_needs():
    """An em dash with no spaces is still a sentence break in model prose.

    Collapsing it to a bare hyphen produced "confidence is low-the segment did
    not match", which reads as a compound word rather than the pause it was.
    """
    assert strip_dashes(f"Confidence is low{EM}the segment did not match.") == (
        "Confidence is low - the segment did not match."
    )


def test_en_dash_in_a_range_becomes_a_bare_hyphen():
    """A range is not a pause, so it must not gain spaces."""
    assert strip_dashes(f"A range of 5{EN}7% is normal.") == "A range of 5-7% is normal."


def test_minus_sign_before_a_number_becomes_a_plain_hyphen():
    assert strip_dashes(f"The gap is {MINUS}0.4 pp.") == "The gap is -0.4 pp."


def test_a_dash_opening_a_line_is_a_bullet_and_is_dropped():
    """The prompts forbid bullets; a leading dash is one wearing a disguise."""
    assert strip_dashes(f"{EM} A bullet point.") == "A bullet point."


def test_prose_without_dashes_is_returned_unchanged():
    text = "Similar homes in this area typically return 6.5%."
    assert strip_dashes(text) == text


def test_reasoning_markup_is_removed_before_dashes_are_rewritten():
    """Order matters: a dash inside a <think> block must not be rewritten into
    the output on its way to being deleted."""
    assert clean_output(f"<think>a {EM} b</think>Answer {EM} final.") == (
        "Answer - final."
    )


def test_dash_constants_are_escapes_not_literal_characters():
    """The regression guard for the failure described in the module docstring.

    These constants are the one place in the project that must hold dash
    characters. Written as literals they are a target for exactly the pass that
    broke them before; written as escapes the source contains no dash to find.
    """
    import pathlib

    import llm.provider as provider

    source = pathlib.Path(provider.__file__).read_text(encoding="utf-8")
    for dash in ALL_DASHES:
        assert dash not in source, f"literal {dash!r} in llm/provider.py"
    assert provider._EM_DASHES == EM + HORIZONTAL_BAR
    assert provider._EN_DASHES == EN + FIGURE_DASH + MINUS


def test_empty_and_none_like_input_is_safe():
    assert strip_dashes("") == ""
    assert clean_output("") == ""
