"""Tests for the tolerant classification parser (F19).

Every case below is a REAL response shape measured against `gpt-oss:20b-cloud`,
where Ollama's JSON-schema `format` is silently ignored. The parser exists
because `with_structured_output` raised OUTPUT_PARSING_FAILURE on all of them.

The governing rule: recover what is genuinely recoverable, and return None for
anything else. A wrong finish tier shown to an investor is worse than a blank.
"""

from __future__ import annotations

import pytest

import config
from llm.chains import classify_listing as cl


def test_plain_json():
    f = cl.parse_features('{"finish_tier":"Luxury","view_type":"Sea","amenity_score":4}')
    assert (f.finish_tier, f.view_type, f.amenity_score) == ("Luxury", "Sea", 4)


def test_fenced_json():
    f = cl.parse_features(
        'Here you go:\n```json\n{"finish_tier":"Mid","view_type":"Pool","amenity_score":2}\n```'
    )
    assert (f.finish_tier, f.view_type, f.amenity_score) == ("Mid", "Pool", 2)


def test_json_embedded_in_prose():
    f = cl.parse_features(
        'Sure! {"finish_tier":"High","view_type":"Skyline","amenity_score":3} Hope that helps.'
    )
    assert (f.finish_tier, f.view_type, f.amenity_score) == ("High", "Skyline", 3)


def test_markdown_bullets_are_recovered():
    """Measured output: the model ignored `format` and answered in bullets."""
    f = cl.parse_features(
        "- **finish_tier**: Luxury  \n- **view_type**: Sea  \n- **amenity_score**: 5"
    )
    assert (f.finish_tier, f.view_type, f.amenity_score) == ("Luxury", "Sea", 5)


def test_out_of_range_score_is_rescaled_not_clamped():
    """Measured: the model answered '9 (out of 10)' against a 0-5 schema.

    Rescaling to 5 preserves the model's intent; clamping would assert a
    top-of-range score it never gave.
    """
    assert cl.parse_features("- **amenity_score**: 9 (out of 10)").amenity_score == 5
    assert cl.parse_features('{"amenity_score": 8}').amenity_score == 4


def test_invented_keys_yield_none_not_a_guess():
    """Measured: valid JSON, but the model chose its own key names."""
    f = cl.parse_features('{"fit_out":"bespoke luxury","marble":"Italian","view":"sea"}')
    assert f.finish_tier is None
    assert f.amenity_score is None


def test_prose_answer_yields_nothing():
    f = cl.parse_features("This stunning residence offers unparalleled coastal living.")
    assert (f.finish_tier, f.view_type, f.amenity_score) == (None, None, None)


def test_reasoning_block_is_stripped_before_parsing():
    f = cl.parse_features(
        '<think>The user wants JSON. Let me decide the tier.</think>'
        '{"finish_tier":"Basic","view_type":"None","amenity_score":0}'
    )
    assert f.finish_tier == "Basic"
    assert f.amenity_score == 0


def test_value_outside_vocabulary_is_rejected():
    f = cl.parse_features('{"finish_tier":"Ultra-Premium","view_type":"Desert"}')
    assert f.finish_tier is None
    assert f.view_type is None


def test_matching_is_case_insensitive():
    f = cl.parse_features('{"finish_tier":"luxury","view_type":"SEA"}')
    assert (f.finish_tier, f.view_type) == ("Luxury", "Sea")


def test_none_view_is_a_valid_value():
    assert cl.parse_features('{"view_type":"None"}').view_type == "None"


def test_empty_and_garbage_never_raise():
    for text in ("", None, "   ", "{{{", "null", "[]"):
        f = cl.parse_features(text)
        assert f.finish_tier is None


def test_classify_raises_when_llm_unconfigured(monkeypatch, unconfigured_llm):
    with pytest.raises(RuntimeError, match="not configured"):
        cl.classify("A nice flat.")
