import pytest

from analysis_engine import extract_facts


def test_extract_facts_basic():
    text = "The Eiffel Tower is 324 meters tall. I think this is cool."
    facts = extract_facts(text)
    assert isinstance(facts, list)
    assert any("Eiffel Tower" in f or "324" in f for f in facts)


def test_extract_empty():
    assert extract_facts("") == []
