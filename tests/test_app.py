"""Headless tests of the Streamlit app using Streamlit's own test runner."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parent.parent / "streamlit_app.py")


@pytest.fixture
def app():
    test = AppTest.from_file(APP, default_timeout=60)
    test.run()
    return test


def test_app_loads_the_sample_and_shows_the_score(app):
    assert not app.exception
    labels = [metric.label for metric in app.metric]
    assert "Quality score" in labels
    assert any("sample_messy.csv" in markdown.value for markdown in app.markdown)


def test_issues_are_listed_as_expanders(app):
    assert len(app.expander) >= 10


def test_cleaning_raises_the_score(app):
    button = next(b for b in app.button if b.label == "Clean the data")
    button.click()
    app.run()

    assert not app.exception
    scores = [m for m in app.metric if m.label == "Quality score"]
    assert len(scores) == 2  # before (Profile tab) and after (Clean tab)
    assert float(scores[1].value) > float(scores[0].value.split()[0])


def test_manual_query_runs_without_a_model(app):
    next(b for b in app.button if b.label == "Run query").click()
    app.run()

    assert not app.exception


@pytest.fixture
def without_agent_extras(monkeypatch):
    """Pretend langgraph and ollama are not installed, as on the free hosting tier."""
    import sys

    for name in ("langgraph", "ollama"):
        monkeypatch.setitem(sys.modules, name, None)
    for name in [m for m in sys.modules if m.startswith("autoeda.agent")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.delitem(sys.modules, "autoeda", raising=False)


def test_app_runs_without_the_agent_extras(without_agent_extras):
    test = AppTest.from_file(APP, default_timeout=60)
    test.run()

    assert not test.exception
    next(b for b in test.button if b.label == "Run query").click()
    test.run()
    assert not test.exception


def test_asking_without_the_agent_extras_explains_what_is_missing(without_agent_extras):
    test = AppTest.from_file(APP, default_timeout=60)
    test.run()
    test.text_input[0].set_value("total sales?")
    next(b for b in test.button if b.label == "Ask").click()
    test.run()

    assert not test.exception
    assert any("agent extras" in error.value for error in test.error)


def test_profile_tab_points_to_the_cleaned_score_after_cleaning(app):
    next(b for b in app.button if b.label == "Clean the data").click()
    app.run()

    assert any("After cleaning, the score is" in info.value for info in app.info)
