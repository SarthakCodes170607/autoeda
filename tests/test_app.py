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
