from pathlib import Path

import numpy as np
import pandas as pd

from autoeda import clean, compare, profile
from autoeda.report import render_drift, render_report

SAMPLE = Path(__file__).parent.parent / "examples" / "sample_messy.csv"


def test_report_contains_the_main_sections():
    report = profile(SAMPLE)

    html = render_report(report, cleaned=clean(SAMPLE))

    for expected in ("Data quality report", "Issues (13)", "After cleaning", "Columns"):
        assert expected in html
    assert "empty_column:internal_flag" in html
    assert "df.drop(columns=" in html


def test_plotly_script_is_loaded_before_the_charts():
    html = render_report(profile(SAMPLE))

    assert html.index("cdn.plot.ly") < html.index("Plotly.newPlot")


def test_inline_mode_embeds_the_library_and_avoids_the_cdn():
    html = render_report(profile(SAMPLE), plotly_js="inline")

    assert '<script src="https://cdn.plot.ly' not in html
    assert len(html) > 1_000_000


def test_hostile_text_in_the_data_cannot_inject_markup():
    evil = "<script>alert(1)</script>"
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            evil: rng.normal(size=100),
            "city": [evil, "Pune", "Delhi", "Mumbai"] * 25,
            "other": rng.normal(size=100),
        }
    )

    html = render_report(profile(frame))

    assert "<script>alert(1)" not in html


def test_drift_page_renders():
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({"x": rng.normal(0, 1, 400)})
    current = pd.DataFrame({"x": rng.normal(2, 1, 400), "new": 1.0})

    html = render_drift(compare(reference, current))

    assert "Data drift report" in html
    assert "New in the current data" in html
    assert "PSI" in html


def test_report_for_a_tiny_table_renders():
    html = render_report(profile(pd.DataFrame({"a": [1.5, 2.5, 3.5]})))

    assert "Data quality report" in html
