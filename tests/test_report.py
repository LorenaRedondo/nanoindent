"""Display helpers for notebooks (`nanoindent.report`).

These run headless, which is also the code path the notebook's ``.py``
authoring source takes during development — so the fallback is exercised by
default and the HTML path is asserted on the returned renderable.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoindent import report as rp

pd = pytest.importorskip("pandas", reason="report tables need pandas")


@pytest.fixture(autouse=True)
def _reset():
    rp.reset_counters()


# ------------------------------------------------------------ input shapes --
def test_accepts_a_sequence_of_dicts():
    frame = rp.table([{"model": "Hertz", "E": 58.0},
                      {"model": "Garcia", "E": 6.9}], display=False)
    assert list(frame.columns) == ["model", "E"]
    assert len(frame) == 2


def test_accepts_a_flat_mapping():
    frame = rp.table({"curves": 114, "h": 6.6}, display=False)
    assert list(frame.columns) == ["quantity", "value"]
    assert frame["quantity"].tolist() == ["curves", "h"]


def test_accepts_a_nested_mapping():
    frame = rp.table({"3.7 nm": {"Garcia": 1.44, "Dimitriadis": 3.14},
                      "6.6 nm": {"Garcia": 6.88, "Dimitriadis": 10.41}},
                     display=False)
    assert len(frame) == 2
    assert "Garcia" in frame.columns


def test_accepts_a_dataframe_without_mutating_it():
    original = pd.DataFrame({"a": [1, 2]})
    returned = rp.table(original, display=False)
    returned["a"] = [9, 9]
    assert original["a"].tolist() == [1, 2]      # a copy, not a view


def test_distinct_keys_are_not_collapsed():
    """Regression: truncating labels made two columns share one dict key.

    A sensitivity sweep over 'Dimitriadis (bonded)' and
    'Dimitriadis (unbonded)' silently lost one of them.
    """
    rows = [{"h": 3.7, "Dimitriadis (bonded)": 3.1,
             "Dimitriadis (unbonded)": 5.8}]
    frame = rp.table(rows, display=False)
    assert "Dimitriadis (bonded)" in frame.columns
    assert "Dimitriadis (unbonded)" in frame.columns
    assert len(frame.columns) == 3


# -------------------------------------------------------------- formatting --
@pytest.mark.parametrize("values,expected", [
    ([1.5, 2.5], "{:.2f}"),            # small floats
    ([150.0, 250.0], "{:.1f}"),        # hundreds
    ([1.5e7, 2.5e7], "{:.3e}"),        # very large
    ([1.5e-5, 2e-5], "{:.3e}"),        # very small
    ([1, 2, 3], "{:d}"),               # integers
])
def test_auto_format_picks_by_magnitude(values, expected):
    frame = pd.DataFrame({"x": values})
    assert rp._auto_format(frame)["x"] == expected


def test_non_numeric_columns_are_left_alone():
    frame = pd.DataFrame({"name": ["a", "b"], "x": [1.0, 2.0]})
    fmts = rp._auto_format(frame)
    assert "name" not in fmts and "x" in fmts


def test_headless_output_is_formatted(capsys):
    """The plain-text path must apply the same formatting as the Styler."""
    rp.table([{"E": 57.993483212}], caption="x")
    out = capsys.readouterr().out
    assert "57.99" in out
    assert "57.993483" not in out


def test_explicit_formats_override_the_default(capsys):
    rp.table([{"E": 57.993483}], formats={"E": "{:.4f}"})
    assert "57.9935" in capsys.readouterr().out


def test_nan_renders_as_a_dash(capsys):
    rp.table([{"x": 1.0, "y": float("nan")}])
    assert "—" in capsys.readouterr().out


# ---------------------------------------------------------------- captions --
def test_tables_and_figures_number_independently(capsys):
    rp.table([{"a": 1}], caption="first")
    rp.figure_caption("a figure")
    rp.table([{"a": 1}], caption="second")
    rp.figure_caption("another figure")
    out = capsys.readouterr().out
    assert "Table 1. first" in out and "Table 2. second" in out
    assert "Figure 1. a figure" in out and "Figure 2. another figure" in out


def test_numbering_can_be_suppressed(capsys):
    rp.table([{"a": 1}], caption="unnumbered", number=False)
    out = capsys.readouterr().out
    assert "unnumbered" in out and "Table 1." not in out


def test_reset_counters():
    rp.table([{"a": 1}], caption="x", display=False)
    rp.reset_counters()
    assert rp._COUNTERS == {"figure": 0, "table": 0, "equation": 0}


# --------------------------------------------------------------- callouts ---
@pytest.mark.parametrize("kind", sorted(rp.CALLOUT_STYLES))
def test_every_callout_kind_renders(kind, capsys):
    rp.callout("body text", kind=kind)
    out = capsys.readouterr().out
    assert "body text" in out
    assert rp.CALLOUT_STYLES[kind][2].upper() in out


def test_unknown_callout_kind_raises():
    with pytest.raises(ValueError, match="unknown kind"):
        rp.callout("x", kind="nope")


def test_callout_title_can_be_overridden(capsys):
    rp.callout("x", kind="open", title="Q1 — blocking")
    assert "Q1 — BLOCKING" in capsys.readouterr().out


# --------------------------------------------------------------- metrics ----
def test_metrics_accepts_bare_values_and_value_unit_pairs(capsys):
    rp.metrics({"curves": 114, "thickness": (6.6, "nm")})
    out = capsys.readouterr().out
    assert "114" in out and "6.6" in out and "nm" in out


# ---------------------------------------------------------------- LaTeX -----
def test_equation_strips_delimiters(capsys):
    rp.equation(r"$\chi = \sqrt{R\delta}/h$")
    out = capsys.readouterr().out
    assert r"\chi = \sqrt{R\delta}/h" in out
    assert "$" not in out


# ------------------------------------------------------- return contract ----
@pytest.mark.parametrize("call", [
    lambda: rp.table([{"a": 1}], caption="x"),
    lambda: rp.metrics({"a": 1}),
    lambda: rp.callout("x"),
    lambda: rp.equation("x"),
    lambda: rp.figure_caption("x"),
])
def test_displaying_returns_none(call, capsys):
    """Otherwise a trailing helper call leaves a stray repr in the notebook."""
    assert call() is None
    capsys.readouterr()


def test_display_false_returns_the_renderable():
    assert isinstance(rp.table([{"a": 1}], display=False), pd.DataFrame)
    assert isinstance(rp.callout("hello", display=False), str)
    assert isinstance(rp.figure_caption("cap", display=False), str)


def test_in_notebook_is_false_under_pytest():
    assert rp.in_notebook() is False
