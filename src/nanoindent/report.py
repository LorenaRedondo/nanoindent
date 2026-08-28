"""Display helpers for analysis notebooks and reports.

Notebooks accumulate `print()` calls with hand-aligned f-strings, which are
tedious to write, awkward to restyle and impossible to reuse. This module
replaces them with a handful of functions that render **real HTML tables**,
captioned figures-style blocks and LaTeX, and that **fall back to readable
plain text** when the same code runs as a script.

That fallback is the point: the notebook's ``.py`` authoring source is executed
directly during development, so every helper has to work headless.

    >>> from nanoindent import report
    >>> report.table([{"model": "Hertz", "E": 58.0}], caption="Fitted moduli")
    ... # doctest: +SKIP

Available output
----------------
:func:`table`
    A list of dicts, a DataFrame or a dict-of-dicts as a captioned table with
    per-column formatting and optional row highlighting.
:func:`metrics`
    A compact strip of labelled scalars for run summaries.
:func:`callout`
    A coloured note / warning / result block.
:func:`equation`
    Display LaTeX, optionally numbered.
:func:`figure_caption`
    A numbered, italicised caption to sit under a plot.

All of them accept ``display=False`` to return the renderable instead of
showing it, which is what the tests use. When they *do* display they return
``None``, so a trailing helper call in a notebook cell does not leave a stray
``execute_result`` repr under the rendered output.
"""

from __future__ import annotations

import html as _html
from collections.abc import Mapping
from typing import Any, Optional

import numpy as np

__all__ = ["in_notebook", "table", "metrics", "callout", "equation",
           "figure_caption", "reset_counters", "CALLOUT_STYLES"]

#: Colours for :func:`callout`, keyed by ``kind``. Chosen to stay legible on
#: both light and dark notebook themes.
CALLOUT_STYLES = {
    "note":    ("#0072B2", "rgba(0,114,178,0.08)",  "Note"),
    "result":  ("#009E73", "rgba(0,158,115,0.08)",  "Result"),
    "warning": ("#D55E00", "rgba(213,94,0,0.08)",   "Caveat"),
    "open":    ("#CC79A7", "rgba(204,121,167,0.08)", "Open question"),
}

_COUNTERS = {"figure": 0, "table": 0, "equation": 0}


def reset_counters() -> None:
    """Zero the figure/table/equation numbering (call once per notebook run)."""
    for key in _COUNTERS:
        _COUNTERS[key] = 0


def _next(kind: str) -> int:
    _COUNTERS[kind] += 1
    return _COUNTERS[kind]


def in_notebook() -> bool:
    """True when running under an IPython kernel that can render HTML."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


def _show(obj, plain: str, display: bool):
    """Render ``obj`` in a notebook, print ``plain`` otherwise."""
    if not display:
        return obj
    if in_notebook():
        from IPython.display import display as _d
        _d(obj)
    else:
        print(plain)
    return obj


# ------------------------------------------------------------------ tables --
def _to_frame(data):
    import pandas as pd

    if hasattr(data, "columns"):                     # already a DataFrame
        return data.copy()
    if isinstance(data, Mapping):
        first = next(iter(data.values()), None)
        if isinstance(first, Mapping):               # dict of dicts -> rows
            frame = pd.DataFrame(data).T
            frame.insert(0, "", list(data))
            return frame.reset_index(drop=True)
        return pd.DataFrame({"quantity": list(data), "value": list(data.values())})
    return pd.DataFrame(list(data))


def _auto_format(frame) -> dict:
    """Sensible per-column float formatting based on magnitude."""
    import numpy as np
    import pandas as pd

    fmts = {}
    for col in frame.columns:
        series = frame[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        if pd.api.types.is_integer_dtype(series):
            fmts[col] = "{:d}"
            continue
        finite = series[np.isfinite(series)] if len(series) else series
        if not len(finite):
            fmts[col] = "{:.3g}"
            continue
        scale = float(np.nanmax(np.abs(finite)))
        if scale and (scale >= 1e5 or scale < 1e-3):
            fmts[col] = "{:.3e}"
        elif scale >= 100:
            fmts[col] = "{:.1f}"
        else:
            fmts[col] = "{:.2f}"
    return fmts


def table(data, caption: Optional[str] = None, formats: Optional[Mapping] = None,
          index: bool = False, highlight=None, number: bool = True,
          note: Optional[str] = None, display: bool = True):
    """Render tabular data as a captioned table.

    Parameters
    ----------
    data:
        A DataFrame, a sequence of dicts, a flat ``{name: value}`` mapping, or a
        nested ``{row: {column: value}}`` mapping.
    caption:
        Caption text. Numbered ``Table N.`` automatically unless
        ``number=False``.
    formats:
        Per-column format strings, e.g. ``{"E (MPa)": "{:.2f}"}``. Columns not
        listed get a magnitude-appropriate default.
    highlight:
        ``callable(row) -> bool``; matching rows are tinted. Useful for marking
        the measured case in a sensitivity sweep.
    note:
        A short line rendered under the table, for provenance or a caveat.
    """
    frame = _to_frame(data)
    fmt = dict(_auto_format(frame))
    if formats:
        fmt.update(formats)

    label = ""
    if caption is not None:
        label = f"Table {_next('table')}. " if number else ""

    if not in_notebook() or not display:
        # Apply the same formatting the Styler would, so the headless output
        # of the .py source is as readable as the rendered notebook.
        shown = frame.copy()
        for col, spec in fmt.items():
            if col in shown.columns:
                shown[col] = [spec.format(x)
                              if isinstance(x, (int, float))
                              and not (isinstance(x, float) and np.isnan(x))
                              else "—"
                              for x in shown[col]]
        lines = []
        if caption:
            lines.append(f"{label}{caption}")
        lines.append(shown.to_string(index=index))
        if note:
            lines.append(f"  ({note})")
        plain = "\n".join(lines)
        if not display:
            return frame
        print(plain)
        return None

    styler = frame.style.format(fmt, na_rep="—")
    if not index:
        styler = styler.hide(axis="index")
    if caption:
        styler = styler.set_caption(f"<b>{label}</b>{_html.escape(caption)}")
    styler = styler.set_table_styles([
        {"selector": "caption",
         "props": [("caption-side", "top"), ("text-align", "left"),
                   ("padding", "0.4em 0"), ("font-size", "0.92em"),
                   ("line-height", "1.4")]},
        {"selector": "th",
         "props": [("text-align", "right"), ("padding", "0.3em 0.7em"),
                   ("border-bottom", "1.5px solid currentColor"),
                   ("font-weight", "600")]},
        {"selector": "td",
         "props": [("text-align", "right"), ("padding", "0.25em 0.7em"),
                   ("font-variant-numeric", "tabular-nums")]},
        {"selector": "th.col_heading:first-child, td:first-child",
         "props": [("text-align", "left")]},
        {"selector": "", "props": [("border-collapse", "collapse")]},
    ])
    if highlight is not None:
        def _row_style(row):
            hit = highlight(row)
            return ["background-color: rgba(0,158,115,0.13)" if hit else ""] * len(row)
        styler = styler.apply(_row_style, axis=1)

    from IPython.display import HTML
    from IPython.display import display as _d
    _d(styler)
    if note:
        _d(HTML(f'<div style="font-size:0.85em;opacity:0.75;'
                f'margin:-0.4em 0 1em 0">{_html.escape(note)}</div>'))
    return None


# ----------------------------------------------------------------- metrics --
def metrics(values: Mapping[str, Any], caption: Optional[str] = None,
            display: bool = True):
    """A compact strip of labelled scalars, for run summaries.

    ``values`` maps a label to either a value or a ``(value, unit)`` pair.
    """
    items = []
    for label, raw in values.items():
        value, unit = raw if isinstance(raw, tuple) else (raw, "")
        text = f"{value:.4g}" if isinstance(value, float) else str(value)
        items.append((label, text, unit))

    if not in_notebook() or not display:
        plain = "\n".join(f"  {lbl:34} {val}{(' ' + u) if u else ''}"
                          for lbl, val, u in items)
        if caption:
            plain = f"{caption}\n{plain}"
        if not display:
            return items
        print(plain)
        return None

    cells = "".join(
        f'<div style="padding:0.45em 1.1em 0.45em 0;min-width:8.5em">'
        f'<div style="font-size:0.76em;opacity:0.7;text-transform:uppercase;'
        f'letter-spacing:0.04em">{_html.escape(lbl)}</div>'
        f'<div style="font-size:1.18em;font-variant-numeric:tabular-nums">'
        f'{_html.escape(val)}'
        f'<span style="font-size:0.7em;opacity:0.7"> {_html.escape(u)}</span>'
        f'</div></div>' for lbl, val, u in items)
    head = (f'<div style="font-size:0.9em;opacity:0.8;margin-bottom:0.3em">'
            f'{_html.escape(caption)}</div>' if caption else "")
    from IPython.display import HTML
    from IPython.display import display as _d
    _d(HTML(f'{head}<div style="display:flex;flex-wrap:wrap;'
            f'border-left:3px solid #0072B2;padding-left:0.9em;'
            f'margin:0.5em 0 1em 0">{cells}</div>'))
    return None


# ---------------------------------------------------------------- callouts --
def callout(text: str, kind: str = "note", title: Optional[str] = None,
            display: bool = True):
    """A coloured block for a note, result, caveat or open question.

    ``kind`` is one of :data:`CALLOUT_STYLES`. Markdown emphasis and ``$…$``
    maths in ``text`` are passed through to the notebook renderer.
    """
    if kind not in CALLOUT_STYLES:
        raise ValueError(f"unknown kind {kind!r}; expected "
                         f"{sorted(CALLOUT_STYLES)}")
    colour, background, default_title = CALLOUT_STYLES[kind]
    heading = title if title is not None else default_title

    if not in_notebook() or not display:
        plain = f"[{heading.upper()}] {text}"
        if not display:
            return plain
        print(plain)
        return None

    from IPython.display import Markdown
    from IPython.display import display as _d
    _d(Markdown(
        f'<div style="border-left:4px solid {colour};background:{background};'
        f'padding:0.7em 1em;margin:0.8em 0;border-radius:0 4px 4px 0">'
        f'<div style="font-weight:600;color:{colour};font-size:0.82em;'
        f'text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.3em">'
        f'{_html.escape(heading)}</div>\n\n{text}\n\n</div>'))
    return None


# ----------------------------------------------------------------- LaTeX ----
def equation(latex: str, label: Optional[str] = None, number: bool = False,
             display: bool = True):
    """Display a LaTeX equation.

    Markdown cells already render ``$…$`` via MathJax; this is for equations
    produced *by code*, e.g. a fitted model written out with its coefficients.
    """
    tag = ""
    if number:
        tag = f"\\qquad ({_next('equation')})"
    body = latex.strip().strip("$")

    if not in_notebook() or not display:
        plain = f"    {body}" + (f"   [{label}]" if label else "")
        if not display:
            return body
        print(plain)
        return None

    from IPython.display import Math
    from IPython.display import display as _d
    _d(Math(body + tag))
    return None


def figure_caption(text: str, number: bool = True, display: bool = True):
    """An italicised, numbered caption to place under a plot."""
    label = f"Figure {_next('figure')}. " if number else ""
    if not in_notebook() or not display:
        plain = f"{label}{text}"
        if not display:
            return plain
        print(plain)
        return None

    from IPython.display import Markdown
    from IPython.display import display as _d
    _d(Markdown(f'<div style="font-size:0.9em;line-height:1.5;'
                f'margin:0.2em 0 1.4em 0"><b>{label}</b>{text}</div>'))
    return None
