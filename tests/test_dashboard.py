"""Smoke test for the Streamlit dashboard (skipped when streamlit is absent).

Verifies the app script loads and lays out without raising. The FEM-driven
"Run" path is exercised manually / in the AppTest harness locally; here we keep
it to the initial (no-solve) render so the test stays fast and does not require
streamlit in CI.
"""

import os
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(__file__), "..", "dashboard", "app.py")


def test_dashboard_loads_without_error():
    at = AppTest.from_file(os.path.abspath(APP), default_timeout=60)
    at.run()
    assert not at.exception
    # sidebar controls and the run button are present
    assert len(at.button) >= 1
    assert any("Run simulation" in b.label for b in at.button)
