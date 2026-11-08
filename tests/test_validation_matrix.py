"""Every test referenced by docs/validation.md must exist."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_validation_matrix_references_exist():
    text = (ROOT / "docs" / "validation.md").read_text()
    refs = re.findall(r"`(tests/[\w./]+\.py)(?:::(\w+))?`", text)
    assert refs
    missing = []
    for path, name in refs:
        f = ROOT / path
        if not f.exists():
            missing.append(path)
        elif name and f"def {name}" not in f.read_text():
            missing.append(f"{path}::{name}")
    assert not missing, missing
