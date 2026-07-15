import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval_icl import build_glossary


def test_glossary_caps_and_formats():
    g = build_glossary(max_terms=5)
    lines = [ln for ln in g.splitlines() if ln.strip()]
    assert len(lines) <= 5
    assert all(" = " in ln for ln in lines)  # "term = meaning"
