"""Shared readers for slang terms / term-meaning pairs from the dictionary CSVs."""
from __future__ import annotations
import pandas as pd
from config import DICT_DIR, DICT_SOURCES


def load_dict_pairs() -> list[tuple[str, str]]:
    """Unique (term, meaning) pairs from non-emoji dictionaries, in source order."""
    pairs, seen = [], set()
    for src in DICT_SOURCES:
        if src.get("emoji"):
            continue
        p = DICT_DIR / src["file"]
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if src["term_col"] not in df.columns or src["meaning_col"] not in df.columns:
            continue
        for _, row in df.iterrows():
            t = str(row.get(src["term_col"], "")).strip()
            m = str(row.get(src["meaning_col"], "")).strip()
            if not t or not m or t.lower() in seen:
                continue
            seen.add(t.lower())
            pairs.append((t, m))
    return pairs


def load_dict_terms() -> list[str]:
    """Unique slang terms (term_col only, length 1-40), sorted."""
    terms: set[str] = set()
    for src in DICT_SOURCES:
        if src.get("emoji"):
            continue
        p = DICT_DIR / src["file"]
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if src["term_col"] in df.columns:
            terms |= {str(t).strip() for t in df[src["term_col"]].dropna()}
    return sorted({t for t in terms if t and 1 <= len(t) <= 40})
