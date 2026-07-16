import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from slang_terms import load_dict_pairs, load_dict_terms


def test_load_dict_terms_nonempty_sorted_unique():
    terms = load_dict_terms()
    assert terms, "expected a non-empty term list"
    assert all(isinstance(t, str) for t in terms)
    assert terms == sorted(terms)
    assert len(terms) == len(set(terms))


def test_load_dict_pairs_nonempty_unique_lowercased_terms():
    pairs = load_dict_pairs()
    assert pairs, "expected non-empty (term, meaning) pairs"
    assert all(isinstance(t, str) and isinstance(m, str) and t and m
               for t, m in pairs)
    lowered = [t.lower() for t, _ in pairs]
    assert len(lowered) == len(set(lowered))
