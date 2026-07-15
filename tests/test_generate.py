import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sdg.attributes import Recipe
from sdg.generate import build_prompt


def test_build_prompt_mentions_term_and_json():
    r = Recipe(direction="to_english", term="delulu", tone="playful",
               difficulty="clear", context="texting a friend", is_hard_negative=False)
    system, user = build_prompt(r)
    assert "delulu" in user
    assert "playful" in user
    assert "JSON" in user or "json" in user
    assert '"slang"' in user and '"english"' in user


def test_build_prompt_hard_negative_instruction():
    r = Recipe(direction="to_english", term="fire", tone="deadpan",
               difficulty="edge", context="group chat", is_hard_negative=True)
    _, user = build_prompt(r)
    assert "literal" in user.lower() or "not slang" in user.lower()
