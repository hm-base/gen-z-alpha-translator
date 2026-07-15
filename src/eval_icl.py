"""Stage 1 — ICL ceiling test.

Runs the BASE model over the frozen eval set twice: plain, and with a slang
glossary injected into the prompt. Scores per-direction with the NVIDIA judge.
If the glossary fixes slang->English, prompting is enough; if not, we've proven
that direction needs training data (justifies Stage 2 SDG).

Usage:  uv run python src/eval_icl.py
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from config import DICT_DIR, DICT_SOURCES, EVAL_PATH
from teacher import chat, extract_json, get_client


def build_glossary(max_terms: int = 40) -> str:
    terms: list[tuple[str, str]] = []
    seen = set()
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
            terms.append((t, m))
            if len(terms) >= max_terms:
                return "\n".join(f"{t} = {m}" for t, m in terms)
    return "\n".join(f"{t} = {m}" for t, m in terms)


def judge_translation(client, direction: str, source: str, reference: str,
                      candidate: str) -> int:
    lang = "plain English" if direction == "to_english" else "Gen Z slang"
    prompt = (
        f"You are grading a translation into {lang}.\n"
        f"Source: {source}\nReference answer: {reference}\n"
        f"Candidate answer: {candidate}\n"
        "Is the candidate correct IN MEANING (wording may differ)? "
        "Reply with ONLY a JSON object: {\"correct\": true} or {\"correct\": false}."
    )
    out = chat(client, prompt, temperature=0.0, max_tokens=200)
    j = extract_json(out) or {}
    return 1 if j.get("correct") is True else 0


def _base_translate(client_unused, model, tokenizer, tag: str, text: str,
                    glossary: str) -> str:
    from translate_core import generate_translation
    prompt_text = text if not glossary else f"Glossary:\n{glossary}\n\n{text}"
    return generate_translation(model, tokenizer, tag, prompt_text)


def main() -> int:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    eval_rows = [json.loads(l) for l in open(EVAL_PATH, encoding="utf-8")]
    translate_rows = [r for r in eval_rows if r.get("type") == "translate"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        max_seq_length=1024, dtype=None, load_in_4bit=True)
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    FastLanguageModel.for_inference(model)

    glossary = build_glossary()
    client = get_client()

    results = {("plain", "to_english"): [], ("plain", "to_slang"): [],
               ("glossary", "to_english"): [], ("glossary", "to_slang"): []}
    for cond, gloss in [("plain", ""), ("glossary", glossary)]:
        for r in translate_rows:
            cand = _base_translate(None, model, tokenizer, r["tag"], r["input"], gloss)
            score = judge_translation(client, r["direction"], r["input"],
                                      r["reference"], cand)
            results[(cond, r["direction"])].append(score)

    print("\n=== ICL CEILING TEST (base model, judged per direction) ===")
    for direction in ("to_english", "to_slang"):
        p = results[("plain", direction)]
        g = results[("glossary", direction)]
        pa = sum(p) / len(p) if p else 0
        ga = sum(g) / len(g) if g else 0
        print(f"{direction:>11}: plain {pa:.0%}  ->  +glossary {ga:.0%}  (n={len(p)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
