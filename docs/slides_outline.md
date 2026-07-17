# Presentation Outline — ~5 min, 7 slides (every member speaks)

Maps to the assignment (problem → gap → method → results → error analysis → recommendation)
plus one methodology slide on why slang translation is hard. Aim ~40–45 sec per slide.
**Note:** brief said 5–6 max — keep slide 6 brisk (~40s) so you stay near 5 minutes.

---

## Slide 1 — Problem (speaker: **Coordinator**)  ~40s
**Title:** Gen Z / Alpha Slang ↔ English Translator
- One model, **both directions**, chosen by a tag.
- Base: **Llama 3.2 3B Instruct**; method: **QLoRA**.

---

## Slide 2 — The gap (speaker: **Analysis**)  ~50s
**Title:** Where the base model falls short
- Show `6Y` → “six years old” failure.
- Knowledge + behaviour gap → SFT, not RAG.

---

## Slide 3 — What we did (speaker: **Modelling**)  ~50s
**Title:** SFT / QLoRA on Llama 3.2 3B
- Data → freeze 70-item eval → QLoRA → human + auto eval.
- Why SFT not DPO/RAG.

---

## Slide 4 — Results (speaker: **Data & eval**)  ~50s
**Title:** Before vs after (honest numbers)

| | Base | Tuned |
|---|---|---|
| Slang → English | 37% | 37% |
| English → Slang | 60% | 20% |
| Overall | **48%** | **28%** |

- Agreement **71%**. Abstain (auto) **10% → 100%**.

---

## Slide 5 — Error analysis (speaker: **Analysis**)  ~40s
**Title:** What it still gets wrong
- English→slang acronym junk (`SIR`, `YLYITN`, `TNTF`).
- Rare acronym / sense misses.

---

## Slide 6 — Why it’s tricky (speaker: **Analysis** or **Modelling**)  ~40s
**Title:** Why slang translation is so tricky
- Many valid answers (esp. English→slang).
- Meaning ≠ surface form (decode / prompts matter).
- Data style leaks into behaviour (acronym soup).
- Base already paraphrases well — fine-tune can hurt.
- Eval design matters (gloss vs held-out).
- Two directions ≠ one skill.

*This slide explains why numbers can look “bad” even with careful work.*

---

## Slide 7 — Recommendation (speaker: **Coordinator**)  ~40s
**Title:** ITERATE (team call with Huimin)
- Not deploy yet: human translate accuracy fell (48% → 28%); English→slang broke.
- Not hold/quit: keep abstain win (10% → 100%); hard domain → keep improving.
- Next: looser decode, less acronym-heavy data, more natural pairs, re-grade.

---

### Speaking split
- **Coordinator:** 1 + 7
- **Analysis:** 2 + 5 (+ 6 if they take it)
- **Modelling:** 3 (+ 6 if Analysis is overloaded)
- **Data & eval:** 4

### Timing
- 5 min + 2 min Q&A. Practice once with a timer.
- Have grading sheet / metrics JSON ready for Q&A.
