# Gen Z/Alpha Slang ↔ English Translator — Design Spec (plain-language version)

**Date:** 2026-07-13
**Team:** Group 2 — Jony Ling, Huimin Goh, Hongming Xu, Maxton Huang
**Course:** Week 5 — LLM Training & Fine-tuning (mini-project, 60% of grade)
**Due:** Friday 2026-07-17, 11:30 PM (xsite dropbox)

> **Jargon mini-glossary** (used throughout):
> - **Base model** — the off-the-shelf AI we start with (Llama 3.2 3B), before we teach it anything.
> - **Fine-tuning** — showing the model lots of examples so it *learns new behaviour*. The knowledge ends up baked into the model itself.
> - **SFT (Supervised Fine-Tuning)** — the simplest kind of fine-tuning: we give it pairs of (input, correct answer) and it learns to copy that pattern.
> - **LoRA / QLoRA** — a cheap, memory-light way to fine-tune. Instead of retraining the whole model (needs an expensive GPU), we train a tiny "add-on" layer. **QLoRA** = LoRA + compressing the model to 4-bit so it fits on a small **T4** GPU (the free Colab one).
> - **RAG** — an *alternative* to fine-tuning where the model looks facts up in a document at answer-time. We are deliberately **not** doing this (the assignment asks us to put the skill "in the weights").
> - **Held-out / eval set** — a set of test examples we hide from the model during training, so we can fairly check if it actually learned.

---

## 1. What we're building (task & the problem it solves)

A **two-way translator** between Gen Z/Alpha slang and normal English, built as **one single AI model**. We tell it which direction we want by adding a tag at the start of the prompt:

- `Translate to English: <slang text>` → gives plain English
- `Translate to Gen Z slang: <plain text>` → gives slang

Doing both directions inside *one* model (rather than training two separate ones) keeps the project small — one thing to train, one thing to hand in.

**The problem we're solving (the "gap"):** the base model is not good at recent slang (2023–2026). It guesses or makes up meanings for words like `delulu`, `rizz`, `ate`, `villain era`, `on god`. Our whole project is about closing that gap.

*What we must capture:* at least one clear example of the base model getting slang wrong — this is our "before" evidence for Friday.

## 2. How we'll fix it (our method, and why)

We use **SFT via QLoRA** on **Llama 3.2 3B Instruct**. In plain terms: we feed the model thousands of slang↔English example pairs until the meanings are baked into it, using the cheap LoRA method so it runs on a free T4 GPU.

**Why this method and not others** (the assignment explicitly asks us to justify this):
- The model is *missing knowledge/behaviour* → fine-tuning (SFT/LoRA) is the right tool.
- We are **not** using **RAG** (looking words up in a document) — the assignment wants the skill learned into the model, and slang meanings are short and messy, which RAG handles poorly.
- We are **not** using **DPO** (a technique for picking between two already-good answers based on taste/style) — our problem is "right vs wrong meaning," not "which nice answer is nicer."

## 3. The data (what we train on)

We already have plenty of slang data collected. We'll combine the **clean** files into one training set that covers both directions.

**Files we'll use:**
| File | What's in it | How we use it |
|---|---|---|
| `genz_dataset.csv` (~1000) | pairs of normal ↔ slang sentences | Both directions (just flip which side is input) |
| `genz_dataset_augmented_1500.csv` (~1650) | slang sentence ↔ normal sentence + the meaning | Both directions + meaning reference |
| `genz_synthetic_dataset.csv` (~14.7k) | slang → English, tagged easy/medium/hard | Extra volume for slang→English |
| `all_slangs.csv`, `genz_slang.csv`, `gen_zz_words.csv` | slang → definition dictionaries | To build and double-check test answers (not as sentence training) |

**File we'll skip:** `genz_slang_merged.csv.xlsx` — its column headers are broken/unlabeled, and we already have cleaner data. Not worth the cleanup time.

**Steps:**
1. Reshape every file into one common format.
2. Remove duplicates (the same sentence appearing twice).
3. Convert into the chat format Llama expects, with the direction tag as the instruction.
4. **Set aside the test examples FIRST** (see §4) so the model never sees them during training — otherwise the test is cheating.

*Note:* we don't have to use all 14,700 synthetic rows. A few thousand clean examples is plenty; more isn't always better.

### 3a. Adding new datasets later (drop-in data folder)

The model never reads raw files directly. There's always a prep step in the middle:

```
data/raw/  (anyone drops files here)  →  [prep script normalizes + dedupes]  →  data/processed/train.jsonl  →  model trains
```

**Folder layout:**
- `data/raw/` — any CSV/XLSX a teammate finds online goes here, untouched.
- `data/processed/` — the auto-generated clean training file + frozen test file. Never edited by hand.

**Why raw files can't be auto-consumed as-is:** every dataset names its columns differently (we already have `normal`/`gen_z`, `slang_sentence`/`normal_sentence`, `input_text`/`target_text`, `Word`/`Meaning`). The prep script can't guess which column is slang and which is English. So each source needs a tiny mapping.

**The mapping (the one thing to maintain):** a small list — one short entry per file — declaring which columns mean what. Example shape:

```python
SOURCES = [
    {"file": "genz_dataset.csv",                "slang_col": "gen_z",           "english_col": "normal"},
    {"file": "genz_dataset_augmented_1500.csv", "slang_col": "slang_sentence",  "english_col": "normal_sentence"},
    {"file": "genz_synthetic_dataset.csv",      "slang_col": "input_text",      "english_col": "target_text"},
    # a teammate adds a new file like this:
    # {"file": "new_kaggle_slang.csv",          "slang_col": "<col>",           "english_col": "<col>"},
]
```

**To add a dataset later — 3 steps, no manual reformatting:**
1. Drop the file into `data/raw/`.
2. Add one entry to `SOURCES` naming its slang column and its English column.
3. Re-run the prep cell → it rebuilds `data/processed/train.jsonl` automatically (dedupe + both-direction expansion happen for free).

**Ownership:** the Data & eval person owns `SOURCES` and the prep script; anyone can contribute a dataset by dropping a file and adding one entry, without touching the training code. The prep script skips/loudly-warns on any file not listed in `SOURCES`, so nothing sneaks in unmapped.

### 3b. Data contribution checklist (before adding a dataset)

A copy of this checklist lives in `data/README_BEFORE_UPLOAD.md` so teammates see it in the folder itself. Source URLs and "why-not-used" reasons are tracked in `data/source info.csv`.

**Must pass ALL of these to go into `data/raw/` (else it goes to `data/unused/` with a one-line reason):**
1. **Has pairs, not just words** — every row needs both a slang side and a plain-English side of the *same* meaning. Slang-only files (no translation) can't train a translator.
2. **Columns are identifiable** — you can say "this column = slang, this column = English." Broken/missing headers = fix first or send to `unused/`.
3. **Sentences or phrases, not single-word definitions** — word→meaning dictionaries go in `data/dictionaries/` (used for the test-set reference), not `raw/`.
4. **Readable text** — open it, eyeball ~5 rows. Real language, not mojibake / HTML junk.

**Should have (good practice, and the assignment rewards honest data):**
5. **Source noted** — add the file + its URL to `data/source info.csv`.
6. **Right domain** — Gen Z / Alpha slang in English.

**Then wire it in (from §3a):**
7. Add one entry to the `SOURCES` mapping (file + slang column + English column).
8. Re-run the prep cell.

**If it fails a must-have:** put it in `data/unused/` and add a one-line "why not" to `data/source info.csv` so nobody re-adds it by mistake.

**Note on `final dataset.xlsx` (future data idea):** it holds real scraped slang sentences but no English pairings. If a teammate later pairs those with plain-English translations, it becomes a strong addition — exactly the drop-in case §3a is built for.

## 4. How we'll measure success (the eval)

**Test set:** about **30 slang→English + 30 English→slang** (~60 total), picked to include easy, medium, hard, and a few rare/tricky words. Hidden from training. Each has a "correct meaning" written down for reference.

**Main score — humans grade it (this is the number we defend):**
- Two teammates each mark every answer **correct or incorrect** vs the reference meaning.
- We report the score **for each direction**, and **base model vs fine-tuned model** side by side.
- We also report how often the two graders **agreed** (shows the grading was consistent, not random).

**Bonus score — a second AI grades it (optional, drop if we run out of time):**
- We run a bigger local model (Llama 3.1 8B) as an automatic grader on the same 60 items.
- We report how often the AI grader **agreed with the humans** — a nice "this could scale up" story for the slides.
- No paid API keys needed — everything runs locally on the T4.

**Being honest:** we must also report what did *not* improve, plus 1–2 examples the fine-tuned model still gets wrong and why.

## 5. Tools & setup

**Two environments — this matters:**
- **Local (development):** the author builds and tests here, using **uv** to manage Python packages and versions so the setup is pinned and reproducible.
- **Colab (delivery):** the final notebook is *handed to teammates on Google Colab (T4)*, so it **must run cleanly on Colab from a fresh start** — this is the environment that gets graded/shared.

**Requirement:** the notebook must work in *both* places. To achieve that:
- The notebook's first cell **detects whether it's on Colab or local** and installs dependencies accordingly (`pip install` on Colab; on local it uses the uv-managed environment).
- We pin exact package versions (from the uv lockfile) so Colab installs the *same* versions that worked locally — no "works on my machine" surprises.
- File paths are written to work both locally and on Colab (e.g. a configurable data folder, and a Google Drive mount cell for Colab).
- **Acceptance check:** before hand-off, do one clean run on a fresh Colab T4 runtime, top to bottom, with no manual fixes.

**Libraries & model:**
- **Unsloth** — makes QLoRA training ~2× faster and small enough for a T4. (Backup plan: plain **PEFT + TRL** if Unsloth gives trouble on Colab.)
- **Base model:** `Llama 3.2 3B Instruct`, compressed to 4-bit so it fits the T4. (Gated on Hugging Face — a team member must accept the license and use a free HF token; do this early.)
- **GPU:** T4 (16 GB) on both local and Colab.
- Final notebook runs start-to-finish with a short "how to use it" manual.

## 6. Who does what (the 4-person split)

The assignment defines four roles — one per person:

| Person | Role | What they own |
|---|---|---|
| Data & eval | Data & eval | Combine + clean the data, build and freeze the ~60 test examples, run the human grading |
| Modelling | Modelling | Set up uv, run the QLoRA training on the T4, save checkpoints, generate the before/after answers |
| Analysis & write-up | Analysis | Error analysis, before-vs-after tables, the recommendation write-up |
| Coordinator | Coordinator | Keep scope from ballooning, stitch the notebook together, own the slides, make sure everyone speaks |

Roles can overlap — the point is everyone owns a piece and no one carries it alone.

## 7. Keeping it small (so we actually finish)

The #1 reason teams fail this is doing too much. Our limits:

**We WILL do:**
- One model, two directions via a tag.
- One frozen ~60-item test set.
- Human grading as the main score.
- One training run (maybe try a couple of settings, no more).

**If Thursday gets tight, we CUT in this order:**
1. Drop the second AI grader → humans only (still fully valid).
2. Drop the English→slang direction → ship slang→English only.
3. No app, no website, no live demo tool.
4. Don't chase all 14.7k rows if a smaller clean set already works.

## 8. What we hand in (Friday 11:30 PM, zipped)

- The runnable notebook + a short user manual.
- The data (combined training set + the frozen test set).
- 5–6 slides; **every team member speaks**; 5-min talk + 2-min Q&A.
- The recommendation: **deploy / keep improving / hold**, with reasons.

## 9. What "done well" looks like

- The fine-tuned model clearly beats the base model on the hidden test set (human score), shown per direction.
- We show at least one concrete base-model failure ("before").
- Honest error analysis, including what didn't improve.
- A recommendation we can actually defend in Q&A.
