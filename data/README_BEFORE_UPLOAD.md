# READ BEFORE ADDING A DATASET

This folder feeds our slang↔English translator. **The model does not read these files directly** — a prep script converts them into one clean training file:

```
data/raw/  (you drop files here)  →  [prep script]  →  data/processed/train.jsonl  →  model trains
```

## Folder layout
- `raw/` — datasets that WILL be trained on (must pass the checklist below).
- `dictionaries/` — slang→meaning word lists + emoji meanings. Each entry becomes a short term↔meaning training example (see `DICT_SOURCES` in `src/config.py`); slang dicts go both directions, emoji dicts are emoji→meaning only.
- `processed/` — auto-generated clean files. **Do not edit by hand.**
- `unused/` — files we decided not to use, kept for reference.
- `source info.csv` — where each file came from (URL) + reasons for anything unused.

## Checklist — a file goes into `raw/` only if it passes ALL of these
Otherwise it goes into `unused/` with a one-line reason in `source info.csv`.

1. **Has pairs, not just words.** Every row needs BOTH a slang side and a plain-English side of the same meaning. A list of slang words with no translation can't train a translator.
2. **Columns are identifiable.** You can point to "this column = slang, this column = English." Broken/missing headers → fix first, or send to `unused/`.
3. **Sentences or phrases, not single-word definitions.** Word→meaning dictionaries go in `dictionaries/`, not `raw/`.
4. **Readable text.** Open it, eyeball ~5 rows — real language, not garbled characters or HTML junk.

Good practice (do these too):
5. **Note the source** — add the file name + URL to `source info.csv`.
6. **Right domain** — Gen Z / Alpha slang in English.

## After it passes — wire it in (3 steps, no hand-reformatting)
1. Drop the file in `data/raw/`.
2. Add ONE entry to the `SOURCES` list in the prep script/notebook:
   ```python
   {"file": "your_new_file.csv", "slang_col": "<slang column name>", "english_col": "<english column name>"},
   ```
3. Re-run the prep cell. It rebuilds `data/processed/train.jsonl` (dedupe + both-direction expansion happen automatically).

Files not listed in `SOURCES` are skipped with a warning — so nothing sneaks in unmapped.

## Currently NOT used (see `source info.csv` for full reasons)
- `genz_slang_merged.csv.xlsx` — headers lost in the merge (columns show as 0,1,2…).
- `final dataset.xlsx` — slang sentences but no English pairing (future: pair them and it's usable).
- `KamusGenAlpa.xlsx` — ~68 rows, unlabeled columns.
- `genz_slang_usage_2020_2025.csv` — 99 MB of usage/frequency data, not translation pairs.
- `genz_dataset_model_ready.csv` / `genz_slang_dataset_final(2020_2026).csv` — duplicates of the augmented file.
- `links.csv` — empty.

> Full design rationale: `docs/superpowers/specs/2026-07-13-genz-slang-translator-design.md` (§3, §3a, §3b).
