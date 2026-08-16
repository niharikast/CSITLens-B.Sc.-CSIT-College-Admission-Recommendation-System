# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A CSIT (Computer Science and Information Technology) college admission recommendation system for
Nepal's entrance exam process. Given a student's entrance rank and an ordered list of preferred CSIT
colleges, it predicts which colleges the student is likely to be admitted to. The pipeline goes: raw
multi-year admission records -> cleaned/standardized master dataset -> cutoff forecasting model +
admit/reject classifier -> recommendation engine.

```bash
pip install -r requirements.txt
```

## Pipeline order (each stage's output feeds the next)

Run in this order from a clean state — later stages read files written by earlier ones:

```bash
# 1. Clean raw admission records, standardize college names, engineer features
python build_master_dataset.py --input fyp_dataset_combine_79_80_81_82_merged.xlsx --outdir .
#    writes: name_mapping_audit.csv, final_master_dataset.csv, college_master_ranking.csv

# 2. Build a labeled admit/reject dataset from students' priority lists + actual outcomes
python build_priority_training_data.py --raw-xlsx fyp_dataset_combine_79_80_81_82_merged.xlsx \
    --priority-file combined_priority_lists_split__1.xlsx --ranking college_master_ranking.csv --outdir .
#    writes: labeled_admissions.csv

# 3. Train the cutoff-forecasting regressor (predicts next year's cutoff per college)
python train_model.py --dataset final_master_dataset.csv --outdir . --test-year 2082 --next-year 2083
#    writes: leakfree_training_features.csv, predicted_cutoffs_2083.csv

# 4. Train the admit/reject classifier (real ground truth, replaces the hand-designed formula)
python train_classifier.py --labeled labeled_admissions.csv --dataset final_master_dataset.csv --outdir .
#    writes: admission_classifier.pkl

# 5. Run a recommendation for a specific student
python recommendation_engine.py --rank 1307 --ranking college_master_ranking.csv \
    --predicted predicted_cutoffs_2083.csv --classifier admission_classifier.pkl \
    --preferences "Pulchowk" "Thapathali" "Padma Kanya" "Kathford" "Himalaya"

# Standalone comparison test for the Source College matching fix
python test_source_college_logic.py --ranking college_master_ranking.csv --predicted predicted_cutoffs_2083.csv
```

`--predicted` and `--classifier` are optional in `recommendation_engine.py`: omitting either falls back
to the latest actual cutoff / the rule-based sigmoid formula respectively, so the engine can run right
after step 1 without steps 2-4.

## Architecture

**Rank convention**: lower number = better score. Rank 1 is the best. A college's "cutoff" is the
*worst* (numerically highest) rank admitted that year — the last student who got a seat.

### `build_master_dataset.py` — canonicalization and feature engineering
- `build_alias_map()` is the single source of truth for turning ~134 raw, inconsistently-formatted
  college name strings into canonical names (typo fixes, then grouping by base name before the first
  comma). It is re-imported by `build_priority_training_data.py` so admitted-college names and
  priority-list names resolve to the *same* canonical strings. Don't duplicate this logic elsewhere.
- `DO_NOT_MERGE_BASES` guards colleges that share a base name but are genuinely different institutions
  in different districts (e.g. "Mahendra Multiple Campus" in Dang vs. Nepalgunj) — the grouping key
  keeps the full name instead of collapsing to the base.
- Produces two shapes of output: `final_master_dataset.csv` (one row per College x Year, with
  leakage-*prone* engineered columns like `Cutoff_Trend`/`Rolling_Avg_Cutoff` computed using each row's
  own value — fine for exploration/reporting, **not safe as model input**) and
  `college_master_ranking.csv` (one row per college, aggregated across all years, including
  `College_Master_Rank` — rank 1 = most competitive — used throughout the recommendation engine).

### `build_priority_training_data.py` — real admit/reject labels from priority lists
Nepal's CEE allocation is sequential per student: colleges listed *above* their actual admitted college
in their priority list were wanted more but not obtained (label 0); the admitted college is label 1;
colleges listed *after* it were never reached by the allocation process and are excluded (not labeled
either way — we don't know what would have happened). Two data-quality problems this file works around:
- ~16% of raw priority cells have two college names crammed into one cell (a PDF-extraction artifact).
  `get_tokens()` flattens all `priority_1..39` cells per student and re-splits on whitespace rather than
  trusting column boundaries.
- Priority-list names are truncated to a single prefix word (e.g. "Amrit-"), and some prefixes are
  ambiguous across multiple real colleges. Ambiguous/unmatched prefixes are excluded by default rather
  than guessed — except when the student's *known* admitted college (ground truth from the raw xlsx) is
  one of the ambiguous prefix's candidates, in which case that specific occurrence can be safely
  resolved without guessing (see `ambiguous_candidates` / "confirmed ambiguous" recovery in
  `build_labeled_dataset()`).

### `train_model.py` — leakage-free cutoff forecasting
`build_leakfree_features()` is the critical shared function (also imported by `train_classifier.py`):
for each (College, Year) row it computes features using *only* that college's cutoff history from years
strictly before that year (via slicing prior to the row's own index, not `.shift()` on the full
pre-aggregated columns from `final_master_dataset.csv`, which would leak the row's own value). Do not
feed the pre-computed `Cutoff_Trend`/`Rolling_Avg_Cutoff`/etc. columns from `final_master_dataset.csv`
directly into any model — rebuild features via `build_leakfree_features()` instead.

Uses a time-based train/test split (train on years before `--test-year`, test on `--test-year`) rather
than a random split, since shuffling years would let the model see the future. Evaluates against a
naive "predict last year's cutoff unchanged" baseline — a real model should beat this.

### `train_classifier.py` — real admit/reject classifier
Reuses `build_leakfree_features()` from `train_model.py`, merges it onto `labeled_admissions.csv` by
`(college, year)`, and trains a Random Forest classifier on genuine outcomes — replacing the
hand-designed sigmoid formula in `recommendation_engine.py`. Also uses the same time-based split
(test year 2082) and reports Accuracy/Precision/Recall/F1/ROC-AUC against the test-set admit-rate
baseline. Saved via `joblib.dump({'model': rf, 'feature_cols': FEATURE_COLS}, ...)` — both the model and
its expected feature column order/list are bundled together so inference code doesn't have to guess
column order.

### `recommendation_engine.py` — inference / prediction layer
- `find_source_college()`: the "Source College" is the most competitive college the student could
  *still get into* — i.e., among colleges whose cutoff >= student rank, the one with the smallest such
  cutoff. This is intentionally **not** nearest-by-absolute-distance: a college 199 positions too
  competitive and one 238 positions comfortably safe are opposite outcomes, not similarly close matches.
  Falls back to the least competitive college (flagged as a weak match) only if literally nothing is
  achievable. `test_source_college_logic.py` exists specifically to demonstrate this fix against the old
  naive absolute-distance logic — read it before touching this function.
- Admission probability comes from one of two sources, chosen automatically based on what's passed in:
  the trained classifier (`--classifier`, preferred — real outcomes) or a hand-designed logistic sigmoid
  over `Destination_Master_Rank - Source_Master_Rank` (`score_to_probability()`, fallback). When the
  classifier is used, `Destination_Rank`/`Score` columns are dropped from output rather than shown
  alongside a probability they didn't actually produce.
- `match_college_name()` fuzzy-matches free-text user input (e.g. "Kathford") to a canonical college
  name using rapidfuzz's `partial_ratio` specifically — `WRatio`/`token_sort_ratio` were tried and found
  to mismatch short queries during testing.
- `top5_by_cutoff()` computes the Source-Destination score explicitly for every college and returns the
  Source College plus the next 4 easiest colleges (Score >= 0, ascending). Can legitimately return fewer
  than 5 rows if the Source College is near the bottom of the master ranking.
- `destination_cutoff(row)` — best available cutoff for a college row (`Predicted_Cutoff_Rank` >
  `Latest_Cutoff_Rank` > `Avg_Cutoff_Rank_AllYears`, first non-NaN wins). Used to populate
  `Destination_Cutoff_Rank`/`Rank_Gap` on every `preferences` row unconditionally (added for the
  frontend — the CLI `run()` path doesn't need these).
- `attach_admission_probability(candidates_df, student_rank, classifier=None, classifier_features=None,
  priority_position=1)` — bolts `Admission_Probability`/`Probability_Label` onto a dataframe that didn't
  go through `recommend_from_preference_list()` (specifically, `top5_by_cutoff()`'s output — colleges
  suggested to the student, not part of their submitted priority list). Since there's no real priority
  position for a suggested college, it defaults to `priority_position=1`. Falls back to the rule-based
  sigmoid (using the dataframe's existing `Score` column) if no classifier is supplied, mirroring
  `recommend_from_preference_list()`'s own fallback.

### `api.py` — FastAPI backend
Wraps `recommendation_engine.py` as a JSON API for the (future) web frontend. Loads
`college_master_ranking.csv`, `predicted_cutoffs_2083.csv`, and `admission_classifier.pkl` once at
startup via a `lifespan` context manager (not per-request) into a module-level `state` dict.

```bash
uvicorn api:app --reload
```

Endpoints:
- `GET /health` — liveness check.
- `GET /colleges` — dropdown data: one row per college (`College`, `College_Master_Rank`,
  `Rank_Category`, `Latest_Cutoff_Rank`, `Predicted_Cutoff_Rank`, `Avg_Cutoff_Rank_AllYears`), sorted
  most-competitive first.
- `POST /recommend` — body `{"rank": int, "preferences": [str, ...]}` (rank must be > 0, preferences
  non-empty — both enforced by Pydantic, return 422 otherwise). Returns:
  - `source_college` — a `college_master_ranking.csv` row plus `Source_Match_Gap`/`Source_Match_Is_Weak`.
  - `preferences` — one row per input college: `User_Input, Matched_College, Admission_Probability,
    Probability_Label, Name_Match_Confidence, Probability_Source, Destination_Cutoff_Rank, Rank_Gap,
    Preference_Order`. `Matched_College`/`Admission_Probability` are `null` for unmatched free-text
    input (`Probability_Label` becomes `"Not Found"`). Sorted by probability descending — re-sort by
    `Preference_Order` client-side to restore the student's own list order.
  - `best_college` — same shape as one `preferences` row, or `null` if nothing matched.
  - `top5_recommendations` — `college_master_ranking.csv` columns + `Score` + `Admission_Probability`/
    `Probability_Label` (5 rows, or fewer near the bottom of the ranking).

`_records()`/`_record()` sanitize pandas DataFrames/Series to JSON-safe output (NaN -> `null`, numpy
dtypes -> native Python) via `df.to_json()` + `json.loads()`, rather than `.to_dict()`, which chokes on
numpy scalar types and leaves NaN as an invalid JSON token.

**`Destination_Cutoff_Rank`/`Rank_Gap` and top5's probability fields were added specifically for the
frontend** (see `recommendation_engine.py` notes below) — they don't exist in the CLI (`run()`) output.

**Known gap** (left as-is per explicit decision, not an oversight): `top5_by_cutoff()` has no
`preference_list` parameter, so `/recommend`'s `top5_recommendations` can include colleges the student
already listed in `preferences` — it is not deduplicated against their own list.

### `frontend/` — static UI, served by `api.py`
`index.html` + `app.js`, plain HTML/CSS/JS (no build step, no framework). Mounted at `/` via
`StaticFiles` at the bottom of `api.py` (after all `@app.get`/`@app.post` routes, so it doesn't shadow
them) — visiting `http://127.0.0.1:8000/` after `uvicorn api:app --reload` serves the whole thing from
one process/port.

- All data comes from `fetch('/colleges')` and `fetch('/recommend', ...)` — no hardcoded college list or
  client-side probability math. `TONE_BY_LABEL` in `app.js` maps the backend's 5 probability tiers to 5
  distinct badge colors (`Very High`->sky, `High`->blue, `Moderate`->maroon, `Low`->rust, `Very
  Low`->red) — all reused from the existing CSS custom-property palette rather than new colors.
- The ticket list ("Where your priority list stands") re-sorts the API's `preferences` array by
  `Preference_Order` before rendering, since the API itself returns it sorted by probability — the UI
  intentionally shows the student's own priority order, not a probability leaderboard.
- `resetResults()` clears the best-college callout, ticket list, and recommendations together on every
  new evaluation attempt (including client-side validation failures like rank <= 0 or an empty
  preference list). This was a real bug found during manual browser testing: showing a validation error
  in the ticket list while a *previous* evaluation's results stayed visible elsewhere on the page was
  misleading. Always route result-area updates through this function rather than clearing individual
  `innerHTML`s piecemeal.

## Data files (not source — regenerated by the pipeline, but check in the working tree for current state)

- `fyp_dataset_combine_79_80_81_82_merged.xlsx` — raw per-student admission records, entrance years
  2079-2082 BS. Primary input to stage 1.
- `combined_priority_lists_split__1.xlsx` — per-student ordered college preference lists. Primary input
  to stage 2, alongside the raw admission xlsx above.
- `final_master_dataset.csv`, `college_master_ranking.csv`, `name_mapping_audit.csv` — stage 1 outputs.
- `labeled_admissions.csv` — stage 2 output.
- `leakfree_training_features.csv`, `predicted_cutoffs_2083.csv` — stage 3 outputs.
- `admission_classifier.pkl` — stage 4 output (joblib bundle: `{'model', 'feature_cols'}`).

When changing name-canonicalization logic in `build_master_dataset.py`, re-run the full pipeline (all
4 stages) rather than patching individual CSVs — `college_master_ranking.csv`,
`labeled_admissions.csv`, and the trained model all derive from canonical names and will silently
drift out of sync otherwise.
