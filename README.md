# CSITLens

Data-driven college recommendation system for TU B.Sc. CSIT admissions: predicts cutoff ranks and
admission probability from real IOST historical data using Random Forest, served via FastAPI.

Given a student's entrance rank and an ordered list of preferred CSIT colleges, it predicts which
colleges they're likely to be admitted to. The pipeline goes: raw multi-year admission records ->
cleaned/standardized master dataset -> cutoff forecasting model + admit/reject classifier ->
recommendation engine.

## Setup

```bash
pip install -r requirements.txt
```

## How the pipeline works

The project runs in five stages, from a clean state, where each stage reads the files the previous
one wrote:

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

`--predicted` and `--classifier` are both optional in `recommendation_engine.py`. Leave either one out
and it falls back to the latest actual cutoff, or the rule-based sigmoid formula, so you can run the
engine right after stage 1 without needing to train anything first.

## How it works

**Rank convention**: lower is better. Rank 1 is the top scorer. A college's "cutoff" is the *worst*
rank it admitted that year — the last student who still got a seat.

### `build_master_dataset.py` — cleaning and standardizing college names

The raw admission records use around 134 different spellings and formats for the same set of
colleges — typos, inconsistent punctuation, abbreviations, you name it. `build_alias_map()` cleans
this up: it fixes known typos first, then groups similar names by their base name (the part before
the first comma) into one canonical form. This same function is reused by
`build_priority_training_data.py`, so a college's name always resolves to the same string whether it
came from the admission records or a student's priority list.

One exception worth knowing about: colleges that share a base name but are actually different
institutions in different cities — like "Mahendra Multiple Campus" in Dang versus Nepalgunj — are kept
separate on purpose. A list called `DO_NOT_MERGE_BASES` protects these from being collapsed together.

This script produces two outputs. `final_master_dataset.csv` has one row per college per year,
including a few engineered columns like `Cutoff_Trend` and `Rolling_Avg_Cutoff` that are handy for
exploring the data but shouldn't be fed into a model — they're computed using each row's own value, so
they'd leak information a model shouldn't have yet. `college_master_ranking.csv` aggregates each
college across all years into a single row, including `College_Master_Rank` (1 = most competitive),
which is the ranking used throughout the recommendation engine.

### `build_priority_training_data.py` — turning priority lists into training labels

IOST's BSc CSIT seat allocation works sequentially: a student is placed into the highest college on
their list that still has a seat left for their rank. That gives us real labels to learn from. Any
college listed above the one a student actually got admitted to was wanted more but not obtained, so
it's labeled a rejection. The admitted college itself is labeled an acceptance. Colleges listed below
it were never reached by the allocation process at all, so we genuinely don't know what would have
happened — those are simply left out rather than guessed at.

Getting to clean labels meant working around two quirks in the raw data. First, around 16% of the
priority-list cells have two college names squeezed into one, a leftover artifact from how the
original PDFs were extracted into spreadsheets. `get_tokens()` handles this by flattening every
`priority_1` through `priority_39` cell for a student and re-splitting on whitespace, instead of
trusting that each cell holds exactly one name.

Second, the priority lists only record a truncated prefix of each college's name (e.g. "Amrit-"), and
some prefixes match more than one real college. Rather than guess, ambiguous or unmatched prefixes are
normally excluded — except when we already know, from the student's actual admitted college in the
raw data, which of the ambiguous candidates it has to be. In that one case it's safe to resolve
without guessing, which is what the "confirmed ambiguous" recovery logic in `build_labeled_dataset()`
does.

### `train_model.py` — forecasting next year's cutoffs

The main thing to get right here is avoiding data leakage. `build_leakfree_features()` — a function
shared with `train_classifier.py` — builds each college's features using only its cutoff history from
years strictly before the one being predicted. This matters because the ready-made columns in
`final_master_dataset.csv`, like `Cutoff_Trend` and `Rolling_Avg_Cutoff`, are computed using each row's
own value, which would let a model see information it shouldn't have yet. Any model in this project
should be trained on features rebuilt through `build_leakfree_features()`, never on those pre-computed
columns directly.

The train/test split is time-based rather than random: the model trains on years before the chosen
test year and is only evaluated on that test year, since shuffling years around would let it peek into
the future. Its performance is checked against a simple baseline — predicting that next year's cutoff
will be the same as last year's — just to confirm the model is actually adding value.

### `train_classifier.py` — learning admit/reject from real outcomes

This reuses the same leak-free feature-building logic from `train_model.py`, merges it with the
labeled dataset from stage 2 by college and year, and trains a Random Forest classifier on genuine
admission outcomes. It replaces an earlier hand-designed formula (a logistic sigmoid) that
`recommendation_engine.py` still falls back to when no trained classifier is available. It uses the
same time-based split as the cutoff model — training before 2082, testing on 2082 — and reports the
usual classification metrics (accuracy, precision, recall, F1, ROC-AUC) against the baseline admit
rate in the test set. The trained model and the exact feature columns it expects are saved together in
one file, so whatever loads it later doesn't have to guess the column order.

### `recommendation_engine.py` — turning predictions into recommendations

This is where the actual recommendation logic lives.

**Finding the student's realistic starting point.** `find_source_college()` looks at every college
whose cutoff is at or above the student's rank — meaning they could plausibly get in — and picks the
most competitive one among those. That's deliberately different from just picking whichever college is
numerically closest to the student's rank: a college that's 199 ranks too competitive and one that's
238 ranks comfortably within reach aren't "similarly close" outcomes, even though the raw distance
might suggest otherwise. If nothing is realistically within reach, it falls back to the least
competitive college available and flags the result as a weak match. `test_source_college_logic.py`
exists specifically to show why this beats a naive nearest-distance approach.

**Estimating admission probability.** This comes from whichever of two sources is available: a trained
Random Forest classifier, if one is supplied (preferred, since it's based on real outcomes), or a
hand-designed logistic sigmoid over the rank gap between student and college as a fallback. When the
classifier is used, its output is a genuine model prediction, so the fallback method's score columns
are dropped rather than shown next to a probability they didn't actually produce.

**Matching free-text college names.** Students type college names all sorts of ways — "Kathford"
instead of the full name, for instance — so `match_college_name()` fuzzy-matches user input against
the canonical college list using rapidfuzz's partial-ratio scoring, chosen after testing showed it
handled short queries better than the alternatives. Some fragments, like "College" or "Multiple
Campus," are common enough to score a perfect match against several colleges at once. When that
happens, the function returns every tied candidate instead of silently picking one, and the frontend
shows this as a warning on the relevant ticket.

**Suggesting alternative colleges.** `top5_by_cutoff()` scores every college and returns the student's
source college plus the four next-easiest options. It can come back with fewer than five if the source
college is already near the bottom of the ranking — there's simply nothing easier left to suggest.

**Filling in display data.** `destination_cutoff()` picks the best available cutoff figure for a
college — preferring the predicted cutoff, then the latest actual cutoff, then the multi-year average,
whichever is available first — and this feeds the "Destination Cutoff" and "Rank Gap" shown for every
college in a student's list.

**Handling suggested colleges.** Colleges recommended by `top5_by_cutoff()` weren't part of the
student's own submitted list, so they don't have a real "priority position" to plug into the
probability calculation — `attach_admission_probability()` defaults to treating them as if they were
the student's top choice.

### `api.py` — the web API

This wraps `recommendation_engine.py` as a JSON API for the frontend. The college ranking data,
predicted cutoffs, and trained classifier are all loaded once when the server starts, not on every
request, so responses stay fast.

```bash
uvicorn api:app --reload
```

Endpoints:
- `GET /health` — liveness check.
- `GET /colleges` — dropdown data: one row per college (`College`, `College_Master_Rank`,
  `Rank_Category`, `Latest_Cutoff_Rank`, `Predicted_Cutoff_Rank`, `Avg_Cutoff_Rank_AllYears`), sorted
  most-competitive first.
- `POST /recommend` — body `{"rank": int, "preferences": [str, ...]}` (rank must be > 0, preferences
  non-empty, both enforced by Pydantic). Returns:
  - `source_college` — a `college_master_ranking.csv` row plus `Source_Match_Gap`/`Source_Match_Is_Weak`.
  - `preferences` — one row per input college: `User_Input, Matched_College, Admission_Probability,
    Probability_Label, Name_Match_Confidence, Probability_Source, Destination_Cutoff_Rank, Rank_Gap,
    Preference_Order`. Unmatched free-text input gets `null` for `Matched_College`/`Admission_Probability`
    and `"Not Found"` for `Probability_Label`. Sorted by probability descending — re-sort by
    `Preference_Order` client-side to restore the student's own list order.
  - `best_college` — same shape as one `preferences` row, or `null` if nothing matched.
  - `top5_recommendations` — `college_master_ranking.csv` columns plus `Score` and
    `Admission_Probability`/`Probability_Label` (5 rows, or fewer near the bottom of the ranking).

Pandas DataFrames and Series get converted to JSON through `df.to_json()` followed by `json.loads()`,
rather than the more obvious `.to_dict()` — that built-in doesn't handle numpy's number types well and
turns missing values into an invalid JSON token instead of `null`.

Two fields, `Destination_Cutoff_Rank` and `Rank_Gap`, along with the probability fields on the top-5
recommendations, exist only in the API — they were added specifically for the frontend and aren't part
of the command-line tool's output.

One known limitation: the top-5 suggested colleges aren't checked against what the student already put
in their own preference list, so the same college can occasionally show up in both places. That's a
deliberate simplification rather than a bug that slipped through.

### `frontend/` — the web interface

Just plain HTML, CSS, and JavaScript — no build tools, no framework. It's served directly by `api.py`,
so running the API server also serves the whole site from the same address and port.

Every piece of data on the page — the college list, the recommendations, the probabilities — comes
from the API; nothing is hardcoded or computed client-side. The student's priority list is shown back
to them in the order they originally entered it, not re-sorted by probability, since that section's
job is to answer "where do my own choices stand," not act as a leaderboard.

One small but important behavior: every time a student submits a new rank/preference combination, all
of the previous results get cleared before new ones (or an error) are shown. This came out of a real
bug found during testing, where a validation error message would appear next to results left over from
a completely different, earlier submission — confusing and easy to misread.

## Data files

Most of the files in this repo aren't source code — they're either the raw input data or files
regenerated by running the pipeline. Here's what each one is:

- `fyp_dataset_combine_79_80_81_82_merged.xlsx` — the raw per-student admission records for entrance
  years 2079-2082 BS. This is the actual starting point for the whole pipeline.
- `combined_priority_lists_split__1.xlsx` — each student's ordered list of preferred colleges, used
  alongside the file above to build the training labels.
- `final_master_dataset.csv`, `college_master_ranking.csv`, `name_mapping_audit.csv` — generated by
  stage 1.
- `labeled_admissions.csv` — generated by stage 2.
- `leakfree_training_features.csv`, `predicted_cutoffs_2083.csv` — generated by stage 3.
- `admission_classifier.pkl` — generated by stage 4 (a bundle containing both the trained model and
  the feature columns it expects).

If the name-standardization logic in `build_master_dataset.py` ever changes, it's worth re-running the
whole pipeline rather than hand-editing these files — several of them are derived from each other, and
patching just one risks leaving the rest out of sync.
