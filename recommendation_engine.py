"""
recommendation_engine.py
=========================
College Recommendation System — Prediction / Recommendation Layer

Uses college_master_ranking.csv (produced by build_master_dataset.py) to:
  1. Find a student's "Source College" — the college whose cutoff is
     closest to the student's entrance rank.
  2. Score every college in the student's preference list using the
     Destination - Source rule, and turn that into an admission
     probability.
  3. Recommend the best college from the student's preference list.
  4. Auto-recommend a Top-5 list (closest-cutoff college + the next four
     colleges below it in the master ranking).

ASSUMPTION / DESIGN NOTE (read before using in production):
  The spec describes the Destination-Source score qualitatively as:
      Large negative -> very low prob   Zero -> "highest" prob
      Small negative  -> low prob        Small positive -> high prob
                                          Large positive -> "very high" prob
  Taken completely literally this isn't a single monotonic curve (it says
  "highest" at zero, but "very high" — implied lower than "highest" — for
  large positive, which would make probability non-monotonic and then
  decrease again for very easy colleges, which doesn't match how admission
  chances actually behave). We interpret the *intent* rather than the
  exact wording: admission probability should rise monotonically as the
  preferred college gets easier relative to the student's own level, and
  fall monotonically as it gets harder. We implement that with a logistic
  (sigmoid) curve centered at score=0, which:
    - gives ~50% probability at score = 0 (evenly matched college)
    - approaches 1.0 as the college gets much easier than the student's level
    - approaches 0.0 as the college gets much harder
  `PROB_SCALE` controls how quickly probability moves away from 50% per
  rank-position of difference — tune it against real outcome data if you
  have it (e.g. via logistic regression) rather than guessing.
"""

import argparse
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz

PROB_SCALE = 8.0  # rank-positions of "difference" needed to roughly double/halve the odds


def load_master_ranking(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_classifier(path: str):
    """Load the trained Random Forest admission classifier (from
    train_classifier.py). Returns (model, feature_cols)."""
    import joblib
    bundle = joblib.load(path)
    return bundle['model'], bundle['feature_cols']


def classifier_probability(student_rank: int, college_row: pd.Series,
                            priority_position: int, model, feature_cols) -> float:
    """Predict P(admitted=1) for a specific (student_rank, college,
    priority_position) using the trained classifier, rather than the
    hand-designed sigmoid formula. Uses the same leak-free-style features
    the model was trained on, taken from the college's known history."""
    prev_cutoff = college_row.get('Latest_Cutoff_Rank', college_row.get('Avg_Cutoff_Rank_AllYears'))
    row = {
        'student_rank': student_rank,
        'priority_position': priority_position,
        'Prev_Cutoff': prev_cutoff,
        'Rolling_Avg_Prior': college_row.get('Avg_Cutoff_Rank_AllYears', prev_cutoff),
        'Cutoff_Trend_Prior': 0.0,  # not available at inference time from master_df; neutral default
        'Cutoff_Volatility_Prior': 0.0,
        'Years_of_Prior_Data': college_row.get('Years_of_Data', 1),
        'Rank_Minus_Prev_Cutoff': student_rank - prev_cutoff,
    }
    X = pd.DataFrame([row])[feature_cols]
    return float(model.predict_proba(X)[0, 1])


def attach_predicted_cutoffs(master_df: pd.DataFrame, predicted_path: str) -> pd.DataFrame:
    """Merge the ML model's predicted next-year cutoffs (from
    train_model.py) onto the master ranking table, as a new
    'Predicted_Cutoff_Rank' column. Falls back to each college's latest
    known actual cutoff if it's missing from the predictions file for any
    reason, so the engine never breaks due to a partial predictions file."""
    predicted = pd.read_csv(predicted_path)[['College', 'Predicted_Cutoff_Rank']]
    merged = master_df.merge(predicted, on='College', how='left')
    missing = merged['Predicted_Cutoff_Rank'].isna()
    if missing.any():
        print(f"NOTE: {missing.sum()} college(s) had no ML prediction — "
              f"falling back to their latest known actual cutoff instead.")
        merged.loc[missing, 'Predicted_Cutoff_Rank'] = merged.loc[missing, 'Latest_Cutoff_Rank']
    return merged


def match_college_name(user_input: str, master_df: pd.DataFrame, threshold: int = 75):
    """Fuzzy-match a user-typed college name to the closest canonical name.

    Uses partial_ratio rather than WRatio/token_sort_ratio: users commonly
    type a short fragment of the full name (e.g. "Kathford" instead of
    "Kathford International College of Engineering and Management,
    Balkumari"). partial_ratio scores how well the short query matches
    *some substring* of each candidate, which handles this correctly —
    WRatio and token_sort_ratio were both found to mismatch short queries
    like "Kathford" to unrelated colleges during testing.
    Returns (canonical_name, score) or (None, 0) if nothing crosses the
    threshold.
    """
    choices = master_df['College'].tolist()
    result = process.extractOne(user_input, choices, scorer=fuzz.partial_ratio)
    if result is None:
        return None, 0
    match, score, _ = result
    if score < threshold:
        return None, 0
    return match, score


SOURCE_MATCH_GAP_THRESHOLD = 150  # rank positions; beyond this, the "closest"
                                   # college is still flagged as a weak match


def find_source_college(student_rank: int, master_df: pd.DataFrame,
                         cutoff_col: str = None) -> pd.Series:
    """The college whose cutoff is numerically closest to the student's
    rank.

    Which cutoff number to compare against, in order of preference:
      1. 'Predicted_Cutoff_Rank'   -> ML model's forecast for next year
         (best choice: a student applying THIS year cares about what the
         cutoff will likely be, not what it was in the past)
      2. 'Latest_Cutoff_Rank'      -> most recent actual year on record
      3. 'Avg_Cutoff_Rank_AllYears'-> multi-year average, as a last resort

    Pass cutoff_col explicitly to override; otherwise the best available
    column is auto-selected.

    IMPORTANT (fixed after review): this does NOT simply pick whichever
    college has the smallest absolute distance to the student's rank.
    That naive approach treats "199 positions too competitive" and "238
    positions comfortably safe" as equally good matches, which is wrong —
    they mean opposite things. A college whose cutoff is BELOW the
    student's rank is one they would NOT have gotten into (out of reach);
    a college whose cutoff is AT OR ABOVE the student's rank is one they
    WOULD have gotten into (achievable).

    So the real logic is:
      1. Prefer the most competitive ACHIEVABLE college — i.e. among all
         colleges whose cutoff >= student_rank, pick the one with the
         SMALLEST such cutoff (the hardest college this student could
         still realistically get into).
      2. Only if no college is achievable at all (the student's rank is
         worse than every single college's cutoff), fall back to the
         least competitive college available (best of a bad situation),
         and flag it as a weak match.

    This always returns *some* college — the caller decides how to react
    via 'Source_Match_Is_Weak' and 'Source_Match_Gap'.
    """
    if cutoff_col is None:
        if 'Predicted_Cutoff_Rank' in master_df.columns:
            cutoff_col = 'Predicted_Cutoff_Rank'
        elif 'Latest_Cutoff_Rank' in master_df.columns:
            cutoff_col = 'Latest_Cutoff_Rank'
        else:
            cutoff_col = 'Avg_Cutoff_Rank_AllYears'

    achievable = master_df[master_df[cutoff_col] >= student_rank]

    if len(achievable) > 0:
        # Most competitive college the student could still get into.
        # A large gap here means the student comfortably clears this
        # college's cutoff -- that is GOOD news (extra safety margin),
        # never a "weak match" worth warning about, regardless of size.
        idx = achievable[cutoff_col].idxmin()
        row = master_df.loc[idx].copy()
        gap = float(row[cutoff_col] - student_rank)
        is_weak = False
    else:
        # Student's rank is worse than every college's cutoff — nothing
        # is achievable. Fall back to the least competitive (easiest)
        # college as the closest available reference point. This IS a
        # genuinely concerning case, so it's flagged as a weak match.
        idx = master_df[cutoff_col].idxmax()
        row = master_df.loc[idx].copy()
        gap = float(student_rank - row[cutoff_col])
        is_weak = gap > SOURCE_MATCH_GAP_THRESHOLD

    row['Source_Match_Gap'] = gap
    row['Source_Match_Is_Weak'] = is_weak
    return row


def score_to_probability(score: float) -> float:
    """Destination - Source score -> admission probability in [0, 1]
    via a sigmoid centered at 0. See module docstring for rationale."""
    return float(1 / (1 + np.exp(-score / PROB_SCALE)))


def probability_label(prob: float) -> str:
    if prob < 0.15:
        return "Very Low"
    elif prob < 0.35:
        return "Low"
    elif prob < 0.65:
        return "Moderate"
    elif prob < 0.85:
        return "High"
    return "Very High"


def destination_cutoff(row: pd.Series):
    """Best available cutoff number for a college row, in the same
    preference order used elsewhere (predicted next-year > latest actual >
    multi-year average). Returns None if none of those columns are present
    or all are NaN, rather than raising."""
    for col in ('Predicted_Cutoff_Rank', 'Latest_Cutoff_Rank', 'Avg_Cutoff_Rank_AllYears'):
        if col in row.index and pd.notna(row[col]):
            return float(row[col])
    return None


def recommend_from_preference_list(student_rank: int, preference_list: list,
                                    master_df: pd.DataFrame,
                                    classifier=None, classifier_features=None) -> pd.DataFrame:
    """Core recommendation logic: map each preferred college to the master
    ranking, and score each one.

    If a trained classifier is supplied, probability comes from the REAL
    model (trained on genuine admit/reject outcomes from priority-list
    data). Otherwise falls back to the Destination-Source sigmoid formula.
    """
    source_row = find_source_college(student_rank, master_df)
    source_rank_pos = int(source_row['College_Master_Rank'])

    rows = []
    for i, pref in enumerate(preference_list):
        matched_name, match_score = match_college_name(pref, master_df)
        if matched_name is None:
            row = {
                'User_Input': pref, 'Matched_College': None,
                'Admission_Probability': None, 'Probability_Label': 'Not Found',
                'Name_Match_Confidence': 0, 'Probability_Source': None,
                'Destination_Cutoff_Rank': None, 'Rank_Gap': None,
            }
            if classifier is None:
                row['Destination_Rank'] = None
                row['Score'] = None
            rows.append(row)
            continue
        dest_row = master_df.loc[master_df['College'] == matched_name].iloc[0]
        destination_rank_pos = int(dest_row['College_Master_Rank'])
        score = destination_rank_pos - source_rank_pos
        dest_cutoff = destination_cutoff(dest_row)

        if classifier is not None:
            prob = classifier_probability(student_rank, dest_row, i + 1,
                                           classifier, classifier_features)
            prob_source = 'ML Classifier'
        else:
            prob = score_to_probability(score)
            prob_source = 'Rule-based formula'

        row = {
            'User_Input': pref,
            'Matched_College': matched_name,
            'Admission_Probability': round(prob, 3),
            'Probability_Label': probability_label(prob),
            'Name_Match_Confidence': match_score,
            'Probability_Source': prob_source,
            # Cutoff number + numeric rank gap (cutoff - student_rank) for
            # this destination college. Populated unconditionally (unlike
            # Destination_Rank/Score below) because the frontend needs a
            # real cutoff number and gap to display regardless of which
            # probability method is active.
            'Destination_Cutoff_Rank': dest_cutoff,
            'Rank_Gap': (dest_cutoff - student_rank) if dest_cutoff is not None else None,
        }
        # Destination_Rank/Score only shown when they're actually driving the
        # result (rule-based mode). When the classifier is used, these
        # columns would be misleading leftovers -- the classifier does not
        # use them to produce Admission_Probability, so they're dropped
        # rather than displayed as if they still mattered.
        if classifier is None:
            row['Destination_Rank'] = destination_rank_pos
            row['Score'] = score
        rows.append(row)

    result = pd.DataFrame(rows)
    result['Preference_Order'] = range(1, len(result) + 1)
    result = result.sort_values(
        by=['Admission_Probability', 'Preference_Order'],
        ascending=[False, True]
    ).reset_index(drop=True)

    return result, source_row


def top5_by_cutoff(student_rank: int, master_df: pd.DataFrame,
                    cutoff_col: str = None) -> Tuple[pd.DataFrame, Optional[str]]:
    """Top-5 auto-recommendation, using the Source-Destination score
    EXPLICITLY (as specified) rather than a plain rank-position lookup.

    For every college in the master ranking, treat it as a candidate
    "Destination" and compute:
        Score = Destination_Master_Rank - Source_Master_Rank
    The Source College itself always scores 0 (Destination == Source).
    Colleges easier than the Source College score positive (Destination
    rank number is higher, i.e. less competitive); colleges harder score
    negative.

    Per the spec, this recommendation looks at colleges AT or BELOW the
    student's own level (Score >= 0) — the Source College plus the next
    4 easiest colleges by Score, sorted ascending (closest-to-zero
    first). This is mathematically equivalent to walking down the master
    ranking one position at a time, but is now computed explicitly via
    the Source-Destination score for every candidate, and the Score
    column is returned so it's visible, not just implied.

    No sliding window: colleges harder than the Source College (Score < 0)
    are never included to pad the list back up to 5 — 'eligible' excludes
    them before the .head(5) truncation, so a short result stays short
    rather than backfilling with harder options.

    Returns (result_df, note). 'note' is None when 5 rows were returned.
    When the Source College is near (or at) the bottom of the master
    ranking, fewer than 5 rows come back and 'note' is a user-facing
    string explaining why — distinct wording for "exactly the last
    college, zero easier ones exist" vs. "some easier ones exist, just
    not four of them" — so the caller (CLI / API) can surface it instead
    of it silently disappearing into a server-side print().
    """
    total_colleges = len(master_df)
    source_row = find_source_college(student_rank, master_df, cutoff_col)
    source_rank_pos = int(source_row['College_Master_Rank'])

    candidates = master_df.copy()
    candidates['Score'] = candidates['College_Master_Rank'] - source_rank_pos

    # Only consider the Source College itself and colleges below it
    # (Score >= 0), matching "the Source college + the next four
    # colleges below it" from the spec.
    eligible = candidates[candidates['Score'] >= 0].sort_values('Score')
    result = eligible.head(5).reset_index(drop=True)

    note = None
    if len(result) == 1:
        note = (f"{source_row['College']} is the least selective college on "
                 f"record (Master Rank #{source_rank_pos} of {total_colleges}) "
                 f"— there are no easier colleges left to recommend below it.")
    elif len(result) < 5:
        note = (f"Only {len(result)} colleges, including the Source College, "
                 f"are at or below Master Rank #{source_rank_pos} of "
                 f"{total_colleges} — there aren't enough easier colleges on "
                 f"record to fill a Top-5 list.")

    return result, note


def attach_admission_probability(candidates_df: pd.DataFrame, student_rank: int,
                                  classifier=None, classifier_features=None,
                                  priority_position: int = 1) -> pd.DataFrame:
    """Attach Admission_Probability/Probability_Label to a college dataframe
    that didn't go through recommend_from_preference_list() -- e.g. the
    output of top5_by_cutoff(), where colleges aren't part of an actual
    submitted priority list, so there's no real priority_position for them.
    Defaults to priority_position=1 (treat each as if it were a standalone,
    top-priority choice), since that's the only position-independent framing
    that makes sense for a college being suggested rather than ranked by the
    student themselves.

    Falls back to the rule-based Destination-Source sigmoid (using the
    existing 'Score' column) if no classifier is supplied, mirroring
    recommend_from_preference_list()'s fallback behavior.
    """
    df = candidates_df.copy()
    if classifier is not None:
        probs = [
            classifier_probability(student_rank, row, priority_position,
                                    classifier, classifier_features)
            for _, row in df.iterrows()
        ]
    elif 'Score' in df.columns:
        probs = [score_to_probability(s) for s in df['Score']]
    else:
        probs = [None] * len(df)

    df['Admission_Probability'] = [round(p, 3) if p is not None else None for p in probs]
    df['Probability_Label'] = [probability_label(p) if p is not None else None for p in probs]
    return df


def run(student_rank: int, preference_list: list, ranking_path: str,
        predicted_path: str = None, classifier_path: str = None):
    master_df = load_master_ranking(ranking_path)

    if predicted_path:
        master_df = attach_predicted_cutoffs(master_df, predicted_path)
        cutoff_source_note = "ML-PREDICTED next-year cutoff"
    else:
        cutoff_source_note = "latest ACTUAL known cutoff (no ML predictions supplied)"

    classifier, classifier_features = (None, None)
    if classifier_path:
        classifier, classifier_features = load_classifier(classifier_path)
        prob_source_note = "trained ML classifier (real admit/reject outcomes)"
    else:
        prob_source_note = "rule-based Destination-Source sigmoid formula"

    print(f"Student Rank: {student_rank}")
    print(f"Source College matching is using: {cutoff_source_note}")
    print(f"Admission probability is using: {prob_source_note}\n")

    ranked_prefs, source_row = recommend_from_preference_list(
        student_rank, preference_list, master_df, classifier, classifier_features
    )
    cutoff_used = (source_row['Predicted_Cutoff_Rank'] if predicted_path
                   else source_row['Latest_Cutoff_Rank'])
    print(f"Source College (closest cutoff to rank {student_rank}): "
          f"{source_row['College']}  (Master Rank #{int(source_row['College_Master_Rank'])}, "
          f"Cutoff used = {cutoff_used})")
    if source_row['Source_Match_Is_Weak']:
        print(f"WARNING: This is a WEAK match — the nearest college's cutoff is "
              f"{source_row['Source_Match_Gap']:.0f} rank positions away from the "
              f"student's rank. No college closely matches this rank; treat the "
              f"Top-5 list below with caution.")
    print()

    print("Preference List — Ranked by Admission Probability:")
    print(ranked_prefs.to_string(index=False))

    valid = ranked_prefs.dropna(subset=['Admission_Probability'])
    if not valid.empty:
        best = valid.iloc[0]
        print(f"\nBest Possible College From Your List: {best['Matched_College']} "
              f"(Probability: {best['Admission_Probability']:.1%}, "
              f"{best['Probability_Label']})")

    print("\nTop 5 Recommended Colleges (using Source-Destination score):")
    top5, top5_note = top5_by_cutoff(student_rank, master_df)
    display_cols = ['College', 'College_Master_Rank', 'Score']
    display_cols += ['Predicted_Cutoff_Rank'] if predicted_path else ['Latest_Cutoff_Rank']
    display_cols += ['Avg_Cutoff_Rank_AllYears']
    print(top5[display_cols].to_string(index=False))
    if top5_note:
        print(f"\nNOTE: {top5_note}")

    return ranked_prefs, top5


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank', type=int, required=True, help='Student entrance rank')
    parser.add_argument('--ranking', default='college_master_ranking.csv')
    parser.add_argument('--predicted', default=None,
                         help='Optional path to predicted_cutoffs_2083.csv from '
                              'train_model.py. If given, Source College matching '
                              'uses the ML-predicted next-year cutoff instead of '
                              'the latest actual known cutoff.')
    parser.add_argument('--classifier', default=None,
                         help='Optional path to admission_classifier.pkl from '
                              'train_classifier.py. If given, admission '
                              'probability comes from the trained model instead '
                              'of the rule-based formula.')
    parser.add_argument('--preferences', nargs='+', required=True,
                         help='Preferred colleges in order, e.g. --preferences '
                              '"Pulchowk" "Thapathali" "Padma Kanya" "Kathford" "Himalaya"')
    args = parser.parse_args()
    run(args.rank, args.preferences, args.ranking, args.predicted, args.classifier)
