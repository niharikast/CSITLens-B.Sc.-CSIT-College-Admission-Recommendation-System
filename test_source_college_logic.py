"""
test_source_college_logic.py
==============================
Standalone test demonstrating the Source College matching fix.

Shows, with real numbers from your dataset, the difference between:
  - The OLD (buggy) logic: pick whichever college has the smallest
    absolute distance to the student's rank, ignoring direction.
  - The NEW (fixed) logic: prefer the most competitive ACHIEVABLE
    college (cutoff >= student rank), and only fall back to the easiest
    available college if literally nothing is achievable.

Run:
    python3 test_source_college_logic.py --ranking college_master_ranking.csv --predicted predicted_cutoffs_2083.csv
"""

import argparse
import pandas as pd
from recommendation_engine import (load_master_ranking, attach_predicted_cutoffs,
                                    find_source_college)


def old_buggy_source_college(student_rank, master_df, cutoff_col='Predicted_Cutoff_Rank'):
    """Recreated exactly as the OLD logic worked, for side-by-side comparison only.
    Picks smallest |cutoff - rank|, regardless of direction (achievable or not)."""
    diffs = (master_df[cutoff_col] - student_rank).abs()
    idx = diffs.idxmin()
    return master_df.loc[idx]


def explain(student_rank, row_old, row_new, cutoff_col='Predicted_Cutoff_Rank'):
    old_cutoff = row_old[cutoff_col]
    new_cutoff = row_new[cutoff_col]

    old_achievable = old_cutoff >= student_rank
    new_achievable = new_cutoff >= student_rank

    print(f"\nStudent Rank: {student_rank}")
    print(f"  OLD logic picked : {row_old['College']}")
    print(f"                     cutoff={old_cutoff:.0f}  "
          f"-> {'ACHIEVABLE (would have gotten in)' if old_achievable else 'NOT ACHIEVABLE (would NOT have gotten in)'}")
    print(f"  NEW logic picks  : {row_new['College']}")
    print(f"                     cutoff={new_cutoff:.0f}  "
          f"-> {'ACHIEVABLE (would have gotten in)' if new_achievable else 'NOT ACHIEVABLE (would NOT have gotten in)'}")

    if row_old['College'] != row_new['College']:
        print(f"  >>> DIFFERENT RESULT — the fix changed the answer for this rank.")
    else:
        print(f"  >>> Same result — both logics agree here (this rank had a clean match).")


def run(ranking_path, predicted_path, test_ranks):
    master_df = load_master_ranking(ranking_path)
    master_df = attach_predicted_cutoffs(master_df, predicted_path)

    print("=" * 70)
    print("SOURCE COLLEGE MATCHING: OLD (buggy) vs NEW (fixed) LOGIC")
    print("=" * 70)

    for rank in test_ranks:
        row_old = old_buggy_source_college(rank, master_df)
        row_new = find_source_college(rank, master_df)
        explain(rank, row_old, row_new)
        if row_new['Source_Match_Is_Weak']:
            print(f"  WARNING (new logic): weak match, gap = "
                  f"{row_new['Source_Match_Gap']:.0f} rank positions.")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ranking', default='college_master_ranking.csv')
    parser.add_argument('--predicted', default='predicted_cutoffs_2083.csv')
    parser.add_argument('--ranks', nargs='+', type=int,
                         default=[600, 850, 1307, 5800],
                         help='Student ranks to test (default includes the '
                              'known gap case at 600)')
    args = parser.parse_args()
    run(args.ranking, args.predicted, args.ranks)
