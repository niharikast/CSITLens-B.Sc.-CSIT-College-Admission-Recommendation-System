"""
build_priority_training_data.py
=================================
Builds a REAL labeled classification dataset from the full student
priority lists + their actual admission outcome.

For each student, their priority list tells us all the colleges they
wanted, in order. Their actual admitted college (from the original
roll-level xlsx) tells us where the automated merit-based allocation
process stopped. Nepal's CEE allocation is sequential: a student is
placed at the highest-priority college where a seat is still available
for their rank. So:

  - Every college listed ABOVE the student's actual admitted college in
    their priority list = they wanted it more, but did NOT get in
    (label = 0)
  - The college they were actually admitted to = label = 1
  - Colleges listed AFTER the admitted one are never reached by the
    allocation process, so we don't know what would have happened —
    these are excluded (neither labeled 0 nor 1).

This gives real ground-truth (rank, college, priority position) -> 
admitted-or-not examples, instead of a hand-designed formula.

Data quality note: ~16% of the raw priority cells have two college names
crammed into one cell (a PDF-extraction column-alignment artifact). This
is fixed by reading all priority_1..39 cells for a student, splitting each
on whitespace, and concatenating — this recovers the true ordered
sequence of names regardless of which column each one landed in.

Name matching note: priority-list college names are truncated to a single
prefix word (e.g. "Amrit-" for "Amrit Campus, Lainchaur"). Some prefixes
are genuinely ambiguous (e.g. "Bhaktapur" matches 3 different colleges).
Ambiguous or unmatched prefixes are EXCLUDED rather than guessed, to avoid
corrupting the labels with wrong matches.
"""

import re
import argparse
import pandas as pd

# Manually verified unambiguous fixes for abbreviations / typos / formatting
# that automatic single-word matching couldn't resolve on its own.
MANUAL_PREFIX_FIXES = {
    'ChitwanCT': 'Chitwan College of Technology, Chitwan',
    'KCT': 'Kathmandu College of Technology, Bhaktapur',
    'NCCS': 'National College of Computer Studies, Paknajol',
    'St.Lawrence': 'St. Lawrence College, Chabahil',
    'St.Xaviers': 'St. Xaviers College, Maitighar',
    'Trubhuvan': 'Tribhuvan Multiple Campus, Palpa',
}


def get_tokens(row, priority_cols):
    """Flatten all priority_1..39 cells for one student into a single
    ordered list of clean name tokens, ignoring which column each token
    originally landed in (fixes the crammed-cell extraction bug)."""
    toks = []
    for c in priority_cols:
        v = row[c]
        if pd.notna(v):
            toks.extend(str(v).split())
    return [t.rstrip('-').strip() for t in toks if t.strip('-').strip()]


def build_prefix_map(all_prefixes, canon_list):
    """Match each priority-list prefix to exactly one canonical college
    name. Only keeps unambiguous matches; ambiguous/unmatched prefixes
    map to None and are excluded downstream (unless independently
    confirmed elsewhere -- see ambiguous_candidates below).

    Returns (prefix_map, ambiguous_candidates):
      prefix_map: prefix -> canonical name, or None if ambiguous/unmatched
      ambiguous_candidates: prefix -> list of candidate canonical names,
        for prefixes that matched 2+ colleges (empty/absent for unmatched
        prefixes with 0 candidates)
    """
    prefix_map = {}
    ambiguous_candidates = {}
    for p in all_prefixes:
        if p in MANUAL_PREFIX_FIXES:
            prefix_map[p] = MANUAL_PREFIX_FIXES[p]
            continue
        pl = p.lower()
        matches = []
        for c in canon_list:
            words = re.split(r'[ ,.]+', c.lower())
            if pl in words:
                matches.append(c)
        if len(matches) == 1:
            prefix_map[p] = matches[0]
        else:
            prefix_map[p] = None
            if len(matches) > 1:
                ambiguous_candidates[p] = matches
    return prefix_map, ambiguous_candidates


def build_alias_map_from_raw(raw_names):
    """Re-use the same canonicalization logic as build_master_dataset.py
    so admitted-college names match the same canonical College strings
    used in college_master_ranking.csv."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from build_master_dataset import build_alias_map
    return build_alias_map(raw_names)


def load_priority_file(priority_path):
    """Load the priority list file, whichever format it's in (the old CSV
    with columns like roll_no/year_bs/priority_1..39, or the newer xlsx
    with columns like RollNo/year /Priority1..20), and normalize it to a
    consistent set of column names: roll_no, year, rank, and a list of
    priority_cols.
    """
    if str(priority_path).lower().endswith(('.xlsx', '.xls')):
        pri = pd.read_excel(priority_path)
    else:
        pri = pd.read_csv(priority_path, low_memory=False)

    # normalize column names: strip whitespace, so 'year ' -> 'year'
    pri.columns = [str(c).strip() for c in pri.columns]

    # map whichever naming convention is present to a standard one
    rename_map = {}
    for c in pri.columns:
        cl = c.lower()
        if cl in ('roll_no', 'rollno', 'roll no'):
            rename_map[c] = 'roll_no'
        elif cl in ('year_bs', 'year'):
            rename_map[c] = 'year'
        elif cl == 'rank':
            rename_map[c] = 'rank'
    pri = pri.rename(columns=rename_map)

    # priority columns: anything starting with "priority" (case-insensitive),
    # covers both 'priority_1' and 'Priority1' style naming
    priority_cols = [c for c in pri.columns if c.lower().startswith('priority')]

    # RollNo sometimes comes through as float (e.g. 794701.0) -> cast to
    # int so it matches the admission Excel's integer Roll No column
    pri['roll_no'] = pd.to_numeric(pri['roll_no'], errors='coerce').astype('Int64')
    pri['year'] = pd.to_numeric(pri['year'], errors='coerce').astype('Int64')
    pri['rank'] = pd.to_numeric(pri['rank'], errors='coerce')

    return pri, priority_cols


def build_labeled_dataset(raw_xlsx, priority_path, master_ranking_csv):
    print("Loading data ...")
    raw = pd.read_excel(raw_xlsx)
    pri, priority_cols = load_priority_file(priority_path)
    master = pd.read_csv(master_ranking_csv)
    canon_list = master['College'].tolist()

    # canonicalize admitted-college names (same logic as the main pipeline)
    alias_map = build_alias_map_from_raw(raw['College'].unique())
    raw['College_Canon'] = raw['College'].map(alias_map)
    admitted_lookup = {
        (int(row['Roll No']), int(row['Year'])): row['College_Canon']
        for _, row in raw.iterrows()
    }

    print("Extracting ordered priority tokens per student ...")
    pri['tokens'] = pri.apply(lambda r: get_tokens(r, priority_cols), axis=1)

    all_prefixes = set()
    for t in pri['tokens']:
        all_prefixes.update(t)
    prefix_map, ambiguous_candidates = build_prefix_map(all_prefixes, canon_list)
    n_matched = sum(1 for v in prefix_map.values() if v is not None)
    print(f"College-name prefixes: {len(prefix_map)} total, "
          f"{n_matched} unambiguously matched, "
          f"{len(prefix_map) - n_matched} excluded (ambiguous/unmatched)")

    rows = []
    skipped_no_admit_in_list = 0
    skipped_missing_key = 0
    recovered_via_confirmed_ambiguous = 0
    for _, r in pri.iterrows():
        if pd.isna(r['roll_no']) or pd.isna(r['year']):
            skipped_missing_key += 1
            continue
        key = (int(r['roll_no']), int(r['year']))
        admitted_canon = admitted_lookup.get(key)
        if admitted_canon is None:
            skipped_missing_key += 1
            continue

        raw_tokens = r['tokens']
        matched_tokens = [prefix_map.get(t) for t in raw_tokens]

        # find first occurrence of the admitted college in this student's list
        admit_idx = None
        for i, m in enumerate(matched_tokens):
            if m == admitted_canon:
                admit_idx = i
                break

        # If not found via a direct unambiguous match, check whether an
        # AMBIGUOUS prefix at this position could be independently
        # confirmed: we already know this student's real admitted college
        # from the admission Excel (ground truth, not a guess). If that
        # known college is one of the ambiguous prefix's candidates, this
        # occurrence can be safely resolved -- we're confirming against an
        # independent source, not guessing which candidate was meant.
        if admit_idx is None:
            for i, tok in enumerate(raw_tokens):
                candidates = ambiguous_candidates.get(tok)
                if candidates and admitted_canon in candidates:
                    admit_idx = i
                    matched_tokens[i] = admitted_canon  # confirmed for this row only
                    recovered_via_confirmed_ambiguous += 1
                    break

        if admit_idx is None:
            skipped_no_admit_in_list += 1
            continue

        for j in range(admit_idx + 1):
            college = matched_tokens[j]
            if college is None:
                continue  # ambiguous/unmatched prefix, skip this entry
            rows.append({
                'roll_no': r['roll_no'],
                'year': r['year'],
                'student_rank': r['rank'],
                'college': college,
                'priority_position': j + 1,
                'admitted': 1 if j == admit_idx else 0,
            })

    labeled = pd.DataFrame(rows)
    print(f"\nStudents skipped (Roll No/Year not found in admission data): "
          f"{skipped_missing_key}")
    print(f"Recovered via independently-confirmed ambiguous match "
          f"(admitted college was one of an ambiguous prefix's candidates, "
          f"confirmed against the admission Excel — not guessed): "
          f"{recovered_via_confirmed_ambiguous}")
    print(f"Students skipped (admitted college not found in their listed "
          f"priorities, even after confirmed-ambiguous recovery): "
          f"{skipped_no_admit_in_list}")
    print(f"Labeled examples built: {len(labeled)}")
    print(f"  admitted=1: {(labeled['admitted']==1).sum()}")
    print(f"  admitted=0: {(labeled['admitted']==0).sum()}")

    return labeled


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-xlsx', default='fyp_dataset_combine_79_80_81_82_merged.xlsx',
                         help='Path to the raw student-level admission Excel file')
    parser.add_argument('--priority-file', default='combined_priority_lists_split__1.xlsx',
                         help='Path to the priority list file (.xlsx or .csv, either format works)')
    parser.add_argument('--ranking', default='college_master_ranking.csv',
                         help='Path to college_master_ranking.csv (from build_master_dataset.py)')
    parser.add_argument('--outdir', default='.', help='Where to save labeled_admissions.csv')
    args = parser.parse_args()

    labeled = build_labeled_dataset(args.raw_xlsx, args.priority_file, args.ranking)
    out_path = f"{args.outdir}/labeled_admissions.csv"
    labeled.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
