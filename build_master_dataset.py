"""
build_master_dataset.py
========================
College Recommendation System — Master Dataset Builder

Reads the raw student-level admission records (Roll No, Name, Rank, Year,
College) covering entrance years 2079-2082, cleans and standardizes college
names, aggregates into a College x Year cutoff table, engineers ML-ready
features, and writes three output files:

    1. name_mapping_audit.csv     -> raw_name -> canonical_name (for review)
    2. final_master_dataset.csv   -> one row per (College, Year), ML-ready
    3. college_master_ranking.csv -> one row per College (aggregate), used
                                      by the recommendation engine

Run:
    python3 build_master_dataset.py --input <path_to_raw_xlsx> --outdir <dir>

Design notes / assumptions (documented so they can be changed later):
  * "Cutoff Rank" for a College+Year = the WORST (i.e. numerically highest)
    Rank among students who got admitted there that year. In Nepal's
    entrance-rank convention, Rank 1 is the best score, so the last student
    admitted defines the true cutoff. This is a real, defensible cutoff —
    not an approximation — because the source data is actual admitted
    students, not a published "average cutoff".
  * College_Master_Rank is computed once, from each college's overall
    (all-year) average cutoff, ascending (rank 1 = most competitive/
    hardest to get into). This is the static reference list the
    recommendation engine uses for Source/Destination scoring.
"""

import argparse
import re
import sys
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# STEP 2a — Name standardization rules
# --------------------------------------------------------------------------
# Regex-based typo/format fixes applied BEFORE grouping. Each was found by
# diffing the 134 raw college-name strings against each other (fuzzy match
# + manual verification against per-year enrollment counts, so that we only
# merge names that are truly the same institution, not just similarly named
# ones — e.g. "Mahendra Multiple Campus, Dang" and "..., Nepalgunj" are two
# DIFFERENT real colleges and must stay separate).
TYPO_FIXES = [
    (r'Jawalakhel', 'Jaulakhel'),
    (r'Butuwal', 'Butwal'),
    (r'Prithivi', 'Prithvi'),
    (r'Sambriddhi', 'Samriddhi'),
    (r'Paknajhol', 'Paknajol'),
    (r'\bTech\.', 'Tech'),
    (r'\s*&\s*', ' and '),
    (r'St\.\s*Lawrence', 'St. Lawrence'),
    (r'St\.\s*Xaviers', 'St. Xaviers'),
    # One-off full-string replacements for names that differ too much for
    # regex substitution to bridge (confirmed same institution by matching
    # yearly enrollment counts — see project notes):
    (r'^Siddhanta Science Campus$', 'Siddhanath Science Campus, Mahendranagar'),
    (r'^Mount Annapurna Campus$', 'Mount Annapurna College, Pokhara'),
    (r'^Soch College of IT, Pokhara$', 'Soch College, Pokhara'),
    (r'^Lumbini ICT Campus$', 'Lumbini ICT College, Gaindakota'),
    (r'^Lumbini ICT Campus, Gaindakota$', 'Lumbini ICT College, Gaindakota'),
]

KATHFORD_CANON = 'Kathford International College of Engineering and Management, Balkumari'

# Base names that are shared by genuinely DIFFERENT colleges in different
# districts. For these, keep the location as part of the grouping key
# instead of merging everything that shares the base name.
DO_NOT_MERGE_BASES = {'mahendra multiple campus'}


def pre_clean(name: str) -> str:
    """Apply whitespace cleanup + known typo/format fixes to a raw college name."""
    s = str(name).strip()
    s = re.sub(r'\s+', ' ', s)
    for pat, repl in TYPO_FIXES:
        s = re.sub(pat, repl, s, flags=re.I)
    s = s.strip().rstrip('.').strip()
    if 'kathford' in s.lower():
        return KATHFORD_CANON
    return s


def base_of(name: str) -> str:
    """Part of the name before the first comma, lowercased — used as the
    grouping key for merging formatting/spelling variants of the same
    college."""
    return name.split(',')[0].strip().lower()


def build_alias_map(raw_names) -> dict:
    """Map every raw college-name string to one canonical name.

    Strategy: pre-clean each name, then group by base name (the part
    before the first comma) and pick the longest variant in each group as
    canonical — longer names carry more location detail, which is more
    useful downstream. The exception is DO_NOT_MERGE_BASES, where we keep
    the full cleaned name as the grouping key so distinct campuses in
    different districts are not merged together.
    """
    cleaned = {n: pre_clean(n) for n in raw_names}

    groups = {}
    for n, c in cleaned.items():
        b = base_of(c)
        key = c if b in DO_NOT_MERGE_BASES else b
        groups.setdefault(key, []).append(c)

    canon_for_key = {
        key: max(set(variants), key=lambda x: (len(x), x))
        for key, variants in groups.items()
    }

    alias_map = {}
    for n, c in cleaned.items():
        b = base_of(c)
        key = c if b in DO_NOT_MERGE_BASES else b
        alias_map[n] = canon_for_key[key]
    return alias_map


# --------------------------------------------------------------------------
# STEP 1 — Data quality inventory (printed, not silently fixed)
# --------------------------------------------------------------------------
def print_data_quality_report(df: pd.DataFrame):
    print("=" * 70)
    print("STEP 1 — DATA QUALITY REPORT")
    print("=" * 70)
    print(f"Rows: {len(df)}   Columns: {list(df.columns)}")
    print(f"Years present: {sorted(df['Year'].unique())}")
    print(f"Unique raw college-name strings: {df['College'].nunique()}")
    print("\nMissing values per column:")
    print(df.isnull().sum().to_string())
    print(f"\nExact duplicate rows: {df.duplicated().sum()}")
    print(f"Duplicate Roll No (should be 0, one record per candidate): "
          f"{df['Roll No'].duplicated().sum()}")
    print("\nRank sanity check:")
    print(f"  min={df['Rank'].min()}  max={df['Rank'].max()}  "
          f"any <= 0: {(df['Rank'] <= 0).any()}")
    print("=" * 70 + "\n")


# --------------------------------------------------------------------------
# STEP 2 + 3 — Standardize + merge all years into one long table
# --------------------------------------------------------------------------
def standardize_and_aggregate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    # --- clean types ---
    df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce').astype('Int64')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
    df = df.dropna(subset=['Rank', 'Year', 'College'])

    # --- invalid rank / outlier flags (kept, not silently dropped) ---
    invalid_rank_mask = df['Rank'] <= 0
    if invalid_rank_mask.any():
        print(f"WARNING: {invalid_rank_mask.sum()} rows with non-positive "
              f"Rank found and removed.")
        df = df[~invalid_rank_mask]

    # --- standardize college names ---
    alias_map = build_alias_map(df['College'].unique())
    df['College_Raw'] = df['College']
    df['College'] = df['College'].map(alias_map)

    audit = (
        pd.DataFrame({'raw_name': list(alias_map.keys()),
                       'canonical_name': list(alias_map.values())})
        .sort_values(['canonical_name', 'raw_name'])
        .reset_index(drop=True)
    )

    # --- remove exact duplicate student records ---
    before = len(df)
    df = df.drop_duplicates(subset=['Roll No', 'Year'])
    if before != len(df):
        print(f"Removed {before - len(df)} duplicate (Roll No, Year) rows.")

    # --- STEP 3: aggregate to one row per (College, Year) ---
    agg = (
        df.groupby(['College', 'Year'])['Rank']
        .agg(Cutoff_Rank='max',        # worst rank admitted = official cutoff
             Best_Rank='min',          # best rank admitted that year
             Avg_Rank='mean',
             Num_Admitted='count')
        .reset_index()
        .sort_values(['College', 'Year'])
        .reset_index(drop=True)
    )
    agg['Avg_Rank'] = agg['Avg_Rank'].round(1)

    return agg, audit


# --------------------------------------------------------------------------
# STEP 4 — Feature engineering
# --------------------------------------------------------------------------
def engineer_features(agg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    agg = agg.sort_values(['College', 'Year']).reset_index(drop=True)

    # Year-over-year change in cutoff, within each college
    agg['Cutoff_Diff_YoY'] = agg.groupby('College')['Cutoff_Rank'].diff()

    # Rolling average cutoff (2-year window, expanding at the start)
    agg['Rolling_Avg_Cutoff'] = (
        agg.groupby('College')['Cutoff_Rank']
        .transform(lambda s: s.rolling(window=2, min_periods=1).mean())
        .round(1)
    )

    # Trend = slope of cutoff vs. year (linear fit) per college.
    # NaN for colleges with only one year of data (slope undefined).
    def _slope(group):
        if len(group) < 2:
            return pd.Series([np.nan] * len(group), index=group.index)
        years = group['Year'].astype(float).values
        cutoffs = group['Cutoff_Rank'].astype(float).values
        m, _ = np.polyfit(years, cutoffs, 1)
        return pd.Series([round(m, 2)] * len(group), index=group.index)

    agg['Cutoff_Trend'] = (
        agg.groupby('College', group_keys=False).apply(_slope)
    )

    # Volatility = std dev of cutoff across all years for that college
    # (constant per college, repeated on every year-row)
    volatility = agg.groupby('College')['Cutoff_Rank'].transform('std').round(1)
    agg['Cutoff_Volatility'] = volatility

    # How many years of history exist for this college
    agg['Years_of_Data'] = agg.groupby('College')['Year'].transform('count')

    # Percentile position of this college's cutoff WITHIN its year
    # (0 = most competitive that year, 1 = least competitive that year)
    agg['Cutoff_Percentile'] = (
        agg.groupby('Year')['Cutoff_Rank'].rank(pct=True, method='average').round(3)
    )

    # --- Master ranking table (one row per college, all-year aggregate) ---
    master = (
        agg.groupby('College')
        .agg(Avg_Cutoff_Rank_AllYears=('Cutoff_Rank', 'mean'),
             Best_Cutoff_Rank_Ever=('Cutoff_Rank', 'min'),
             Worst_Cutoff_Rank_Ever=('Cutoff_Rank', 'max'),
             Years_of_Data=('Year', 'nunique'),
             Latest_Year=('Year', 'max'))
        .reset_index()
    )
    master['Avg_Cutoff_Rank_AllYears'] = master['Avg_Cutoff_Rank_AllYears'].round(1)

    # Also attach each college's most recent cutoff (more relevant than the
    # multi-year average for "closest college to my rank today" matching)
    latest_cutoffs = (
        agg.sort_values('Year')
        .groupby('College')
        .tail(1)[['College', 'Year', 'Cutoff_Rank']]
        .rename(columns={'Year': 'Latest_Cutoff_Year', 'Cutoff_Rank': 'Latest_Cutoff_Rank'})
    )
    master = master.merge(latest_cutoffs, on='College', how='left')

    # Rank 1 = most competitive (lowest average cutoff number)
    master = master.sort_values('Avg_Cutoff_Rank_AllYears').reset_index(drop=True)
    master['College_Master_Rank'] = master.index + 1

    # Rank_Category via tertiles of the master rank
    n = len(master)
    def _category(rank_pos):
        if rank_pos <= n / 3:
            return 'Top-tier'
        elif rank_pos <= 2 * n / 3:
            return 'Mid-tier'
        return 'Lower-tier'
    master['Rank_Category'] = master['College_Master_Rank'].apply(_category)

    # Merge Master_Rank + Rank_Category back onto the per-year table
    agg = agg.merge(
        master[['College', 'College_Master_Rank', 'Rank_Category']],
        on='College', how='left'
    )

    return agg, master


# --------------------------------------------------------------------------
# STEP 5 — Final cleaning / sanity checks
# --------------------------------------------------------------------------
def final_clean(agg: pd.DataFrame) -> pd.DataFrame:
    before = len(agg)
    agg = agg.drop_duplicates(subset=['College', 'Year'])
    if len(agg) != before:
        print(f"Dropped {before - len(agg)} duplicate College-Year rows.")

    # Outlier flag: cutoff far outside the observed overall range
    lo, hi = agg['Cutoff_Rank'].quantile([0.01, 0.99])
    agg['Is_Outlier_Cutoff'] = ~agg['Cutoff_Rank'].between(lo, hi)
    n_out = agg['Is_Outlier_Cutoff'].sum()
    if n_out:
        print(f"NOTE: {n_out} College-Year rows flagged as statistical "
              f"outliers (outside 1st-99th percentile of cutoffs). "
              f"Kept in the dataset with a flag rather than dropped, "
              f"since a very high or low cutoff can be genuine.")

    return agg.reset_index(drop=True)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def build(input_path: str, outdir: str):
    print(f"Loading raw data from {input_path} ...")
    raw = pd.read_excel(input_path)

    print_data_quality_report(raw)

    print("STEP 2/3 — Standardizing names and merging years ...")
    agg, audit = standardize_and_aggregate(raw)
    print(f"  {raw['College'].nunique()} raw name variants -> "
          f"{agg['College'].nunique()} canonical colleges")

    print("STEP 4 — Engineering features ...")
    agg, master = engineer_features(agg)

    print("STEP 5 — Final cleaning ...")
    agg = final_clean(agg)

    # STEP 6 — final ML-ready dataset
    audit_path = f"{outdir}/name_mapping_audit.csv"
    dataset_path = f"{outdir}/final_master_dataset.csv"
    ranking_path = f"{outdir}/college_master_ranking.csv"

    audit.to_csv(audit_path, index=False)
    agg.to_csv(dataset_path, index=False)
    master.to_csv(ranking_path, index=False)

    print("\nSTEP 6 — Outputs written:")
    print(f"  {audit_path}      ({len(audit)} raw->canonical mappings)")
    print(f"  {dataset_path}    ({len(agg)} rows, {agg.shape[1]} columns)")
    print(f"  {ranking_path}  ({len(master)} colleges)")

    return agg, master, audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='fyp_dataset_combine_79_80_81_82_merged.xlsx')
    parser.add_argument('--outdir', default='.')
    args = parser.parse_args()
    build(args.input, args.outdir)
