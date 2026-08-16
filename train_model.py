"""
train_model.py
================
College Recommendation System — ML Model Training

Trains a model to predict a college's Cutoff_Rank for a given year, using
ONLY information that would genuinely have been available *before* that
year happened. This avoids data leakage: the pre-computed columns in
final_master_dataset.csv (Cutoff_Trend, Rolling_Avg_Cutoff, etc.) were
built using each row's OWN cutoff value, so they cannot be used as model
inputs directly. This script rebuilds a clean, leakage-free feature table
from scratch, using only Cutoff_Rank history from strictly earlier years.

Pipeline:
  1. Load final_master_dataset.csv, keep only (College, Year, Cutoff_Rank)
     — the raw ground truth — and discard the leaky pre-computed columns.
  2. For every (College, Year) row, compute features using only that
     college's cutoffs from years BEFORE the target year (via .shift(1)
     and expanding windows that exclude the current row).
  3. Drop rows with no prior year at all (nothing to predict from —
     equivalent to "first year we ever saw this college").
  4. Time-based train/test split: train on all transitions up to and
     including 2081->predicting old years, test on predicting 2082
     (the most recent actual year) — NOT a random split, because
     shuffling years would let the model "see the future" during
     training, which is unrealistic for a forecasting problem.
  5. Train Linear Regression (baseline) and Random Forest (main model),
     evaluate both with MAE / RMSE / R^2.
  6. Use the best model + ALL available history (2079-2082) to predict
     each college's cutoff for the upcoming year (2083).

Run:
    python3 train_model.py --dataset final_master_dataset.csv --outdir .
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------------------------------
# Leakage-free feature engineering
# --------------------------------------------------------------------------
def build_leakfree_features(df: pd.DataFrame) -> pd.DataFrame:
    """For each (College, Year) row, compute features using ONLY that
    college's Cutoff_Rank values from years strictly BEFORE Year.
    The target column (Cutoff_Rank) is the actual value for that year,
    kept separately so it's never accidentally used as an input."""
    df = df[['College', 'Year', 'Cutoff_Rank']].drop_duplicates()
    df = df.sort_values(['College', 'Year']).reset_index(drop=True)

    rows = []
    for college, group in df.groupby('College'):
        group = group.sort_values('Year').reset_index(drop=True)
        years = group['Year'].tolist()
        cutoffs = group['Cutoff_Rank'].tolist()

        for i in range(len(group)):
            target_year = years[i]
            target_cutoff = cutoffs[i]

            # Everything strictly BEFORE this row's year:
            prior_years = years[:i]
            prior_cutoffs = cutoffs[:i]

            if len(prior_cutoffs) == 0:
                # No history at all yet -> nothing to predict from, skip
                continue

            prev_cutoff = prior_cutoffs[-1]  # most recent known cutoff
            years_of_prior_data = len(prior_cutoffs)

            if len(prior_cutoffs) >= 2:
                rolling_avg_prior = np.mean(prior_cutoffs[-2:])
                volatility_prior = np.std(prior_cutoffs, ddof=0)
                slope, _ = np.polyfit(prior_years, prior_cutoffs, 1)
            else:
                rolling_avg_prior = prev_cutoff
                volatility_prior = 0.0
                slope = 0.0  # not enough points for a trend yet

            rows.append({
                'College': college,
                'Year': target_year,
                'Prev_Cutoff': prev_cutoff,
                'Rolling_Avg_Prior': rolling_avg_prior,
                'Cutoff_Trend_Prior': round(slope, 2),
                'Cutoff_Volatility_Prior': round(volatility_prior, 2),
                'Years_of_Prior_Data': years_of_prior_data,
                'Target_Cutoff_Rank': target_cutoff,   # ground truth to predict
            })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Train / evaluate
# --------------------------------------------------------------------------
FEATURE_COLS = ['Prev_Cutoff', 'Rolling_Avg_Prior', 'Cutoff_Trend_Prior',
                 'Cutoff_Volatility_Prior', 'Years_of_Prior_Data', 'Year']


def time_based_split(features: pd.DataFrame, test_year: int):
    """Train on every row predicting a year BEFORE test_year, test on
    rows predicting test_year. This mirrors real use: you only ever have
    past data available, and the test set is the most recent real year."""
    train = features[features['Year'] < test_year]
    test = features[features['Year'] == test_year]
    return train, test


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"  {name:20s}  MAE={mae:8.1f}   RMSE={rmse:8.1f}   R2={r2:6.3f}")
    return mae, rmse, r2


def train_and_evaluate(features: pd.DataFrame, test_year: int):
    train, test = time_based_split(features, test_year)
    print(f"Train rows: {len(train)} (predicting years before {test_year})")
    print(f"Test rows:  {len(test)} (predicting year {test_year})\n")

    X_train, y_train = train[FEATURE_COLS], train['Target_Cutoff_Rank']
    X_test, y_test = test[FEATURE_COLS], test['Target_Cutoff_Rank']

    print("Model performance on held-out year", test_year, ":")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    evaluate("Linear Regression", lr, X_test, y_test)

    # Baseline sanity check: "naive" model that just predicts last year's
    # cutoff unchanged. A real ML model should beat this, or it isn't
    # adding value over doing nothing.
    naive_preds = X_test['Prev_Cutoff']
    naive_mae = mean_absolute_error(y_test, naive_preds)
    naive_rmse = np.sqrt(mean_squared_error(y_test, naive_preds))
    naive_r2 = r2_score(y_test, naive_preds)
    print(f"  {'Naive (=last year)':20s}  MAE={naive_mae:8.1f}   "
          f"RMSE={naive_rmse:8.1f}   R2={naive_r2:6.3f}")

    rf = RandomForestRegressor(n_estimators=200, max_depth=4,
                                random_state=42, min_samples_leaf=3)
    rf.fit(X_train, y_train)
    evaluate("Random Forest", rf, X_test, y_test)

    print("\nRandom Forest feature importances:")
    for col, imp in sorted(zip(FEATURE_COLS, rf.feature_importances_),
                            key=lambda x: -x[1]):
        print(f"  {col:25s} {imp:.3f}")

    return lr, rf


def predict_next_year(model, features: pd.DataFrame, raw_cutoffs: pd.DataFrame,
                       next_year: int) -> pd.DataFrame:
    """Build one feature row per college using ALL currently known history
    (2079-2082), then predict next_year's cutoff for every college."""
    rows = []
    for college, group in raw_cutoffs.groupby('College'):
        group = group.sort_values('Year')
        years = group['Year'].tolist()
        cutoffs = group['Cutoff_Rank'].tolist()

        prev_cutoff = cutoffs[-1]
        if len(cutoffs) >= 2:
            rolling_avg_prior = np.mean(cutoffs[-2:])
            volatility_prior = np.std(cutoffs, ddof=0)
            slope, _ = np.polyfit(years, cutoffs, 1)
        else:
            rolling_avg_prior = prev_cutoff
            volatility_prior = 0.0
            slope = 0.0

        rows.append({
            'College': college,
            'Prev_Cutoff': prev_cutoff,
            'Rolling_Avg_Prior': rolling_avg_prior,
            'Cutoff_Trend_Prior': round(slope, 2),
            'Cutoff_Volatility_Prior': round(volatility_prior, 2),
            'Years_of_Prior_Data': len(cutoffs),
            'Year': next_year,
        })

    future = pd.DataFrame(rows)
    future['Predicted_Cutoff_Rank'] = model.predict(future[FEATURE_COLS]).round(0)
    return future.sort_values('Predicted_Cutoff_Rank').reset_index(drop=True)


# --------------------------------------------------------------------------
def run(dataset_path: str, outdir: str, test_year: int = 2082, next_year: int = 2083):
    print(f"Loading {dataset_path} ...")
    df = pd.read_csv(dataset_path)
    raw_cutoffs = df[['College', 'Year', 'Cutoff_Rank']].drop_duplicates()

    print("Building leakage-free feature table "
          "(each row uses ONLY prior-year data) ...\n")
    features = build_leakfree_features(df)
    features.to_csv(f"{outdir}/leakfree_training_features.csv", index=False)
    print(f"Leakage-free feature table: {len(features)} rows "
          f"(saved to leakfree_training_features.csv)\n")

    lr, rf = train_and_evaluate(features, test_year)

    print(f"\nPredicting {next_year} cutoffs for every college using the "
          f"Random Forest model ...")
    future = predict_next_year(rf, features, raw_cutoffs, next_year)
    out_path = f"{outdir}/predicted_cutoffs_{next_year}.csv"
    future[['College', 'Predicted_Cutoff_Rank', 'Prev_Cutoff',
            'Cutoff_Trend_Prior', 'Years_of_Prior_Data']].to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(future[['College', 'Predicted_Cutoff_Rank']].head(10).to_string(index=False))

    return lr, rf, features, future


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='final_master_dataset.csv')
    parser.add_argument('--outdir', default='.')
    parser.add_argument('--test-year', type=int, default=2082)
    parser.add_argument('--next-year', type=int, default=2083)
    args = parser.parse_args()
    run(args.dataset, args.outdir, args.test_year, args.next_year)
