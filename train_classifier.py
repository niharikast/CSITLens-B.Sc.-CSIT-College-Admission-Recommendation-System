"""
train_classifier.py
=====================
Trains a REAL classification model on genuine admit/reject outcomes
(from build_priority_training_data.py), instead of the hand-designed
sigmoid formula used in recommendation_engine.py.

For every labeled (student_rank, college, priority_position, admitted)
example, we attach leakage-free historical features for that college
(reusing the same prior-years-only logic as train_model.py), then train
a classifier to predict P(admitted = 1).

Same leakage-avoidance rule as before: features for a (college, year)
example use ONLY that college's cutoff history from years strictly
before year — never that year's own actual outcome.

Run:
    python3 train_classifier.py
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_model import build_leakfree_features  # reuse the same leak-free logic

FEATURE_COLS = ['student_rank', 'priority_position', 'Prev_Cutoff',
                 'Rolling_Avg_Prior', 'Cutoff_Trend_Prior',
                 'Cutoff_Volatility_Prior', 'Years_of_Prior_Data',
                 'Rank_Minus_Prev_Cutoff']


def build_training_table(labeled_path, dataset_path):
    print("Loading labeled admissions and building leak-free college features ...")
    labeled = pd.read_csv(labeled_path)
    dataset = pd.read_csv(dataset_path)

    leakfree = build_leakfree_features(dataset)  # College, Year, Prev_Cutoff, ...
    leakfree = leakfree.rename(columns={'Year': 'year', 'College': 'college'})

    merged = labeled.merge(leakfree, on=['college', 'year'], how='inner')
    dropped = len(labeled) - len(merged)
    print(f"Merged with historical features: {len(merged)} rows kept, "
          f"{dropped} dropped (no prior-year history available yet for "
          f"that college — usually its first appearance in the data).")

    merged['Rank_Minus_Prev_Cutoff'] = merged['student_rank'] - merged['Prev_Cutoff']
    return merged


def time_split(df, test_year=2082):
    train = df[df['year'] < test_year]
    test = df[df['year'] == test_year]
    return train, test


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, probs)
    print(f"\n{name}")
    print(f"  Accuracy={acc:.3f}  Precision={prec:.3f}  Recall={rec:.3f}  "
          f"F1={f1:.3f}  ROC-AUC={auc:.3f}")
    cm = confusion_matrix(y_test, preds)
    print(f"  Confusion matrix [[TN FP] [FN TP]]:\n{cm}")
    return {'accuracy': acc, 'precision': prec, 'recall': rec,
            'f1': f1, 'roc_auc': auc}


def run(labeled_path='labeled_admissions.csv', dataset_path='final_master_dataset.csv',
        outdir='.'):
    merged = build_training_table(labeled_path, dataset_path)
    train, test = time_split(merged, test_year=2082)
    print(f"\nTrain rows: {len(train)}   Test rows (held-out year 2082): {len(test)}")
    print(f"Test set admit rate: {test['admitted'].mean():.1%}  "
          f"(this is the baseline a model must beat)")

    X_train, y_train = train[FEATURE_COLS], train['admitted']
    X_test, y_test = test[FEATURE_COLS], test['admitted']

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    evaluate("Logistic Regression", lr, X_test, y_test)

    rf = RandomForestClassifier(n_estimators=300, max_depth=6,
                                 min_samples_leaf=5, random_state=42,
                                 class_weight='balanced')
    rf.fit(X_train, y_train)
    evaluate("Random Forest Classifier", rf, X_test, y_test)

    print("\nRandom Forest feature importances:")
    for col, imp in sorted(zip(FEATURE_COLS, rf.feature_importances_),
                            key=lambda x: -x[1]):
        print(f"  {col:28s} {imp:.3f}")

    import joblib
    model_path = f"{outdir}/admission_classifier.pkl"
    joblib.dump({'model': rf, 'feature_cols': FEATURE_COLS}, model_path)
    print(f"\nSaved trained model: {model_path}")

    return lr, rf, merged


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labeled', default='labeled_admissions.csv',
                         help='Path to labeled_admissions.csv (from build_priority_training_data.py)')
    parser.add_argument('--dataset', default='final_master_dataset.csv',
                         help='Path to final_master_dataset.csv (from build_master_dataset.py)')
    parser.add_argument('--outdir', default='.', help='Where to save admission_classifier.pkl')
    args = parser.parse_args()
    run(args.labeled, args.dataset, args.outdir)
