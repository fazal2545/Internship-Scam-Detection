"""
Member 03 (Fesal) - ML model: XGBoost on TF-IDF features.
Class imbalance handled with scale_pos_weight (per Section 3 justification).

Run:  python member3_fesal/train_xgboost.py --data dataset/fake_job_postings.csv
"""

import argparse
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing import load_and_clean, train_test_split_stratified
from src.evaluate import evaluate_model, plot_confusion_matrix

from sklearn.feature_extraction.text import TfidfVectorizer

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def main(data_path: str):
    from xgboost import XGBClassifier

    df = load_and_clean(data_path)
    X_train, X_test, y_train, y_test = train_test_split_stratified(df)

    tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2),
                            sublinear_tf=True, min_df=3)
    Xtr = tfidf.fit_transform(X_train)
    Xte = tfidf.transform(X_test)

    # scale_pos_weight = n_negative / n_positive
    spw = float((y_train == 0).sum()) / max((y_train == 1).sum(), 1)
    print(f"scale_pos_weight = {spw:.2f}")

    clf = XGBClassifier(
        n_estimators=600,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    print("Training XGBoost ...")
    clf.fit(Xtr, y_train, eval_set=[(Xte, y_test)], verbose=100)

    y_pred = clf.predict(Xte)
    y_prob = clf.predict_proba(Xte)[:, 1]
    m = evaluate_model("XGBoost", y_test, y_pred, y_prob)
    plot_confusion_matrix("XGBoost", m["confusion_matrix"])

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump({"vectorizer": tfidf, "model": clf},
                os.path.join(MODELS_DIR, "xgboost.joblib"))
    print("Saved -> models/xgboost.joblib")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="dataset/fake_job_postings.csv")
    main(p.parse_args().data)
