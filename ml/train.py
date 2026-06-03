import argparse
import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from ml.features import extract_url_features
from ml.brand_features import brand_similarity_features


def build_feature_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Expect df with columns: url, label (0 legit, 1 phishing)."""
    urls = df["url"].astype(str).tolist()

    feats = []
    for u in urls:
        f1 = extract_url_features(u)
        f2 = brand_similarity_features(u)
        # Merge dicts
        f1.update({k: v for k, v in f2.items() if k != "brand_best_match"})
        # For brand_best_match string, drop for modeling
        feats.append(f1)

    X = pd.DataFrame(feats)
    y = df["label"].astype(int)
    return X, y


def train_models(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    numeric_cols = X.columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_cols)],
        remainder="drop",
    )

    candidates = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "rf": RandomForestClassifier(
            n_estimators=400,
            random_state=random_state,
            class_weight="balanced_subsample",
        ),
        "svm": SVC(probability=True, kernel="rbf", class_weight="balanced", random_state=random_state),
        "xgb": XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=random_state,
            eval_metric="logloss",
        ),
        "lgbm": LGBMClassifier(
            n_estimators=800,
            learning_rate=0.03,
            num_leaves=64,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            random_state=random_state,
        ),
    }

    results = {}
    fitted = {}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    for name, model in candidates.items():
        clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        clf.fit(X_train, y_train)

        proba = clf.predict_proba(X_test)
        phishing_idx = 1  # assuming labels {0,1}
        y_pred = np.argmax(proba, axis=1)

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, proba[:, phishing_idx])),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", pos_label=1)
        metrics.update({"precision": float(pr), "recall": float(rc), "f1": float(f1)})
        results[name] = metrics
        fitted[name] = clf

    return results, fitted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="CSV with columns: url,label")
    parser.add_argument("--out_dir", default="models")
    parser.add_argument("--best_model", default="lgbm")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    if "url" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain columns: url, label")

    X, y = build_feature_frame(df)
    results, fitted = train_models(X, y)

    # Save results
    joblib.dump(results, os.path.join(args.out_dir, "metrics.joblib"))

    # Pick best by ROC-AUC
    best = max(results.items(), key=lambda kv: kv[1]["roc_auc"])[0]
    chosen = args.best_model if args.best_model in fitted else best

    joblib.dump(
        {
            "model": fitted[chosen],
            "feature_columns": X.columns.tolist(),
            "chosen_model": chosen,
            "all_results": results,
        },
        os.path.join(args.out_dir, "phishguard_model.joblib"),
    )

    print("Training complete. Chosen model:", chosen)
    print("Top results by ROC-AUC:")
    for k, v in sorted(results.items(), key=lambda kv: kv[1]["roc_auc"], reverse=True):
        print(k, v)


if __name__ == "__main__":
    main()

