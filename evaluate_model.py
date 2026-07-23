import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from ml.features import extract_url_features
from ml.brand_features import brand_similarity_features


def _detect_columns(df: pd.DataFrame) -> Tuple[str, str]:
    url_col_candidates = [
        "url",
        "URL",
        "link",
        "Link",
        "website",
        "Website",
        "phishing_url",
        "Phishing_URL",
        "full_url",
        "Full_URL",
    ]

    label_col_candidates = [
        "phishing",
        "label",
        "Phishing",
        "Label",
        "is_phishing",
        "Is_phishing",
        "target",
        "Target",
        "result",
        "Result",
    ]

    url_col = next((c for c in url_col_candidates if c in df.columns), None)
    if url_col is None:
        for c in df.columns:
            if "url" in str(c).lower():
                url_col = c
                break

    label_col = next((c for c in label_col_candidates if c in df.columns), None)
    if label_col is None:
        for c in df.columns:
            cl = str(c).lower()
            if "label" in cl or "phish" in cl:
                label_col = c
                break

    if url_col is None or label_col is None:
        raise ValueError(
            "Could not auto-detect required columns. "
            "Expected a URL column and a label column. "
            f"Found columns: {list(df.columns)}"
        )

    return url_col, label_col


def _coerce_label_series(y: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(y):
        return y.astype(int).to_numpy()

    y_str = y.astype(str).str.strip().str.lower()

    mapping = {
        "1": 1,
        "0": 0,
        "phishing": 1,
        "dangerous": 1,
        "danger": 1,
        "malicious": 1,
        "true": 1,
        "yes": 1,
        "safe": 0,
        "legitimate": 0,
        "benign": 0,
        "false": 0,
        "no": 0,
    }

    # Try numeric conversion first
    try:
        as_num = pd.to_numeric(y_str, errors="raise")
        if set(as_num.unique()).issubset({0, 1}):
            return as_num.astype(int).to_numpy()
    except Exception:
        pass

    y_mapped = y_str.map(mapping)
    if y_mapped.isna().any():
        missing_examples = y.loc[y_mapped.isna()].head(10).tolist()
        raise ValueError(f"Could not map some labels to binary values. Examples: {missing_examples}")

    return y_mapped.astype(int).to_numpy()


def _extract_all_features_for_url(url: str) -> Dict[str, Any]:
    # Training in ml/train.py uses engineered numeric features from the training CSV.
    # Those engineered columns are produced from URL heuristics in ml/features.py.
    # We reconstruct exactly the expected feature columns (by name + order) from the artifact.
    f_url = extract_url_features(url)
    f_brand = brand_similarity_features(url)
    return {**f_url, **f_brand}


def _build_feature_matrix(urls: List[str], feature_columns: List[str], batch_size: int) -> np.ndarray:
    n = len(urls)
    X_out = np.empty((n, len(feature_columns)), dtype=float)

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        rows: List[Dict[str, Any]] = []
        for u in urls[start:end]:
            f = _extract_all_features_for_url(u)
            # IMPORTANT: align by exact trained column names
            aligned = {col: f.get(col, 0.0) for col in feature_columns}
            rows.append(aligned)
        X_batch = pd.DataFrame.from_records(rows, columns=feature_columns).to_numpy(dtype=float)
        X_out[start:end, :] = X_batch

    return X_out


def evaluate(
    model_path: str,
    excel_path: str,
    out_csv: str,
    url_col: Optional[str] = None,
    label_col: Optional[str] = None,
    batch_size: int = 5000,
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel dataset not found: {excel_path}")

    bundle = joblib.load(model_path)
    if isinstance(bundle, dict) and "model" in bundle:
        model = bundle["model"]
        feature_columns: List[str] = bundle.get("feature_columns", [])
    else:
        model = bundle
        feature_columns = []

    if not feature_columns:
        raise ValueError(
            "Model artifact does not contain 'feature_columns'. "
            "Cannot reconstruct the exact feature order used during training."
        )

    df = pd.read_excel(excel_path)

    if url_col is None or label_col is None:
        detected_url_col, detected_label_col = _detect_columns(df)
        url_col = detected_url_col if url_col is None else url_col
        label_col = detected_label_col if label_col is None else label_col

    if url_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"Detected columns not present. url_col={url_col}, label_col={label_col}. "
            f"Columns: {list(df.columns)}"
        )

    df = df.loc[df[url_col].notna() & df[label_col].notna()].copy()
    urls = df[url_col].astype(str).tolist()
    y_true = _coerce_label_series(df[label_col])

    # Reconstruct features from the SAME extraction functions, aligned to the model's expected columns.
    X = _build_feature_matrix(urls, feature_columns, batch_size=batch_size)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        phishing_p = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        y_pred = (phishing_p >= 0.5).astype(int)
        y_score_phishing = phishing_p.astype(float)
    else:
        y_pred = np.asarray(model.predict(X)).astype(int)
        y_score_phishing = np.full_like(y_pred, np.nan, dtype=float)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().tolist()

    fp_rate = fp / (fp + tn) if (fp + tn) else 0.0
    fn_rate = fn / (fn + tp) if (fn + tp) else 0.0

    eval_df = pd.DataFrame(
        {
            "url": urls,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_score_phishing": y_score_phishing,
            "is_false_positive": (y_true == 0) & (y_pred == 1),
            "is_false_negative": (y_true == 1) & (y_pred == 0),
        }
    )

    eval_df["accuracy"] = acc
    eval_df["precision"] = prec
    eval_df["recall"] = rec
    eval_df["f1_score"] = f1
    eval_df["confusion_matrix_json"] = str(cm.tolist())
    eval_df["false_positives"] = fp
    eval_df["false_negatives"] = fn

    eval_df.to_csv(out_csv, index=False, encoding="utf-8")

    print("================== Academic Evaluation Report ==================")
    print(f"Dataset (rows evaluated): {len(eval_df)}")
    print(f"URL column: {url_col}")
    print(f"Label column: {label_col}")
    print(f"Model artifact: {model_path}")
    print(f"Saved detailed CSV: {out_csv}")
    print("---------------------------------------------------------------")
    print("Classification Metrics (binary: phishing=1, legitimate=0):")
    print(f"  Accuracy : {acc:.6f}")
    print(f"  Precision: {prec:.6f}")
    print(f"  Recall   : {rec:.6f}")
    print(f"  F1 Score : {f1:.6f}")
    print("---------------------------------------------------------------")
    print("Confusion Matrix (labels [0,1]):")
    print(f"  TN={tn}  FP={fp}")
    print(f"  FN={fn}  TP={tp}")
    print("---------------------------------------------------------------")
    print("Error Analysis:")
    print(f"  False Positive Count: {fp}  (FP rate={fp_rate:.6f})")
    print(f"  False Negative Count: {fn}  (FN rate={fn_rate:.6f})")
    print("===============================================================")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained phishing model on an Excel dataset.")
    parser.add_argument(
        "--model",
        default=os.path.join("models", "phishguard_model.joblib"),
        help="Path to trained model joblib artifact.",
    )
    parser.add_argument(
        "--data",
        default=os.path.join("data", "phishing url data.xlsx"),
        help="Path to the Excel file containing URLs and labels.",
    )
    parser.add_argument(
        "--out",
        default="evaluation_report.csv",
        help="Output CSV path for evaluation results.",
    )
    parser.add_argument("--url-col", default=None, help="Optional: name of the URL column")
    parser.add_argument("--label-col", default=None, help="Optional: name of the label column")
    parser.add_argument("--batch-size", type=int, default=5000, help="Batch size for feature extraction/prediction")

    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        excel_path=args.data,
        out_csv=args.out,
        url_col=args.url_col,
        label_col=args.label_col,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

