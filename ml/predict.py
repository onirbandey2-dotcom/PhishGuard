import os
from typing import Dict, Any, List, Optional

import joblib
import numpy as np

from ml.features import extract_url_features
from ml.brand_features import brand_similarity_features


ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "phishguard_model.joblib")


def _fallback_heuristic(url: str) -> Dict[str, Any]:
    f = extract_url_features(url)
    b = brand_similarity_features(url)

    score = 0.0
    reasons: List[Dict[str, Any]] = []

    # Weighting based on intuition
    weights = {
        "has_suspicious_keyword": 18,
        "suspicious_keyword_count": 3,
        "url_shortening": 20,
        "num_special_chars": 0.7,
        "num_hyphens": 1.8,
        "url_length": 0.25,
        "path_length": 0.08,
        "has_at": 18,
        "has_double_slash_redirection": 12,
        "is_ip_address": 30,
        "has_typosquatting": 22,
        "brand_similarity": 45,
        "homoglyph_like": 15,
        "https_usage": -5.0,  # phishing can still use https, but keep slight penalty
    }

    merged = {**f, **b}
    for k, w in weights.items():
        if k not in merged:
            continue
        val = merged.get(k)
        try:
            val_num = float(val)
        except Exception:
            continue
        delta = w * val_num
        score += delta
        if val_num and (k in ["has_suspicious_keyword", "url_shortening", "has_at", "has_typosquatting", "is_ip_address", "homoglyph_like"]):
            reasons.append({"reason": k, "weight": float(w)})

    # Clamp score to 0..100 via logistic transform
    risk = 100.0 / (1.0 + np.exp(-score / 50.0))
    pred = "phishing" if risk >= 50 else "legitimate"

    # Provide top reasons
    reasons = sorted(reasons, key=lambda x: abs(x["weight"]), reverse=True)
    return {
        "prediction": pred,
        "risk_score": float(risk),
        "probabilities": {"phishing": float(risk / 100.0), "legitimate": float(1 - risk / 100.0)},
        "reasons": reasons[:5],
        "features": merged,
    }


def _compute_reasons_from_linearish_model(model_obj, X_row: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
    # For tree models we won't have stable per-feature weights; SHAP is preferred.
    # This function attempts to provide a simple explanation from available coefficients if present.
    reasons: List[Dict[str, Any]] = []

    clf = model_obj
    try:
        # sklearn Pipeline: preprocessor + model
        inner_model = clf.named_steps.get("model")
        pre = clf.named_steps.get("preprocessor")

        # Attempt coefficients
        coef = getattr(inner_model, "coef_", None)
        if coef is None:
            return []

        feature_names = clf.named_steps["preprocessor"].transformers_[0][2]
        coef_vec = coef[0] if coef.ndim > 1 else coef

        # Transform feature row as used by model is non-trivial; but for a basic reason list,
        # we use raw feature values multiplied by coef (approx.)
        for name, c in zip(feature_names, coef_vec):
            val = X_row.get(name, 0.0)
            try:
                val = float(val)
            except Exception:
                val = 0.0
            weight = c * val
            reasons.append({"reason": name, "weight": float(weight)})

        reasons = sorted(reasons, key=lambda x: abs(x["weight"]), reverse=True)
        return reasons[:top_k]
    except Exception:
        return []


def predict_with_explanations(url: str, top_k_reasons: int = 5) -> Dict[str, Any]:
    url = (url or "").strip()
    merged = {**extract_url_features(url), **{k: v for k, v in brand_similarity_features(url).items() if k != "brand_best_match"}}

    if not os.path.exists(ARTIFACT_PATH):
        # No artifacts present: use fallback heuristic.
        return _fallback_heuristic(url)

    bundle = joblib.load(ARTIFACT_PATH)
    model = bundle["model"]
    feature_columns = bundle.get("feature_columns", list(merged.keys()))

    # Build X row in expected column order
    X_row = {col: merged.get(col, 0.0) for col in feature_columns}
    # sklearn expects DataFrame
    import pandas as pd

    X_df = pd.DataFrame([X_row], columns=feature_columns)

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_df)[0]
        # index 1 = phishing
        phishing_p = float(proba[1]) if len(proba) > 1 else float(proba[0])
        risk = phishing_p * 100.0
        pred = "phishing" if phishing_p >= 0.5 else "legitimate"
        probabilities = {"phishing": phishing_p, "legitimate": float(1 - phishing_p)}
    else:
        pred = str(model.predict(X_df)[0])
        risk = None
        probabilities = {}

    # Explanation attempt: if model has coefficients return approximate reasons.
    reasons = _compute_reasons_from_linearish_model(model, X_row, top_k=top_k_reasons)

    if not reasons:
        # Provide heuristic reasons even when ML model exists.
        heuristic = _fallback_heuristic(url)
        reasons = heuristic.get("reasons", [])[:top_k_reasons]

    return {
        "prediction": pred,
        "risk_score": float(risk) if risk is not None else None,
        "probabilities": probabilities,
        "reasons": reasons,
        "features": merged,
    }

