"""
ml/explain.py — SHAP-based explainability for PhishGuard.

Falls back gracefully to feature-weight heuristic when SHAP is unavailable
or the model type does not support it.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ARTIFACT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "phishguard_model.joblib"
)


def _shap_explanations(
    bundle: Dict[str, Any],
    X_df: pd.DataFrame,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Return SHAP-based top-k reasons for a single row prediction."""
    try:
        import shap  # imported lazily — not required at module load
    except ImportError:
        return []

    model = bundle["model"]
    feature_columns = bundle.get("feature_columns", X_df.columns.tolist())

    try:
        # sklearn Pipeline: extract the fitted preprocessor output
        pre = model.named_steps.get("preprocessor")
        inner = model.named_steps.get("model")

        X_transformed = pre.transform(X_df)
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        # Tree-based models (RF, XGB, LGBM) → TreeExplainer (fast)
        try:
            explainer = shap.TreeExplainer(inner)
            shap_values = explainer.shap_values(X_transformed)

            # Binary classification: shap_values may be list[2] or ndarray
            if isinstance(shap_values, list):
                sv = shap_values[1][0]  # class 1 (phishing), first row
            else:
                sv = shap_values[0]

        except Exception:
            # Fall back to KernelExplainer for non-tree models (slower)
            bg_data = np.zeros((1, X_transformed.shape[1]))
            explainer = shap.KernelExplainer(
                lambda x: inner.predict_proba(x)[:, 1], bg_data
            )
            shap_values = explainer.shap_values(X_transformed, nsamples=50)
            sv = shap_values[0]

        reasons: List[Dict[str, Any]] = []
        for name, val in zip(feature_columns, sv):
            reasons.append({"reason": name, "weight": float(val)})

        reasons.sort(key=lambda x: abs(x["weight"]), reverse=True)
        return reasons[:top_k]

    except Exception:
        return []


def explain_prediction(
    url: str,
    top_k: int = 5,
    bundle: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Return top-k feature explanations for a URL prediction.

    Parameters
    ----------
    url : str
        The URL to explain.
    top_k : int
        How many top reasons to return.
    bundle : dict, optional
        Pre-loaded model bundle (avoids re-loading from disk).

    Returns
    -------
    List of dicts with keys ``reason`` (feature name) and ``weight`` (SHAP value).
    """
    from ml.features import extract_url_features
    from ml.brand_features import brand_similarity_features

    url = (url or "").strip()
    merged = {
        **extract_url_features(url),
        **{
            k: v
            for k, v in brand_similarity_features(url).items()
            if k != "brand_best_match"
        },
    }

    if bundle is None:
        if not os.path.exists(ARTIFACT_PATH):
            return _heuristic_reasons(merged, top_k)
        import joblib
        bundle = joblib.load(ARTIFACT_PATH)

    feature_columns = bundle.get("feature_columns", list(merged.keys()))
    X_row = {col: merged.get(col, 0.0) for col in feature_columns}
    X_df = pd.DataFrame([X_row], columns=feature_columns)

    reasons = _shap_explanations(bundle, X_df, top_k)
    if reasons:
        return reasons

    # No SHAP available → fall back to heuristic
    return _heuristic_reasons(merged, top_k)


def _heuristic_reasons(features: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
    """Simple weight × value heuristic used when SHAP is unavailable.

    IMPORTANT: Must return *exactly* `top_k` reasons.
    We generate contributions for all known heuristic feature signals,
    then sort by absolute contribution magnitude and pad/truncate.
    """

    WEIGHTS: Dict[str, float] = {
        "is_ip_address": 30.0,
        "has_typosquatting": 22.0,
        "brand_similarity": 45.0,
        "url_shortening": 20.0,
        "has_suspicious_keyword": 18.0,
        "has_at": 18.0,
        "has_double_slash_redirection": 12.0,
        "homoglyph_like": 15.0,
        "suspicious_keyword_count": 3.0,
        "num_hyphens": 1.8,
        "num_special_chars": 0.7,
        "url_length": 0.25,
        "https_usage": -5.0,
    }

    # Ensure top_k is sane
    try:
        top_k_int = int(top_k)
    except Exception:
        top_k_int = 5
    top_k_int = max(1, top_k_int)

    reasons_all: List[Dict[str, Any]] = []
    for k, w in WEIGHTS.items():
        val = features.get(k, 0)
        try:
            val_num = float(val)
        except Exception:
            val_num = 0.0
        weight = float(w * val_num)
        # Always include the feature signal even if val_num == 0 so we can pad.
        reasons_all.append({"reason": k, "weight": weight})

    # Sort by magnitude (positive + negative both allowed)
    reasons_all.sort(key=lambda x: abs(x["weight"]), reverse=True)

    # Truncate or pad deterministically.
    if len(reasons_all) >= top_k_int:
        return reasons_all[:top_k_int]

    # Padding: use next highest-ranked feature signals.
    # (Since we already included all WEIGHTS, this should rarely trigger.)
    pad_needed = top_k_int - len(reasons_all)
    if pad_needed > 0 and len(reasons_all) > 0:
        i = 0
        while len(reasons_all) < top_k_int:
            reasons_all.append(reasons_all[i % len(reasons_all)])
            i += 1

    return reasons_all[:top_k_int]

