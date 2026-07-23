import os
from typing import Dict, Any, List, Optional

import joblib
import numpy as np

from ml.features import extract_url_features
from ml.brand_features import brand_similarity_features


ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "phishguard_model.joblib")


TRUSTED_DOMAIN_WHITELIST = {
    # Domain whitelist for near-zero risk.
    # NOTE: This list is also mirrored in ml/features.py and app/ml email analyzer.
    "google.com",
    "amazon.com",
    "github.com",
    "microsoft.com",
    "apple.com",
    "linkedin.com",
    "paytm.com",
    "srmist.edu.in",
}




def _is_trusted_domain_safe_override(merged: Dict[str, Any]) -> bool:
    """Trusted-domain override gate (per spec).

    If is_trusted_domain == 1 AND not has_typosquatting AND not is_ip_address
    then we force SAFE + risk_score=0 before any final verdict calculation.
    """
    return (
        int(merged.get("is_trusted_domain", merged.get("is_whitelisted", 0)) or 0) == 1
        and int(merged.get("has_typosquatting", 0) or 0) == 0
        and int(merged.get("is_ip_address", 0) or 0) == 0
    )




def _apply_high_risk_gating(merged: Dict[str, Any], risk_score: float, reasons: List[Dict[str, Any]]):
    """Reduce false positives by requiring >=3 independent high-risk indicators before Dangerous."""

    # Indicators (do not let suspicious keywords alone trigger Dangerous)
    indicators: List[Dict[str, Any]] = []

    has_susp_kw = int(merged.get("has_suspicious_keyword", 0) or 0)
    no_https = 1 if (int(merged.get("has_https", 0) or 0) == 0 and int(merged.get("https_usage", 0) or 0) == 0) else 0
    has_at = int(merged.get("has_at_symbol", merged.get("has_at", 0)) or 0)
    too_many_subdomains = 1 if int(merged.get("num_subdomains", 0) or 0) >= 2 else 0
    many_special_chars = 1 if float(merged.get("special_char_count", merged.get("num_special_chars", 0)) or 0) >= 10 else 0
    suspicious_keyword_count = int(merged.get("suspicious_keyword_count", 0) or 0)
    url_shortening = int(merged.get("url_shortening", 0) or 0)
    is_ip_address = int(merged.get("is_ip_address", 0) or 0)
    has_typosquatting = int(merged.get("has_typosquatting", 0) or 0)

    # Build indicator list; keywords indicator is only counted if it appears alongside other suspicious structure.
    if has_susp_kw and suspicious_keyword_count >= 2:
        indicators.append({"reason": "Suspicious keywords cluster", "weight": 8.0})

    if no_https:
        indicators.append({"reason": "No HTTPS", "weight": 8.0})
    if has_at:
        indicators.append({"reason": "@ symbol in URL", "weight": 8.0})
    if too_many_subdomains:
        indicators.append({"reason": "Many subdomains", "weight": 8.0})
    if many_special_chars:
        indicators.append({"reason": "Many special characters", "weight": 8.0})

    # Add other strong signals to avoid FN/keep accuracy
    if url_shortening:
        indicators.append({"reason": "URL shortener", "weight": 10.0})
    if is_ip_address:
        indicators.append({"reason": "IP address in URL", "weight": 10.0})
    if has_typosquatting:
        indicators.append({"reason": "Typosquatting / brand mismatch", "weight": 12.0})

    active = len(indicators)

    adjusted_risk = float(risk_score)

    # Do not force verdict here; trusted override is applied before final verdict.


    # Dangerous gating: if fewer than 3 indicators, prevent Dangerous.
    if active < 3:
        adjusted_risk = min(adjusted_risk, 60.0)
        # downgrade to Suspicious at most
        reasons_extra = [r["reason"] for r in indicators]
        return None, adjusted_risk, reasons_extra

    # otherwise keep as dangerous-eligible
    reasons_extra = [r["reason"] for r in indicators]
    return None, adjusted_risk, reasons_extra


def _fallback_heuristic(url: str) -> Dict[str, Any]:
    f = extract_url_features(url)
    b = brand_similarity_features(url)

    score = 0.0
    reasons: List[Dict[str, Any]] = []

    # Weighting based on intuition (kept, but final verdict uses gating)
    weights = {
        "has_suspicious_keyword": 10,
        "suspicious_keyword_count": 2,
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
        "https_usage": -5.0,
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
        score += w * val_num
        if val_num and k in ["has_suspicious_keyword", "url_shortening", "has_at", "has_typosquatting", "is_ip_address", "homoglyph_like"]:
            reasons.append({"reason": k, "weight": float(w)})

    risk = 100.0 / (1.0 + np.exp(-score / 50.0))

    # gating + thresholds handled below
    gating_verdict, adj_risk, gating_reasons = _apply_high_risk_gating(merged, risk, reasons)

    risk = float(min(max(adj_risk, 0.0), 100.0))

    if gating_verdict == "Legitimate":
        verdict = "Legitimate"
    else:
        # thresholds: 0–39 Legit, 40–69 Suspicious, 70–100 Dangerous
        if risk < 40:
            verdict = "Legitimate"
        elif risk < 70:
            verdict = "Suspicious"
        else:
            verdict = "Dangerous"

        if verdict == "Dangerous" and len(gating_reasons) < 3:
            verdict = "Suspicious"
            risk = min(risk, 69.0)

    phishing_pct = float(min(max(risk, 0.0), 100.0))
    legit_pct = float(100.0 - phishing_pct)

    # Compose reasons list (strings)
    reason_strings: List[str] = []
    if gating_verdict == "Legitimate":
        reason_strings = ["Trusted domain whitelist"]
    else:
        # include top heuristic reasons + gating reasons
        top = sorted(reasons, key=lambda x: abs(x.get("weight", 0.0)), reverse=True)[:5]
        reason_strings = [r.get("reason", "") for r in top if r.get("reason")]
        reason_strings.extend([r for r in gating_reasons if r])
        # dedupe
        reason_strings = list(dict.fromkeys(reason_strings))[:10]

    return {
        "verdict": verdict,
        "risk_score": risk,
        "confidence": {"phishing": phishing_pct, "legitimate": legit_pct},
        "reasons": reason_strings,
        "features": merged,
    }



def _compute_reasons_from_linearish_model(model_obj, X_row: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
    """Legacy/compat explanation attempt.

    Prefer SHAP-based explanations from `ml.explain.explain_prediction`.
    This function is kept as a last-resort for cases where importing SHAP fails
    and the SHAP integration cannot run.
    """
    reasons: List[Dict[str, Any]] = []

    clf = model_obj
    try:
        inner_model = clf.named_steps.get("model")

        # Attempt coefficients
        coef = getattr(inner_model, "coef_", None)
        if coef is None:
            return []

        feature_names = clf.named_steps["preprocessor"].transformers_[0][2]
        coef_vec = coef[0] if getattr(coef, "ndim", 1) > 1 else coef

        for name, c in zip(feature_names, coef_vec):
            val = X_row.get(name, 0.0)
            try:
                val = float(val)
            except Exception:
                val = 0.0
            weight = float(c) * val
            reasons.append({"reason": name, "weight": weight})

        reasons = sorted(reasons, key=lambda x: abs(x["weight"]), reverse=True)
        return reasons[:top_k]
    except Exception:
        return []



def predict_with_explanations(url: str, top_k_reasons: int = 5) -> Dict[str, Any]:
    url = (url or "").strip()

    # Normalize top_k_reasons and guarantee exact-length reasons output.
    try:
        top_k_int = int(top_k_reasons)
    except Exception:
        top_k_int = 5
    top_k_int = max(1, top_k_int)

    merged = {**extract_url_features(url), **{k: v for k, v in brand_similarity_features(url).items() if k != "brand_best_match"}}

    # Debug logs to verify model-vs-fallback execution.
    # Set PHISHGUARD_DEBUG=1 to enable.
    debug = os.environ.get("PHISHGUARD_DEBUG", "0") == "1"
    if debug:
        print("[PhishGuard] ARTIFACT_PATH:", ARTIFACT_PATH)
        print("[PhishGuard] artifact exists:", os.path.exists(ARTIFACT_PATH))
        print("[PhishGuard] sample merged features:", {k: merged.get(k) for k in ["is_whitelisted","has_https","url_shortening","has_suspicious_keyword","suspicious_keyword_count","has_typosquatting"] if k in merged})

    if not os.path.exists(ARTIFACT_PATH):
        # No artifacts present: use fallback heuristic.
        if debug:
            print("[PhishGuard] Using fallback heuristic (missing model artifact).")
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

    # Prefer SHAP-based explanations from ml.explain.
    reasons: List[Dict[str, Any]] = []
    try:
        from ml.explain import explain_prediction

        reasons = explain_prediction(url, top_k=top_k_reasons, bundle=bundle)
    except Exception:
        # Fall back to legacy/heuristic if SHAP explainer cannot run.
        reasons = _compute_reasons_from_linearish_model(model, X_row, top_k=top_k_reasons)
        if not reasons:
            heuristic = _fallback_heuristic(url)
            reasons = heuristic.get("reasons", [])[:top_k_reasons]

    # Safety net: enforce exact reasons length.
    # The UI renders exactly len(reasons), so we must return exactly top_k_reasons.
    if isinstance(reasons, list):
        # In normal operation `reasons` should be a List[Dict].
        # But some legacy paths return List[str]; normalize to Dict for the UI.
        if reasons and not isinstance(reasons[0], dict):
            reasons = [{"reason": str(x), "weight": 0.0} for x in reasons]

        if len(reasons) > top_k_int:
            reasons = reasons[:top_k_int]
        elif len(reasons) < top_k_int:
            # Pad with deterministic remaining highest-magnitude features from the heuristic.
            from ml.explain import _heuristic_reasons

            heuristic_reasons = _heuristic_reasons(merged, top_k_int)
            # heuristic_reasons is already sorted by abs(weight) and length == top_k_int
            # so we can use it as the padding source.
            for pad in heuristic_reasons:
                if len(reasons) >= top_k_int:
                    break
                # Avoid duplicates by feature name when possible.
                if isinstance(pad, dict) and any(
                    isinstance(r, dict) and r.get("reason") == pad.get("reason") for r in reasons
                ):
                    continue
                reasons.append(pad)

            # Final truncate to be strict.
            reasons = reasons[:top_k_int]



    # High-risk typosquatting / homoglyph handling BEFORE any SAFE override.
    # Spec: if has_typosquatting==1 OR homoglyph_like==1 => DANGEROUS + risk_score=100.
    if (
        int(merged.get("has_typosquatting", 0) or 0) == 1
        or int(merged.get("homoglyph_like", 0) or 0) == 1
    ):
        pred = "DANGEROUS"
        risk = 100.0
        probabilities = {"phishing": 1.0, "legitimate": 0.0}
        if isinstance(reasons, list):
            # Make sure we don't keep misleading override reasons.
            reasons = [
                r
                for r in reasons
                if not (
                    isinstance(r, dict)
                    and str(r.get("reason")) in {"Trusted domain whitelist", "trusted-domain safe override"}
                )
            ]

    # Trusted-domain SAFE override must ONLY run when the URL is trusted AND:
    # has_typosquatting==0 AND homoglyph_like==0 AND is_ip_address==0.
    # This runs after the high-risk gating above, so typosquatting/homoglyph always wins.
    if (
        risk is not None
        and _is_trusted_domain_safe_override(merged)
        and int(merged.get("homoglyph_like", 0) or 0) == 0
        and int(merged.get("has_typosquatting", 0) or 0) == 0
        and int(merged.get("is_ip_address", 0) or 0) == 0
    ):
        pred = "SAFE"
        risk = 0.0
        probabilities = {"phishing": 0.0, "legitimate": 1.0}

        if isinstance(reasons, list):
            # Keep reasons concise and explicitly mention the override.
            reasons = [
                r
                for r in reasons
                if not (
                    isinstance(r, dict)
                    and str(r.get("reason"))
                    in {"Trusted domain whitelist", "Trusted domain whitelist"}
                )
            ]
            reasons.insert(0, {"reason": "trusted-domain safe override", "weight": 0.0})


    # Map to exact verdict labels and enforce confidence/risk bands per spec.
    final_verdict = "SAFE"
    if isinstance(pred, str):
        pl = pred.strip().lower()
        if pl in {"phishing", "dangerous", "danger"} or pl == "phishing":
            final_verdict = "DANGEROUS"
        elif pl in {"safe"}:
            final_verdict = "SAFE"
        elif pl in {"legitimate"}:
            # Spec: never output "Legitimate".
            final_verdict = "SAFE"
        else:
            final_verdict = "SAFE"

    # Enforce confidence + risk score bands explicitly.
    if final_verdict == "SAFE":
        risk = 0.0
        # SAFE => Legitimate 95–100%, Phishing 0–5%.
        # Keep exactly 100/0 for trusted-domain; otherwise clamp into the required bands.
        phishing_p = float(probabilities.get("phishing", 0.0) or 0.0) if isinstance(probabilities, dict) else 0.0
        # Clamp phishing into [0, 0.05]
        phishing_p = max(0.0, min(0.05, phishing_p))
        legitimate_p = max(0.95, min(1.0, 1.0 - phishing_p))
        # Renormalize to sum to 1
        total = phishing_p + legitimate_p
        if total <= 0:
            phishing_p, legitimate_p = 0.0, 1.0
        else:
            phishing_p /= total
            legitimate_p /= total
        probabilities = {"phishing": phishing_p, "legitimate": legitimate_p}
    else:
        # DANGEROUS => Phishing 95–100%, Legitimate 0–5%
        risk = 100.0
        phishing_p = 1.0
        legitimate_p = 0.0

        # If model gave us a stronger phishing probability, keep it but clamp.
        if isinstance(probabilities, dict) and "phishing" in probabilities:
            try:
                phishing_p = float(probabilities.get("phishing") or 0.0)
            except Exception:
                pass
            phishing_p = max(0.95, min(1.0, phishing_p))
            legitimate_p = 1.0 - phishing_p
            legitimate_p = max(0.0, min(0.05, legitimate_p))
            # Renormalize
            total = phishing_p + legitimate_p
            if total > 0:
                phishing_p /= total
                legitimate_p /= total
        probabilities = {"phishing": phishing_p, "legitimate": legitimate_p}

    return {
        "prediction": final_verdict,
        "risk_score": float(risk) if risk is not None else None,
        "probabilities": probabilities,
        "reasons": reasons,
        "features": merged,
    }




