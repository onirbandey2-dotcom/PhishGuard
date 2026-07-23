# ml package — PhishGuard AI
# Public API surface (import lazily to avoid heavy deps at import time)

__all__ = [
    "predict_with_explanations",  # ml.predict
    "analyze_email",              # ml.email_analyzer
    "analyze_sms",                # ml.sms_analyzer
    "explain_prediction",         # ml.explain
    "extract_url_features",       # ml.features
    "brand_similarity_features",  # ml.brand_features
]
