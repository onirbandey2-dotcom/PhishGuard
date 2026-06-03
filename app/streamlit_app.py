import json
import os
from typing import Dict, Any

import streamlit as st

# Local imports
from ml.predict import predict_with_explanations

st.set_page_config(page_title="PhishGuard", page_icon="🛡️", layout="wide")

st.title("PhishGuard 🛡️")
st.caption("Explainable Real-Time Phishing URL Detection")

with st.sidebar:
    st.header("Input")
    url = st.text_input("Enter a URL", value="https://secure-paytm-login-verification.com")
    top_k = st.slider("Top reasons", min_value=1, max_value=10, value=5)

    st.divider()
    st.subheader("Notes")
    st.markdown(
        "- This demo uses a pre-trained pipeline if present.\n"
        "- If no model artifacts are found, it will run with the fallback heuristic model."
    )

run_btn = st.button("Check URL", type="primary")

if run_btn:
    if not url.strip():
        st.error("Please enter a URL")
        st.stop()

    with st.spinner("Analyzing URL..."):
        result: Dict[str, Any] = predict_with_explanations(url, top_k_reasons=top_k)

    pred = result.get("prediction", "unknown")
    risk = result.get("risk_score", None)
    proba = result.get("probabilities", {})
    reasons = result.get("reasons", [])

    c1, c2, c3 = st.columns([1.2, 1, 1])

    with c1:
        st.subheader("Prediction")
        st.metric(label="Class", value=str(pred).upper())

    with c2:
        st.subheader("Risk score")
        if risk is None:
            st.write("N/A")
        else:
            st.metric(label="Risk", value=f"{risk:.1f}%")

    with c3:
        st.subheader("Confidence")
        phishing_p = float(proba.get("phishing", 0.0))
        legit_p = float(proba.get("legitimate", 0.0))
        st.write(f"Phishing: {phishing_p:.3f}")
        st.write(f"Legitimate: {legit_p:.3f}")

    st.divider()
    st.subheader("Main reasons")
    if reasons:
        for r in reasons:
            st.markdown(f"- **{r.get('reason', '')}** ({r.get('weight', 0.0):.3f})")
    else:
        st.write("No reasons available.")

    st.divider()
    st.subheader("Feature snapshot")
    feats = result.get("features", {})
    # Keep UI readable: show first ~25 features
    items = list(feats.items())[:25]
    if items:
        st.json({k: v for k, v in items})
    else:
        st.write("No features returned.")

