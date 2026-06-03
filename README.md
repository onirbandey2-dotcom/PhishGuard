# PhishGuard: Explainable Real-Time Phishing URL Detection System

## What’s included
- **Streamlit demo**: `app/streamlit_app.py`
- **Feature extraction** (URL/domain heuristics + brand similarity):
  - `ml/features.py`
  - `ml/brand_features.py`
- **Training pipeline** (multiple ML models): `ml/train.py`
- **Prediction pipeline + explanation fallback**: `ml/predict.py`
- **Smoke test**: `scripts/smoke_test.py`

> This repo is scaffolded with a working baseline that runs even before you have trained model artifacts. If `models/phishguard_model.joblib` exists, the app will use it.

## 1) Install dependencies
From this folder, run:

```bash
pip install -r requirements.txt
```

## 2) Run the demo
```bash
streamlit run app/streamlit_app.py
```
Then open the URL shown in the terminal.

## 3) Train a model (optional but recommended)
Prepare a CSV dataset with columns:
- `url`
- `label` (1 = phishing, 0 = legitimate)

Train:
```bash
python ml/train.py --input_csv data.csv --out_dir models --best_model lgbm
```

Artifacts:
- `models/phishguard_model.joblib`
- `models/metrics.joblib`

## 4) Quick sanity check
```bash
python scripts/smoke_test.py
```

## Expected project extension (IIT-level)
To reach a more research-complete scope, extend:
- real brand list + better typosquatting/homoglyph detection
- optional SHAP/LIME explainer module
- dataset ingestion + deduplication + imbalance handling
- richer webpage features (optional; may require scraping policy)

