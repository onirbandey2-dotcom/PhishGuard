# PhishGuard: Explainable Real-Time Phishing URL Detection System

## Overview

PhishGuard is an explainable machine learning-based phishing URL detection system designed for real-time analysis of suspicious URLs. The project combines URL heuristic analysis, domain intelligence, brand similarity detection, and interpretable ML predictions to classify phishing attempts with high accuracy.

The system includes:

* Real-time Streamlit web interface
* Feature engineering pipeline
* Brand similarity and typosquatting detection
* Multi-model ML training framework
* Prediction and explanation module
* Smoke testing utilities

The repository is scaffolded with a working baseline implementation and can run even before training custom models.

---

# Project Structure

```bash
app/
 └── streamlit_app.py          # Streamlit frontend demo

ml/
 ├── features.py               # URL and domain heuristic features
 ├── brand_features.py         # Brand similarity & typo detection
 ├── train.py                  # Model training pipeline
 └── predict.py                # Prediction + explanation pipeline

scripts/
 └── smoke_test.py             # Quick functionality test

models/
 └── phishguard_model.joblib   # Trained model artifact (generated after training)
```

---

# Features

## URL Heuristic Features

The system extracts multiple handcrafted phishing indicators such as:

* URL length
* Number of subdomains
* Presence of IP addresses
* Suspicious keywords
* Entropy analysis
* Special character frequency
* HTTPS usage

## Brand Similarity Detection

Detects potential impersonation attempts using:

* Edit-distance similarity
* Typosquatting patterns
* Homoglyph-based analysis
* Brand keyword matching

## Explainable Predictions

Provides interpretable outputs for predictions through:

* Feature importance fallback explanations
* Probability scoring
* Transparent phishing indicators

## Multi-Model Training Support

Supports training and evaluation of:

* Random Forest
* XGBoost
* LightGBM
* Logistic Regression
* Other scikit-learn compatible models

---

# Installation

## 1. Clone the Repository

```bash
git clone <repository-url>
cd PhishGuard
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Demo

Launch the Streamlit application:

```bash
streamlit run app/streamlit_app.py
```

After execution, open the local URL displayed in the terminal.

---

# Training a Model

## Dataset Format

Prepare a CSV file containing:

| Column | Description                  |
| ------ | ---------------------------- |
| url    | Website URL                  |
| label  | 1 = phishing, 0 = legitimate |

Example:

```csv
url,label
http://secure-paypal-login.com,1
https://github.com,0
```

## Training Command

```bash
python ml/train.py --input_csv data.csv --out_dir models --best_model lgbm
```

## Generated Artifacts

```bash
models/phishguard_model.joblib
models/metrics.joblib
```

---

# Prediction Pipeline

The prediction system:

1. Extracts heuristic and brand-based features
2. Loads trained model artifacts
3. Generates phishing probability scores
4. Produces interpretable explanations

If no trained model exists, the system falls back to baseline heuristic detection.

---

# Smoke Testing

Run a quick sanity test:

```bash
python scripts/smoke_test.py
```

This validates:

* Feature extraction
* Model loading
* Prediction workflow
* End-to-end execution

---

# Research Extension Opportunities

To elevate the project toward an IIT-level research scope, future improvements may include:

## Advanced Brand Intelligence

* Large-scale brand database integration
* Improved homoglyph attack detection
* Unicode spoofing analysis

## Explainable AI Enhancements

* SHAP explainability integration
* LIME-based local explanations
* Feature attribution visualizations

## Dataset Engineering

* Automated phishing feed ingestion
* Dataset deduplication
* Class imbalance handling
* Adversarial sample generation

## Webpage-Level Analysis

* HTML and JavaScript feature extraction
* DOM structure analysis
* Certificate and WHOIS intelligence
* Safe web scraping pipeline

## Deep Learning Extensions

* Transformer-based URL embeddings
* Character-level CNN/LSTM models
* Hybrid ensemble systems

---

# Tech Stack

* Python
* Streamlit
* scikit-learn
* LightGBM
* XGBoost
* pandas
* numpy

---

# Disclaimer

This project is intended strictly for cybersecurity research, educational purposes, and phishing awareness. It should not be used for unauthorized monitoring or malicious activities.

---

# License

Add your preferred open-source license here (MIT, Apache 2.0, etc.).


## Future Enhancements
1. CNN (Convolutional Neural Network) based phishing detection to automatically learn hidden URL patterns and improve detection accuracy.
2. Random Walk / Graph-based analysis for identifying phishing networks and suspicious domain relationships.
3. Transformer-based models for advanced phishing email and SMS detection.
4. Cloud/Web deployment so the application can be accessed online instead of localhost.
5. Integration with real-time threat intelligence feeds and domain reputation services.

*Note:* The current system uses feature-based real-time phishing detection. CNN and Random Walk are proposed future enhancements and are not implemented in the current version.


