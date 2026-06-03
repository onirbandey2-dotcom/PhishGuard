from pathlib import Path
import sys

# Ensure repository root is on PYTHONPATH so `import ml` works when running:
#   python scripts/smoke_test.py
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.predict import predict_with_explanations


if __name__ == "__main__":
    url = "https://secure-paytm-login-verification.com"
    res = predict_with_explanations(url, top_k_reasons=5)
    print(res)


