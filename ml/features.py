import re
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import tldextract

# Suspicious keywords commonly seen in phishing URLs
SUSPICIOUS_KEYWORDS = [
    "login",
    "verify",
    "secure",
    "update",
    "account",
    "bank",
    "free",
    "wallet",
    "payment",
    "signin",
    "confirm",
]

SPECIAL_CHARS_RE = re.compile(r"[^a-zA-Z0-9\-_.:/?&=%+#@\[\]]")


def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def _count_digits(s: str) -> int:
    return sum(ch.isdigit() for ch in s)


def _count_special_chars(s: str) -> int:
    # Count characters not in a conservative URL-safe set
    return len(SPECIAL_CHARS_RE.findall(s or ""))


def extract_url_features(url: str) -> Dict[str, Any]:
    """Extract URL-based features (no network calls)."""
    url = (url or "").strip()
    if not url:
        return {}

    # Normalize missing scheme
    if not re.match(r"^[a-zA-Z]+://", url):
        url_norm = "http://" + url
    else:
        url_norm = url

    # Use tldextract for domain parsing
    ext = tldextract.extract(url_norm)
    subdomain = ext.subdomain or ""
    domain = ext.domain or ""
    suffix = ext.suffix or ""
    full_domain = ".".join([p for p in [subdomain, domain, suffix] if p])

    path = ""
    query = ""
    m = re.match(r"^[a-zA-Z]+://[^/]+(.*)$", url_norm)
    rest = m.group(1) if m else ""
    if "?" in rest:
        path, query = rest.split("?", 1)
    else:
        path = rest

    # Derived lengths
    url_length = len(url_norm)
    domain_length = len(full_domain)
    path_length = len(path or "")

    # Character counts
    num_dots = full_domain.count(".") if full_domain else 0
    num_hyphens = (full_domain + (path or "")).count("-")
    digits_in_url = _count_digits(url_norm)
    special_chars = _count_special_chars(url_norm)

    # Subdomains
    num_subdomains = len([p for p in subdomain.split(".") if p]) if subdomain else 0

    # IP address presence (simple heuristic)
    is_ip_address = 1 if re.search(r"\b(\d{1,3}\.){3}\d{1,3}\b", url_norm) else 0

    has_at = 1 if "@" in url_norm else 0
    has_double_slash = 1 if "//" in url_norm[url_norm.find("//") + 2 :] else 0

    https_usage = 1 if url_norm.lower().startswith("https://") else 0

    url_shortening = 1 if re.search(
        r"bit\.ly|tinyurl\.com|t\.co|goo\.gl|ow\.ly|buff\.ly|is\.gd|shorte\.st|rebrand\.ly",
        url_norm.lower(),
    ) else 0

    # Suspicious keyword flags
    lower = url_norm.lower()
    keyword_hits = [kw for kw in SUSPICIOUS_KEYWORDS if kw in lower]
    has_suspicious_keyword = 1 if keyword_hits else 0
    suspicious_keyword_count = len(keyword_hits)

    features: Dict[str, Any] = {
        "url_length": url_length,
        "domain_length": domain_length,
        "path_length": path_length,
        "num_dots": num_dots,
        "num_hyphens": num_hyphens,
        "num_digits": digits_in_url,
        "num_special_chars": special_chars,
        "num_subdomains": num_subdomains,
        "is_ip_address": is_ip_address,
        "has_at": has_at,
        "has_double_slash_redirection": has_double_slash,
        "https_usage": https_usage,
        "url_shortening": url_shortening,
        "has_suspicious_keyword": has_suspicious_keyword,
        "suspicious_keyword_count": suspicious_keyword_count,
    }

    # Light structural signals
    features["has_login_in_path"] = 1 if "/login" in lower or "login" in path.lower() else 0
    features["has_verify_in_path"] = 1 if "verify" in lower or "verification" in lower else 0

    return features

