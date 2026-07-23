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

SUSPICIOUS_KEYWORDS_URL = [
    "login",
    "verify",
    "secure",
    "update",
    "account",
    "password",
    "bank",
    "free",
    "prize",
]



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


TRUSTED_DOMAIN_WHITELIST = {
    # Required trusted domains (root/registered domain)
    "amazon.com",
    "amazon.in",
    "flipkart.com",
    "paypal.com",
    "google.com",
    "github.com",
    "microsoft.com",
    "linkedin.com",
    "srmist.edu.in",
}

# Conservative list of tracking/marketing query parameter names.
# Used to ignore URL/marketing length signals for trusted domains.
TRACKING_PARAM_PREFIXES = (
    "utm_",
    "ref=",
    "tag=",
    "adgrpid",
    "hv",
    "gad_source",
)

TRACKING_PARAM_EXACT = {
    "ref",
    "tag",
    "gclid",
    "fbclid",
    "adgroupid",
    "adid",
    "campaign",
    "source",
    "medium",
    "content",
    "term",
    "affiliate",
}



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

    # Whitelist check (based on registered domain + suffix)
    registered_domain = domain.lower().strip()
    suffix_norm = (suffix or "").lower().strip()
    registered_domain_full = ".".join([p for p in [registered_domain, suffix_norm] if p])

    is_trusted_domain = 1 if registered_domain_full in TRUSTED_DOMAIN_WHITELIST else 0

    # For trusted domains, ignore marketing/tracking query params and length/https signals.
    # We do this at feature-extraction time so any ML/model/scoring won't over-penalize.
    if is_trusted_domain == 1:
        # keep only non-tracking parameters (very conservative)
        # NOTE: if query is empty, this is a no-op.
        query_to_process = query or ""
        

        kept_parts = []
        for part in query.split("&"):
            if not part:
                continue
            key = part.split("=", 1)[0].strip().lower()

            # Ignore common tracking parameter patterns
            if any(key.startswith(p) for p in TRACKING_PARAM_PREFIXES):
                continue
            if key in TRACKING_PARAM_EXACT:
                continue
            # Ignore common affiliate / campaign style keys
            if key.startswith("utm") or key.startswith("aff") or key.startswith("campaign"):
                continue

            kept_parts.append(part)

        query_effective = "&".join(kept_parts)
        # Reduce length influence from the query
        path_for_features = path
        url_length = len("http://" + registered_domain_full + (path_for_features or ""))
        # Do not let tracking params affect structural features
        special_chars = _count_special_chars(registered_domain_full + (path_for_features or ""))

        path_length = len(path_for_features or "")
        https_usage = 0
        full_domain_for_len = registered_domain_full
        domain_length = len(full_domain_for_len)
    else:
        # Derived lengths
        url_length = len(url_norm)
        domain_length = len(full_domain)
        path_length = len(path or "")
        query_effective = query


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


    # URL shortening detection (match only known shortener hosts to avoid false positives).
    url_shortening_hosts = [
        "bit.ly",
        "tinyurl.com",
        "t.co",
        "goo.gl",
        "is.gd",
        "ow.ly",
        "buff.ly",
        "cutt.ly",
    ]
    url_shortening = 0
    for host in url_shortening_hosts:
        # Accept with or without scheme and with optional www.
        host_l = host.lower()
        if re.search(rf"(^|://)(www\\.)?{re.escape(host_l)}(/|\\?|#|$)", url_norm.lower()):
            url_shortening = 1
            break

    # Suspicious keyword flags
    lower = url_norm.lower()
    keyword_hits = [kw for kw in SUSPICIOUS_KEYWORDS_URL if kw in lower]
    has_suspicious_keyword = 1 if keyword_hits else 0
    suspicious_keyword_count = len(keyword_hits)

    # Root-domain trusted flag (explicit feature)
    is_trusted_domain = int(is_trusted_domain)

    # Backward-compatible alias used by existing model logic
    is_whitelisted = is_trusted_domain


    features: Dict[str, Any] = {
        "url_length": url_length,
        "is_trusted_domain": is_trusted_domain,

        "num_dots": num_dots,
        "num_subdomains": num_subdomains,
        "has_https": https_usage,
        "has_at_symbol": has_at,
        "special_char_count": special_chars,
        "has_suspicious_keyword": has_suspicious_keyword,
        "is_whitelisted": is_whitelisted,

        # Keep existing richer features for the ML model.
        "domain_length": domain_length,
        "path_length": path_length,
        "num_hyphens": num_hyphens,
        "num_digits": digits_in_url,
        "num_special_chars": special_chars,
        "is_ip_address": is_ip_address,
        "has_double_slash_redirection": has_double_slash,
        "https_usage": https_usage,
        "url_shortening": url_shortening,
        "suspicious_keyword_count": suspicious_keyword_count,
    }


    # Light structural signals
    features["has_login_in_path"] = 1 if "/login" in lower or "login" in path.lower() else 0
    features["has_verify_in_path"] = 1 if "verify" in lower or "verification" in lower else 0

    return features

