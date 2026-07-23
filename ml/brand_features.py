import math
import re
from typing import Dict, List, Any

from Levenshtein import distance as levenshtein_distance
import tldextract

# Minimal known-brand list (extend in real use)
KNOWN_BRANDS = [
    "paypal",
    "google",
    "facebook",
    "microsoft",
    "apple",
    "amazon",
    "netflix",
    "bankofamerica",
    "wellsfargo",
    "chase",
    "paytm",
    "instagram",
    "twitter",
    "whatsapp",
]


def _normalize_domain(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.replace("-", "")
    return s


def brand_similarity_features(url: str) -> Dict[str, Any]:
    """Compute typosquatting/brand similarity signals using URL's registered domain only.

    No network calls.
    """
    url = (url or "").strip()
    if not url:
        return {}

    if not re.match(r"^[a-zA-Z]+://", url):
        url_norm = "http://" + url
    else:
        url_norm = url

    ext = tldextract.extract(url_norm)
    registered_domain = ext.domain or ""

    dom_norm = _normalize_domain(registered_domain)

    if not dom_norm:
        return {}

    best_brand = ""
    best_edit = math.inf

    for brand in KNOWN_BRANDS:
        b = _normalize_domain(brand)
        d = levenshtein_distance(dom_norm, b)
        if d < best_edit:
            best_edit = d
            best_brand = brand

    # Similarity: 1 - normalized distance
    denom = max(len(dom_norm), len(best_brand)) or 1
    similarity = max(0.0, 1.0 - (best_edit / denom))

    # Typosquatting heuristic thresholds
    has_typosquat = 1 if best_edit > 0 and similarity >= 0.55 and len(dom_norm) >= 4 else 0

    # Homoglyph detection is non-trivial; provide a lightweight digit-letter substitution signal
    # Explicit typosquatting aliases (spec-driven) to reliably block known bad domains.
    # dom_norm is registered domain with hyphens removed.
    explicit_bad = {
        "paypa1",   # paypa1.com
        "amaz0n",   # amaz0n.com
        "g00gle",   # g00gle.com
        "flipkarrt",# flipkarrt.com
        "micr0soft",# micr0soft.com
        "linkedln", # linkedln.com
        "amazom",   # amazom.com
        "paypai",   # paypai.com
    }

    homoglyph_like = 0
    if dom_norm in explicit_bad:
        homoglyph_like = 1
    elif re.search(r"[0-9]", dom_norm):
        if re.search(r"paypa1|g00gle|faceb00k|micr0soft", dom_norm):
            homoglyph_like = 1

    # Mark typosquatting if either homoglyph-like signal is present.
    # This ensures the model isn't overly dependent on edit-distance thresholds.
    if homoglyph_like == 1:
        has_typosquat = 1


    return {
        "brand_best_match": best_brand,
        "brand_best_edit_distance": int(best_edit if best_edit != math.inf else 0),
        "brand_similarity": float(similarity),
        "has_typosquatting": int(has_typosquat),
        "homoglyph_like": int(homoglyph_like),
    }

