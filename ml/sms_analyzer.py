"""
ml/sms_analyzer.py — SMS / text message scam detection for PhishGuard.

Analyzes a short text message for smishing (SMS phishing) indicators:
  - OTP / verification code request patterns
  - Prize / lottery / reward language
  - Urgency + delivery scam patterns
  - Embedded URL extraction + risk scoring
  - Overall risk score and verdict

No network calls; all analysis is local.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


# ---------------------------------------------------------------------------
# Pattern lists (regex, case-insensitive)
# ---------------------------------------------------------------------------

OTP_PATTERNS: List[str] = [
    r"\botp\b",
    r"\bone.?time.?(password|code|pin)\b",
    r"\bverification\s+code\b",
    r"\bauth(entication)?\s+code\b",
    r"\bdo\s+not\s+share\s+(this|your)\s+(otp|code|pin)\b",
    r"\byour\s+(code|pin|otp)\s+is\b",
]

PRIZE_PATTERNS: List[str] = [
    r"\b(congratulations|congrats)\b",
    r"\byou\s+(have\s+)?(won|are\s+selected|are\s+a\s+winner)\b",
    r"\b(lottery|prize|reward|lucky\s+draw|lucky\s+winner)\b",
    r"\bclaim\s+(your|the)\s+(prize|reward|cash)\b",
    r"\b(cash\s+prize|gift\s+card)\b",
]

DELIVERY_SCAM_PATTERNS: List[str] = [
    r"\b(fedex|dhl|ups|usps|royal\s+mail)\b",
    r"\bparcel\s+(awaiting|on\s+hold|failed)\b",
    r"\bdelivery\s+(attempt|failed|missed)\b",
    r"\breschedule\s+delivery\b",
    r"\bundelivered\s+package\b",
]

URGENCY_PATTERNS: List[str] = [
    r"\burgent\b",
    r"\bact\s+now\b",
    r"\bexpires?\s+(today|soon|in\s+\d+)",
    r"\bwithin\s+\d+\s+(hour|minute|day)",
    r"\byour\s+account\s+(will\s+be\s+)?(suspended|blocked|locked)\b",
    r"\bimmediately\b",
    r"\blast\s+(chance|warning|notice)\b",
]

FINANCIAL_PATTERNS: List[str] = [
    r"\bsend\s+(money|funds|cash)\b",
    r"\bwire\s+transfer\b",
    r"\bgift\s+card\b",
    r"\bbitcoin\b",
    r"\bcrypto\b",
    r"\bpay\s+(now|immediately|today)\b",
    r"\boverdue\s+(balance|payment|invoice)\b",
]

IMPERSONATION_PATTERNS: List[str] = [
    r"\bthis\s+is\s+(your\s+bank|amazon|paypal|google|apple|microsoft|irs|hmrc|income\s+tax)\b",
    r"\b(amazon|paypal|google|apple|microsoft)\s+(support|team|security)\b",
    r"\bgovernment\s+(grant|benefit|refund)\b",
    r"\btax\s+refund\b",
    r"\birs\s+(notice|alert|refund)\b",
]

# URL pattern shared with email analyzer
URL_REGEX = re.compile(
    r"(?:https?://|www\.)[^\s\"'<>\)\]\}]+",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class SMSRiskReport:
    risk_score: float          # 0–100
    verdict: str               # Safe / Suspicious / Dangerous
    otp_hits: List[str] = field(default_factory=list)
    prize_hits: List[str] = field(default_factory=list)
    delivery_hits: List[str] = field(default_factory=list)
    urgency_hits: List[str] = field(default_factory=list)
    financial_hits: List[str] = field(default_factory=list)
    impersonation_hits: List[str] = field(default_factory=list)
    urls_found: List[str] = field(default_factory=list)
    url_risk_scores: Dict[str, float] = field(default_factory=dict)
    top_reasons: List[str] = field(default_factory=list)
    raw_indicators: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scan_patterns(text: str, patterns: List[str]) -> List[str]:
    hits: List[str] = []
    lower = text.lower()
    for pat in patterns:
        if re.search(pat, lower):
            hits.append(pat)
    return hits


def _extract_urls(text: str) -> List[str]:
    return list(dict.fromkeys(URL_REGEX.findall(text)))


def _url_risk_score(url: str) -> float:
    """Reuse URL feature heuristic from features + brand modules."""
    try:
        from ml.features import extract_url_features
        from ml.brand_features import brand_similarity_features
        f = {**extract_url_features(url), **brand_similarity_features(url)}
    except Exception:
        return 50.0

    score = 0.0
    score += f.get("has_suspicious_keyword", 0) * 18
    score += f.get("suspicious_keyword_count", 0) * 3
    score += f.get("url_shortening", 0) * 20
    score += f.get("is_ip_address", 0) * 30
    score += f.get("has_at", 0) * 18
    score += f.get("has_double_slash_redirection", 0) * 12
    score += f.get("has_typosquatting", 0) * 22
    score += float(f.get("brand_similarity", 0)) * 45
    score += f.get("homoglyph_like", 0) * 15
    score += f.get("num_hyphens", 0) * 1.8
    score -= f.get("https_usage", 0) * 5

    return float(100.0 / (1.0 + np.exp(-score / 50.0)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_sms(text: str) -> SMSRiskReport:
    """
    Analyze an SMS message for smishing indicators.

    Parameters
    ----------
    text : str
        The full SMS / text message content.

    Returns
    -------
    SMSRiskReport
        Structured report with risk score, verdict, and all indicators.
    """
    text = (text or "").strip()
    lower = text.lower()

    otp_hits = _scan_patterns(lower, OTP_PATTERNS)
    prize_hits = _scan_patterns(lower, PRIZE_PATTERNS)
    delivery_hits = _scan_patterns(lower, DELIVERY_SCAM_PATTERNS)
    urgency_hits = _scan_patterns(lower, URGENCY_PATTERNS)
    financial_hits = _scan_patterns(lower, FINANCIAL_PATTERNS)
    impersonation_hits = _scan_patterns(lower, IMPERSONATION_PATTERNS)

    urls = _extract_urls(text)
    url_scores: Dict[str, float] = {}
    for u in urls[:10]:
        url_scores[u] = _url_risk_score(u)

    max_url_risk = max(url_scores.values(), default=0.0)

    # --- Composite score ---
    score = 0.0
    reasons: List[str] = []

    # OTP request: could be legitimate 2FA but also smishing — raise only if combined
    if otp_hits and (urgency_hits or impersonation_hits or urls):
        score += 30
        reasons.append("OTP request combined with urgency or impersonation signals")
    elif otp_hits:
        score += 5  # could be legitimate 2FA
        reasons.append("OTP / verification code request (may be legitimate)")

    if prize_hits:
        delta = min(len(prize_hits) * 15, 40)
        score += delta
        reasons.append(f"Prize / lottery language detected ({len(prize_hits)} pattern(s))")

    if delivery_hits:
        delta = min(len(delivery_hits) * 12, 30)
        score += delta
        reasons.append(f"Delivery scam pattern detected ({len(delivery_hits)} pattern(s))")

    if urgency_hits:
        delta = min(len(urgency_hits) * 8, 25)
        score += delta
        reasons.append(f"Urgency language detected ({len(urgency_hits)} pattern(s))")

    if financial_hits:
        delta = min(len(financial_hits) * 10, 25)
        score += delta
        reasons.append(f"Financial coercion language ({len(financial_hits)} pattern(s))")

    if impersonation_hits:
        score += 30
        reasons.append(f"Brand / authority impersonation detected ({len(impersonation_hits)} pattern(s))")

    if max_url_risk > 60:
        score += 20
        reasons.append(f"High-risk URL embedded in message (score {max_url_risk:.0f}/100)")
    elif max_url_risk > 35:
        score += 10
        reasons.append(f"Suspicious URL embedded in message (score {max_url_risk:.0f}/100)")

    score = min(score, 100.0)

    if score >= 65:
        verdict = "Dangerous"
    elif score >= 30:
        verdict = "Suspicious"
    else:
        verdict = "Safe"

    return SMSRiskReport(
        risk_score=float(score),
        verdict=verdict,
        otp_hits=otp_hits,
        prize_hits=prize_hits,
        delivery_hits=delivery_hits,
        urgency_hits=urgency_hits,
        financial_hits=financial_hits,
        impersonation_hits=impersonation_hits,
        urls_found=urls,
        url_risk_scores=url_scores,
        top_reasons=reasons,
        raw_indicators={
            "otp_count": len(otp_hits),
            "prize_count": len(prize_hits),
            "delivery_count": len(delivery_hits),
            "urgency_count": len(urgency_hits),
            "financial_count": len(financial_hits),
            "impersonation_count": len(impersonation_hits),
            "url_count": len(urls),
            "max_url_risk": max_url_risk,
        },
    )
