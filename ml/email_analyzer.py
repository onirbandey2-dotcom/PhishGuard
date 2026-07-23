"""
ml/email_analyzer.py — Phishing email analysis for PhishGuard.

Analyzes raw email text (paste or .eml) for phishing indicators:
  - Suspicious keywords in subject / body
  - Urgency language patterns
  - Spoofed sender heuristics
  - Embedded URL extraction + per-URL risk scoring
  - Suspicious attachment extensions
  - Overall risk score and structured report

No network calls; all analysis is local.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Keyword / pattern lists
# ---------------------------------------------------------------------------

URGENCY_PATTERNS: List[str] = [
    # Tuned to strong phishing phrases only (reduce false positives)
    r"\burgent\b",
    r"\bimmediately\b",
    r"\bverify\s+now\b",
    r"\bact\s+now\b",
    r"\baccount\s+suspended\b",
    r"\bsecurity\s+alert\b",
    r"\bwithin\s+24\s+hours\b",
]


FINANCIAL_PATTERNS: List[str] = [
    r"\bwire\s+transfer\b",
    r"\bgift\s+card\b",
    r"\bbitcoin\b",
    r"\bcrypto(currency)?\b",
    r"\bpaypal\s+account\b",
    r"\bbank\s+transfer\b",
    r"\byou\s+(have\s+)?(won|are\s+selected)\b",
    r"\b(lottery|prize|winner|reward)\b",
    r"\binheritance\b",
    r"\brefund\b",
    r"\binvoice\s+attached\b",
    r"\bpayment\s+(required|overdue|failed)\b",
]

CREDENTIAL_PATTERNS: List[str] = [
    r"\benter\s+your\s+password\b",
    r"\bupdate\s+your\s+(login|credentials|password)\b",
    r"\bsign\s+in\s+to\s+continue\b",
    r"\bverif(y|ication)\s+(code|otp)\b",
    r"\bsocial\s+security\b",
    r"\bcredit\s+card\s+(number|details)\b",
]

# Only flag *actual dangerous* attachment file extensions.
# IMPORTANT: Do NOT treat domain extensions like ".com" as attachments.
SUSPICIOUS_ATTACHMENT_EXTENSIONS = {
    ".exe", ".scr", ".bat", ".cmd", ".zip", ".rar", ".js", ".vbs",
}


# Simple URL regex (covers http/https and www. URLs)
URL_REGEX = re.compile(
    r"(?:https?://|www\.)[^\s\"'<>\)\]\}]+",
    re.IGNORECASE,
)

# Basic email address regex for sender extraction
EMAIL_REGEX = re.compile(r"[\w.\-+]+@[\w.\-]+\.[a-zA-Z]{2,}")

# Display-name spoofing: "PayPal Support <evil@other-domain.com>"
DISPLAY_NAME_SPOOF_RE = re.compile(
    r"([\w\s]+)\s+<([\w.\-+]+@[\w.\-]+\.[a-zA-Z]{2,})>", re.IGNORECASE
)

KNOWN_BRANDS_LOWER = {
    "paypal", "google", "facebook", "microsoft", "apple", "amazon",
    "netflix", "bankofamerica", "wellsfargo", "chase", "paytm",
    "instagram", "twitter", "whatsapp", "linkedin", "dropbox",
    "docusign", "fedex", "ups", "dhl", "irs", "amazon",
}






# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EmailRiskReport:
    risk_score: float                          # 0–100
    verdict: str                               # Safe / Suspicious / Dangerous
    urgency_hits: List[str] = field(default_factory=list)
    financial_hits: List[str] = field(default_factory=list)
    credential_hits: List[str] = field(default_factory=list)
    sender_spoofing: bool = False
    sender_details: Dict[str, str] = field(default_factory=dict)
    urls_found: List[str] = field(default_factory=list)
    url_risk_scores: Dict[str, float] = field(default_factory=dict)
    suspicious_attachments: List[str] = field(default_factory=list)
    top_reasons: List[str] = field(default_factory=list)
    raw_indicators: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _scan_patterns(text: str, patterns: List[str]) -> List[str]:
    """Return all pattern strings that match anywhere in *text* (case-insensitive)."""
    hits: List[str] = []
    lower = text.lower()
    for pat in patterns:
        if re.search(pat, lower):
            hits.append(pat)
    return hits


def _normalize_url_for_domain(url: str) -> str:
    """Return only the scheme/host part (drop path/query/fragment).

    Used to stabilize trusted-domain matching (ignore tracking/query/length penalties).
    """
    u = url.strip()
    # Strip scheme
    u = re.sub(r"^https?://", "", u, flags=re.IGNORECASE)
    # Handle URLs extracted from text without scheme (e.g. www.example.com)
    u = u.split("/", 1)[0].lower()
    if u.startswith("www."):
        u = u[4:]
    return u



def _extract_urls(text: str) -> List[str]:
    return list(dict.fromkeys(URL_REGEX.findall(text)))  # deduplicated, order preserved



def _extract_sender(raw_header: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (display_name, email_address) from a From: header value or None."""
    raw_header = (raw_header or "").strip()
    m = DISPLAY_NAME_SPOOF_RE.search(raw_header)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m2 = EMAIL_REGEX.search(raw_header)
    if m2:
        return None, m2.group(0)
    return None, None


def _check_sender_spoofing(display_name: Optional[str], email: Optional[str]) -> bool:
    """
    Heuristic: display name mentions a known brand but the email domain doesn't match.
    """
    if not display_name or not email:
        return False
    dn_lower = display_name.lower().replace(" ", "")
    for brand in KNOWN_BRANDS_LOWER:
        if brand in dn_lower:
            # Email domain should contain brand name
            domain_part = email.split("@")[-1].lower().replace("-", "").replace(".", "")
            if brand not in domain_part:
                return True
    return False





def _attachment_risks(text: str) -> List[str]:
    """Detect *dangerous* attachment file extensions mentioned in the email.

    Only counts actual filenames/extensions like `invoice.zip`.
    Domain extensions like `example.com` are intentionally ignored.
    """
    found: List[str] = []
    lower = text.lower()

    # Require a filename-like token ending with the dangerous extension.
    # Avoid matching plain domain endings by requiring a '/' or whitespace boundary
    # *before* the token, and an actual extension suffix afterwards.
    for ext in SUSPICIOUS_ATTACHMENT_EXTENSIONS:
        # e.g. " report.exe" or "report.exe" or "report.exe\""
        pattern = r"(?:^|[\s\(\[\{\"'<>/\\])[^\s\(\[\{\"'<>/\\]+" + re.escape(ext) + r"(?:$|[\s\)\]\}\"'<>/\\])"
        if re.search(pattern, lower):
            found.append(ext)

    return list(set(found))



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_email(
    body: str,
    subject: str = "",
    sender: str = "",
) -> EmailRiskReport:

    """
    Analyze an email for phishing indicators.

    Parameters
    ----------
    body : str
        The full email body (plain text or raw .eml content).
    subject : str
        Email subject line.
    sender : str
        Raw From: header value, e.g. "PayPal Support <no-reply@evil.com>".

    Returns
    -------
    EmailRiskReport
        Structured report with risk score, verdict, and all indicators.
    """
    full_text = f"{subject}\n{sender}\n{body}"

    # --- Pattern scanning ---
    urgency_hits = _scan_patterns(full_text, URGENCY_PATTERNS)
    financial_hits = _scan_patterns(full_text, FINANCIAL_PATTERNS)
    credential_hits = _scan_patterns(full_text, CREDENTIAL_PATTERNS)

    # --- Sender analysis ---
    display_name, email_addr = _extract_sender(sender)
    spoofed = _check_sender_spoofing(display_name, email_addr)
    sender_details: Dict[str, str] = {}
    if display_name:
        sender_details["display_name"] = display_name
    if email_addr:
        sender_details["email"] = email_addr

    # --- URL analysis ---
    from ml.predict import predict_with_explanations

    urls = _extract_urls(full_text)

    # Single source of truth: URL risk comes ONLY from predict_with_explanations.
    url_risk_scores: Dict[str, float] = {}
    for url in urls[:20]:  # cap at 20 URLs for performance
        result = predict_with_explanations(url)
        url_risk_scores[url] = float(result["risk_score"])

    max_url_risk = max(url_risk_scores.values(), default=0.0)





    # --- Attachment detection ---
    suspicious_attachments = _attachment_risks(full_text)

    # --- Composite risk score ---
    score = 0.0
    reasons: List[str] = []

    if urgency_hits:
        delta = min(len(urgency_hits) * 10, 30)
        score += delta
        reasons.append(f"Urgency language detected ({len(urgency_hits)} pattern(s))")

    if financial_hits:
        delta = min(len(financial_hits) * 10, 25)
        score += delta
        reasons.append(f"Financial / reward language detected ({len(financial_hits)} pattern(s))")

    if credential_hits:
        delta = min(len(credential_hits) * 12, 30)
        score += delta
        reasons.append(f"Credential harvesting language ({len(credential_hits)} pattern(s))")

    if spoofed:
        score += 35
        reasons.append("Sender display name spoofs a known brand")

    if max_url_risk > 60:
        score += 20
        reasons.append(f"High-risk URL found (score {max_url_risk:.0f}/100)")
    elif max_url_risk > 40:
        score += 10
        reasons.append(f"Suspicious URL found (score {max_url_risk:.0f}/100)")

    if suspicious_attachments:
        score += len(suspicious_attachments) * 15
        reasons.append(f"Suspicious attachment extension(s): {', '.join(suspicious_attachments)}")

    # Clamp
    score = min(score, 100.0)

    if score >= 65:
        verdict = "Dangerous"
    elif score >= 35:
        verdict = "Suspicious"
    else:
        verdict = "Safe"

    if max_url_risk >= 60 and len(urgency_hits) >= 2:
        score = max(score, 75.0)
        verdict = "Dangerous"

    return EmailRiskReport(
        risk_score=float(score),
        verdict=verdict,
        urgency_hits=urgency_hits,
        financial_hits=financial_hits,
        credential_hits=credential_hits,
        sender_spoofing=spoofed,
        sender_details=sender_details,
        urls_found=urls,
        url_risk_scores=url_risk_scores,

        suspicious_attachments=suspicious_attachments,
        top_reasons=reasons,
        raw_indicators={
            "urgency_count": len(urgency_hits),
            "financial_count": len(financial_hits),
            "credential_count": len(credential_hits),
            "url_count": len(urls),
            "max_url_risk": max_url_risk,
            "attachment_count": len(suspicious_attachments),
        },
    )
