
"""PhishGuard AI — Streamlit UI

Tabs:
- URL Scanner
- Dataset Analytics
- Email Analyzer
- SMS Detector

Requirements for IIT demo:
- Keep existing URL phishing prediction feature.
- Add Dataset Analytics page with charts, search, pagination, and caching.

All analysis runs locally.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, Tuple

import streamlit as st

# Ensure repo root is importable regardless of working directory
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd

from ml.predict import predict_with_explanations
from ml.email_analyzer import analyze_email
from ml.sms_analyzer import analyze_sms


# ---------------------------------------------------------------------------
# Historical Knowledge Base (SQLite)
# ---------------------------------------------------------------------------
import sqlite3
import csv
import io
import datetime as _dt
import difflib

# Optional: RapidFuzz for similarity matching (fallback to difflib if unavailable)
try:
    from rapidfuzz import fuzz as _rf_fuzz  # type: ignore
except Exception:
    _rf_fuzz = None


def _normalize_url_for_similarity(u: str) -> str:
    """Normalization used only for similarity/exclusion.

    Keep it minimal to avoid breaking historical matching.
    """
    s = (u or "").strip().lower()
    if s.endswith("/") and len(s) > 1:
        s = s.rstrip("/")
    return s



def _history_db_path() -> str:
    """Return absolute path to history.db (stored at repo root)."""
    return os.path.join(_REPO_ROOT, "history.db")


def _history_table_name() -> str:
    return "history"


def _init_history_db() -> None:
    """Create history table if it doesn't exist."""
    conn = sqlite3.connect(_history_db_path())
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_history_table_name()} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL UNIQUE,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                risk_score REAL,
                scan_time TEXT NOT NULL,
                times_scanned INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _parse_scan_time(dt_str: str) -> _dt.datetime:
    # Store as ISO string for easy parsing/sorting.
    # If parsing fails, fall back to current time.
    try:
        return _dt.datetime.fromisoformat(dt_str)
    except Exception:
        return _dt.datetime.now()


def _format_scan_time(dt_str: Any) -> str:
    """Format ISO timestamps into a human-friendly form.

    Target format example: 18 Jul 2026, 7:28 PM
    """
    if dt_str is None:
        return ""
    try:
        dt = _parse_scan_time(str(dt_str))
        return dt.strftime("%d %b %Y, %I:%M %p").lstrip("0")
    except Exception:
        return str(dt_str)



def _get_url_history(url: str) -> Optional[Dict[str, Any]]:
    """Return history row for URL if it exists."""
    _init_history_db()
    conn = sqlite3.connect(_history_db_path())
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT url, prediction, confidence, risk_score, scan_time, times_scanned
            FROM {_history_table_name()}
            WHERE url = ?
            """,
            (url,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "url": row[0],
            "prediction": row[1],
            "confidence": float(row[2]),
            "risk_score": None if row[3] is None else float(row[3]),
            "first_scan": row[4],
            "last_scan": row[4],
            "times_scanned": int(row[5]),
        }
    finally:
        conn.close()


def _record_scan(url: str, prediction: str, confidence: float, risk_score: Optional[float], scan_time: _dt.datetime) -> None:
    """Upsert scan record: increment times_scanned and update last scan fields."""
    _init_history_db()
    conn = sqlite3.connect(_history_db_path())
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT scan_time, times_scanned
            FROM {_history_table_name()}
            WHERE url = ?
            """,
            (url,),
        )
        existing = cur.fetchone()

        if existing:
            times_scanned = int(existing[1])
            new_times = times_scanned + 1
            # Schema stores one scan_time; we keep it updated to the latest scan_time.
            cur.execute(
                f"""
                UPDATE {_history_table_name()}
                SET prediction = ?,
                    confidence = ?,
                    risk_score = ?,
                    scan_time = ?,
                    times_scanned = ?
                WHERE url = ?
                """,
                (
                    prediction,
                    float(confidence),
                    None if risk_score is None else float(risk_score),
                    scan_time.isoformat(),
                    new_times,
                    url,
                ),
            )
        else:
            cur.execute(
                f"""
                INSERT INTO {_history_table_name()} (url, prediction, confidence, risk_score, scan_time, times_scanned)
                VALUES (?, ?, ?, ?, ?, 1)
                """,
                (
                    url,
                    prediction,
                    float(confidence),
                    None if risk_score is None else float(risk_score),
                    scan_time.isoformat(),
                ),
            )

        conn.commit()
    finally:
        conn.close()



def _export_history_csv() -> bytes:
    _init_history_db()
    conn = sqlite3.connect(_history_db_path())
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT url,prediction,confidence,risk_score,scan_time,times_scanned FROM {_history_table_name()} ORDER BY id ASC")
        rows = cur.fetchall()

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["url", "prediction", "confidence", "risk_score", "scan_time", "times_scanned"])
        writer.writerows(rows)
        return buf.getvalue().encode("utf-8")
    finally:
        conn.close()


def _knowledge_stats() -> Dict[str, int]:
    _init_history_db()
    conn = sqlite3.connect(_history_db_path())
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {_history_table_name()}")
        total_cases = int(cur.fetchone()[0])
        cur.execute(f"SELECT COUNT(DISTINCT url) FROM {_history_table_name()}")
        previously_seen_urls = int(cur.fetchone()[0])

        today = _dt.datetime.now().date().isoformat()
        cur.execute(
            f"SELECT SUM(CASE WHEN DATE(scan_time) = DATE(?) THEN 1 ELSE 0 END) FROM {_history_table_name()}",
            (today,),
        )
        new_cases_added_today = cur.fetchone()[0]
        new_cases_added_today = int(new_cases_added_today or 0)

        cur.execute(f"SELECT SUM(times_scanned) FROM {_history_table_name()}")
        total_scans = cur.fetchone()[0]
        total_scans = int(total_scans or 0)

        # previously_seen_urls is same as total_cases in current upsert schema.
        return {
            "total_cases": total_cases,
            "previously_seen_urls": previously_seen_urls,
            "new_cases_added_today": new_cases_added_today,
            "total_scans": total_scans,
        }
    finally:
        conn.close()


def _history_similar_urls(query_url: str, top_n: int = 5) -> list[dict[str, Any]]:
    """Return list of {url, similarity} for the most similar historical URLs.

    Requirements implemented:
    - Exclude the current scanned URL from results.
    - Use RapidFuzz if available, else fall back to difflib.
    """
    _init_history_db()
    conn = sqlite3.connect(_history_db_path())
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT url FROM {_history_table_name()}")
        all_urls = [r[0] for r in cur.fetchall()]

        q_raw = (query_url or "")
        q = _normalize_url_for_similarity(q_raw)
        q_alt = _normalize_url_for_similarity(q_raw.rstrip("/"))

        scored: list[tuple[str, float]] = []

        for u in all_urls:
            if u is None:
                continue

            us = _normalize_url_for_similarity(str(u))
            if not us:
                continue

            # Exclude exact match of the current scanned URL (including trailing-slash variants)
            if q and (us == q or us == q_alt):
                continue

            if _rf_fuzz is not None:
                sim = float(_rf_fuzz.ratio(q, us))
            else:
                sim = difflib.SequenceMatcher(None, q, us).ratio() * 100.0

            scored.append((u, sim))

        scored.sort(key=lambda x: x[1], reverse=True)

        out: list[dict[str, Any]] = []
        for u, sim in scored[: int(top_n)]:
            out.append({"url": u, "similarity": float(sim)})
        return out
    finally:
        conn.close()



def _search_previous_cases(query_url: str, limit: int = 50) -> pd.DataFrame:
    """Return exact/substring matches of historical URLs."""
    _init_history_db()
    q = (query_url or "").strip().lower()
    conn = sqlite3.connect(_history_db_path())
    try:
        cur = conn.cursor()
        # SQLite LIKE is case-insensitive for ASCII by default; still we lower both.
        cur.execute(
            f"""
            SELECT url, prediction, confidence, risk_score, scan_time, times_scanned
            FROM {_history_table_name()}
            WHERE lower(url) LIKE ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (f"%{q}%", int(limit)),
        )
        rows = cur.fetchall()
        return pd.DataFrame(
            rows,
            columns=["url", "prediction", "confidence", "risk_score", "scan_time", "times_scanned"],
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PhishGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# Custom CSS — minimal, accessible
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .verdict-dangerous  { color: #d32f2f; font-weight: 700; font-size: 1.4rem; }
    .verdict-suspicious { color: #f57c00; font-weight: 700; font-size: 1.4rem; }
    .verdict-safe       { color: #2e7d32; font-weight: 700; font-size: 1.4rem; }
    .risk-bar-wrap { background: #e0e0e0; border-radius: 8px; height: 18px; width: 100%; }
    .risk-bar-fill { height: 18px; border-radius: 8px; }
    /* Make number_input match text_input visual height in URL Scanner */
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input {
        height: 44px;
    }

    /* Remove earlier spacing tweak that made number_input shorter */
    div[data-testid="stNumberInput"] { margin-top: -28px; }

    .pill { padding: 6px 10px; border-radius: 999px; background: #f3f4f6; display: inline-block; }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def verdict_color(verdict: str) -> str:
    v = (verdict or "").strip().lower()
    if v == "dangerous":
        return "verdict-dangerous"
    return "verdict-safe"


def risk_bar(score: float) -> str:
    pct = max(0.0, min(100.0, float(score)))
    if pct >= 65:
        color = "#d32f2f"
    elif pct >= 35:
        color = "#f57c00"
    else:
        color = "#2e7d32"

    return (
        f"<div class='risk-bar-wrap'>"
        f"<div class='risk-bar-fill' style='width:{pct:.1f}%;background:{color};'></div>"
        f"</div><small>{pct:.1f} / 100</small>"
    )


def verdict_badge(verdict: str) -> str:
    css = verdict_color(verdict)
    v = (verdict or "").strip().lower()
    icon = "🟢 "
    if v == "dangerous":
        icon = "🔴 "
    return f"<span class='{css}'>{icon}{(verdict or '').upper()}</span>"


def _normalize_label_value(x: Any) -> Optional[int]:
    """Map label values to {0,1} if possible."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, (int,)):
        return 1 if int(x) == 1 else 0
    s = str(x).strip().lower()
    mapping = {
        "1": 1,
        "0": 0,
        "phishing": 1,
        "dangerous": 1,
        "danger": 1,
        "malicious": 1,
        "true": 1,
        "yes": 1,
        "safe": 0,
        "legitimate": 0,
        "benign": 0,
        "false": 0,
        "no": 0,
    }
    if s in mapping:
        return mapping[s]
    return None


@st.cache_data(show_spinner=False, persist=True)
def load_dataset(excel_path: str) -> pd.DataFrame:
    """Load dataset once and cache it.

    Notes:
    - Using st.cache_data ensures repeated dashboard interactions don't re-read the file.
    - For 450k rows this is still feasible; avoid repeated filters that rebuild huge copies.
    """
    df = pd.read_excel(excel_path)

    # Standardize columns we need
    # Expected columns per prompt: url, label, result
    col_map = {c.lower(): c for c in df.columns}

    # url
    url_col = col_map.get("url")
    if url_col is None:
        # fallback
        for k in df.columns:
            if "url" in str(k).lower():
                url_col = k
                break
    # label
    label_col = col_map.get("label")
    if label_col is None:
        # fallback
        for k in df.columns:
            kl = str(k).lower()
            if "label" in kl or "phish" in kl:
                label_col = k
                break
    if url_col is None or label_col is None:
        raise ValueError(f"Dataset must contain URL and label columns. Columns: {list(df.columns)}")

    # result
    result_col = col_map.get("result")

    out = df[[url_col, label_col] + ([result_col] if result_col else [])].copy()
    out = out.rename(columns={url_col: "url", label_col: "label"})

    # Normalize label to {0,1}
    out["label"] = out["label"].map(_normalize_label_value)
    out = out.dropna(subset=["label"]).copy()
    out["label"] = out["label"].astype(int)

    if result_col:
        out = out.rename(columns={result_col: "result"})

    return out


def dataset_counts(df: pd.DataFrame) -> Tuple[int, int, int]:
    total = len(df)
    benign = int((df["label"] == 0).sum())
    malicious = int((df["label"] == 1).sum())
    return total, benign, malicious


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("# 🛡️ PhishGuard AI")
st.markdown("**Real-time phishing detection for URLs + dataset analytics (IIT demo).**")
st.divider()


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_url, tab_dataset, tab_email, tab_sms = st.tabs(
    ["🔗 URL Scanner", "📊 Dataset Analytics", "📧 Email Analyzer", "📱 SMS Detector"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — URL Scanner (keep existing feature)
# ══════════════════════════════════════════════════════════════════════════════
with tab_url:
    st.subheader("Scan a Suspicious URL")
    st.caption("Paste any URL — no network request is made to the target site.")

    col_input, col_opts = st.columns([3, 1])
    with col_input:
        url_input = st.text_input(
            "URL",
            value="https://secure-paytm-login-verification.com",
            label_visibility="collapsed",
            placeholder="https://example.com/login",
        )
    with col_opts:
        top_k = st.number_input("Top reasons", min_value=1, max_value=6, value=5, step=1)

    scan_btn = st.button("🔍 Scan URL", type="primary", key="scan_url")

    if scan_btn:
        if not url_input.strip():
            st.error("Please enter a URL.")
        else:
            with st.spinner("Analyzing…"):
                result: Dict[str, Any] = predict_with_explanations(url_input.strip(), top_k_reasons=int(top_k))

            pred = result.get("prediction", "SAFE")
            risk = result.get("risk_score")
            proba = result.get("probabilities", {})
            reasons = (result.get("reasons", []) or [])[: int(top_k)]

            st.divider()

            # Verdict row
            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown("**Verdict**")
                st.markdown(verdict_badge(pred), unsafe_allow_html=True)
            with r2:
                st.markdown("**Risk Score**")
                if risk is not None:
                    st.markdown(risk_bar(risk), unsafe_allow_html=True)
                else:
                    st.write("N/A")
            with r3:
                st.markdown("**Confidence**")
                ph = float(proba.get("phishing", 0.0) or 0.0)
                lg = float(proba.get("legitimate", 0.0) or 0.0)
                total = ph + lg
                if total <= 0:
                    ph_pct, lg_pct = 0.0, 100.0
                else:
                    ph_pct = (ph / total) * 100.0
                    lg_pct = (lg / total) * 100.0
                st.write(f"Phishing: `{ph_pct:.1f}%`")
                st.write(f"Legitimate: `{lg_pct:.1f}%`")

            st.divider()

            is_phishing = str(pred).strip().lower() == "phishing" or str(pred).strip().lower() == "dangerous"
            reasons_header = "**Why is this risky?**" if is_phishing else "**Why is this considered safe?**"
            st.markdown(reasons_header)

            # -------------------------------------------------------------------
            # Historical Knowledge Base (enhancement only)
            # -------------------------------------------------------------------
            # 1) Check if URL exists in history.db
            # 2) If found, show Historical Match card
            # 3) If not found, save this new scan into history.db
            # 4) Always show Top 5 Similar Historical URLs + Recommended Action +
            #    Verification Status + Knowledge Base Statistics
            # -------------------------------------------------------------------
            now = _dt.datetime.now()
            normalized_url = str(url_input).strip()
            history_row = _get_url_history(normalized_url)

            # Recommended Action based on phishing probability / risk
            phishing_prob = 0.0
            if isinstance(proba, dict):
                phishing_prob = float(proba.get("phishing", 0.0) or 0.0)

            # Approximate: treat >=0.7 phishing probability OR risk >= 70 as high
            high_risk = phishing_prob >= 0.7 or (risk is not None and float(risk) >= 70)
            suspicious = (not high_risk) and (phishing_prob >= 0.4 or (risk is not None and float(risk) >= 40))

            if history_row is not None:
                # Historical match card
                st.subheader("Historical Knowledge Base")

                first_scan = _format_scan_time(history_row.get("first_scan"))
                last_scan = _format_scan_time(history_row.get("last_scan"))

                import html as _html
                import re as _re

                def _strip_html_tags(text: str) -> str:
                    # Remove any stored HTML tags so they never show up in the UI.
                    # This fixes cases where history.db contains strings like <p>, <b>, <div>.
                    return _re.sub(r"<[^>]*>", "", text or "")

                # Sanitize values before inserting into HTML
                prev_val = str(history_row.get("prediction", ""))
                prev_val = prev_val.replace("DANGEROUS", "Phishing").replace("SAFE", "Legitimate")
                prev_val = _strip_html_tags(prev_val)

                conf_raw = float(history_row.get("confidence", 0.0))
                conf_val = conf_raw * 100.0 if conf_raw <= 1.0 else conf_raw

                first_val = _strip_html_tags(str(first_scan))
                last_val = _strip_html_tags(str(last_scan))

                times_scanned = int(history_row.get("times_scanned", 0))

                # Build ONLY ONE history_html block and render it once.
                history_html = (
                    f'<div style="border:1px solid #e0e0e0;padding:16px;border-radius:10px;margin-top:10px;">'
                    f'<h3 style="margin:0 0 8px 0;">Historical Match Found</h3>'
                    f'<div><b>Previous Verdict:</b> {prev_val}</div>'
                    f'<div><b>Confidence:</b> {conf_val:.1f}%</div>'
                    f'<div><b>First Scan:</b> {first_val}</div>'
                    f'<div><b>Last Scan:</b> {last_val}</div>'
                    f'<div><b>Times Scanned:</b> {times_scanned}</div>'
                    f'</div>'
                )

                st.markdown(history_html, unsafe_allow_html=True)

            else:
                # No history match: record this new scan
                _record_scan(
                    url=normalized_url,
                    prediction=str(pred),
                    confidence=float((proba or {}).get("phishing", 0.0) or 0.0) * 100.0
                    if str(pred).strip().lower() in {"dangerous", "phishing"}
                    else float((proba or {}).get("legitimate", 0.0) or 0.0) * 100.0,
                    risk_score=float(risk) if risk is not None else None,
                    scan_time=now,
                )

            # Update scan count every time user scans (increment times_scanned)
            # To keep behavior aligned, record scan after prediction regardless of match.
            # This still satisfies "save when not found" and improves scan counters.
            if history_row is not None:
                _record_scan(
                    url=normalized_url,
                    prediction=str(pred),
                    confidence=float((proba or {}).get("phishing", 0.0) or 0.0) * 100.0
                    if str(pred).strip().lower() in {"dangerous", "phishing"}
                    else float((proba or {}).get("legitimate", 0.0) or 0.0) * 100.0,
                    risk_score=float(risk) if risk is not None else None,
                    scan_time=now,
                )

            # Top 5 Similar URLs
            st.subheader("Top 5 Similar URLs")
            similar = _history_similar_urls(normalized_url, top_n=5)
            for i, item in enumerate(similar, start=1):
                st.write(f"{i}. {item['url']}")
                st.caption(f"Similarity: {item['similarity']:.0f}%")

            # Recommended Action
            st.subheader("Recommended Action")
            if high_risk:
                st.markdown("""
                <div>
                    ✓ Block URL<br/>
                    ✓ Warn User<br/>
                    ✓ Report to Administrator
                </div>
                """, unsafe_allow_html=True)
            elif suspicious:
                st.markdown("""
                <div>
                    ✓ Monitor<br/>
                    ✓ Use Caution
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div>
                    ✓ Safe to Proceed
                </div>
                """, unsafe_allow_html=True)

            # Verification Status (demo)
            st.subheader("Verification Status")
            if high_risk:
                st.write("Pending Review")
            else:
                st.write("Verified by Security Team")

            # Knowledge Base Statistics
            stats = _knowledge_stats()
            st.subheader("Knowledge Base")
            cts1, cts2, cts3, cts4 = st.columns(4)
            cts1.metric("Total Stored Cases", f"{stats['total_cases']:,}")
            cts2.metric("Previously Seen URLs", f"{stats['previously_seen_urls']:,}")
            cts3.metric("New Cases Added Today", f"{stats['new_cases_added_today']:,}")
            cts4.metric("Total Historical Scans", f"{stats['total_scans']:,}")

            # URL history search + export CSV (within URL Scanner tab)
            st.subheader("Search Previous Cases")
            search_prev = st.text_input("Enter URL to search in history")
            search_prev_btn = st.button("Search", key="search_prev_cases")
            if search_prev_btn and (search_prev or "").strip():
                hist_df = _search_previous_cases(search_prev, limit=50)
                if len(hist_df) == 0:
                    st.warning("No historical records found.")
                else:
                    st.dataframe(hist_df, use_container_width=True, hide_index=True)

            st.download_button(
                label="⬇ Export Historical Database (CSV)",
                data=_export_history_csv(),
                file_name="history.csv",
                mime="text/csv",
                use_container_width=False,
                key="export_history_csv_btn",
            )

            # -------------------------------------------------------------------
            # End Historical Knowledge Base enhancements
            # -------------------------------------------------------------------

            st.subheader("Top Reasons")
            REASON_LABELS = {
                "has_suspicious_keyword": "Contains suspicious phishing keywords",
                "brand_similarity": "Domain resembles a known brand",
                "url_length": "URL is unusually long",
                "suspicious_keyword_count": "Multiple phishing-related keywords detected",
                "num_hyphens": "Contains many hyphens",
                "https_usage": "Uses HTTPS encryption",
                "url_shortening": "Uses a URL shortening service",
                "is_ip_address": "Uses an IP address instead of a domain",
                "has_typosquatting": "Possible brand impersonation detected",
            }

            if reasons:
                for r in reasons:
                    if isinstance(r, dict):
                        reason = r.get("reason", "")
                        label = REASON_LABELS.get(reason, reason.replace("_", " ").title())
                        st.write(f"✓ {label}")
                    else:
                        s = str(r)
                        label = REASON_LABELS.get(s, s.replace("_", " ").title())
                        st.write(f"✓ {label}")
            else:
                st.info("No explanation available.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Dataset Analytics
# ══════════════════════════════════════════════════════════════════════════════
with tab_dataset:
    st.subheader("Dataset Analytics")
    st.caption("Efficiently explore the 450,176 URL dataset used for training/evaluation.")

    # Resolve dataset path relative to repo root
    default_excel = os.path.join(_REPO_ROOT, "data", "phishing url data.xlsx")

    # --- Top controls (clean dashboard layout) ---
    # Same row: Excel path (left) + Preview rows (right)
    col_path, col_preview = st.columns([3, 2], vertical_alignment="bottom")

    with col_path:
        st.text_input(
            "Dataset Excel path",
            value=default_excel,
            disabled=True,
            label_visibility="visible",
        )

    with col_preview:
        preview_rows = st.number_input(
            "Preview rows (sample table + search results)",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            key="dataset_preview_rows",
        )

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # Next row: Search + Filter
    col_search, col_filter = st.columns([3, 2])

    with col_search:
        search_query = st.text_input(
            "Search by URL (substring match)",
            value="",
            key="dataset_search",
        )

    with col_filter:
        filter_mode = st.radio(
            "Filter",
            options=["All", "Benign", "Malicious"],
            horizontal=True,
            index=0,
        )

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # Next row: Pagination
    col_per_page, col_page = st.columns([2, 1])

    with col_per_page:
        per_page = st.number_input(
            "Rows per page",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            key="dataset_per_page",
        )

    with col_page:
        page_idx = st.number_input(
            "Page",
            min_value=1,
            value=1,
            step=1,
            key="dataset_page_idx",
        )

    def _reload_dataset() -> None:
        """(Re)load dataset into session_state.

        Cache is still used via @st.cache_data in load_dataset(), but we allow
        a user-triggered full refresh by clearing the cache.
        """
        try:
            # Clear st.cache_data contents for this app (automatic cache management)
            # so users never need to use Streamlit's developer menu.
            st.cache_data.clear()
        except Exception:
            # Older Streamlit versions may not expose st.cache_data.clear()
            pass

        try:
            with st.spinner("Loading dataset…"):
                df = load_dataset(excel_path=default_excel)
            st.session_state["_dataset_df"] = df
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
            st.stop()

    # Load dataset once (cached) and store in session state
    if st.button("Load dataset", type="primary") or "_dataset_df" not in st.session_state:
        _reload_dataset()

    # Automatic cache management via built-in refresh button
    st.caption("")
    if st.button("🔄 Refresh Dataset"):
        _reload_dataset()
        st.rerun()

    if "_dataset_df" not in st.session_state:
        st.info("Click **Load dataset** to view analytics.")
        st.stop()

    df_all: pd.DataFrame = st.session_state["_dataset_df"]


    # Apply filters instantly
    df_view = df_all

    if filter_mode == "Benign":
        df_view = df_view.loc[df_view["label"] == 0]
    elif filter_mode == "Malicious":
        df_view = df_view.loc[df_view["label"] == 1]

    q = (search_query or "").strip().lower()
    if q:
        mask = df_view["url"].astype(str).str.lower().str.contains(q, na=False)
        df_view = df_view.loc[mask]

    # Total after filter
    st.markdown(
        f"<span class='pill'>Showing <b>{len(df_view):,}</b> record(s)</span>",
        unsafe_allow_html=True,
    )

    # Metrics at top (always reflect the dataset state)
    total, benign, malicious = dataset_counts(df_all)
    m1, m2, m3 = st.columns(3)
    m1.metric("Total URLs", f"{total:,}")
    m2.metric("Total Benign URLs", f"{benign:,}")
    m3.metric("Total Malicious URLs", f"{malicious:,}")

    # Charts for filtered dataset
    filtered_total, filtered_benign, filtered_malicious = dataset_counts(df_view)
    chart_df = pd.DataFrame(
        {
            "class": ["Benign", "Malicious"],
            "count": [filtered_benign, filtered_malicious],
        }
    )

    import matplotlib.pyplot as plt

    fig_pie, ax_pie = plt.subplots(figsize=(4.8, 4.2))
    ax_pie.pie(
        chart_df["count"],
        labels=[f"{c}" for c in chart_df["class"]],
        autopct="%.1f%%",
        startangle=90,
    )
    ax_pie.set_title("Benign vs Malicious (Filtered)")
    st.pyplot(fig_pie, clear_figure=True)

    fig_bar, ax_bar = plt.subplots(figsize=(6.2, 4.2))
    ax_bar.bar(chart_df["class"], chart_df["count"], color=["#2e7d32", "#d32f2f"])
    ax_bar.set_ylabel("Count")
    ax_bar.set_title("Class Distribution (Filtered)")
    st.pyplot(fig_bar, clear_figure=True)

    # Export filtered dataset
    export_cols = ["url", "label"] + (["result"] if "result" in df_view.columns else [])

    export_df = df_view[export_cols].copy()
    st.download_button(
        label="⬇ Export Filtered Dataset (CSV)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_dataset.csv",
        mime="text/csv",
        use_container_width=False,
    )

    # Sample records (top N)
    st.subheader("Sample Records")
    sample_df = export_df.head(int(preview_rows))
    st.dataframe(sample_df, use_container_width=True, hide_index=True)

    # Paginated table
    st.subheader("Dataset Table (Pagination)")

    # Keep pagination stable via deterministic ordering when available
    if "Unnamed: 0" in df_view.columns:
        df_view = df_view.sort_values("Unnamed: 0", kind="mergesort")

    page_idx_int = max(1, int(page_idx))
    per_page_int = max(10, int(per_page))

    total_pages = max(1, (len(df_view) + per_page_int - 1) // per_page_int)
    page_idx_int = min(page_idx_int, total_pages)

    start = (page_idx_int - 1) * per_page_int
    end = start + per_page_int
    page_df = df_view.iloc[start:end][export_cols]

    st.caption(
        f"Page {page_idx_int} / {total_pages} • Rows {start + 1}–{min(end, len(df_view))} of {len(df_view):,}"
    )
    st.dataframe(page_df, use_container_width=True, hide_index=True)



# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Email Analyzer (keep existing)
# ══════════════════════════════════════════════════════════════════════════════
with tab_email:
    st.subheader("Analyze a Suspicious Email")
    st.caption("Paste the email content below. Nothing is sent to external servers.")

    e1, e2 = st.columns([2, 1])
    with e1:
        email_subject = st.text_input(
            "Subject line",
            placeholder="e.g. Urgent: Your account has been suspended",
            key="email_subject",
        )
    with e2:
        email_sender = st.text_input(
            "From / Sender",
            placeholder="e.g. PayPal Support <no-reply@evil-domain.com>",
            key="email_sender",
        )

    email_body = st.text_area(
        "Email body",
        height=200,
        placeholder=(
            "Paste the full email body here.\n\n"
            "Example:\n"
            "Dear Customer,\n"
            "Your account has been suspended. Click here immediately to verify your details:\n"
            "http://paypa1-secure-verify.com/login\n"
        ),
        key="email_body",
    )

    uploaded_email = st.file_uploader(
        "Upload an email file (.txt, .eml, .pdf, .docx)",
        type=["txt", "eml", "pdf", "docx"],
        accept_multiple_files=False,
        key="email_file_uploader",
    )
    st.caption("200MB per file • TXT, EML, PDF, DOCX")

    analyze_email_btn = st.button("🔍 Analyze Email", type="primary", key="analyze_email")

    if analyze_email_btn:
        uploaded_bytes = uploaded_email.getvalue() if uploaded_email is not None else None

        parsed_body = None
        parsed_subject = None
        parsed_sender = None

        if uploaded_bytes:
            try:
                filename = (uploaded_email.name or "").lower()

                if filename.endswith(".txt"):
                    try:
                        parsed_body = uploaded_bytes.decode("utf-8")
                    except UnicodeDecodeError:
                        parsed_body = uploaded_bytes.decode("latin-1", errors="replace")

                elif filename.endswith(".eml"):
                    import email
                    from email import policy
                    from email.parser import BytesParser

                    msg = BytesParser(policy=policy.default).parsebytes(uploaded_bytes)
                    parsed_subject = msg.get("subject", "") or ""
                    parsed_sender = msg.get("from", "") or ""

                    body_parts = []
                    if msg.is_multipart():
                        for part in msg.walk():
                            content_type = part.get_content_type()
                            disp = part.get_content_disposition()
                            if disp == "attachment":
                                continue
                            if content_type == "text/plain":
                                payload = part.get_payload(decode=True)
                                if payload is None:
                                    continue
                                charset = part.get_content_charset() or "utf-8"
                                try:
                                    body_parts.append(payload.decode(charset, errors="replace"))
                                except Exception:
                                    body_parts.append(payload.decode("utf-8", errors="replace"))
                    else:
                        payload = msg.get_payload(decode=True)
                        if payload is not None:
                            charset = msg.get_content_charset() or "utf-8"
                            try:
                                parsed_body = payload.decode(charset, errors="replace")
                            except Exception:
                                parsed_body = payload.decode("utf-8", errors="replace")

                    if body_parts:
                        parsed_body = "\n\n".join(body_parts)

                elif filename.endswith(".pdf"):
                    try:
                        import io
                        import pdfplumber

                        with pdfplumber.open(io.BytesIO(uploaded_bytes)) as pdf:
                            pages_text = []
                            for page in pdf.pages:
                                t = page.extract_text() or ""
                                if t.strip():
                                    pages_text.append(t)
                            parsed_body = "\n".join(pages_text).strip()
                    except Exception:
                        import io
                        from PyPDF2 import PdfReader

                        reader = PdfReader(io.BytesIO(uploaded_bytes))
                        pages_text = []
                        for page in reader.pages:
                            t = page.extract_text() or ""
                            if t.strip():
                                pages_text.append(t)
                        parsed_body = "\n".join(pages_text).strip()

                elif filename.endswith(".docx"):
                    try:
                        import io
                        from docx import Document

                        doc = Document(io.BytesIO(uploaded_bytes))
                        paragraphs = [p.text for p in doc.paragraphs if (p.text or "").strip()]
                        parsed_body = "\n".join(paragraphs).strip()
                    except Exception as e:
                        raise RuntimeError(f"Failed to parse DOCX: {e}")

            except Exception as e:
                st.error(f"Failed to parse uploaded email file: {e}")

        final_body = parsed_body if parsed_body is not None else email_body
        final_subject = parsed_subject if parsed_subject is not None else email_subject
        final_sender = parsed_sender if parsed_sender is not None else email_sender

        if not (final_body or "").strip():
            st.error("Please paste the email body or upload a TXT, EML, PDF, or DOCX file.")
        else:
            with st.spinner("Analyzing email…"):
                report = analyze_email(body=final_body, subject=final_subject, sender=final_sender)

            st.divider()

            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown("**Verdict**")
                st.markdown(verdict_badge(report.verdict), unsafe_allow_html=True)
                st.markdown("**Risk Score**")
                st.markdown(risk_bar(report.risk_score), unsafe_allow_html=True)
            with c2:
                st.markdown("**Top Risk Reasons**")
                if report.top_reasons:
                    for reason in report.top_reasons:
                        st.markdown(f"- {reason}")
                else:
                    st.success("No phishing indicators found.")

            st.divider()

            if report.sender_details or report.sender_spoofing:
                with st.expander("📨 Sender Analysis", expanded=True):
                    if report.sender_spoofing:
                        st.error("⚠️ Sender display name spoofs a known brand!")
                    for k, v in report.sender_details.items():
                        st.write(f"**{k.replace('_', ' ').title()}:** {v}")

            if report.urls_found:
                with st.expander(f"🔗 URLs found in email ({len(report.urls_found)})", expanded=True):
                    for u in report.urls_found:
                        score = report.url_risk_scores.get(u, 0.0)
                        icon = "🔴" if score >= 65 else ("🟠" if score >= 35 else "🟢")
                        st.markdown(f"{icon} `{u}` — risk {score:.0f}/100")

            with st.expander("📊 Pattern breakdown"):
                ind = report.raw_indicators
                cols = st.columns(3)
                cols[0].metric("Urgency patterns", ind.get("urgency_count", 0))
                cols[1].metric("Financial patterns", ind.get("financial_count", 0))
                cols[2].metric("Credential patterns", ind.get("credential_count", 0))

                cols2 = st.columns(3)
                cols2[0].metric("URLs found", ind.get("url_count", 0))
                cols2[1].metric("Max URL risk", f"{ind.get('max_url_risk', 0):.0f}/100")
                cols2[2].metric("Suspicious attachments", ind.get("attachment_count", 0))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SMS Detector (keep existing)
# ══════════════════════════════════════════════════════════════════════════════
with tab_sms:
    st.subheader("Scan a Suspicious SMS / Text Message")
    st.caption("Paste the full message text. Detects smishing (SMS phishing) patterns locally.")

    sms_text = st.text_area(
        "Message text",
        height=150,
        placeholder=(
            "Paste the SMS here.\n\n"
            "Example:\n"
            "URGENT: Your HDFC bank account is blocked. "
            "Verify now at http://hdfc-secure-verify.xyz or lose access permanently."
        ),
        key="sms_text",
    )

    analyze_sms_btn = st.button("🔍 Analyze SMS", type="primary", key="analyze_sms")

    if analyze_sms_btn:
        if not sms_text.strip():
            st.error("Please paste the SMS message.")
        else:
            with st.spinner("Analyzing SMS…"):
                sms_report = analyze_sms(sms_text.strip())

            st.divider()

            s1, s2 = st.columns([1, 2])
            with s1:
                st.markdown("**Verdict**")
                st.markdown(verdict_badge(sms_report.verdict), unsafe_allow_html=True)
                st.markdown("**Risk Score**")
                st.markdown(risk_bar(sms_report.risk_score), unsafe_allow_html=True)
            with s2:
                st.markdown("**Top Risk Reasons**")
                if sms_report.top_reasons:
                    for reason in sms_report.top_reasons:
                        st.markdown(f"- {reason}")
                else:
                    st.success("No smishing indicators detected.")

            st.divider()

            if sms_report.urls_found:
                with st.expander(f"🔗 URLs in message ({len(sms_report.urls_found)})", expanded=True):
                    for u in sms_report.urls_found:
                        score = sms_report.url_risk_scores.get(u, 0.0)
                        icon = "🔴" if score >= 65 else ("🟠" if score >= 35 else "🟢")
                        st.markdown(f"{icon} `{u}` — risk {score:.0f}/100")

            with st.expander("📊 Pattern breakdown"):
                ind = sms_report.raw_indicators
                col1, col2, col3 = st.columns(3)
                col1.metric("Urgency", ind.get("urgency_count", 0))
                col2.metric("Prize/Lottery", ind.get("prize_count", 0))
                col3.metric("Delivery scam", ind.get("delivery_count", 0))

                col4, col5, col6 = st.columns(3)
                col4.metric("Impersonation", ind.get("impersonation_count", 0))
                col5.metric("Financial", ind.get("financial_count", 0))
                col6.metric("URLs found", ind.get("url_count", 0))


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "PhishGuard AI · All analysis runs locally — no data is sent to external servers. "
    "This tool provides heuristic risk signals; always apply human judgment."
)

