"""Gmail integration (OAuth2) for PhishGuard.

Key requirements implemented:
- OAuth 2.0 using Google APIs.
- Does NOT persist tokens/credentials to disk.
- Provides helpers to list recent inbox messages and fetch message content.

Notes:
- OAuth client credentials are read from environment variables:
    GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET

- This module is designed to be used by the Streamlit app.
"""

from __future__ import annotations

import base64
import os
import re
from typing import Dict, List, Optional, Tuple

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow


GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def _get_client_env() -> Tuple[str, str]:
    client_id = os.environ.get("GOOGLE_CLIENT_ID")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError(
            "Missing OAuth environment variables. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET."
        )
    return client_id, client_secret


def get_gmail_service_no_token_storage():
    """Authenticate with OAuth 2.0 and return an authenticated Gmail API client.

    This function deliberately avoids saving tokens to disk.

    Returns
    -------
    googleapiclient.discovery.Resource
        Gmail API client.
    """

    client_id, client_secret = _get_client_env()

    # Build an OAuth flow without any token persistence.
    # InstalledAppFlow opens a browser and returns an in-memory code flow.
    flow = InstalledAppFlow.from_client_config(
        {
            "installed": {
                "client_id": client_id,
                "client_secret": client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        },
        scopes=GMAIL_SCOPES,
    )

    creds: Credentials = flow.run_local_server(port=0)

    # Defensive: ensure it's fresh.
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())

    # Construct Gmail client.
    service = build("gmail", "v1", credentials=creds)
    return service


def list_latest_inbox_messages(service, max_results: int = 10) -> List[Dict]:
    """List latest inbox messages.

    Returns list of dicts with: id, threadId, snippet, internalDate, subject, from.
    """
    try:
        results = (
            service.users()
            .messages()
            .list(userId="me", labelIds=["INBOX"], maxResults=max_results)
            .execute()
        )
    except HttpError as e:
        raise RuntimeError(f"Failed to list Gmail messages: {e}") from e

    messages = results.get("messages", []) or []
    enriched: List[Dict] = []

    # Fetch headers for display (minimal payload).
    for m in messages:
        msg_id = m.get("id")
        if not msg_id:
            continue
        meta = get_message_metadata(service, msg_id)
        enriched.append(meta)

    return enriched


def get_message_metadata(service, message_id: str) -> Dict:
    """Fetch metadata headers for message list rendering."""
    msg = (
        service.users()
        .messages()
        .get(userId="me", id=message_id, format="metadata", metadataHeaders=["From", "Subject"])
        .execute()
    )

    headers = msg.get("payload", {}).get("headers", []) or []
    header_map = {h.get("name"): h.get("value", "") for h in headers if h.get("name")}

    from_val = header_map.get("From", "")
    subject_val = header_map.get("Subject", "")

    return {
        "id": message_id,
        "threadId": msg.get("threadId"),
        "snippet": msg.get("snippet", ""),
        "internalDate": msg.get("internalDate"),
        "from": from_val,
        "subject": subject_val,
    }


def _strip_html(html: str) -> str:
    # Very small/fast HTML tag removal for previews.
    text = re.sub(r"<\s*script[^>]*>.*?<\s*/\s*script\s*>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<\s*style[^>]*>.*?<\s*/\s*style\s*>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _decode_base64url(data: str) -> str:
    raw = base64.urlsafe_b64decode(data.encode("utf-8"))
    # Gmail usually UTF-8; fall back to latin-1.
    try:
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return raw.decode("latin-1", errors="replace")


def get_message_as_text(service, message_id: str, max_bytes: int = 2_000_000) -> Dict[str, str]:
    """Fetch full message and return {subject, from, body}.

    Prefers text/plain; falls back to decoded text/html.
    """
    msg = (
        service.users()
        .messages()
        .get(userId="me", id=message_id, format="full")
        .execute()
    )

    headers = msg.get("payload", {}).get("headers", []) or []
    header_map = {h.get("name"): h.get("value", "") for h in headers if h.get("name")}
    subject = header_map.get("Subject", "") or ""
    from_val = header_map.get("From", "") or ""

    payload = msg.get("payload", {}) or {}

    plain_parts: List[str] = []
    html_parts: List[str] = []

    def walk(part: Dict):
        mime_type = part.get("mimeType", "")
        body = part.get("body", {}) or {}
        filename = part.get("filename") or ""

        # Skip attachments (filename present or mime type not text/*).
        if filename and filename.strip():
            return

        data = body.get("data")
        if data:
            decoded = _decode_base64url(data)
            # Cap for safety.
            if len(decoded) > max_bytes:
                decoded = decoded[:max_bytes]
            if mime_type == "text/plain":
                plain_parts.append(decoded)
            elif mime_type == "text/html":
                html_parts.append(decoded)

        for p in part.get("parts", []) or []:
            walk(p)

    walk(payload)

    if plain_parts:
        body_text = "\n\n".join([p.strip() for p in plain_parts if p and p.strip()])
        if not body_text.strip() and html_parts:
            body_text = _strip_html("\n".join(html_parts))
    else:
        body_text = _strip_html("\n".join(html_parts)) if html_parts else ""

    # Remove common quoted-printable remnants (light touch).
    body_text = body_text.replace("=3D", "=").replace("\r\n", "\n")

    return {"subject": subject, "from": from_val, "body": body_text}

