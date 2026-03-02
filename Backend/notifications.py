"""
notifications.py — Twilio SMS helper for SafeWatch incident pings.

Reads credentials from environment variables.
Auto-prepends +65 (Singapore) to bare numbers that don't have a country code.
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _normalise_sg_number(raw: str) -> str:
    """
    Normalise a phone number to E.164 format.
    If the number has no leading '+', presumes Singapore (+65).
    """
    num = raw.strip().replace(" ", "").replace("-", "")
    if num.startswith("+"):
        return num
    # strip leading 0 if present (local format)
    num = num.lstrip("0")
    return f"+65{num}"


def send_sms(to: str, body: str) -> bool:
    """
    Send a single SMS via Twilio.

    Args:
        to:   Recipient phone number (E.164 or bare SG number).
        body: Message text.

    Returns:
        True if the message was accepted by Twilio, False otherwise.
    """
    account_sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
    auth_token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
    from_number = _normalise_sg_number(os.getenv("TWILIO_PHONE_NUMBER", "").strip())

    if not account_sid or not auth_token:
        logger.error("Twilio credentials not configured (TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN missing).")
        return False

    to_normalised = _normalise_sg_number(to)

    try:
        from twilio.rest import Client  # imported lazily to avoid import error if not installed
        client = Client(account_sid, auth_token)
        message = client.messages.create(
            to=to_normalised,
            from_=from_number,
            body=body,
        )
        logger.info("SMS sent to %s — SID: %s", to_normalised, message.sid)
        return True
    except Exception as exc:
        logger.error("Failed to send SMS to %s: %s", to_normalised, exc)
        return False


def notify_incident(
    incident_id: str,
    category: str,
    risk_score: float,
    lat: float,
    lng: float,
    phone_numbers: list[str],
) -> list[str]:
    """
    Compose and dispatch incident SMS notifications to all provided phone numbers.
    Numbers are sourced from Firestore user documents by the caller.

    Returns a list of E.164 numbers that were successfully pinged.
    """
    if not phone_numbers:
        logger.warning("notify_incident called with no phone numbers — no SMS sent.")
        return []

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_targets: list[str] = []
    for t in phone_numbers:
        normalised = _normalise_sg_number(t)
        if normalised not in seen:
            seen.add(normalised)
            unique_targets.append(t)

    short_id = incident_id[:6].upper()
    body = (
        f"🚨 SAFEWATCH ALERT\n"
        f"Incident #{short_id} — {category.replace('_', ' ').title()}\n"
        f"Risk Score: {risk_score:.1f}\n"
        f"Location: {lat:.5f}, {lng:.5f}\n"
        f"Please respond immediately."
    )

    notified: list[str] = []
    for number in unique_targets:
        if send_sms(number, body):
            notified.append(_normalise_sg_number(number))

    return notified
