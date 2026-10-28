"""Email delivery for verification and password recovery.

Three explicit modes (AI_SOC_EMAIL_MODE):
  * smtp   - real delivery through an operator-configured SMTP server;
  * outbox - development mode: the message is written as an .eml file to
             results/mail_outbox/ and the UI says so. Nothing is sent;
  * off    - no delivery; the UI documents that recovery/verification
             emails cannot be delivered until SMTP is configured.

No mode ever reports a fake delivery success.
"""
from __future__ import annotations

import smtplib
import ssl
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import formatdate, make_msgid

from src.auth import settings as auth_settings


@dataclass
class DeliveryResult:
    mode: str
    delivered: bool
    note: str


def _send_smtp(message: EmailMessage) -> DeliveryResult:
    if not auth_settings.SMTP_HOST:
        return DeliveryResult(
            "smtp", False,
            "SMTP is selected but AI_SOC_SMTP_HOST is not configured; the "
            "message was not sent.")
    try:
        with smtplib.SMTP(auth_settings.SMTP_HOST, auth_settings.SMTP_PORT,
                          timeout=15) as server:
            if auth_settings.SMTP_STARTTLS:
                server.starttls(context=ssl.create_default_context())
            if auth_settings.SMTP_USER:
                server.login(auth_settings.SMTP_USER,
                             auth_settings.SMTP_PASSWORD)
            server.send_message(message)
        return DeliveryResult("smtp", True, "Email sent via SMTP.")
    except Exception as exc:  # delivery errors must not leak message content
        return DeliveryResult(
            "smtp", False,
            f"SMTP delivery failed ({type(exc).__name__}); the message was "
            "not sent. Check the server settings.")


def send_mail(to_email: str, subject: str, body: str) -> DeliveryResult:
    mode = auth_settings.EMAIL_MODE
    if mode == "off":
        return DeliveryResult(
            "off", False,
            "Email delivery is disabled (AI_SOC_EMAIL_MODE=off), so no "
            "message was sent.")

    message = EmailMessage()
    message["From"] = auth_settings.SMTP_FROM
    message["To"] = to_email
    message["Subject"] = subject
    message["Date"] = formatdate(localtime=False)
    message["Message-ID"] = make_msgid(domain="ai-soc.local")
    message.set_content(body)

    if mode == "smtp":
        return _send_smtp(message)

    # Development outbox: write the exact message to disk, clearly labelled.
    outbox = auth_settings.MAIL_OUTBOX_DIR
    outbox.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    path = outbox / f"{stamp}.eml"
    path.write_bytes(message.as_bytes())
    return DeliveryResult(
        "outbox", False,
        "Email delivery is not configured for real sending: the message was "
        "written to the development outbox at results/mail_outbox/ "
        f"({path.name}).")


def reset_link(token: str) -> str:
    return f"{auth_settings.APP_URL}/?auth=reset&token={token}"


def verify_link(token: str) -> str:
    return f"{auth_settings.APP_URL}/?auth=verify&token={token}"


def send_password_reset(to_email: str, token: str) -> DeliveryResult:
    minutes = auth_settings.RESET_TOKEN_TTL_MINUTES
    body = (
        "A password recovery was requested for your AI-SOC account.\n\n"
        f"Open this link within {minutes} minutes to choose a new password:\n"
        f"{reset_link(token)}\n\n"
        "The link can be used once. If you did not request this, ignore "
        "this message - your password has not been changed.\n\n"
        "AI-SOC - Intelligent Security Operations Platform (research "
        "prototype)"
    )
    return send_mail(to_email, "AI-SOC password recovery", body)


def send_verification(to_email: str, token: str) -> DeliveryResult:
    hours = auth_settings.VERIFY_TOKEN_TTL_HOURS
    body = (
        "Welcome to AI-SOC. Confirm your email address to mark your "
        "account as verified.\n\n"
        f"Open this link within {hours} hours:\n{verify_link(token)}\n\n"
        "The link can be used once. If you did not create this account, "
        "ignore this message.\n\n"
        "AI-SOC - Intelligent Security Operations Platform (research "
        "prototype)"
    )
    return send_mail(to_email, "Verify your AI-SOC email address", body)