"""Authentication and access control for the AI-SOC research prototype.

Real credential authentication: argon2id password hashing, server-side
opaque sessions, single-use expiring recovery/verification tokens, rate
limiting, role-based access control and an audit trail. No secrets are
hardcoded; configuration comes from environment variables (see
.env.example and docs/AUTHENTICATION.md).
"""
from src.auth.service import (  # noqa: F401
    AuthResult,
    authenticate,
    change_password,
    request_password_reset,
    register,
    resend_verification,
    reset_password,
    validate_session,
    verify_email,
)