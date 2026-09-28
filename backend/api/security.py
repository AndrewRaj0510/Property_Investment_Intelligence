"""Authentication & authorization for secured inbound endpoints.

Implements a fail-closed JWT scheme for the admin-only endpoints (e.g. manually
triggering a live DLD ingest). Design choices for "highest security level":

* **Fail closed.** If the JWT secret or admin credentials are not configured,
  every protected route and the login route return 503 - never a guessable
  default secret, never an open door.
* **Hashed passwords only.** The admin password is verified against a bcrypt
  hash from the environment; the plaintext is never stored or logged.
* **Short-lived bearer tokens.** Tokens carry ``exp`` and ``iat`` and are signed
  HS256; expiry defaults to 30 minutes.
* **Constant-time comparisons.** Username check uses ``secrets.compare_digest``;
  password check uses passlib's bcrypt verify (constant-time by construction).
* **No information leak.** Login failures return a single generic 401 whether
  the username or the password was wrong.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone

import jwt
import structlog
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext

from config import get_settings

log = structlog.get_logger(__name__)

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

_bearer = HTTPBearer(auto_error=False)

SERVICE_UNAVAILABLE = HTTPException(
    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    detail="Authentication is not configured on this server.",
)
INVALID_CREDENTIALS = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Invalid authentication credentials.",
    headers={"WWW-Authenticate": "Bearer"},
)


def hash_password(plaintext: str) -> str:
    """Helper to generate a bcrypt hash (used when provisioning admin creds)."""
    return _pwd_context.hash(plaintext)


def authenticate_admin(username: str, password: str) -> bool:
    """Verify supplied credentials against the configured admin (fail-closed).

    Raises 503 if auth is not configured. Returns True only on an exact,
    constant-time username match plus a valid bcrypt password verification.
    """
    settings = get_settings()
    if not settings.jwt_configured:
        raise SERVICE_UNAVAILABLE

    username_ok = secrets.compare_digest(
        username.encode("utf-8"), settings.admin_username.encode("utf-8")
    )
    try:
        password_ok = _pwd_context.verify(password, settings.admin_password_hash)
    except (ValueError, TypeError):
        password_ok = False

    return username_ok and password_ok


def create_access_token(subject: str) -> tuple[str, int]:
    """Mint a short-lived signed JWT. Returns (token, expires_in_seconds)."""
    settings = get_settings()
    if not settings.jwt_configured:
        raise SERVICE_UNAVAILABLE

    expires_minutes = settings.jwt_expiry_minutes
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=expires_minutes)
    payload = {
        "sub": subject,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        "scope": "admin",
    }
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return token, expires_minutes * 60


def require_admin(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    """FastAPI dependency: require a valid admin JWT. Returns the subject.

    Fail-closed: 503 if auth isn't configured, 401 for any missing/invalid/
    expired token or wrong scheme. Used to protect admin-only routes.
    """
    settings = get_settings()
    if not settings.jwt_configured:
        raise SERVICE_UNAVAILABLE

    if creds is None or (creds.scheme or "").lower() != "bearer" or not creds.credentials:
        raise INVALID_CREDENTIALS

    try:
        payload = jwt.decode(
            creds.credentials,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            options={"require": ["exp", "iat", "sub"]},
        )
    except jwt.PyJWTError as exc:
        log.warning("jwt_verification_failed", error=str(exc))
        raise INVALID_CREDENTIALS from exc

    if payload.get("scope") != "admin":
        raise INVALID_CREDENTIALS

    return str(payload.get("sub"))
