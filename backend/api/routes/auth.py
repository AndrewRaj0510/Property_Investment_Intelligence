"""POST /api/v1/auth/token - admin login, issues a short-lived JWT.

Rate-limited hard (brute-force protection) and fail-closed: returns 503 when
auth is not configured, a uniform 401 on any bad credential.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from api.rate_limit import limiter
from api.security import (
    INVALID_CREDENTIALS,
    authenticate_admin,
    create_access_token,
)

log = structlog.get_logger(__name__)
router = APIRouter()


class TokenRequest(BaseModel):
    username: str = Field(min_length=1, max_length=150)
    password: str = Field(min_length=1, max_length=300)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


@router.post("/auth/token", response_model=TokenResponse, tags=["auth"])
@limiter.limit("5/minute")
async def login(request: Request, body: TokenRequest) -> TokenResponse:
    if not authenticate_admin(body.username, body.password):
        log.warning("admin_login_failed", client=request.client.host if request.client else "?")
        raise INVALID_CREDENTIALS

    token, expires_in = create_access_token(subject=body.username)
    log.info("admin_login_ok", subject=body.username)
    return TokenResponse(access_token=token, expires_in=expires_in)
