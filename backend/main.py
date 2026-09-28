"""FastAPI application entry point.

Wires routers under the /api/v1 prefix, enables CORS for the separately-built
React frontend, and manages startup/shutdown via the lifespan context. The
APScheduler hookup is stubbed here (M5) but the lifespan structure is in place.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from api.rate_limit import limiter
from api.routes import (
    admin,
    auth,
    communities,
    compare,
    comparison,
    explain,
    health,
    macro,
    market,
    predict,
    score,
    seasonal,
    shap,
)
from config import get_settings
from logging_config import configure_logging
from scheduler.jobs import start_scheduler, stop_scheduler

configure_logging()
log = structlog.get_logger(__name__)

API_PREFIX = "/api/v1"


def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Uniform 429 response when a client exceeds a rate limit."""
    log.warning(
        "rate_limit_exceeded",
        path=request.url.path,
        client=request.client.host if request.client else "?",
    )
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please slow down and retry."},
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    log.info("app_startup", env=settings.app_env)

    if not settings.db_parts_configured:
        log.warning(
            "database_not_configured",
            detail="SUPABASE_DB_* connection parts missing - data endpoints will "
            "be unavailable until they are set.",
        )

    if settings.is_production:
        if settings.cors_origins_list == ["*"]:
            log.error(
                "cors_wildcard_in_production",
                detail="CORS_ALLOWED_ORIGINS is '*' - set a concrete origin "
                "allowlist before exposing this deployment.",
            )
        if not settings.jwt_configured:
            log.error(
                "admin_auth_unconfigured_in_production",
                detail="JWT_SECRET/ADMIN_USERNAME/ADMIN_PASSWORD_HASH are unset - "
                "admin endpoints will refuse every request (fail-closed).",
            )
        if not settings.allowed_hosts_list:
            log.warning(
                "allowed_hosts_unset",
                detail="ALLOWED_HOSTS is empty - any Host header is accepted.",
            )

    if settings.replay_listings:
        log.warning(
            "listing_replay_mode_enabled",
            detail="LISTING_FIXTURE_MODE=replay - /score returns SAVED FIXTURES, "
            "not live listings. Unset it for production.",
        )

    start_scheduler()
    yield
    stop_scheduler()
    log.info("app_shutdown")


_settings = get_settings()

_docs_enabled = not _settings.is_production

app = FastAPI(
    title="Property Investment Intelligence API",
    description=(
        "Predicts and explains Dubai gross rental yield at the community and "
        "property-type level. See /docs for interactive API documentation."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if _docs_enabled else None,
    redoc_url="/redoc" if _docs_enabled else None,
    openapi_url="/openapi.json" if _docs_enabled else None,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
app.add_middleware(SlowAPIMiddleware)

_origins = _settings.cors_origins_list
_wildcard = _origins == ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=not _wildcard,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


if _settings.allowed_hosts_list:
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=_settings.allowed_hosts_list
    )


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    """Reject oversized request bodies before they are parsed.

    Rate limiting caps requests per minute, not bytes per request. ``/explain``
    takes a client-supplied analysis whose string fields (description, title,
    narrative) have no natural bound, so a single request could make the server
    read and validate an arbitrarily large JSON document. ``clamp_for_prompt``
    bounds what reaches the MODEL, but it runs long after Pydantic has already
    materialised the whole payload - measured: a 2 MB field was accepted end to
    end. This guard is the one that bounds the SERVER.

    Content-Length is checked first (the cheap path). A chunked upload that
    declares no length is streamed and aborted the moment it crosses the cap, so
    the header cannot simply be omitted to bypass the limit.
    """
    cap = get_settings().max_request_bytes

    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > cap:
                return _too_large(request, int(declared), cap)
        except ValueError:
            return JSONResponse(
                status_code=400, content={"detail": "Invalid Content-Length header."}
            )
    else:
        body = b""
        async for chunk in request.stream():
            body += chunk
            if len(body) > cap:
                return _too_large(request, len(body), cap)

        async def _receive():
            return {"type": "http.request", "body": body, "more_body": False}

        request = Request(request.scope, _receive)

    return await call_next(request)


def _too_large(request: Request, size: int, cap: int) -> JSONResponse:
    log.warning(
        "request_body_too_large",
        path=request.url.path,
        bytes=size,
        cap=cap,
        client=request.client.host if request.client else "?",
    )
    return JSONResponse(
        status_code=413,
        content={"detail": f"Request body too large (limit {cap} bytes)."},
    )


@app.middleware("http")
async def security_headers(request: Request, call_next):
    """Add baseline hardening headers to every response."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Strict-Transport-Security"] = (
        "max-age=31536000; includeSubDomains"
    )
    if not request.url.path.rstrip("/") in ("/docs", "/redoc", "/openapi.json"):
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; frame-ancestors 'none'; base-uri 'none'; "
            "form-action 'none'"
        )
    response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
    response.headers["Permissions-Policy"] = (
        "geolocation=(), microphone=(), camera=(), interest-cohort=()"
    )
    return response


app.include_router(health.router, prefix=API_PREFIX)
app.include_router(auth.router, prefix=API_PREFIX)
app.include_router(admin.router, prefix=API_PREFIX)
app.include_router(predict.router, prefix=API_PREFIX)
app.include_router(score.router, prefix=API_PREFIX)
app.include_router(communities.router, prefix=API_PREFIX)
app.include_router(compare.router, prefix=API_PREFIX)
app.include_router(shap.router, prefix=API_PREFIX)
app.include_router(comparison.router, prefix=API_PREFIX)
app.include_router(explain.router, prefix=API_PREFIX)
app.include_router(macro.router, prefix=API_PREFIX)
app.include_router(seasonal.router, prefix=API_PREFIX)
app.include_router(market.router, prefix=API_PREFIX)


@app.get("/", tags=["system"])
async def root():
    return {
        "name": "Property Investment Intelligence API",
        "version": app.version,
        "docs": "/docs",
        "health": f"{API_PREFIX}/health",
    }
