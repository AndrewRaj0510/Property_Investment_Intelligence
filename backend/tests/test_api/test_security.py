"""Security tests: JWT auth, fail-closed behavior, protected admin route.

Auth is configured at runtime by overriding the cached settings singleton, so we
can exercise both the unconfigured (fail-closed) and configured paths.
"""

from __future__ import annotations

import pytest

import config
from api.security import hash_password


@pytest.fixture
def admin_env(monkeypatch):
    """Configure JWT + admin creds on the cached settings for one test."""
    s = config.get_settings()
    monkeypatch.setattr(s, "jwt_secret", "test-secret-please-rotate", raising=False)
    monkeypatch.setattr(s, "admin_username", "admin", raising=False)
    monkeypatch.setattr(
        s, "admin_password_hash", hash_password("S3cret!pw"), raising=False
    )
    monkeypatch.setattr(s, "jwt_expiry_minutes", 30, raising=False)
    return s


def test_login_fail_closed_when_unconfigured(client, unconfigured_auth):
    r = client.post("/api/v1/auth/token", json={"username": "x", "password": "y"})
    assert r.status_code == 503


def test_admin_route_fail_closed_when_unconfigured(client, unconfigured_auth):
    r = client.post("/api/v1/admin/ingest/dld")
    assert r.status_code == 503


def test_login_rejects_bad_password(client, admin_env):
    r = client.post(
        "/api/v1/auth/token", json={"username": "admin", "password": "wrong"}
    )
    assert r.status_code == 401


def test_login_rejects_unknown_user(client, admin_env):
    r = client.post(
        "/api/v1/auth/token", json={"username": "nope", "password": "S3cret!pw"}
    )
    assert r.status_code == 401


def test_login_succeeds_and_token_unlocks_admin(client, admin_env, unconfigured_dld):
    r = client.post(
        "/api/v1/auth/token", json={"username": "admin", "password": "S3cret!pw"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["token_type"] == "bearer"
    assert body["expires_in"] == 1800
    token = body["access_token"]

    assert client.post("/api/v1/admin/ingest/dld").status_code == 401

    r2 = client.post(
        "/api/v1/admin/ingest/dld",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r2.status_code == 200
    j = r2.json()
    assert j["dld_configured"] is False
    assert j["status"] == "skipped"


def test_admin_rejects_garbage_token(client, admin_env):
    r = client.post(
        "/api/v1/admin/ingest/dld",
        headers={"Authorization": "Bearer not.a.jwt"},
    )
    assert r.status_code == 401


def test_admin_rejects_wrong_scheme(client, admin_env):
    r = client.post(
        "/api/v1/admin/ingest/dld",
        headers={"Authorization": "Basic YWRtaW46YWRtaW4="},
    )
    assert r.status_code == 401


def test_security_headers_present(client):
    r = client.get("/api/v1/health")
    assert r.headers.get("X-Content-Type-Options") == "nosniff"
    assert r.headers.get("X-Frame-Options") == "DENY"
