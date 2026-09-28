"""Shared SlowAPI rate limiter.

A single ``Limiter`` instance is created here and imported by both ``main.py``
(to register the middleware + 429 handler) and the route modules (to apply
``@limiter.limit(...)`` decorators). Keyed by client IP.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])
