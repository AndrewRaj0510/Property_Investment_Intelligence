"""Response schema for the health endpoint."""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    db: bool
    model_loaded: bool
    last_pipeline_run: str | None = None
