"""Model artefact persistence (M3B-03).

Saves the model bundle (estimator + feature names + target + metadata) with
joblib to the local artefacts dir, and uploads to a Supabase Storage bucket in
live mode. ``load_model`` resolves from local cache first, then Storage.
"""

from __future__ import annotations

import os

import joblib
import structlog

from config import get_settings
from db.client import get_supabase

log = structlog.get_logger(__name__)

BUCKET = "model-artefacts"


def _artefacts_dir() -> str:
    d = get_settings().model_artefacts_dir
    os.makedirs(d, exist_ok=True)
    return d


def save_model(bundle: dict, run_name: str) -> str:
    """Serialise the bundle; return its storage path."""
    filename = f"{run_name}.joblib"
    local_path = os.path.join(_artefacts_dir(), filename)
    joblib.dump(bundle, local_path)
    log.info("model_saved_local", path=local_path)

    client = get_supabase()
    if client is not None:
        try:
            with open(local_path, "rb") as fh:
                client.storage.from_(BUCKET).upload(
                    filename, fh.read(), {"upsert": "true"}
                )
            log.info("model_uploaded", bucket=BUCKET, file=filename)
            return f"{BUCKET}/{filename}"
        except Exception as exc:  # noqa: BLE001
            log.warning("model_upload_failed", error=str(exc))
    return local_path


def load_model(artefact_path: str) -> dict:
    """Load a model bundle from a local path or a Supabase Storage path."""
    if os.path.exists(artefact_path):
        return joblib.load(artefact_path)

    filename = os.path.basename(artefact_path)
    local_path = os.path.join(_artefacts_dir(), filename)
    if os.path.exists(local_path):
        return joblib.load(local_path)

    client = get_supabase()
    if client is not None:
        data = client.storage.from_(BUCKET).download(filename)
        with open(local_path, "wb") as fh:
            fh.write(data)
        return joblib.load(local_path)

    raise FileNotFoundError(f"artefact not found: {artefact_path}")
