"""Dubai DET / DSC tourism client (M1E-05) - live-only.

Parses a manually-downloaded CSV at backend/data/tourism.csv. No synthetic
fallback: if the CSV is absent, this raises so callers never persist fake data.
"""

from __future__ import annotations

import os

import structlog

log = structlog.get_logger(__name__)

DEFAULT_CSV = os.path.join("data", "tourism.csv")


def fetch_tourism(csv_path: str = DEFAULT_CSV) -> list[dict]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Tourism CSV not found at {csv_path}; download the DET/DSC tourism "
            "data first. There is no synthetic fallback."
        )
    import pandas as pd

    df = pd.read_csv(csv_path)
    return [
        {
            "month": str(r["month"]),
            "tourist_arrivals": int(r["tourist_arrivals"]),
            "hotel_occupancy": float(r["hotel_occupancy"]),
        }
        for _, r in df.iterrows()
    ]
