"""Community-level aggregation shared by the communities/compare endpoints."""

from __future__ import annotations

from statistics import median

from ml.persist import active_run_row, feature_matrix_rows, shap_summary_rows


def _median(vals: list) -> float | None:
    clean = [v for v in vals if v is not None]
    return round(median(clean), 3) if clean else None


def community_summaries() -> list[dict]:
    run = active_run_row(phase=1)
    version = run.get("feature_version") if run else None
    rows = feature_matrix_rows(version)
    by_comm: dict[str, list[dict]] = {}
    for r in rows:
        by_comm.setdefault(r["community"], []).append(r)

    top_driver = _global_top_driver()

    out = []
    for community, group in by_comm.items():
        out.append(
            {
                "community": community,
                "median_yield_lt": _median([g.get("gross_yield_lt") for g in group]),
                "median_sale_price_aed": _median(
                    [g.get("sale_price_median_aed") for g in group]
                ),
                "median_rent_aed": _median(
                    [g.get("annual_rent_median_aed") for g in group]
                ),
                "data_points": len(group),
                "top_shap_driver": top_driver,
            }
        )
    out.sort(key=lambda c: (c["median_yield_lt"] or 0), reverse=True)
    return out


def _global_top_driver() -> str | None:
    run = active_run_row()
    if run is None:
        return None
    summ = shap_summary_rows(run["id"])
    return summ[0]["feature_name"] if summ else None
