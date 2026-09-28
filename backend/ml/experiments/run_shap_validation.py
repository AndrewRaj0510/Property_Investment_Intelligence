"""ISOLATED experiment: signed SHAP directions for the literature validation of
Objective 2.3.

Objective 2.3 asks for the SHAP output to be "validated against domain expert
expectations". No expert panel was run (see AIM.md 2.3), so the validation route
taken is against **published literature** instead. That validation needs
something the stored ``shap_summary`` table cannot supply.

``shap_summary`` stores ``mean_abs_shap`` - a MAGNITUDE. The literature makes
*directional* claims ("station proximity raises price", "larger units are priced
at a premium"), and a magnitude can neither agree nor disagree with a direction.
So this script recovers the sign of each driver's effect:

    * ``mean_abs_shap``  - importance, reproducing the production ranking.
    * ``spearman(feature value, its SHAP contribution)`` - the DIRECTION. Rank
      correlation rather than Pearson because a monotone-but-curved response is
      exactly what a tree ensemble produces, and Pearson would understate it.
    * ``low/high tercile mean SHAP`` - the same direction expressed in yield
      percentage points, which is what a write-up can actually quote.

Two properties make the output usable as evidence:

    * It reads the **materialised** ``feature_matrix`` table (the serving path),
      not a live rebuild. F24 recorded the case where the experiment path and the
      serving path read the same data by different routes and a green experiment
      proved nothing about the API - so this reads what production reads.
    * SHAP is computed over **all** rows, not the 400-row sample
      ``shap_explainer.compute_summary`` uses. Sampling is fine for ranking
      (which is all production needs) but the tercile means quoted in a
      dissertation should not move between runs.

Read-only and side-effect-free: NO database writes, NO model persistence, NO run
activation. The live model and the active run are untouched.

Run (Windows):
    cd backend && .venv\\Scripts\\python.exe -m ml.experiments.run_shap_validation
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from ml.artefacts import load_model
from ml.encoding import align_columns, encode
from ml.persist import active_run_row, feature_matrix_rows

DIRECTIONAL = [
    "price_per_sqm",
    "floor_area_median_sqm",
    "rent_per_sqm",
    "bedrooms",
    "parking_rate",
    "year",
    "nearest_metro_km",
    "poi_density_1km",
    "n_sales",
    "n_rents",
]


def _load() -> tuple[dict, pd.DataFrame]:
    run = active_run_row(phase=1)
    if run is None:
        raise SystemExit("no active phase-1 run - nothing to validate")
    bundle = load_model(run["artefact_path"])
    rows = feature_matrix_rows()
    if not rows:
        raise SystemExit("feature_matrix is empty - re-materialise it first")
    df = pd.DataFrame(rows)
    print(f"active run   : {run['run_name']}  (id {run['id']})")
    print(f"artefact     : {run['artefact_path']}")
    print(f"model type   : {bundle['model_type']}, target {bundle['target']}")
    print(f"matrix rows  : {len(df)}")
    return bundle, df


def _tercile_means(values: np.ndarray, shap_col: np.ndarray) -> tuple[float, float]:
    """Mean SHAP contribution in the bottom vs top tercile of the feature."""
    finite = np.isfinite(values)
    if finite.sum() < 6:
        return float("nan"), float("nan")
    lo_cut, hi_cut = np.nanpercentile(values[finite], [33.3, 66.7])
    lo = shap_col[finite & (values <= lo_cut)]
    hi = shap_col[finite & (values >= hi_cut)]
    lo_mean = float(lo.mean()) if lo.size else float("nan")
    hi_mean = float(hi.mean()) if hi.size else float("nan")
    return lo_mean, hi_mean


def main() -> None:
    import shap

    bundle, df = _load()
    X = align_columns(encode(df), bundle["feature_names"])
    print(f"encoded      : {X.shape[0]} rows x {X.shape[1]} features\n")

    shap_values = shap.TreeExplainer(bundle["model"]).shap_values(X)
    mean_abs = np.abs(shap_values).mean(axis=0)

    ranked = sorted(zip(X.columns, mean_abs), key=lambda t: t[1], reverse=True)
    rank_of = {name: i + 1 for i, (name, _) in enumerate(ranked)}

    print("=== GLOBAL IMPORTANCE (mean |SHAP|, all rows) - top 15 ===")
    print(f"{'rank':>4}  {'feature':<34} {'mean|SHAP|':>10}")
    for i, (name, val) in enumerate(ranked[:15], start=1):
        print(f"{i:>4}  {name:<34} {val:>10.4f}")

    print("\n=== SIGNED DIRECTIONS (the literature-validation input) ===")
    print(
        f"{'feature':<26} {'rank':>4} {'mean|SHAP|':>10} {'rho':>7} "
        f"{'p':>9} {'loSHAP':>8} {'hiSHAP':>8} {'dir':>5}"
    )
    out = []
    cols = list(X.columns)
    for feat in DIRECTIONAL:
        if feat not in cols:
            print(f"{feat:<26} {'--':>4}  (not in the trained feature set)")
            continue
        j = cols.index(feat)
        vals = pd.to_numeric(X[feat], errors="coerce").to_numpy(dtype=float)
        shap_col = shap_values[:, j]
        if np.nanstd(vals) == 0:
            print(f"{feat:<26} {rank_of[feat]:>4}  (constant in this matrix)")
            continue
        rho, p = spearmanr(vals, shap_col, nan_policy="omit")
        lo_mean, hi_mean = _tercile_means(vals, shap_col)
        direction = "+" if rho > 0 else "-"
        print(
            f"{feat:<26} {rank_of[feat]:>4} {mean_abs[j]:>10.4f} {rho:>7.3f} "
            f"{p:>9.2e} {lo_mean:>8.3f} {hi_mean:>8.3f} {direction:>5}"
        )
        out.append(
            {
                "feature": feat,
                "shap_rank": rank_of[feat],
                "mean_abs_shap": round(float(mean_abs[j]), 4),
                "spearman_rho": round(float(rho), 3),
                "p_value": float(p),
                "lo_tercile_shap": round(lo_mean, 3),
                "hi_tercile_shap": round(hi_mean, 3),
                "direction": direction,
            }
        )

    print(
        "\nrho = Spearman(feature value, its SHAP contribution). Positive means a"
        "\nhigher feature value pushes the predicted gross yield UP. loSHAP/hiSHAP"
        "\nare mean contributions in the bottom/top tercile, in yield percentage"
        "\npoints, so the gap is directly quotable."
    )
    print(f"\n{len(out)} directional features measured over {len(X)} rows.")


if __name__ == "__main__":
    main()
