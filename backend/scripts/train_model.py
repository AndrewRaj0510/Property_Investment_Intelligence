"""Train the Phase-1 yield model end-to-end and persist everything (psycopg).

This is the offline "weekly job" the app relies on: it builds the feature matrix
from the raw DLD tables, trains + cross-validates LightGBM / Random Forest /
XGBoost, saves the model artefact locally, writes the per-fold comparison report,
and persists the feature matrix, model run, and SHAP summary to Postgres via the
direct psycopg connection (the Supabase ORM is not configured in this project).
The freshly trained run is marked active so the app serves it.

Run (Windows):
    cd backend && .venv\\Scripts\\python.exe -m scripts.train_model
Flags:
    --phase N            (default 1)
    --no-activate        train + persist but do not activate the run
    --no-persist-db      skip DB writes (local artefact + report only)
"""

from __future__ import annotations

import argparse
import os

import structlog

from config import get_settings
from db.queries.feature_matrix_dld import build_feature_matrix
from ml.artefacts import save_model
from ml.baseline import evaluate_baselines
from ml.evaluate import comparison_summary, report
from ml.features import to_records
from ml.persist import (
    activate_run,
    ensure_ml_tables,
    insert_model_run,
    replace_feature_matrix_rows,
    replace_shap_summary,
)
from ml.shap_explainer import compute_summary
from ml.train import train_and_evaluate

log = structlog.get_logger(__name__)


def _write_comparison_report(metrics: dict, target: str, phase: int) -> str | None:
    try:
        d = get_settings().model_artefacts_dir
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"model_comparison_phase{phase}.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(comparison_summary(metrics, target))
        log.info("comparison_report_written", path=path)
        return path
    except Exception as exc:  # noqa: BLE001
        log.warning("comparison_report_failed", error=str(exc))
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Train + persist the yield model.")
    ap.add_argument("--phase", type=int, default=1)
    ap.add_argument("--no-activate", action="store_true")
    ap.add_argument("--no-persist-db", action="store_true")
    args = ap.parse_args()
    phase = args.phase
    version = 1

    rows = build_feature_matrix(feature_version=version)
    if len(rows) < 10:
        raise SystemExit(f"insufficient feature rows: {len(rows)}")
    print(f"feature matrix: {len(rows)} rows (phase {phase}, version {version})")

    if not args.no_persist_db:
        ensure_ml_tables()
        replace_feature_matrix_rows(to_records(rows), version)

    baselines = evaluate_baselines(feature_version=version)
    print("\nNaive baselines (5-fold CV):")
    print("baseline           MAE     RMSE     R2")
    for name, m in baselines.items():
        print(f"{name:<18} {m['mae']:.3f}  {m['rmse']:.3f}  {m['r_squared']:.3f}")

    result = train_and_evaluate(phase=phase, feature_version=version)
    print("\n" + report(result["metrics"]))

    model_type = result["model_type"]
    print(f"\nselected model (CV winner): {model_type}")
    if result.get("rf_best_params"):
        print(f"RF tuned params:  {result['rf_best_params']}")
    if result.get("xgb_best_params"):
        print(f"XGB tuned params: {result['xgb_best_params']}")
    run_name = f"phase{phase}_v{version}_{model_type}"
    bundle = {
        "model": result["model"],
        "feature_names": result["feature_names"],
        "target": result["target"],
        "phase": phase,
        "feature_version": version,
        "model_type": model_type,
    }
    artefact_path = save_model(bundle, run_name)
    print(f"artefact saved: {artefact_path}")

    _write_comparison_report(result["metrics"], result["target"], phase)

    summary = compute_summary(result["model"], result["X"])

    if not args.no_persist_db:
        m = result["best_metrics"]
        run_id = insert_model_run(
            {
                "run_name": run_name,
                "phase": phase,
                "model_type": model_type,
                "feature_version": version,
                "mae": round(m["mae"], 4),
                "rmse": round(m["rmse"], 4),
                "r_squared": round(m["r_squared"], 4),
                "cv_folds": m["cv_folds"],
                "artefact_path": artefact_path,
                "params_json": {
                    "all_models": result["metrics"],
                    "rf_best_params": result.get("rf_best_params"),
                    "xgb_best_params": result.get("xgb_best_params"),
                },
                "is_active": False,
            }
        )
        replace_shap_summary(run_id, summary)
        if not args.no_activate:
            activate_run(run_id, phase)
        print(f"model_run persisted: {run_id} (active={not args.no_activate})")

    print("\nTop 10 SHAP drivers:")
    for s in summary[:10]:
        print(f"  {s['rank']:>2}. {s['feature_name']:<45} {s['mean_abs_shap']}")


if __name__ == "__main__":
    main()
