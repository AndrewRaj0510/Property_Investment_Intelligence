"""APScheduler jobs (M5-04, M5-05).

Started/stopped from the FastAPI lifespan. Times are GST (Asia/Dubai).

**What is scheduled, and what deliberately is not.** There used to be a monthly
``run_pipeline`` job here that rebuilt the model end to end. It was removed
because it could never succeed: ``run_pipeline`` raised ``SupabaseNotConfigured``
on its first statement (the ORM run-registry write, which sits outside the
function's own try block), so every month it crashed before a single stage ran -
silently, since the exception escaped into the scheduler's error handler. Beyond
that it orchestrated the wrong flow entirely; see DISSERTATION_FINDINGS F34.

Model rebuilds are therefore a DELIBERATE, human-run operation:

    python -m scripts.load_transactions_csv     # DDA portal sales CSV
    python -m scripts.load_rent_contracts_csv   # DDA portal Ejari CSV
    python -m scripts.refresh_yield_medians     # verdict baseline
    python -m scripts.train_model               # train + SHAP + activate

That is the sequence that actually produced the deployed model, and retraining is
not something that should happen unattended on a cron - it swaps the model
serving live predictions. What remains scheduled is only the low-risk, additive
refresh of the macro context series.
"""

from __future__ import annotations

import structlog

from scheduler.tasks import refresh_macro

log = structlog.get_logger(__name__)

_scheduler = None


async def run_macro_refresh_only() -> None:
    """Weekly job: refresh the macro context series. Does not retrain anything."""
    log.info("macro_refresh_start")
    result = await refresh_macro()
    log.info("macro_refresh_done", errors=result.get("errors"))


def start_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        return
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger

        _scheduler = AsyncIOScheduler(timezone="Asia/Dubai")
        _scheduler.add_job(
            run_macro_refresh_only,
            CronTrigger(day_of_week="mon", hour=3, minute=0),
            id="weekly_macro_refresh",
            replace_existing=True,
        )
        _scheduler.start()
        log.info("scheduler_started", jobs=["weekly_macro_refresh"])
    except Exception as exc:  # noqa: BLE001
        log.warning("scheduler_start_failed", error=str(exc))


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        try:
            _scheduler.shutdown(wait=False)
        except Exception:  # noqa: BLE001
            pass
        _scheduler = None
        log.info("scheduler_stopped")
