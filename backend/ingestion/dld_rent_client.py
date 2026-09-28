"""DLD Ejari rent-contracts client - LIVE source (data.dubai / DDA).

The rental counterpart to :mod:`ingestion.dld_client`. Ejari is Dubai's rental
registration system; the DDA publishes registered rent contracts as
``dld_rent_contracts-open-api`` under the same ``dld`` entity. Joined to the
sales transactions on area + property type, these contracts give the real
annual rent needed to compute gross rental yield - the training label for the
yield model.

This client is **live-only** (no synthetic fallback) and is a thin wrapper over
:mod:`ingestion.dld_client`: it is the *same* API (same base URL, entity,
credentials, and approved quota of 20 req/min + 500 req/day), so auth, token
caching, rate limiting, retries and paging are all reused; only the dataset
name differs.

Verified live on staging: 41 columns incl. ``contract_id`` (natural key,
string-sortable like a transaction_id), ``contract_start_date`` (the date
cursor), ``contract_amount``, ``annual_amount``, ``actual_area``,
``area_name_en``, ``ejari_property_type_en``, and shared geography keys
(``nearest_metro_en``, ``nearest_mall_en``).
"""

from __future__ import annotations

from config import get_settings
from ingestion import dld_client


async def fetch_raw_limited(
    limit: int | None = None,
    order_by: str | None = None,
    order_dir: str | None = None,
    filter_expr: str | None = None,
) -> list[dict]:
    """Fetch up to ``limit`` raw rent-contract records (default: no cap → one page).

    :func:`ingestion.dld_client.fetch_raw_limited` pointed at the rent-contracts
    dataset, so the backfill script can use either dataset interchangeably.
    ``order_by``/``order_dir`` sort server-side (e.g. ``contract_start_date``
    desc); ``filter_expr`` is a DDA filter expression.
    """
    return await dld_client.fetch_raw_limited(
        limit=limit,
        order_by=order_by,
        order_dir=order_dir,
        filter_expr=filter_expr,
        dataset=get_settings().dld_rent_dataset,
    )
