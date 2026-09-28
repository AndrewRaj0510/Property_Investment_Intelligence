"""Canonical psycopg connection helpers - the one place the data layer connects.

Data access in this project runs over a DIRECT psycopg connection; the Supabase
ORM is present but unusable in this deployment (no SUPABASE_URL/KEY), so every
query path ends up here. See DISSERTATION_FINDINGS F6.

**Why this module exists.** These two helpers were defined in
``scripts/pull_and_store_dld.py`` - a CLI backfill script - and imported from
there by sixteen modules, including the entire ``db/queries`` package,
``ml/persist``, ``services/listing_scorer`` and the ``/health`` route. That
inverted the dependency: the API server could not read a row without importing a
batch-ingest script, which as a side effect of module import

  * reconfigured the process's ``sys.stdout`` encoding, and
  * pulled in the live DLD network client and its credentials handling,

neither of which a read query needs. It also meant sixteen call sites depended on
names marked private with a leading underscore.

The helpers live here now and the script imports them back, so the dependency
runs app-code → db, the direction it should.
"""

from __future__ import annotations

import psycopg


def connect(settings):
    """Open a psycopg connection (connection PARTS preferred, URL as fallback).

    Parts are preferred because the Supabase password routinely contains
    characters (``@``, ``:``, ``/``) that have to be percent-encoded inside a URL
    and silently produce an auth failure when they are not.
    """
    if settings.db_parts_configured:
        return psycopg.connect(**settings.db_connection_kwargs)
    return psycopg.connect(settings.supabase_db_url)


def quote_ident(name: str) -> str:
    """Safely quote a SQL identifier (table/column name).

    Identifiers cannot be passed as query parameters, so anything interpolated
    into SQL as a table or column name must come through here. Doubling embedded
    quotes is what makes an identifier containing ``"`` inert rather than a way
    out of the quoted context.
    """
    return '"' + name.replace('"', '""') + '"'
