"""Table name constants for the Supabase schema.

Centralising table names avoids stringly-typed queries scattered across the
codebase and gives one place to rename if the schema evolves.
"""

DLD_TRANSACTIONS = "dld_transactions"
RENTAL_LISTINGS = "rental_listings"
AIRBNB_LISTINGS = "airbnb_listings"
COMMUNITY_FEATURES = "community_features"
FEATURE_MATRIX = "feature_matrix"
MACRO_TIMESERIES = "macro_timeseries"
MODEL_RUNS = "model_runs"
SHAP_SUMMARY = "shap_summary"
PIPELINE_RUNS = "pipeline_runs"

ALL_TABLES = [
    DLD_TRANSACTIONS,
    RENTAL_LISTINGS,
    AIRBNB_LISTINGS,
    COMMUNITY_FEATURES,
    FEATURE_MATRIX,
    MACRO_TIMESERIES,
    MODEL_RUNS,
    SHAP_SUMMARY,
    PIPELINE_RUNS,
]
