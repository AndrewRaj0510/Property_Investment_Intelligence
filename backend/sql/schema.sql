CREATE TABLE IF NOT EXISTS rental_listings (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source          TEXT NOT NULL,
    listing_url     TEXT UNIQUE,
    community       TEXT NOT NULL,
    property_type   TEXT NOT NULL,
    bedrooms        INTEGER,
    floor_area_sqft NUMERIC,
    annual_rent_aed NUMERIC NOT NULL,
    description     TEXT,
    amenities_raw   TEXT[],
    
    finish_tier     TEXT,
    view_type       TEXT,
    amenity_score   INTEGER,
    llm_processed   BOOLEAN DEFAULT FALSE,
    scraped_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS airbnb_listings (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    airbnb_id       TEXT UNIQUE,
    community       TEXT NOT NULL,
    property_type   TEXT NOT NULL,
    bedrooms        INTEGER,
    nightly_rate_aed NUMERIC,
    occupancy_rate  NUMERIC,
    annual_str_income_aed NUMERIC,
    data_month      DATE NOT NULL,
    ingested_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS community_features (
    community           TEXT PRIMARY KEY,
    centroid_lat        DOUBLE PRECISION,
    centroid_lon        DOUBLE PRECISION,
    centroid_source     TEXT,
    nearest_metro_km    DOUBLE PRECISION,
    nearest_metro_name  TEXT,
    poi_density_1km     INTEGER,
    osm_refreshed_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS feature_matrix (
    id                     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    community              TEXT NOT NULL,
    property_type          TEXT NOT NULL,
    bedrooms               INTEGER NOT NULL,
    year                   INTEGER NOT NULL,
    gross_yield_lt         NUMERIC,
    sale_price_median_aed  NUMERIC,
    annual_rent_median_aed NUMERIC,
    floor_area_median_sqm  NUMERIC,
    price_per_sqm          NUMERIC,
    rent_per_sqm           NUMERIC,
    nearest_metro          TEXT,
    parking_rate           NUMERIC,
    n_sales                INTEGER,
    n_rents                INTEGER,
    nearest_metro_km       NUMERIC,
    poi_density_1km        INTEGER,
    feature_version        INTEGER DEFAULT 1,
    computed_at            TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (community, property_type, bedrooms, year, feature_version)
);

CREATE TABLE IF NOT EXISTS macro_timeseries (
    month           DATE PRIMARY KEY,
    eibor_3m        NUMERIC,
    eibor_3m_n_days INTEGER,
    weather_n_days  INTEGER,
    uae_gdp_growth  NUMERIC,
    uae_cpi         NUMERIC,
    brent_crude_usd NUMERIC,
    temp_max_c      NUMERIC,
    humidity_pct    NUMERIC,
    thermal_index   NUMERIC,
    tourist_arrivals INTEGER,
    hotel_occupancy NUMERIC,
    ramadan_flag    BOOLEAN DEFAULT FALSE,
    expo_dummy      BOOLEAN DEFAULT FALSE,
    ingested_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS macro_annual (
    year               INTEGER PRIMARY KEY,
    eibor_3m           NUMERIC,
    eibor_3m_n_days    INTEGER,
    eibor_3m_n_months  INTEGER,
    eibor_partial      BOOLEAN,
    brent_crude_usd    NUMERIC,
    brent_n_months     INTEGER,
    brent_partial      BOOLEAN,
    uae_gdp_growth     NUMERIC,
    uae_inflation_pct  NUMERIC,
    uae_cpi_index      NUMERIC,

    temp_max_c         NUMERIC,
    humidity_pct       NUMERIC,
    thermal_index      NUMERIC,
    weather_n_days     INTEGER,
    weather_partial    BOOLEAN,
    ingested_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_runs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_name        TEXT NOT NULL,
    phase           INTEGER NOT NULL,
    model_type      TEXT NOT NULL,
    feature_version INTEGER,
    mae             NUMERIC,
    rmse            NUMERIC,
    r_squared       NUMERIC,
    cv_folds        INTEGER DEFAULT 5,
    artefact_path   TEXT,
    params_json     JSONB,
    is_active       BOOLEAN DEFAULT FALSE,
    trained_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS shap_summary (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_run_id    UUID REFERENCES model_runs(id),
    feature_name    TEXT NOT NULL,
    mean_abs_shap   NUMERIC,
    rank            INTEGER,
    feature_group   TEXT
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    status          TEXT NOT NULL,
    stage           TEXT,
    error_message   TEXT,
    started_at      TIMESTAMPTZ DEFAULT NOW(),
    completed_at    TIMESTAMPTZ
);
