"""Application settings loaded from environment variables.

Uses Pydantic BaseSettings so the app boots even when optional credentials
are missing (mock-first development). Only non-secret defaults are hard-coded;
all secrets default to empty strings and are validated lazily by the services
that need them.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    supabase_url: str = ""
    supabase_key: str = ""
    supabase_db_url: str = ""
    supabase_db_host: str = ""
    supabase_db_port: int = 5432
    supabase_db_user: str = ""
    supabase_db_password: str = ""
    supabase_db_name: str = "postgres"

    @property
    def db_connection_kwargs(self) -> dict:
        """psycopg connection kwargs built from parts (no URL escaping needed)."""
        return {
            "host": self.supabase_db_host,
            "port": self.supabase_db_port,
            "user": self.supabase_db_user,
            "password": self.supabase_db_password,
            "dbname": self.supabase_db_name,
            "sslmode": "require",
        }

    @property
    def db_parts_configured(self) -> bool:
        return bool(
            self.supabase_db_host and self.supabase_db_user and self.supabase_db_password
        )

    llm_provider: str = "openai"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = ""
    openai_reasoning_headroom_tokens: int = 2000
    openai_reasoning_effort: str = "low"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gpt-oss:120b-cloud"
    ollama_api_key: str = ""
    ollama_request_timeout_s: int = 120
    ollama_enrich_timeout_s: int = 20
    ollama_narrate_timeout_s: int = 60
    llm_max_field_chars: int = 2000
    llm_max_list_items: int = 40
    ollama_enforces_schema: bool = False

    @property
    def llm_request_timeout_s(self) -> int:
        """Request budget for a user-initiated LLM call, whichever provider."""
        return self.ollama_request_timeout_s

    @property
    def llm_enrich_timeout_s(self) -> int:
        """Request budget for the /score CLASSIFICATION call, whichever provider.

        Kept much shorter than a user-initiated call: classify and narrate run
        sequentially, so at the full budget each they could add minutes to a
        request the frontend has already abandoned.
        """
        return self.ollama_enrich_timeout_s

    @property
    def llm_narrate_timeout_s(self) -> int:
        """Request budget for the /score NARRATIVE call, whichever provider.

        Larger than the classification budget on purpose: one produces three
        short labels, the other produces the explanation the results page is
        built around. A budget too small for it does not surface as an error, it
        surfaces as a missing explanation, which is why the two are now separate
        numbers rather than one shared leash.
        """
        return self.ollama_narrate_timeout_s

    dubai_pulse_api_key: str = ""
    dubai_pulse_base_url: str = "https://api.data.dubai.gov.ae"

    dld_base_url: str = "https://apis.data.dubai"
    dld_entity: str = "dld"
    dld_dataset: str = "dld_transactions-open-api"
    dld_rent_dataset: str = "dld_rent_contracts-open-api"
    dld_client_id: str = ""
    dld_client_secret: str = ""
    dld_security_identifier: str = ""
    dld_max_requests_per_minute: int = 20
    dld_max_requests_per_day: int = 500
    dld_page_size: int = 1000
    dld_request_timeout_s: int = 30
    dld_fetch_limit: int = 0
    dld_max_per_date: int = 350
    dld_records_per_run: int = 500
    dld_rent_max_per_load_ts: int = 5000
    dld_rent_records_per_run: int = 5000

    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 30
    admin_username: str = ""
    admin_password_hash: str = ""
    cors_allowed_origins: str = "*"
    allowed_hosts: str = ""
    max_request_bytes: int = 262_144

    @property
    def allowed_hosts_list(self) -> list[str]:
        return [h.strip() for h in self.allowed_hosts.split(",") if h.strip()]

    @property
    def is_production(self) -> bool:
        return self.app_env.strip().lower() in ("production", "prod")

    worldbank_base_url: str = "https://api.worldbank.org/v2"

    eia_api_key: str = ""
    eia_base_url: str = "https://api.eia.gov/v2"

    open_meteo_base_url: str = "https://archive-api.open-meteo.com/v1"

    serp_api_key: str = ""

    apify_token: str = ""
    bayut_apify_actor: str = "memo23~apify-bayut-scraper"
    dubizzle_apify_actor: str = "memo23~dubizzle-search-listing-s-scraper"
    listing_request_timeout_s: int = 90

    listing_fixture_mode: str = "live"

    @property
    def listings_configured(self) -> bool:
        """True when live listing lookups can run (single Apify token)."""
        return bool(self.apify_token)

    @property
    def replay_listings(self) -> bool:
        return self.listing_fixture_mode.strip().lower() == "replay"

    @property
    def fixtures_allowed(self) -> bool:
        """Whether a ``fixture://`` URL may resolve to a saved capture.

        The fixture short-circuit sits BEFORE the portal-domain allowlist in
        ``fetch_listing``, so while it is open any anonymous caller can reach the
        replay path in a live deployment by pasting ``fixture://<name>``. The
        result is labelled ``source_mode="fixture"``, so it was never passed off
        as live - but an offline development affordance should not be reachable
        in production at all. Allowed only when replay mode is explicitly on, or
        outside production.
        """
        return self.replay_listings or not self.is_production

    app_env: str = "development"
    log_level: str = "INFO"
    model_artefacts_dir: str = "./artefacts"

    @property
    def supabase_configured(self) -> bool:
        """True when both URL and key are present - services can go live."""
        return bool(self.supabase_url and self.supabase_key)

    @property
    def uses_openai(self) -> bool:
        """True when the OpenAI path is selected."""
        return self.llm_provider.strip().lower() == "openai"

    @property
    def llm_configured(self) -> bool:
        """True when the LLM layer can run.

        OpenAI needs a key; Ollama is reachable-or-not and so reduces to having
        a server URL and a model name. Reachability is NOT checked here (that
        would mean a network call on a settings property); an unreachable server
        or a rejected key surfaces at call time, where each chain already
        handles failure.
        """
        if self.uses_openai:
            return bool(self.openai_api_key and self.openai_model)
        return bool(self.ollama_base_url and self.ollama_model)

    @property
    def dld_configured(self) -> bool:
        """True only when all DDA credentials needed for a live fetch exist."""
        return bool(
            self.dld_client_id
            and self.dld_client_secret
            and self.dld_security_identifier
        )

    @property
    def jwt_configured(self) -> bool:
        """True when the secured endpoints can safely operate (fail-closed)."""
        return bool(
            self.jwt_secret and self.admin_username and self.admin_password_hash
        )

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
