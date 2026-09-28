
import type {
  CommunitySummary,
  ExplainResponse,
  MacroSnapshot,
  ModelComparison,
  PriceHistory,
  SeasonalCalendar,
  PredictRequest,
  PredictResponse,
  ScoreResponse,
} from "./types";

const BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") || "/api/v1";

async function req<T>(
  path: string,
  init?: RequestInit,
  timeoutMs = 12000
): Promise<T | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${BASE}${path}`, {
      ...init,
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        ...(init?.headers || {}),
      },
    });
    if (!res.ok) return null;
    return (await res.json()) as T;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

export function getMacroSnapshot(): Promise<MacroSnapshot | null> {
  return req<MacroSnapshot>("/macro/snapshot");
}

export function getModelComparison(): Promise<ModelComparison | null> {
  return req<ModelComparison>("/models/comparison");
}

export function getPriceHistory(
  area: string,
  propertyType: string,
  fromYear = 2019
): Promise<PriceHistory | null> {
  const q = new URLSearchParams({
    area,
    property_type: propertyType,
    from_year: String(fromYear),
  });
  return req<PriceHistory>(`/market/price-history?${q}`, undefined, 25000);
}

export function getSeasonalCalendar(): Promise<SeasonalCalendar | null> {
  return req<SeasonalCalendar>("/seasonal/calendar");
}

export function getCommunities(
  community?: string
): Promise<CommunitySummary[] | null> {
  const q = community ? `?community=${encodeURIComponent(community)}` : "";
  return req<CommunitySummary[]>(`/communities${q}`);
}

export function predict(
  body: PredictRequest
): Promise<PredictResponse | null> {
  return req<PredictResponse>("/predict", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export type ScoreOutcome =
  | { ok: true; data: ScoreResponse }
  | { ok: false; reason: string };

const SCORE_TIMEOUT_MS = 240_000;

export async function scoreListing(url: string): Promise<ScoreOutcome> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), SCORE_TIMEOUT_MS);
  try {
    const res = await fetch(`${BASE}/score`, {
      method: "POST",
      body: JSON.stringify({ url }),
      signal: controller.signal,
      headers: { "Content-Type": "application/json" },
    });
    if (res.ok) return { ok: true, data: (await res.json()) as ScoreResponse };

    let detail = "";
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
    }
    if (res.status === 429)
      return {
        ok: false,
        reason:
          "Too many analyses in a short time. Wait a minute and try again.",
      };
    if (res.status === 503)
      return {
        ok: false,
        reason:
          detail ||
          "The scoring service is not fully configured on this deployment.",
      };
    return {
      ok: false,
      reason: detail || `The listing could not be scored (error ${res.status}).`,
    };
  } catch (err) {
    const aborted = err instanceof DOMException && err.name === "AbortError";
    return {
      ok: false,
      reason: aborted
        ? "The analysis took longer than three minutes and was stopped. The portal or the model may be slow right now - try again."
        : "Could not reach the analysis service. Check that the backend is running.",
    };
  } finally {
    clearTimeout(timer);
  }
}

export function askAboutScore(
  question: string,
  context: ScoreResponse,
  priceHistory?: PriceHistory | null
): Promise<ExplainResponse | null> {
  return req<ExplainResponse>(
    "/explain",
    {
      method: "POST",
      body: JSON.stringify({
        question,
        context,
        price_history: priceHistory ?? null,
      }),
    },
    60000
  );
}
