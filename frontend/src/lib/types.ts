
export interface ShapDriver {
  feature: string;
  contribution: number;
  label: string;
}

export interface ShapExplanation {
  top_drivers: ShapDriver[];
  narrative: string;
  property_level?: ShapDriver[] | null;
  macro_level?: ShapDriver[] | null;
  macro_note?: string | null;
  property_share_pct?: number | null;
}

export interface YieldRange {
  low: number;
  high: number;
}

export interface PredictResponse {
  predicted_yield_lt: number;
  predicted_yield_st?: number | null;
  yield_range: YieldRange;
  community_median_yield?: number | null;
  shap_explanation: ShapExplanation;
  model_phase: number;
  model_run_id: string;
}

export interface PredictRequest {
  community: string;
  property_type: string;
  bedrooms: number;
  floor_area_sqft: number;
  use_extended_model: boolean;
}


export interface ScoredListing {
  source?: string | null;
  listing_url?: string | null;
  listing_id?: string | null;
  title?: string | null;
  description?: string | null;
  property_type?: string | null;
  bedrooms?: number | null;
  bathrooms?: number | null;
  area_sqft?: number | null;
  price?: number | null;
  purpose?: string | null;
  rent_frequency?: string | null;
  annual_rent?: number | null;
  community?: string | null;
  latitude?: number | null;
  longitude?: number | null;
  amenities?: string[] | null;
  detail_gated?: boolean | null;
}

export interface ScoreModelBlock {
  expected_gross_yield?: number | null;
  segment_matched: boolean;
  segment_fallback?: boolean;
  shap?: Record<string, number> | null;
}

export interface ScoringBlock {
  actual_gross_yield: number;
  expected_gross_yield: number;
  expected_basis: string;
  annual_rent: number;
  sale_price: number;
  rent_basis: string;
  price_basis: string;
  yield_vs_expected_pp: number;
}

export interface ScoreVerdict {
  recommend: boolean;
  label: string;
  reason: string;
  confidence: string;
}

export interface ScoreBaselineBlock {
  area: string;
  property_type: string;
  median_gross_yield: number;
  median_sale_price: number;
  median_annual_rent: number;
  n_sales: number;
  n_rents: number;
  low_sample: boolean;
  p25_sale_price?: number | null;
  p75_sale_price?: number | null;
  p90_sale_price?: number | null;
  max_sale_price?: number | null;
  p25_annual_rent?: number | null;
  p75_annual_rent?: number | null;
  p90_annual_rent?: number | null;
  max_annual_rent?: number | null;
}

export interface ListingCharacter {
  finish_tier?: string | null;
  view_type?: string | null;
  amenity_score?: number | null;
}

export interface NarrativePoint {
  label: string;
  detail: string;
}

export interface NarrativeDetail {
  intro: string;
  points: NarrativePoint[];
  action: string;
}

export interface ScoreResponse {
  listing: ScoredListing;
  scoring: ScoringBlock;
  verdict: ScoreVerdict;
  model?: ScoreModelBlock | null;
  baseline?: ScoreBaselineBlock | null;
  character?: ListingCharacter | null;
  narrative?: string | null;
  narrative_detail?: NarrativeDetail | null;
  source_mode?: string;
}


export interface ExplainResponse {
  answer: string;
  question: string;
}

export interface MacroSnapshot {
  month: string;
  eibor_3m?: number | null;
  uae_gdp_growth?: number | null;
  uae_cpi?: number | null;
  brent_crude_usd?: number | null;
  temp_max_c?: number | null;
  humidity_pct?: number | null;
  thermal_index?: number | null;
  tourist_arrivals?: number | null;
  hotel_occupancy?: number | null;
  ramadan_flag?: boolean;
  expo_dummy?: boolean;
  model_phase: number;
  eibor_12m_avg?: number | null;
  brent_12m_avg?: number | null;
  eibor_as_of?: string | null;
  brent_as_of?: string | null;
}

export interface CommunitySummary {
  community: string;
  median_yield_lt?: number | null;
  median_yield_st?: number | null;
  median_sale_price_aed?: number | null;
  median_rent_aed?: number | null;
  data_points: number;
  top_shap_driver?: string | null;
}

export interface SeasonalMonth {
  month_of_year: number;
  month_name: string;
  avg_temp_max_c?: number | null;
  avg_thermal_index?: number | null;
  thermal_index_relative?: number | null;
  avg_ramadan_days?: number | null;
  months_observed: number;
}

export interface SeasonalCalendar {
  months: SeasonalMonth[];
  basis: string;
  note: string;
}


export interface PhaseMetrics {
  run_id: string;
  run_name: string;
  model_type: string;
  mae: number | null;
  rmse: number | null;
  r_squared: number | null;
  cv_folds: number | null;
  n_features: number;
}

export interface MacroDriver {
  feature: string;
  mean_abs_shap: number;
}

export interface MetricDelta {
  mae: number | null;
  rmse: number | null;
  r_squared: number | null;
}

export interface ModelComparison {
  phase1: PhaseMetrics | null;
  phase2: PhaseMetrics | null;
  delta: MetricDelta | null;
  macro_attribution_pct: number | null;
  macro_drivers: MacroDriver[];
  macro_feature_count: number;
  interpretation: string | null;
  available: boolean;
  detail: string | null;
}


export interface PricePoint {
  period: string;
  year: number;
  quarter: number;
  median_aed_per_sqm: number;
  median_aed_per_sqft: number;
  ma4_aed_per_sqft?: number | null;
  n_sales: number;
}

export interface PriceHistory {
  area: string;
  area_requested: string;
  property_type: string;
  points: PricePoint[];
  change_pct?: number | null;
  yoy_change_pct?: number | null;
  direction?: string | null;
  first_period?: string | null;
  last_period?: string | null;
  total_sales: number;
  coverage_to?: string | null;
  basis: string;
  note: string;
  available: boolean;
  detail?: string | null;
}
