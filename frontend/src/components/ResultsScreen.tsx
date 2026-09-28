"use client";

import { useEffect, useRef, useState } from "react";
import { askAboutScore } from "@/lib/api";
import type {
  MacroSnapshot,
  NarrativeDetail,
  PriceHistory,
  ScoreResponse,
} from "@/lib/types";
import { DEMO_LISTING } from "@/lib/demo";
import { driverLabel } from "@/lib/labels";
import { FocusInput } from "./FocusInput";
import { HoverButton } from "./HoverButton";
import { ListingMap } from "./ListingMap";
import { PriceTrendChart } from "./PriceTrendChart";

interface ShapRow {
  name: string;
  value: string;
  width: string;
  color: string;
  delay: string;
  note: string;
}

function shapRows(score: ScoreResponse | null): ShapRow[] {
  const shap = score?.model?.shap;
  if (!shap || Object.keys(shap).length === 0) return [];

  const entries = Object.entries(shap)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 5);
  const maxAbs = Math.max(...entries.map(([, v]) => Math.abs(v)), 0.01);
  return entries.map(([feature, contribution], i) => {
    const positive = contribution >= 0;
    const pct = ((Math.abs(contribution) / maxAbs) * 92).toFixed(0);
    return {
      name: driverLabel(feature),
      value: `${positive ? "+" : "−"}${Math.abs(contribution).toFixed(2)}`,
      width: `${pct}%`,
      color: positive ? "var(--green)" : "var(--red)",
      delay: `.${i + 1}s`,
      note: `${driverLabel(feature)} ${positive ? "adds to" : "reduces"} the yield`,
    };
  });
}

export function ResultsScreen({
  analyzedUrl,
  yieldDisplay,
  gaugeOffset,
  score,
  macro,
  onReset,
  onOpenScenario,
}: {
  analyzedUrl: string;
  yieldDisplay: string;
  gaugeOffset: number;
  score: ScoreResponse | null;
  macro: MacroSnapshot | null;
  onReset: () => void;
  onOpenScenario: () => void;
}) {
  const live = score;
  const shap = shapRows(score);

  const [priceHistory, setPriceHistory] = useState<PriceHistory | null>(null);

  const metroDriver = (() => {
    const keys = Object.keys(score?.model?.shap ?? {});
    const k = keys.find((n) => n.startsWith("nearest_metro_") && (score!.model!.shap![n] ?? 0) !== 0)
      ?? keys.find((n) => n.startsWith("nearest_metro_"));
    return k ? k.replace("nearest_metro_", "") : null;
  })();

  const predicted = score ? score.scoring.actual_gross_yield * 100 : 6.2;
  const expected = score ? score.scoring.expected_gross_yield * 100 : 5.5;
  const median = score?.baseline
    ? score.baseline.median_gross_yield * 100
    : null;
  const delta = predicted - expected;
  const narrative =
    score?.narrative ||
    score?.verdict?.reason ||
    "This is the demo dataset the design ships with - no live listing was scored.";
  const narrativeDetail = score?.narrative_detail ?? null;

  return (
    <div
      style={{
        flex: 1,
        padding: "26px 28px 48px",
        maxWidth: 1360,
        width: "100%",
        margin: "0 auto",
        boxSizing: "border-box",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          marginBottom: 20,
          animation: "fadeUp .5s both",
        }}
      >
        <span
          style={{
            font: "500 11.5px 'JetBrains Mono',monospace",
            color: "var(--faint)",
            letterSpacing: "0.06em",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
            maxWidth: 520,
          }}
        >
          {analyzedUrl}
        </span>
        <span
          style={{
            font: "600 9.5px 'JetBrains Mono',monospace",
            color: live ? "var(--gold)" : "var(--green)",
            background: live ? "var(--goldsoft)" : "var(--greensoft)",
            border: `1px solid ${live ? "var(--gold)" : "var(--green)"}`,
            borderRadius: 999,
            padding: "3px 9px",
            letterSpacing: "0.08em",
            whiteSpace: "nowrap",
            flex: "none",
          }}
        >
          {live ? "FETCHED · LIVE MODEL" : "FETCHED · DEMO DATA"}
        </span>
        <HoverButton
          onClick={onReset}
          style={{
            marginLeft: "auto",
            background: "transparent",
            border: "1px solid var(--line)",
            color: "var(--muted)",
            borderRadius: 9,
            padding: "8px 15px",
            font: "500 12.5px 'DM Sans',sans-serif",
            cursor: "pointer",
            transition: "all .25s",
          }}
          hoverStyle={{ borderColor: "var(--linegold)", color: "var(--text)" }}
        >
          ← Analyze another
        </HoverButton>
      </div>

      <ListingCard score={score} />

      <InvestmentSummary
        narrative={narrative}
        detail={narrativeDetail}
        verdictLabel={score?.verdict?.label ?? null}
        confidence={score?.verdict?.confidence ?? null}
        predicted={predicted}
        expected={expected}
        median={median}
        delta={delta}
        hasScore={!!score}
        onOpenScenario={onOpenScenario}
      />

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit,minmax(min(460px,100%),1fr))",
          gap: 18,
          alignItems: "stretch",
          marginTop: 18,
        }}
      >
        <div style={{ display: "flex", flexDirection: "column", gap: 18, minWidth: 0 }}>
          <YieldGauge
            yieldDisplay={yieldDisplay}
            gaugeOffset={gaugeOffset}
            expected={expected}
            delta={delta}
            median={median}
            verdictLabel={score?.verdict?.label ?? null}
            confidence={score?.verdict?.confidence ?? "demo"}
          />
          <ShapCard shap={shap} />
          <MacroContextCard macro={macro} fill />
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 18, minWidth: 0 }}>
          <PriceTrendChart
            area={score?.baseline?.area ?? score?.listing?.community}
            propertyType={
              score?.baseline?.property_type ?? score?.listing?.property_type
            }
            onData={setPriceHistory}
          />
          <ListingMap
            lat={score?.listing?.latitude}
            lng={score?.listing?.longitude}
            community={score?.listing?.community}
            metro={metroDriver}
          />
        </div>
      </div>

      <div style={{ maxWidth: 820, margin: "18px auto 0" }}>
        <AskPanel score={score} priceHistory={priceHistory} />
      </div>
    </div>
  );
}

function ListingCard({ score }: { score: ScoreResponse | null }) {
  const lst = score?.listing;
  const character = score?.character ?? null;
  const isRent = lst?.purpose === "for-rent";
  const priceVal = lst ? (isRent ? lst.annual_rent : lst.price) ?? lst.price : null;
  const fmtAed = (n?: number | null) =>
    n == null ? null : `AED ${Math.round(n).toLocaleString()}`;

  const L = lst
    ? {
        type: lst.property_type ?? "n/a",
        community: lst.community ?? "n/a",
        purpose: isRent ? "For rent" : "For sale",
        title: lst.title ?? "Untitled listing",
        description: lst.description ?? null,
        detailGated: lst.detail_gated ?? false,
        beds: lst.bedrooms === 0 ? "Studio" : (lst.bedrooms ?? "n/a"),
        baths: lst.bathrooms ?? "n/a",
        sqft: lst.area_sqft ? Math.round(lst.area_sqft).toLocaleString() : "n/a",
        sqm: lst.area_sqft ? Math.round(lst.area_sqft / 10.7639).toLocaleString() : null,
        price: fmtAed(priceVal) ?? "n/a",
        priceLabel: isRent ? "annual rent" : "asking price",
        pricePerSqft:
          priceVal && lst.area_sqft
            ? `AED ${Math.round(priceVal / lst.area_sqft).toLocaleString()}/sqft`
            : "n/a",
        amenities: lst.amenities ?? [],
        url: lst.listing_url ?? null,
        source: lst.source ?? null,
      }
    : {
        type: DEMO_LISTING.type,
        community: DEMO_LISTING.community,
        purpose: "For sale",
        title: DEMO_LISTING.title,
        description: null as string | null,
        detailGated: false,
        beds: DEMO_LISTING.beds ?? "n/a",
        baths: DEMO_LISTING.baths ?? "n/a",
        sqft: DEMO_LISTING.sqft ?? "n/a",
        sqm: null as string | null,
        price: DEMO_LISTING.price,
        priceLabel: "asking price",
        pricePerSqft: DEMO_LISTING.pricePerSqft,
        amenities: [] as string[],
        url: null as string | null,
        source: null as string | null,
      };

  const stats = [
    { v: L.beds, k: "BEDROOMS" },
    { v: L.baths, k: "BATHROOMS" },
    { v: L.sqft, k: "SQFT", sub: L.sqm ? `${L.sqm} m²` : undefined },
    { v: L.purpose, k: "LISTED FOR" },
  ];

  return (
    <div
      style={{
        background: "var(--card)",
        border: "1px solid var(--linegold)",
        borderRadius: 18,
        overflow: "hidden",
        boxShadow: "var(--shadow)",
        animation: "fadeUp .6s .05s cubic-bezier(.22,1,.36,1) both",
        transition: "background-color .35s ease",
      }}
    >
      <div style={{ padding: "26px 30px 24px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", marginBottom: 14 }}>
          <Chip tone="gold">{L.type}</Chip>
          <Chip tone="line">{L.community}</Chip>
          <Chip tone={L.purpose === "For rent" ? "line" : "green"}>{L.purpose}</Chip>
          {L.source && <Chip tone="line">via {L.source}</Chip>}
        </div>

        <div
          style={{
            display: "flex",
            gap: 24,
            flexWrap: "wrap",
            alignItems: "flex-start",
            justifyContent: "space-between",
          }}
        >
          <div style={{ minWidth: 0, flex: "1 1 320px" }}>
            <h2
              style={{
                font: "700 27px/1.2 'DM Sans',sans-serif",
                letterSpacing: "-0.02em",
                margin: "0 0 8px",
                textWrap: "balance",
              }}
            >
              {L.title}
            </h2>
            {!L.description && L.detailGated && (
              <p
                style={{
                  margin: 0,
                  font: "400 12.5px/1.6 'DM Sans',sans-serif",
                  textAlign: "justify",
                  color: "var(--faint)",
                  maxWidth: 620,
                }}
              >
                The portal did not release this listing&rsquo;s description text,
                so the written details below come from its structured fields
                only.
              </p>
            )}
            {L.description && (
              <p
                style={{
                  margin: 0,
                  font: "400 13px/1.6 'DM Sans',sans-serif",
                  textAlign: "justify",
                  color: "var(--muted)",
                  maxWidth: 620,
                  textWrap: "pretty",
                  whiteSpace: "pre-line",
                  display: "-webkit-box",
                  WebkitLineClamp: 5,
                  WebkitBoxOrient: "vertical",
                  overflow: "hidden",
                }}
              >
                {L.description}
              </p>
            )}
          </div>
          <div style={{ textAlign: "right", flex: "0 0 auto" }}>
            <div
              style={{
                font: "700 30px 'JetBrains Mono',monospace",
                color: "var(--text)",
                letterSpacing: "-0.03em",
                lineHeight: 1.1,
              }}
            >
              {L.price}
            </div>
            <div
              style={{
                font: "500 11px 'JetBrains Mono',monospace",
                color: "var(--muted)",
                marginTop: 4,
              }}
            >
              {L.pricePerSqft} · {L.priceLabel}
            </div>
          </div>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit,minmax(120px,1fr))",
            gap: 2,
            marginTop: 22,
            border: "1px solid var(--line)",
            borderRadius: 12,
            overflow: "hidden",
            background: "var(--line)",
          }}
        >
          {stats.map((s) => (
            <div key={s.k} style={{ background: "var(--card2)", padding: "14px 16px" }}>
              <div style={{ font: "700 18px 'JetBrains Mono',monospace", color: "var(--text)" }}>
                {s.v}
              </div>
              <div
                style={{
                  font: "500 9.5px 'JetBrains Mono',monospace",
                  color: "var(--faint)",
                  letterSpacing: "0.12em",
                  marginTop: 3,
                }}
              >
                {s.k}
                {s.sub ? ` · ${s.sub}` : ""}
              </div>
            </div>
          ))}
        </div>

        {L.amenities.length > 0 && (
          <div style={{ marginTop: 22 }}>
            <div
              style={{
                font: "600 10px 'JetBrains Mono',monospace",
                letterSpacing: "0.14em",
                color: "var(--faint)",
                marginBottom: 11,
              }}
            >
              AMENITIES · {L.amenities.length}
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
              {L.amenities.map((a) => (
                <span
                  key={a}
                  style={{
                    font: "500 11.5px 'DM Sans',sans-serif",
                    color: "var(--muted)",
                    background: "var(--card2)",
                    border: "1px solid var(--line)",
                    borderRadius: 8,
                    padding: "6px 11px",
                    whiteSpace: "nowrap",
                  }}
                >
                  {a}
                </span>
              ))}
            </div>
          </div>
        )}

        {character && (
          <div style={{ marginTop: 22 }}>
            <div
              style={{
                font: "600 10px 'JetBrains Mono',monospace",
                letterSpacing: "0.14em",
                color: "var(--faint)",
                marginBottom: 4,
              }}
            >
              READ FROM THE DESCRIPTION
            </div>
            <div
              style={{
                font: "400 11.5px/1.5 'DM Sans',sans-serif",
                textAlign: "justify",
                color: "var(--faint)",
                marginBottom: 11,
                maxWidth: 560,
              }}
            >
              How the advertiser presents this unit. Not used to calculate the
              yield.
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
              {character.finish_tier && (
                <CharacterChip
                  label="Finish"
                  value={character.finish_tier}
                />
              )}
              {character.view_type && (
                <CharacterChip
                  label="View"
                  value={
                    character.view_type.toLowerCase() === "none"
                      ? "No notable view"
                      : character.view_type
                  }
                />
              )}
              {character.amenity_score != null && (
                <CharacterChip
                  label="Amenities"
                  value={`${character.amenity_score} of 5`}
                />
              )}
            </div>
          </div>
        )}

        {L.url && (
          <a
            href={L.url}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              display: "inline-block",
              marginTop: 20,
              font: "500 11.5px 'JetBrains Mono',monospace",
              color: "var(--gold)",
              textDecoration: "none",
              borderBottom: "1px solid transparent",
            }}
            onMouseOver={(e) => (e.currentTarget.style.borderBottomColor = "var(--gold)")}
            onMouseOut={(e) => (e.currentTarget.style.borderBottomColor = "transparent")}
          >
            View original listing ↗
          </a>
        )}
      </div>
    </div>
  );
}

function Chip({
  children,
  tone,
}: {
  children: React.ReactNode;
  tone: "gold" | "green" | "line";
}) {
  const styles: Record<string, React.CSSProperties> = {
    gold: { color: "var(--gold)", background: "var(--goldsoft)", border: "1px solid transparent" },
    green: { color: "var(--green)", background: "var(--greensoft)", border: "1px solid transparent" },
    line: { color: "var(--muted)", background: "transparent", border: "1px solid var(--line)" },
  };
  return (
    <span
      style={{
        font: "600 10px 'JetBrains Mono',monospace",
        letterSpacing: "0.1em",
        borderRadius: 6,
        padding: "5px 9px",
        ...styles[tone],
      }}
    >
      {children}
    </span>
  );
}

function MacroContextCard({
  macro,
  fill = false,
}: {
  macro: MacroSnapshot | null;
  fill?: boolean;
}) {
  const rows: { dot: string; body: React.ReactNode }[] = [];

  if (macro?.eibor_3m != null) {
    rows.push({
      dot: "var(--amber)",
      body: (
        <>
          3-month EIBOR is{" "}
          <span style={{ font: "600 12.5px 'JetBrains Mono',monospace", color: "var(--text)" }}>
            {macro.eibor_3m.toFixed(2)}%
          </span>
          {macro.eibor_as_of ? ` (as of ${macro.eibor_as_of})` : ""}
          {macro.eibor_12m_avg != null ? (
            <>
              , against a 12-month average of{" "}
              <span style={{ color: "var(--text)", fontWeight: 600 }}>
                {macro.eibor_12m_avg.toFixed(2)}%
              </span>
            </>
          ) : null}
          . Borrowing cost context only - it is not an input to this yield model.
        </>
      ),
    });
  }

  if (macro?.brent_crude_usd != null) {
    rows.push({
      dot: "var(--muted)",
      body: (
        <>
          Brent crude is{" "}
          <span style={{ font: "600 12.5px 'JetBrains Mono',monospace", color: "var(--text)" }}>
            ${macro.brent_crude_usd.toFixed(1)}
          </span>
          {macro.brent_as_of ? ` (as of ${macro.brent_as_of})` : ""}
          {"."} A standard proxy for regional liquidity.
        </>
      ),
    });
  }

  if (macro?.thermal_index != null || macro?.temp_max_c != null) {
    rows.push({
      dot: "var(--red)",
      body: (
        <>
          Dubai climate for {macro.month}:{" "}
          {macro.temp_max_c != null ? (
            <span style={{ color: "var(--text)", fontWeight: 600 }}>
              {Math.round(macro.temp_max_c)}°C max
            </span>
          ) : null}
          {macro.temp_max_c != null && macro.thermal_index != null ? ", " : ""}
          {macro.thermal_index != null ? (
            <>
              thermal index{" "}
              <span style={{ color: "var(--text)", fontWeight: 600 }}>
                {macro.thermal_index.toFixed(1)}
              </span>
            </>
          ) : null}
          . Seasonal climate, not a seasonal yield claim - this model predicts at
          annual grain, so it measures no monthly yield effect.
        </>
      ),
    });
  }

  if (rows.length === 0) {
    rows.push({
      dot: "var(--muted)",
      body: (
        <>
          No macroeconomic series is available right now. Nothing is shown rather
          than a placeholder figure.
        </>
      ),
    });
  }

  return (
    <div
      style={{
        background: "var(--card)",
        border: "1px solid var(--line)",
        borderRadius: 16,
        padding: "22px 24px",
        boxShadow: "var(--shadow)",
        animation: "fadeUp .6s .25s cubic-bezier(.22,1,.36,1) both",
        ...(fill
          ? {
              flex: 1,
              display: "flex",
              flexDirection: "column" as const,
            }
          : {}),
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 9,
          marginBottom: 14,
        }}
      >
        <div style={{ font: "600 13px 'DM Sans',sans-serif" }}>
          Macroeconomic context
        </div>
        <span
          style={{
            font: "600 9px 'JetBrains Mono',monospace",
            letterSpacing: "0.12em",
            color: "var(--muted)",
            border: "1px solid var(--line)",
            borderRadius: 5,
            padding: "3px 7px",
          }}
        >
          MARKET CONTEXT
        </span>
      </div>
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          justifyContent: fill ? "space-between" : "flex-start",
          gap: 11,
          flex: fill ? 1 : undefined,
        }}
      >
        {rows.map((r, i) => (
          <div
            key={i}
            style={{ display: "flex", gap: 12, alignItems: "flex-start" }}
          >
            <span
              style={{
                width: 7,
                height: 7,
                borderRadius: "50%",
                background: r.dot,
                marginTop: 6,
                flex: "none",
              }}
            />
            <span
              style={{
                font: "400 13.5px/1.55 'DM Sans',sans-serif",
                textAlign: "justify",
                color: "var(--muted)",
                textWrap: "pretty",
              }}
            >
              {r.body}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function YieldGauge({
  yieldDisplay,
  gaugeOffset,
  expected,
  delta,
  median,
  confidence,
  verdictLabel,
}: {
  yieldDisplay: string;
  gaugeOffset: number;
  expected: number;
  delta: number;
  median: number | null;
  confidence: string;
  verdictLabel: string | null;
}) {
  const hasBaseline = median != null;
  const negligible = Math.abs(delta) < 0.05;
  const beats = delta >= 0;
  const good = hasBaseline && !negligible && beats;
  return (
    <div
      style={{
        background: "linear-gradient(160deg, var(--card2), var(--card))",
        border: "1px solid var(--linegold)",
        borderRadius: 16,
        padding: "26px 24px 20px",
        boxShadow: "var(--shadow)",
        animation: "fadeUp .6s .1s cubic-bezier(.22,1,.36,1) both",
        position: "relative",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          position: "absolute",
          inset: 0,
          background:
            "radial-gradient(ellipse 70% 50% at 50% -10%, var(--goldsoft), transparent 70%)",
          pointerEvents: "none",
          animation: "glowPulse 4s ease-in-out infinite",
        }}
      />
      <div style={{ position: "relative" }}>
        <div
          style={{
            font: "600 10.5px 'JetBrains Mono',monospace",
            letterSpacing: "0.2em",
            color: "var(--muted)",
            textTransform: "uppercase",
            textAlign: "center",
            marginBottom: 6,
          }}
        >
          Predicted Gross Yield
        </div>
        <div
          style={{
            font: "400 11.5px/1.5 'DM Sans',sans-serif",
            color: "var(--faint)",
            textAlign: "center",
            marginBottom: 10,
            textWrap: "pretty",
          }}
        >
          A year of rent as a share of the purchase price, before costs
        </div>
        <div style={{ display: "grid", placeItems: "center" }}>
          <svg
            viewBox="0 0 220 122"
            style={{ width: 250, maxWidth: "100%", overflow: "visible" }}
          >
            <path
              d="M 20 110 A 90 90 0 0 1 200 110"
              fill="none"
              stroke="var(--line)"
              strokeWidth={11}
              strokeLinecap="round"
            />
            <path
              d="M 20 110 A 90 90 0 0 1 200 110"
              fill="none"
              stroke="var(--gold)"
              strokeWidth={11}
              strokeLinecap="round"
              strokeDasharray={283}
              strokeDashoffset={gaugeOffset}
              style={{
                transition: "stroke-dashoffset 1.4s cubic-bezier(.22,1,.36,1)",
              }}
            />
            <circle cx={110} cy={110} r={3} fill="var(--gold)" />
          </svg>
        </div>
        <div style={{ textAlign: "center", marginTop: -72 }}>
          <div
            style={{
              font: "700 46px 'JetBrains Mono',monospace",
              letterSpacing: "-0.04em",
              color: "var(--text)",
            }}
          >
            {yieldDisplay}
            <span style={{ fontSize: 24, color: "var(--gold)" }}>%</span>
          </div>
          <div
            style={{
              font: "500 12px 'JetBrains Mono',monospace",
              color: "var(--muted)",
              marginTop: 2,
            }}
          >
            model expects {expected.toFixed(1)}% · {confidence} confidence
          </div>
        </div>
        <div
          style={{
            marginTop: 20,
            padding: "12px 14px",
            background: good ? "var(--greensoft)" : "var(--inputbg)",
            border: `1px solid ${good ? "var(--green)" : "var(--line)"}`,
            borderRadius: 10,
            display: "flex",
            alignItems: "center",
            gap: 10,
          }}
        >
          {hasBaseline && (
            <span
              style={{
                font: "700 13px 'JetBrains Mono',monospace",
                color: negligible
                  ? "var(--faint)"
                  : beats
                    ? "var(--green)"
                    : "var(--red)",
              }}
            >
              {negligible ? "" : beats ? "+" : "−"}
              {Math.abs(delta).toFixed(1)}%
            </span>
          )}
          <span
            style={{
              font: "500 12.5px 'DM Sans',sans-serif",
              color: "var(--text)",
            }}
          >
            {hasBaseline ? (
              <>
                vs model-expected {expected.toFixed(1)}% (area median{" "}
                {median.toFixed(1)}%),{" "}
                {negligible
                  ? "in line with comparable stock."
                  : beats
                    ? "beats comparable stock."
                    : "below comparable stock."}
              </>
            ) : (
              <>
                {verdictLabel ? `${verdictLabel}: ` : ""}there are no Dubai Land
                Department sales or rent records for this area and property
                type, so this yield cannot be compared against local stock.
              </>
            )}
          </span>
        </div>
      </div>
    </div>
  );
}

function CharacterChip({ label, value }: { label: string; value: string }) {
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "baseline",
        gap: 6,
        font: "500 11.5px 'DM Sans',sans-serif",
        color: "var(--muted)",
        background: "var(--card2)",
        border: "1px dashed var(--line)",
        borderRadius: 8,
        padding: "6px 11px",
        whiteSpace: "nowrap",
      }}
    >
      <span
        style={{
          font: "600 9.5px 'JetBrains Mono',monospace",
          letterSpacing: "0.1em",
          color: "var(--faint)",
        }}
      >
        {label.toUpperCase()}
      </span>
      {value}
    </span>
  );
}

function ShapCard({ shap }: { shap: ShapRow[] }) {
  return (
    <div
      style={{
        background: "var(--card)",
        border: "1px solid var(--line)",
        borderRadius: 16,
        padding: "22px 24px",
        boxShadow: "var(--shadow)",
        animation: "fadeUp .6s .2s cubic-bezier(.22,1,.36,1) both",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "baseline",
          justifyContent: "space-between",
          marginBottom: 4,
        }}
      >
        <div style={{ font: "600 13px 'DM Sans',sans-serif" }}>Why this yield?</div>
        <span
          style={{
            font: "600 9px 'JetBrains Mono',monospace",
            letterSpacing: "0.12em",
            color: "var(--muted)",
          }}
        >
          SHAP DRIVERS
        </span>
      </div>
      <div
        style={{
          font: "400 12px/1.5 'DM Sans',sans-serif",
          textAlign: "justify",
          color: "var(--faint)",
          marginBottom: 18,
          textWrap: "pretty",
        }}
      >
        What pushed the model&rsquo;s estimate up or down for a home like this
        one, strongest first
      </div>
      {shap.length === 0 && (
        <div
          style={{
            font: "400 12px/1.6 'DM Sans',sans-serif",
            textAlign: "justify",
            color: "var(--faint)",
            padding: "6px 0",
          }}
        >
          No driver attributions available for this listing. The scoring call did
          not return SHAP values; nothing is shown rather than a placeholder.
        </div>
      )}
      <div style={{ display: "flex", flexDirection: "column", gap: 15 }}>
        {shap.map((d) => (
          <div
            key={d.name}
            style={{ display: "flex", flexDirection: "column", gap: 5 }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "baseline",
              }}
            >
              <span style={{ font: "600 13px 'DM Sans',sans-serif" }}>
                {d.name}
              </span>
              <span
                style={{
                  font: "700 13px 'JetBrains Mono',monospace",
                  color: d.color,
                }}
              >
                {d.value}
              </span>
            </div>
            <div
              style={{
                height: 7,
                background: "var(--inputbg)",
                borderRadius: 99,
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  height: "100%",
                  width: d.width,
                  background: d.color,
                  borderRadius: 99,
                  transformOrigin: "left",
                  animation: `growX .9s ${d.delay} cubic-bezier(.22,1,.36,1) both`,
                }}
              />
            </div>
            <div
              style={{
                font: "400 11.5px/1.45 'DM Sans',sans-serif",
                textAlign: "justify",
                color: "var(--faint)",
              }}
            >
              {d.note}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function InvestmentSummary({
  narrative,
  detail,
  verdictLabel,
  confidence,
  predicted,
  expected,
  median,
  delta,
  hasScore,
  onOpenScenario,
}: {
  narrative: string;
  detail: NarrativeDetail | null;
  verdictLabel: string | null;
  confidence: string | null;
  predicted: number;
  expected: number;
  median: number | null;
  delta: number;
  hasScore: boolean;
  onOpenScenario: () => void;
}) {
  const negligible = Math.abs(delta) < 0.05;
  const hasBaseline = median != null;
  const beats = delta >= 0;
  const good = hasScore && hasBaseline && !negligible && beats;

  const headline = !hasScore
    ? "This is the demo dataset the design ships with."
    : !hasBaseline
      ? "This property cannot be compared against local sales records."
      : negligible
        ? "This property earns about the going rate for its area."
        : beats
          ? "This property earns more rent for its price than similar ones nearby."
          : "This property earns less rent for its price than similar ones nearby.";

  return (
    <div
      style={{
        background: "linear-gradient(150deg, var(--card2), var(--card))",
        border: `1px solid ${good ? "var(--green)" : "var(--linegold)"}`,
        borderRadius: 18,
        padding: "26px 30px 24px",
        boxShadow: "var(--shadow)",
        marginTop: 18,
        animation: "fadeUp .6s .12s cubic-bezier(.22,1,.36,1) both",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 9,
          flexWrap: "wrap",
          marginBottom: 12,
        }}
      >
        <span
          style={{
            font: "600 9.5px 'JetBrains Mono',monospace",
            letterSpacing: "0.14em",
            color: "var(--faint)",
          }}
        >
          WHAT THIS MEANS
        </span>
        {verdictLabel && (
          <span
            style={{
              font: "600 10px 'JetBrains Mono',monospace",
              letterSpacing: "0.08em",
              color: good ? "var(--green)" : "var(--gold)",
              background: good ? "var(--greensoft)" : "var(--goldsoft)",
              borderRadius: 6,
              padding: "4px 9px",
            }}
          >
            {verdictLabel.toUpperCase()}
          </span>
        )}
      </div>

      <h3
        style={{
          margin: "0 0 12px",
          font: "700 21px/1.35 'DM Sans',sans-serif",
          letterSpacing: "-0.015em",
          color: "var(--text)",
          textWrap: "balance",
          maxWidth: 780,
        }}
      >
        {headline}
      </h3>

      {detail ? (
        <>
          {detail.intro && <p style={introStyle}>{detail.intro}</p>}

          {detail.points.length > 0 && (
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: 14,
                marginTop: 18,
                maxWidth: 820,
              }}
            >
              {detail.points.map((point, i) => (
                <div
                  key={`${point.label}-${i}`}
                  style={{ display: "flex", gap: 12, alignItems: "flex-start" }}
                >
                  <span
                    style={{
                      width: 6,
                      height: 6,
                      borderRadius: "50%",
                      background: "var(--gold)",
                      marginTop: 8,
                      flex: "none",
                    }}
                  />
                  <div style={{ minWidth: 0 }}>
                    {point.label && (
                      <div
                        style={{
                          font: "600 13px 'DM Sans',sans-serif",
                          color: "var(--text)",
                          marginBottom: 2,
                        }}
                      >
                        {point.label}
                      </div>
                    )}
                    <div
                      style={{
                        font: "400 13.5px/1.65 'DM Sans',sans-serif",
                        color: "var(--muted)",
                        textAlign: "justify",
                        textWrap: "pretty",
                      }}
                    >
                      {point.detail}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {detail.action && (
            <div
              style={{
                marginTop: 20,
                padding: "16px 18px",
                background: "var(--inputbg)",
                border: "1px solid var(--linegold)",
                borderRadius: 12,
                maxWidth: 820,
              }}
            >
              <div
                style={{
                  font: "600 9.5px 'JetBrains Mono',monospace",
                  letterSpacing: "0.14em",
                  color: "var(--gold)",
                  marginBottom: 7,
                }}
              >
                WHAT TO DO NEXT
              </div>
              <p
                style={{
                  margin: 0,
                  font: "400 13.5px/1.7 'DM Sans',sans-serif",
                  color: "var(--text)",
                  opacity: 0.92,
                  textAlign: "justify",
                  textWrap: "pretty",
                }}
              >
                {detail.action}
              </p>
            </div>
          )}
        </>
      ) : (
        <p style={introStyle}>{narrative}</p>
      )}

      {hasScore && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit,minmax(min(260px,100%),1fr))",
            gap: 12,
            marginTop: 20,
          }}
        >
          <PlainFact
            label="What it earns"
            value={`${predicted.toFixed(1)}% a year`}
            body={`For every AED 100 of the purchase price, the rent brings in about AED ${predicted.toFixed(
              1
            )} a year, before service charges, agency fees and maintenance.`}
          />
          <PlainFact
            label="What the model expected"
            value={`${expected.toFixed(1)}% a year`}
            body={
              hasBaseline
                ? `Similar homes in this area typically return ${median!.toFixed(
                    1
                  )}%, so this one is ${
                    negligible
                      ? "in line with them"
                      : `${Math.abs(delta).toFixed(1)} points ${
                          beats ? "above" : "below"
                        } what was expected`
                  }.`
                : "There are no recorded sales or rent contracts for this area and property type to compare against."
            }
          />
        </div>
      )}

      {confidence && (
        <p
          style={{
            margin: "16px 0 0",
            font: "400 12.5px/1.6 'DM Sans',sans-serif",
            textAlign: "justify",
            color: "var(--muted)",
            textWrap: "pretty",
            maxWidth: 780,
          }}
        >
          <strong style={{ color: "var(--text)", fontWeight: 600 }}>
            How much to trust this:
          </strong>{" "}
          {confidenceText(confidence)}
        </p>
      )}

      <div style={{ display: "flex", gap: 10, marginTop: 20, flexWrap: "wrap" }}>
        <HoverButton
          onClick={onOpenScenario}
          style={{
            background: "var(--gold)",
            color: "#0A1628",
            border: "none",
            borderRadius: 10,
            padding: "13px 22px",
            font: "700 13px 'DM Sans',sans-serif",
            cursor: "pointer",
            transition: "transform .2s, box-shadow .2s",
            boxShadow: "0 4px 16px rgba(201,168,76,0.25)",
          }}
          hoverStyle={{
            transform: "translateY(-1px)",
            boxShadow: "0 8px 24px rgba(201,168,76,0.4)",
          }}
        >
          Run my scenario
        </HoverButton>
        <HoverButton
          style={{
            background: "transparent",
            border: "1px solid var(--line)",
            color: "var(--text)",
            borderRadius: 10,
            padding: "13px 22px",
            font: "600 13px 'DM Sans',sans-serif",
            cursor: "pointer",
            transition: "border-color .25s",
          }}
          hoverStyle={{ borderColor: "var(--linegold)" }}
        >
          Export PDF
        </HoverButton>
      </div>
    </div>
  );
}

function confidenceText(confidence: string): string {
  switch (confidence) {
    case "high":
      return "The model matched this exact community, property type and bedroom count against Dubai Land Department records, so this is as solid as the data here gets.";
    case "normal":
      return "The comparable sample for this segment is on the small side, so treat the figures as a good guide rather than a precise number.";
    case "low":
      return "Treat this as indicative only. Either the model had no exact match for this segment, or the listing left out details it needs.";
    case "very_low":
      return "There is no independent sale price for this area and property type, so the two yields above are not genuinely comparable and no judgement on the asking price is possible.";
    default:
      return "This is demonstration data, not a live analysis of a real listing.";
  }
}

const introStyle: React.CSSProperties = {
  margin: 0,
  font: "400 15px/1.7 'DM Sans',sans-serif",
  color: "var(--text)",
  opacity: 0.88,
  textAlign: "justify",
  textWrap: "pretty",
  maxWidth: 820,
};

function PlainFact({
  label,
  value,
  body,
}: {
  label: string;
  value: string;
  body: string;
}) {
  return (
    <div
      style={{
        background: "var(--inputbg)",
        border: "1px solid var(--line)",
        borderRadius: 12,
        padding: "14px 16px",
      }}
    >
      <div
        style={{
          font: "600 9.5px 'JetBrains Mono',monospace",
          letterSpacing: "0.12em",
          color: "var(--faint)",
        }}
      >
        {label.toUpperCase()}
      </div>
      <div
        style={{
          font: "700 19px 'JetBrains Mono',monospace",
          color: "var(--text)",
          letterSpacing: "-0.02em",
          margin: "5px 0 6px",
        }}
      >
        {value}
      </div>
      <div
        style={{
          font: "400 12px/1.55 'DM Sans',sans-serif",
          textAlign: "justify",
          color: "var(--muted)",
          textWrap: "pretty",
        }}
      >
        {body}
      </div>
    </div>
  );
}


interface Turn {
  question: string;
  answer: string | null;
  failed: boolean;
  pending: boolean;
}

function AskPanel({
  score,
  priceHistory,
}: {
  score: ScoreResponse | null;
  priceHistory: PriceHistory | null;
}) {
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const busy = turns.some((t) => t.pending);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (turns.length) {
      endRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [turns]);

  if (!score) return null;

  const ask = async () => {
    const q = question.trim();
    if (!q || busy) return;
    setQuestion("");
    const index = turns.length;
    setTurns((prev) => [
      ...prev,
      { question: q, answer: null, failed: false, pending: true },
    ]);
    const res = await askAboutScore(q, score, priceHistory);
    setTurns((prev) =>
      prev.map((t, i) =>
        i === index
          ? { ...t, answer: res?.answer ?? null, failed: !res, pending: false }
          : t
      )
    );
  };

  return (
    <div
      style={{
        background: "var(--card)",
        border: "1px solid var(--line)",
        borderRadius: 16,
        padding: "22px 24px",
        boxShadow: "var(--shadow)",
        animation: "fadeUp .6s .3s cubic-bezier(.22,1,.36,1) both",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 9, marginBottom: 6 }}>
        <div style={{ font: "600 13px 'DM Sans',sans-serif" }}>
          Ask about this analysis
        </div>
      </div>
      <div
        style={{
          font: "400 11.5px/1.5 'DM Sans',sans-serif",
          textAlign: "justify",
          color: "var(--faint)",
          marginBottom: turns.length ? 18 : 12,
        }}
      >
        Answered only from the figures on this page, including the price history
        above. If the analysis doesn&apos;t cover it, you&apos;ll be told that
        rather than given a guess.
      </div>

      {turns.length > 0 && (
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 16,
            marginBottom: 18,
            paddingBottom: 18,
            borderBottom: "1px solid var(--line)",
          }}
        >
          {turns.map((turn, i) => (
            <div
              key={i}
              style={{
                display: "flex",
                flexDirection: "column",
                gap: 8,
                animation: "fadeUp .4s cubic-bezier(.22,1,.36,1) both",
              }}
            >
              <div style={{ display: "flex", justifyContent: "flex-end" }}>
                <div
                  style={{
                    maxWidth: "86%",
                    background: "var(--goldsoft)",
                    border: "1px solid var(--linegold)",
                    borderRadius: "12px 12px 3px 12px",
                    padding: "9px 13px",
                    font: "500 13px/1.55 'DM Sans',sans-serif",
                    color: "var(--text)",
                    textAlign: "left",
                  }}
                >
                  {turn.question}
                </div>
              </div>

              <div style={{ display: "flex", justifyContent: "flex-start" }}>
                <div
                  style={{
                    maxWidth: "92%",
                    background: "var(--inputbg)",
                    border: "1px solid var(--line)",
                    borderRadius: "12px 12px 12px 3px",
                    padding: "11px 14px",
                    font: "400 13px/1.65 'DM Sans',sans-serif",
                    textAlign: "justify",
                    color: turn.pending || turn.failed ? "var(--faint)" : "var(--text)",
                    textWrap: "pretty",
                  }}
                >
                  {turn.pending
                    ? "Reading the analysis…"
                    : turn.failed
                      ? "Couldn't answer that right now, the explanation service is unavailable. The figures above are unaffected."
                      : turn.answer}
                </div>
              </div>
            </div>
          ))}
          <div ref={endRef} />
        </div>
      )}

      <div style={{ display: "flex", gap: 8 }}>
        <FocusInput
          value={question}
          onChange={setQuestion}
          onKeyDown={(e) => {
            if (e.key === "Enter") ask();
          }}
          maxLength={500}
          disabled={busy}
          placeholder={
            turns.length
              ? "Ask another question"
              : "e.g. what is the highest rent in this community?"
          }
          style={{
            flex: 1,
            minWidth: 0,
            background: "var(--inputbg)",
            border: "1px solid var(--line)",
            borderRadius: 10,
            padding: "11px 12px",
            font: "500 13px 'DM Sans',sans-serif",
            color: "var(--text)",
            outline: "none",
            transition: "border-color .25s",
          }}
          focusStyle={{ borderColor: "var(--gold)" }}
        />
        <HoverButton
          onClick={ask}
          style={{
            background: busy ? "var(--inputbg)" : "var(--gold)",
            color: busy ? "var(--faint)" : "#0A1628",
            border: "none",
            borderRadius: 10,
            padding: "11px 18px",
            font: "700 13px 'DM Sans',sans-serif",
            cursor: busy ? "default" : "pointer",
            whiteSpace: "nowrap",
          }}
          hoverStyle={busy ? {} : { transform: "translateY(-1px)" }}
        >
          {busy ? "Asking…" : "Ask"}
        </HoverButton>
      </div>
    </div>
  );
}
