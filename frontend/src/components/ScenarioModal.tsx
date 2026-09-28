"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { getCommunities, predict } from "@/lib/api";
import { SCN, scnModel, type ScenarioState } from "@/lib/demo";
import { driverLabel } from "@/lib/labels";
import type { CommunitySummary, PredictRequest } from "@/lib/types";
import { HoverButton } from "./HoverButton";

const LISTING_BASELINE_FALLBACK = 6.2;

const BUDGET_MIN = 400_000;
const BUDGET_MAX = 15_000_000;
const BUDGET_STEP = 50_000;

function fmtAed(v: number): string {
  if (v >= 1_000_000) return `AED ${(v / 1_000_000).toFixed(v >= 10_000_000 ? 0 : 2)}M`;
  return `AED ${Math.round(v / 1000)}K`;
}

function toRequest(s: ScenarioState): PredictRequest {
  const bedrooms =
    s.scnBeds === "Studio" ? 0 : s.scnBeds === "4+" ? 4 : Number(s.scnBeds);
  return {
    community: s.scnCommunity,
    property_type: s.scnType,
    bedrooms,
    floor_area_sqft: s.scnArea,
    use_extended_model: false,
  };
}

export function ScenarioModal({
  listingYieldPct,
  onClose,
}: {
  listingYieldPct: number | null;
  onClose: () => void;
}) {
  const [s, setS] = useState<ScenarioState>({
    scnCommunity: "Dubai Marina",
    scnType: "Apartment",
    scnBeds: "2",
    scnFinish: "High",
    scnView: "Sea",
    scnArea: 1234,
    scnBudget: 2_000_000,
  });

  const [summaries, setSummaries] = useState<CommunitySummary[] | null>(null);
  useEffect(() => {
    let live = true;
    getCommunities().then((res) => {
      if (live) setSummaries(res);
    });
    return () => {
      live = false;
    };
  }, []);

  const [liveYield, setLiveYield] = useState<number | null>(null);
  const [split, setSplit] = useState<{
    propertyPct: number | null;
    macroCount: number;
    note: string | null;
  } | null>(null);
  const [liveContribs, setLiveContribs] = useState<
    { name: string; value: string; color: string }[] | null
  >(null);
  const reqSeq = useRef(0);

  const demo = useMemo(() => scnModel(s), [s]);

  useEffect(() => {
    const seq = ++reqSeq.current;
    const t = setTimeout(async () => {
      const res = await predict(toRequest(s));
      if (seq !== reqSeq.current) return;
      if (res) {
        setLiveYield(res.predicted_yield_lt);
        const ex = res.shap_explanation;
        setSplit({
          propertyPct: ex?.property_share_pct ?? null,
          macroCount: (ex?.macro_level ?? []).length,
          note: ex?.macro_note ?? null,
        });
        const drivers = ex?.top_drivers ?? [];
        setLiveContribs(
          drivers.slice(0, 6).map((d) => {
            const pos = d.contribution >= 0;
            return {
              name: driverLabel(d.feature),
              value: `${pos ? "+" : "−"}${Math.abs(d.contribution).toFixed(2)}%`,
              color: pos
                ? "var(--green)"
                : d.contribution < 0
                ? "var(--red)"
                : "var(--faint)",
            };
          })
        );
      } else {
        setLiveYield(null);
        setLiveContribs(null);
      }
    }, 350);
    return () => clearTimeout(t);
  }, [s]);

  const total = liveYield ?? demo.total;
  const baseline = listingYieldPct ?? LISTING_BASELINE_FALLBACK;
  const delta = total - baseline;
  const range = `${(total - 0.45).toFixed(1)}% - ${(total + 0.45).toFixed(1)}%`;

  const fmt = (v: number) =>
    (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2) + "%";
  const demoContribs = demo.parts.map((p) => ({
    name: p.name,
    value: p.base ? p.v.toFixed(1) + "%" : fmt(p.v),
    color: p.base
      ? "var(--text)"
      : p.v > 0.001
      ? "var(--green)"
      : p.v < -0.001
      ? "var(--red)"
      : "var(--faint)",
  }));
  const contribs = liveContribs ?? demoContribs;

  const note =
    "Long-term estimate assumes a 12-month contract at community median rent, 30 days vacancy.";

  const priceByCommunity = useMemo(() => {
    const m = new Map<string, number>();
    for (const c of summaries ?? []) {
      if (c.median_sale_price_aed != null) m.set(c.community, c.median_sale_price_aed);
    }
    return m;
  }, [summaries]);

  const allCommunities = useMemo(
    () =>
      summaries?.length
        ? summaries.map((c) => c.community).sort()
        : Object.keys(SCN.community),
    [summaries]
  );

  const affordable = useMemo(
    () =>
      allCommunities.filter((c) => {
        const p = priceByCommunity.get(c);
        return p == null || p <= s.scnBudget;
      }),
    [allCommunities, priceByCommunity, s.scnBudget]
  );

  useEffect(() => {
    if (affordable.length && !affordable.includes(s.scnCommunity)) {
      setS((p) => ({ ...p, scnCommunity: affordable[0] }));
    }
  }, [affordable, s.scnCommunity]);

  const selectedPrice = priceByCommunity.get(s.scnCommunity) ?? null;
  const expectedAnnualRent = (total / 100) * s.scnBudget;

  const fields: { label: string; key: keyof ScenarioState; options: string[] }[] =
    [
      { label: "Community", key: "scnCommunity", options: affordable },
      { label: "Property type", key: "scnType", options: ["Apartment", "Villa", "Townhouse"] },
      { label: "Bedrooms", key: "scnBeds", options: ["Studio", "1", "2", "3", "4+"] },
      { label: "Finish tier", key: "scnFinish", options: ["Basic", "Mid", "High", "Luxury"] },
      { label: "View", key: "scnView", options: ["None", "Community", "Pool", "Skyline", "Sea"] },
    ];

  return (
    <>
      <div
        onClick={onClose}
        style={{
          position: "fixed",
          inset: 0,
          background: "rgba(4,10,20,0.6)",
          backdropFilter: "blur(6px)",
          zIndex: 100,
          animation: "fadeIn .25s both",
        }}
      />
      <div
        style={{
          position: "fixed",
          inset: 0,
          zIndex: 101,
          display: "grid",
          placeItems: "center",
          padding: 24,
          pointerEvents: "none",
          boxSizing: "border-box",
        }}
      >
        <div
          style={{
            pointerEvents: "auto",
            width: 940,
            maxWidth: "100%",
            maxHeight: "calc(100vh - 48px)",
            overflow: "auto",
            background: "var(--card)",
            border: "1px solid var(--linegold)",
            borderRadius: 18,
            boxShadow: "var(--shadow)",
            animation: "fadeUp .4s cubic-bezier(.22,1,.36,1) both",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              padding: "20px 26px",
              borderBottom: "1px solid var(--line)",
              position: "sticky",
              top: 0,
              background: "var(--card)",
              zIndex: 2,
            }}
          >
            <div>
              <div
                style={{
                  font: "700 17px 'DM Sans',sans-serif",
                  letterSpacing: "-0.01em",
                }}
              >
                Yield scenario
              </div>
              <div
                style={{
                  font: "400 12px 'DM Sans',sans-serif",
                  color: "var(--faint)",
                }}
              >
                Adjust the inputs - the model re-predicts live.
                {liveYield == null ? " Demo model, placeholder numbers." : " Live model."}
              </div>
            </div>
            <HoverButton
              onClick={onClose}
              style={{
                width: 34,
                height: 34,
                border: "1px solid var(--line)",
                background: "var(--card2)",
                color: "var(--muted)",
                borderRadius: 9,
                font: "600 15px 'DM Sans',sans-serif",
                cursor: "pointer",
                transition: "all .2s",
              }}
              hoverStyle={{ borderColor: "var(--gold)", color: "var(--text)" }}
            >
              ✕
            </HoverButton>
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "minmax(0,1fr) 320px",
            }}
          >
            <div
              style={{
                padding: "24px 26px",
                display: "flex",
                flexDirection: "column",
                gap: 20,
                borderRight: "1px solid var(--line)",
              }}
            >
              {fields.map((f) => (
                <div
                  key={f.key}
                  style={{ display: "flex", flexDirection: "column", gap: 8 }}
                >
                  <label
                    style={{
                      font: "600 10.5px 'JetBrains Mono',monospace",
                      letterSpacing: "0.14em",
                      textTransform: "uppercase",
                      color: "var(--muted)",
                    }}
                  >
                    {f.label}
                  </label>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 7 }}>
                    {f.options.map((opt) => {
                      const active = s[f.key] === opt;
                      return (
                        <HoverButton
                          key={opt}
                          onClick={() => setS((p) => ({ ...p, [f.key]: opt }))}
                          style={{
                            background: active ? "var(--gold)" : "var(--inputbg)",
                            color: active ? "#0A1628" : "var(--muted)",
                            border: `1px solid ${active ? "var(--gold)" : "var(--line)"}`,
                            borderRadius: 9,
                            padding: "9px 15px",
                            font: "600 12.5px 'DM Sans',sans-serif",
                            cursor: "pointer",
                            transition: "all .2s",
                            whiteSpace: "nowrap",
                          }}
                          hoverStyle={active ? {} : { borderColor: "var(--gold)" }}
                        >
                          {opt}
                        </HoverButton>
                      );
                    })}
                  </div>
                </div>
              ))}
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "baseline",
                  }}
                >
                  <label
                    style={{
                      font: "600 10.5px 'JetBrains Mono',monospace",
                      letterSpacing: "0.14em",
                      textTransform: "uppercase",
                      color: "var(--muted)",
                    }}
                  >
                    Floor area
                  </label>
                  <span
                    style={{
                      font: "700 13px 'JetBrains Mono',monospace",
                      color: "var(--gold)",
                    }}
                  >
                    {s.scnArea.toLocaleString()} sqft
                  </span>
                </div>
                <input
                  type="range"
                  min={400}
                  max={4000}
                  step={50}
                  value={s.scnArea}
                  onChange={(e) =>
                    setS((p) => ({ ...p, scnArea: Number(e.target.value) }))
                  }
                  style={{
                    width: "100%",
                    accentColor: "var(--gold)",
                    cursor: "pointer",
                  }}
                />
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    font: "500 10px 'JetBrains Mono',monospace",
                    color: "var(--faint)",
                  }}
                >
                  <span>400 sqft</span>
                  <span>4,000 sqft</span>
                </div>
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "baseline",
                  }}
                >
                  <label
                    style={{
                      font: "600 10.5px 'JetBrains Mono',monospace",
                      letterSpacing: "0.14em",
                      textTransform: "uppercase",
                      color: "var(--muted)",
                    }}
                  >
                    Budget
                  </label>
                  <span
                    style={{
                      font: "700 13px 'JetBrains Mono',monospace",
                      color: "var(--gold)",
                    }}
                  >
                    {fmtAed(s.scnBudget)}
                  </span>
                </div>
                <input
                  type="range"
                  min={BUDGET_MIN}
                  max={BUDGET_MAX}
                  step={BUDGET_STEP}
                  value={s.scnBudget}
                  onChange={(e) =>
                    setS((p) => ({ ...p, scnBudget: Number(e.target.value) }))
                  }
                  style={{
                    width: "100%",
                    accentColor: "var(--gold)",
                    cursor: "pointer",
                  }}
                />
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    font: "500 10px 'JetBrains Mono',monospace",
                    color: "var(--faint)",
                  }}
                >
                  <span>
                    {summaries?.length
                      ? `${affordable.length}/${allCommunities.length} communities in budget`
                      : `${fmtAed(BUDGET_MIN)} min`}
                  </span>
                  <span>
                    {selectedPrice != null
                      ? `median here ${fmtAed(selectedPrice)}`
                      : "median price n/a"}
                  </span>
                </div>
              </div>
            </div>

            <div
              style={{
                padding: "24px 26px",
                display: "flex",
                flexDirection: "column",
                gap: 18,
                background: "var(--card2)",
              }}
            >
              <div style={{ textAlign: "center", padding: "18px 0 6px" }}>
                <div
                  style={{
                    font: "600 10px 'JetBrains Mono',monospace",
                    letterSpacing: "0.2em",
                    color: "var(--muted)",
                    textTransform: "uppercase",
                    marginBottom: 8,
                  }}
                >
                  Predicted Yield
                </div>
                <div
                  style={{
                    font: "700 44px 'JetBrains Mono',monospace",
                    letterSpacing: "-0.04em",
                    color: "var(--gold)",
                    transition: "color .3s",
                  }}
                >
                  {total.toFixed(1)}
                  <span style={{ fontSize: 22 }}>%</span>
                </div>
                <div
                  style={{
                    font: "500 11.5px 'JetBrains Mono',monospace",
                    color: "var(--muted)",
                    marginTop: 3,
                  }}
                >
                  range {range}
                </div>
                <div
                  style={{
                    marginTop: 10,
                    font: "500 11.5px 'JetBrains Mono',monospace",
                    color: "var(--muted)",
                  }}
                >
                  ≈{" "}
                  <span style={{ color: "var(--text)", fontWeight: 700 }}>
                    {Math.round(expectedAnnualRent).toLocaleString()} AED
                  </span>{" "}
                  a year on {fmtAed(s.scnBudget)}
                </div>
                <div
                  style={{
                    marginTop: 12,
                    display: "inline-block",
                    font: "600 11.5px 'JetBrains Mono',monospace",
                    color: delta >= 0 ? "var(--green)" : "var(--red)",
                    background: "var(--inputbg)",
                    border: "1px solid var(--line)",
                    borderRadius: 999,
                    padding: "5px 12px",
                  }}
                >
                  {delta >= 0 ? "+" : "−"}
                  {Math.abs(delta).toFixed(1)}% vs this listing (
                  {baseline.toFixed(1)}%)
                </div>
              </div>
              <div
                style={{
                  borderTop: "1px solid var(--line)",
                  paddingTop: 16,
                  display: "flex",
                  flexDirection: "column",
                  gap: 10,
                }}
              >
                <div
                  style={{
                    font: "600 10px 'JetBrains Mono',monospace",
                    letterSpacing: "0.16em",
                    color: "var(--muted)",
                    textTransform: "uppercase",
                  }}
                >
                  Contributions
                </div>
                {contribs.map((c, i) => (
                  <div
                    key={`${c.name}-${i}`}
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "baseline",
                      gap: 10,
                    }}
                  >
                    <span
                      style={{
                        font: "500 12.5px 'DM Sans',sans-serif",
                        color: "var(--text)",
                      }}
                    >
                      {c.name}
                    </span>
                    <span
                      style={{
                        font: "700 12px 'JetBrains Mono',monospace",
                        color: c.color,
                      }}
                    >
                      {c.value}
                    </span>
                  </div>
                ))}

                {split && liveContribs && (
                  <div
                    style={{
                      marginTop: 14,
                      paddingTop: 12,
                      borderTop: "1px solid var(--line)",
                    }}
                  >
                    <div
                      style={{
                        font: "600 9.5px 'JetBrains Mono',monospace",
                        letterSpacing: "0.1em",
                        color: "var(--faint)",
                        marginBottom: 8,
                      }}
                    >
                      TWO-LEVEL DECOMPOSITION
                    </div>
                    <div
                      style={{
                        display: "flex",
                        height: 8,
                        borderRadius: 4,
                        overflow: "hidden",
                        background: "var(--line)",
                      }}
                    >
                      <div
                        style={{
                          width: `${split.propertyPct ?? 100}%`,
                          background: "var(--gold)",
                        }}
                        title="Property-level contribution"
                      />
                      <div
                        style={{
                          width: `${100 - (split.propertyPct ?? 100)}%`,
                          background: "var(--amber)",
                        }}
                        title="Macro / environmental contribution"
                      />
                    </div>
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        marginTop: 7,
                        font: "600 10.5px 'JetBrains Mono',monospace",
                      }}
                    >
                      <span style={{ color: "var(--gold)" }}>
                        Property {(split.propertyPct ?? 100).toFixed(0)}%
                      </span>
                      <span
                        style={{
                          color: split.macroCount ? "var(--amber)" : "var(--faint)",
                        }}
                      >
                        Macro {(100 - (split.propertyPct ?? 100)).toFixed(0)}%
                      </span>
                    </div>
                    {split.note && (
                      <p
                        style={{
                          font: "500 10px/1.55 'JetBrains Mono',monospace",
                          color: "var(--faint)",
                          margin: "9px 0 0",
                        }}
                      >
                        {split.note}
                      </p>
                    )}
                  </div>
                )}
              </div>
              <div
                style={{
                  marginTop: "auto",
                  font: "400 11.5px/1.5 'DM Sans',sans-serif",
                  textAlign: "justify",
                  color: "var(--faint)",
                  borderTop: "1px solid var(--line)",
                  paddingTop: 14,
                  textWrap: "pretty",
                }}
              >
                {note}
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
