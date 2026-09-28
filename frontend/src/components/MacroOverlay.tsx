"use client";

import { useEffect, useState } from "react";
import type {
  MacroSnapshot,
  ModelComparison,
  SeasonalCalendar,
} from "@/lib/types";
import { getModelComparison, getSeasonalCalendar } from "@/lib/api";
import { ComparisonPanel } from "./ComparisonPanel";


function fmtMonth(iso?: string | null): string {
  if (!iso) return "";
  const [y, m] = iso.split("-");
  const names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const idx = Number(m) - 1;
  return names[idx] ? `${names[idx]} ${y}` : iso.slice(0, 7);
}

function Metric({
  label,
  value,
  unit,
  asOf,
  context,
  unavailable,
}: {
  label: string;
  value?: number | null;
  unit?: string;
  asOf?: string | null;
  context?: string;
  unavailable?: string;
}) {
  const missing = value == null;
  return (
    <div
      style={{
        padding: "12px 14px",
        border: "1px solid var(--line)",
        borderRadius: 10,
        background: "var(--card)",
        minWidth: 0,
      }}
    >
      <div
        style={{
          font: "600 9.5px 'JetBrains Mono',monospace",
          letterSpacing: "0.1em",
          color: "var(--faint)",
          marginBottom: 6,
        }}
      >
        {label}
      </div>
      <div
        style={{
          font: "600 20px 'JetBrains Mono',monospace",
          color: missing ? "var(--faint)" : "var(--text)",
          lineHeight: 1.1,
        }}
        title={missing ? unavailable : undefined}
      >
        {missing ? "n/a" : `${value.toFixed(2)}${unit ?? ""}`}
      </div>
      <div
        style={{
          font: "500 10px 'JetBrains Mono',monospace",
          color: "var(--faint)",
          marginTop: 5,
        }}
      >
        {missing ? unavailable ?? "no data" : context ?? (asOf ? `as of ${fmtMonth(asOf)}` : "")}
      </div>
    </div>
  );
}

function SeasonalBars({ cal }: { cal: SeasonalCalendar }) {
  const vals = cal.months
    .map((m) => m.thermal_index_relative)
    .filter((v): v is number => v != null);
  if (!vals.length) return null;
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  const span = max - min || 1;

  return (
    <div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(12, minmax(0, 1fr))",
          gap: 5,
          alignItems: "end",
          height: 130,
        }}
      >
        {cal.months.map((m) => {
          const v = m.thermal_index_relative;
          const h = v == null ? 0 : 16 + ((v - min) / span) * 100;
          const hot = (v ?? 100) >= 100;
          return (
            <div key={m.month_of_year} style={{ display: "grid", gap: 5, justifyItems: "center" }}>
              <span
                style={{
                  font: "600 9px 'JetBrains Mono',monospace",
                  color: "var(--faint)",
                }}
              >
                {v == null ? "n/a" : Math.round(v)}
              </span>
              <div
                style={{
                  width: "100%",
                  height: h,
                  borderRadius: 4,
                  background: hot ? "var(--amber)" : "var(--green)",
                  opacity: v == null ? 0.15 : 0.55 + 0.45 * Math.abs((v ?? 100) - 100) / (span / 2 || 1) * 0.5,
                }}
                title={
                  m.avg_thermal_index != null
                    ? `${m.month_name}: index ${m.avg_thermal_index}, max temp ${m.avg_temp_max_c}°C, ${m.months_observed} yrs observed`
                    : `${m.month_name}: no data`
                }
              />
              <span
                style={{
                  font: "600 9.5px 'JetBrains Mono',monospace",
                  color: "var(--muted)",
                }}
              >
                {m.month_name.slice(0, 3)}
              </span>
            </div>
          );
        })}
      </div>
      <p
        style={{
          font: "500 10.5px/1.6 'JetBrains Mono',monospace",
          color: "var(--faint)",
          margin: "14px 0 0",
        }}
      >
        {cal.basis}
      </p>
      <p
        style={{
          font: "500 10.5px/1.6 'JetBrains Mono',monospace",
          color: "var(--amber)",
          margin: "8px 0 0",
        }}
      >
        {cal.note}
      </p>
    </div>
  );
}

export function MacroOverlay({
  macro,
  onClose,
}: {
  macro: MacroSnapshot | null;
  onClose: () => void;
}) {
  const [cal, setCal] = useState<SeasonalCalendar | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [cmp, setCmp] = useState<ModelComparison | null>(null);
  const [cmpLoading, setCmpLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    getSeasonalCalendar()
      .then((c) => {
        if (alive) {
          setCal(c);
          setLoaded(true);
        }
      })
      .catch(() => alive && setLoaded(true));
    getModelComparison()
      .then((c) => {
        if (alive) setCmp(c);
      })
      .finally(() => {
        if (alive) setCmpLoading(false);
      });
    return () => {
      alive = false;
    };
  }, []);

  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,.45)",
        display: "grid",
        placeItems: "center",
        zIndex: 100,
        padding: 20,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: "min(860px, 100%)",
          maxHeight: "88vh",
          overflowY: "auto",
          background: "var(--surface)",
          border: "1px solid var(--line)",
          borderRadius: 14,
          padding: 26,
          boxShadow: "var(--shadow)",
        }}
      >
        <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
          <h2
            style={{
              font: "600 16px 'JetBrains Mono',monospace",
              color: "var(--text)",
              margin: 0,
            }}
          >
            Market context
          </h2>
          <span
            style={{
              font: "500 10px 'JetBrains Mono',monospace",
              color: "var(--faint)",
              letterSpacing: "0.08em",
            }}
          >
            MACROECONOMIC &amp; ENVIRONMENTAL OVERLAY
          </span>
          <button
            onClick={onClose}
            style={{
              marginLeft: "auto",
              background: "none",
              border: "1px solid var(--line)",
              borderRadius: 8,
              color: "var(--muted)",
              font: "600 11px 'JetBrains Mono',monospace",
              padding: "6px 12px",
              cursor: "pointer",
            }}
          >
            CLOSE
          </button>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
            gap: 10,
            marginTop: 18,
          }}
        >
          <Metric
            label="EIBOR 3M"
            value={macro?.eibor_3m}
            unit="%"
            asOf={macro?.eibor_as_of}
            unavailable="not ingested"
          />
          <Metric
            label="BRENT"
            value={macro?.brent_crude_usd}
            unit=" $"
            asOf={macro?.brent_as_of}
            unavailable="not ingested"
          />
          <Metric
            label="MAX TEMP"
            value={macro?.temp_max_c}
            unit="°C"
            context={macro?.month ? `as of ${fmtMonth(macro.month)}` : undefined}
            unavailable="not ingested"
          />
          <Metric
            label="THERMAL INDEX"
            value={macro?.thermal_index}
            context="apparent temp, °C"
            unavailable="not ingested"
          />
        </div>

        <p
          style={{
            font: "500 10.5px/1.6 'JetBrains Mono',monospace",
            color: "var(--faint)",
            margin: "14px 0 0",
          }}
        >
          These series are ingested and shown as market context. They are{" "}
          <strong style={{ color: "var(--amber)" }}>
            not inputs to the model that scores your listing
          </strong>{" "}
          - that is the Phase-1 baseline. They <em>are</em> inputs to the Phase-2
          model compared below.
        </p>

        <ComparisonPanel cmp={cmp} loading={cmpLoading} />

        <div
          style={{
            height: 1,
            background: "var(--line)",
            margin: "22px 0 20px",
          }}
        />

        <h3
          style={{
            font: "600 13px 'JetBrains Mono',monospace",
            color: "var(--text)",
            margin: "0 0 4px",
          }}
        >
          Seasonal calendar
        </h3>
        <div
          style={{
            font: "500 10px 'JetBrains Mono',monospace",
            color: "var(--faint)",
            letterSpacing: "0.08em",
            marginBottom: 16,
          }}
        >
          THERMAL DISCOMFORT BY MONTH · 100 = ANNUAL MEAN
        </div>

        {!loaded && (
          <div style={{ font: "500 11px 'JetBrains Mono',monospace", color: "var(--faint)" }}>
            Loading…
          </div>
        )}
        {loaded && (!cal || !cal.months.length) && (
          <div style={{ font: "500 11px 'JetBrains Mono',monospace", color: "var(--faint)" }}>
            - Seasonal data unavailable.
          </div>
        )}
        {loaded && cal && cal.months.length > 0 && <SeasonalBars cal={cal} />}
      </div>
    </div>
  );
}
