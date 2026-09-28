"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { getPriceHistory } from "@/lib/api";
import type { PriceHistory, PricePoint } from "@/lib/types";


const RANGES = [
  { label: "3Y", quarters: 12 },
  { label: "5Y", quarters: 20 },
  { label: "Max", quarters: 0 },
] as const;

const CHART_H = 200;
const PAD = { top: 16, right: 14, bottom: 26, left: 52 };

const aed = (n: number) => `AED ${Math.round(n).toLocaleString()}`;

export function PriceTrendChart({
  area,
  propertyType,
  onData,
}: {
  area?: string | null;
  propertyType?: string | null;
  onData?: (data: PriceHistory | null) => void;
}) {
  const [data, setData] = useState<PriceHistory | null>(null);
  const [state, setState] = useState<"loading" | "ready" | "empty">("loading");
  const [range, setRange] = useState<number>(20);

  const onDataRef = useRef(onData);
  onDataRef.current = onData;

  useEffect(() => {
    if (!area || !propertyType) {
      setState("empty");
      onDataRef.current?.(null);
      return;
    }
    let cancelled = false;
    setState("loading");
    getPriceHistory(area, propertyType).then((res) => {
      if (cancelled) return;
      if (!res || !res.available || res.points.length < 2) {
        setData(res);
        setState("empty");
        onDataRef.current?.(null);
        return;
      }
      setData(res);
      setState("ready");
      onDataRef.current?.(res);
    });
    return () => {
      cancelled = true;
    };
  }, [area, propertyType]);

  const points = useMemo(() => {
    if (!data) return [];
    return range > 0 ? data.points.slice(-range) : data.points;
  }, [data, range]);

  if (state === "loading") {
    return (
      <Card>
        <Head title="Price history" />
        <div style={skeleton} />
      </Card>
    );
  }

  if (state === "empty" || points.length < 2) {
    return (
      <Card>
        <Head title="Price history" />
        <p style={emptyText}>
          {data?.detail ??
            "No transacted price history could be loaded for this community and property type, so no chart is drawn."}
        </p>
      </Card>
    );
  }

  return <Chart data={data!} points={points} range={range} onRange={setRange} />;
}

function Chart({
  data,
  points,
  range,
  onRange,
}: {
  data: PriceHistory;
  points: PricePoint[];
  range: number;
  onRange: (q: number) => void;
}) {
  const wrap = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(560);
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      setW(Math.max(280, Math.round(entry.contentRect.width)));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const last = points[points.length - 1];
  const first = points[0];
  const windowChange =
    first.median_aed_per_sqft > 0
      ? (last.median_aed_per_sqft / first.median_aed_per_sqft - 1) * 100
      : null;

  const up = (windowChange ?? 0) >= 0;
  const accent = windowChange == null ? "var(--gold)" : up ? "var(--green)" : "var(--red)";

  const innerW = w - PAD.left - PAD.right;
  const innerH = CHART_H - PAD.top - PAD.bottom;

  const values = points.flatMap((p) =>
    p.ma4_aed_per_sqft != null
      ? [p.median_aed_per_sqft, p.ma4_aed_per_sqft]
      : [p.median_aed_per_sqft]
  );
  const rawMin = Math.min(...values);
  const rawMax = Math.max(...values);
  const span = Math.max(rawMax - rawMin, rawMax * 0.05 || 1);
  const yMin = rawMin - span * 0.12;
  const yMax = rawMax + span * 0.12;

  const x = (i: number) =>
    PAD.left + (points.length === 1 ? innerW / 2 : (i / (points.length - 1)) * innerW);
  const y = (v: number) =>
    PAD.top + innerH - ((v - yMin) / (yMax - yMin)) * innerH;

  const line = points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${x(i).toFixed(1)} ${y(p.median_aed_per_sqft).toFixed(1)}`)
    .join(" ");
  const areaPath = `${line} L ${x(points.length - 1).toFixed(1)} ${(PAD.top + innerH).toFixed(1)} L ${x(0).toFixed(1)} ${(PAD.top + innerH).toFixed(1)} Z`;

  const maPoints = points
    .map((p, i) => ({ p, i }))
    .filter(({ p }) => p.ma4_aed_per_sqft != null);
  const maLine = maPoints
    .map(({ p, i }, k) => `${k === 0 ? "M" : "L"} ${x(i).toFixed(1)} ${y(p.ma4_aed_per_sqft!).toFixed(1)}`)
    .join(" ");

  const yearTicks = points
    .map((p, i) => ({ p, i }))
    .filter(({ p, i }) => i === 0 || p.year !== points[i - 1].year);

  const active = hover != null ? (points[hover] ?? null) : null;
  const gradId = "priceTrendFill";

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const box = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - box.left;
    const ratio = (px - PAD.left) / innerW;
    const i = Math.round(ratio * (points.length - 1));
    setHover(Math.max(0, Math.min(points.length - 1, i)));
  };

  return (
    <Card>
      <div
        style={{
          display: "flex",
          alignItems: "flex-start",
          justifyContent: "space-between",
          gap: 12,
          flexWrap: "wrap",
          marginBottom: 14,
        }}
      >
        <div style={{ minWidth: 0 }}>
          <Head
            title="Price history"
            sub={`${data.area}${
              data.area.toLowerCase() !== data.area_requested.toLowerCase()
                ? ` (listed as ${data.area_requested})`
                : ""
            } · ${data.property_type}s`}
          />
          <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginTop: 10, flexWrap: "wrap" }}>
            <span
              style={{
                font: "700 25px 'JetBrains Mono',monospace",
                letterSpacing: "-0.03em",
                color: "var(--text)",
              }}
            >
              {aed(last.median_aed_per_sqft)}
            </span>
            <span style={{ font: "500 11px 'JetBrains Mono',monospace", color: "var(--faint)" }}>
              /sqft · {last.period}
            </span>
          </div>
          {windowChange != null && (
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 8 }}>
              <span
                style={{
                  font: "700 12.5px 'JetBrains Mono',monospace",
                  color: accent,
                  background: up ? "var(--greensoft)" : "rgba(229,84,75,0.12)",
                  border: `1px solid ${accent}`,
                  borderRadius: 7,
                  padding: "3px 8px",
                }}
              >
                {up ? "▲" : "▼"} {Math.abs(windowChange).toFixed(1)}%
              </span>
              <span style={{ font: "400 12px 'DM Sans',sans-serif", color: "var(--muted)" }}>
                since {first.period}
              </span>
            </div>
          )}
        </div>

        <div style={{ display: "flex", gap: 5, flex: "none" }}>
          {RANGES.map((r) => {
            const on = range === r.quarters;
            return (
              <button
                key={r.label}
                onClick={() => {
                  setHover(null);
                  onRange(r.quarters);
                }}
                style={{
                  font: "600 10.5px 'JetBrains Mono',monospace",
                  letterSpacing: "0.06em",
                  color: on ? "var(--text)" : "var(--faint)",
                  background: on ? "var(--inputbg)" : "transparent",
                  border: `1px solid ${on ? "var(--linegold)" : "var(--line)"}`,
                  borderRadius: 7,
                  padding: "5px 9px",
                  cursor: "pointer",
                  transition: "all .2s",
                }}
              >
                {r.label}
              </button>
            );
          })}
        </div>
      </div>

      <div ref={wrap} style={{ position: "relative", width: "100%" }}>
        <svg
          width={w}
          height={CHART_H}
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
          style={{ display: "block", cursor: "crosshair", touchAction: "none" }}
        >
          <defs>
            <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={accent} stopOpacity={0.24} />
              <stop offset="100%" stopColor={accent} stopOpacity={0} />
            </linearGradient>
          </defs>

          {[0, 0.5, 1].map((t) => {
            const v = yMax - t * (yMax - yMin);
            const gy = PAD.top + t * innerH;
            return (
              <g key={t}>
                <line
                  x1={PAD.left}
                  x2={w - PAD.right}
                  y1={gy}
                  y2={gy}
                  stroke="var(--line)"
                  strokeWidth={1}
                />
                <text
                  x={PAD.left - 8}
                  y={gy + 3.5}
                  textAnchor="end"
                  style={{ font: "500 9.5px 'JetBrains Mono',monospace", fill: "var(--faint)" }}
                >
                  {Math.round(v).toLocaleString()}
                </text>
              </g>
            );
          })}

          <path d={areaPath} fill={`url(#${gradId})`} />
          <path
            d={line}
            fill="none"
            stroke={accent}
            strokeWidth={1.6}
            strokeOpacity={0.55}
            strokeLinejoin="round"
            strokeLinecap="round"
          />
          {maLine && (
            <path
              d={maLine}
              fill="none"
              stroke={accent}
              strokeWidth={2.4}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          )}

          {yearTicks.map(({ p, i }) => (
            <text
              key={p.period}
              x={x(i)}
              y={CHART_H - 8}
              textAnchor={i === 0 ? "start" : "middle"}
              style={{ font: "500 9.5px 'JetBrains Mono',monospace", fill: "var(--faint)" }}
            >
              {p.year}
            </text>
          ))}

          {active && hover != null && (
            <g>
              <line
                x1={x(hover)}
                x2={x(hover)}
                y1={PAD.top}
                y2={PAD.top + innerH}
                stroke="var(--faint)"
                strokeWidth={1}
                strokeDasharray="3 3"
              />
              <circle
                cx={x(hover)}
                cy={y(active.median_aed_per_sqft)}
                r={4}
                fill="var(--card)"
                stroke={accent}
                strokeWidth={2}
              />
            </g>
          )}

          <circle
            cx={x(points.length - 1)}
            cy={y(last.median_aed_per_sqft)}
            r={3.5}
            fill={accent}
          />
        </svg>

        {active && hover != null && (
          <div
            style={{
              position: "absolute",
              left: Math.min(Math.max(x(hover) - 70, 0), Math.max(w - 150, 0)),
              top: 4,
              pointerEvents: "none",
              background: "var(--card)",
              border: "1px solid var(--linegold)",
              borderRadius: 9,
              padding: "8px 11px",
              boxShadow: "var(--shadow)",
              minWidth: 128,
            }}
          >
            <div style={{ font: "600 10px 'JetBrains Mono',monospace", color: "var(--faint)", letterSpacing: "0.1em" }}>
              {active.period}
            </div>
            <div style={{ font: "700 13px 'JetBrains Mono',monospace", color: "var(--text)", marginTop: 3 }}>
              {aed(active.median_aed_per_sqft)}/sqft
            </div>
            <div style={{ font: "400 10.5px 'DM Sans',sans-serif", color: "var(--muted)", marginTop: 2 }}>
              median of {active.n_sales} sale{active.n_sales === 1 ? "" : "s"}
            </div>
          </div>
        )}
      </div>

      <Legend accent={accent} hasMa={maPoints.length > 0} />

      <p style={readText}>
        {directionSentence(data)}
      </p>
      <p style={basisText}>
        {data.basis} Based on {data.total_sales.toLocaleString()} registered sales
        {data.coverage_to ? `, recorded up to ${data.coverage_to}` : ""}. {data.note}
      </p>
    </Card>
  );
}

function directionSentence(d: PriceHistory): string {
  const where = `${d.property_type === "villa" ? "Villas" : "Apartments"} in ${d.area}`;
  if (d.yoy_change_pct == null || !d.direction) {
    return `${where} do not have enough quarters of recorded sales to describe a trend, so the line above is shown without one.`;
  }
  const move = Math.abs(d.yoy_change_pct).toFixed(1);
  if (d.direction === "flat") {
    return `${where} sold for about the same price over the last four quarters as in the four before them (${move}% apart), so prices here have been broadly flat.`;
  }
  const verb = d.direction === "rising" ? "higher" : "lower";
  return `${where} sold for ${move}% ${verb} per square foot over the last four recorded quarters than in the four before them. That is what has already happened, not a prediction of what comes next.`;
}


function Card({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        background: "var(--card)",
        border: "1px solid var(--line)",
        borderRadius: 16,
        padding: "22px 24px",
        boxShadow: "var(--shadow)",
        animation: "fadeUp .6s .15s cubic-bezier(.22,1,.36,1) both",
        minWidth: 0,
      }}
    >
      {children}
    </div>
  );
}

function Head({ title, sub }: { title: string; sub?: string }) {
  return (
    <div>
      <div style={{ font: "600 13px 'DM Sans',sans-serif" }}>{title}</div>
      {sub && (
        <div
          style={{
            font: "500 10.5px 'JetBrains Mono',monospace",
            color: "var(--faint)",
            letterSpacing: "0.04em",
            marginTop: 3,
          }}
        >
          {sub}
        </div>
      )}
    </div>
  );
}

function Legend({ accent, hasMa }: { accent: string; hasMa: boolean }) {
  return (
    <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginTop: 10 }}>
      <LegendItem color={accent} faded label="Quarterly median" />
      {hasMa && <LegendItem color={accent} label="4-quarter average" />}
    </div>
  );
}

function LegendItem({
  color,
  label,
  faded,
}: {
  color: string;
  label: string;
  faded?: boolean;
}) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
      <span
        style={{
          width: 14,
          height: faded ? 1.6 : 2.6,
          background: color,
          opacity: faded ? 0.55 : 1,
          borderRadius: 2,
        }}
      />
      <span style={{ font: "400 11px 'DM Sans',sans-serif", color: "var(--muted)" }}>
        {label}
      </span>
    </span>
  );
}

const skeleton: React.CSSProperties = {
  height: CHART_H,
  marginTop: 14,
  borderRadius: 10,
  background:
    "linear-gradient(90deg, var(--inputbg) 25%, var(--card2) 50%, var(--inputbg) 75%)",
  backgroundSize: "200% 100%",
  animation: "shimmer 1.4s infinite",
};

const emptyText: React.CSSProperties = {
  margin: "14px 0 0",
  font: "400 12.5px/1.6 'DM Sans',sans-serif",
  textAlign: "justify",
  color: "var(--faint)",
  textWrap: "pretty",
};

const readText: React.CSSProperties = {
  margin: "14px 0 0",
  font: "400 13px/1.6 'DM Sans',sans-serif",
  textAlign: "justify",
  color: "var(--text)",
  textWrap: "pretty",
};

const basisText: React.CSSProperties = {
  margin: "8px 0 0",
  font: "400 11px/1.55 'DM Sans',sans-serif",
  textAlign: "justify",
  color: "var(--faint)",
  textWrap: "pretty",
};
