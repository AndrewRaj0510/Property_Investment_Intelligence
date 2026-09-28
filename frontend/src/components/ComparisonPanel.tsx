"use client";

import type { ModelComparison } from "@/lib/types";


const MONO = "'JetBrains Mono',monospace";

export function ComparisonPanel({
  cmp,
  loading = false,
}: {
  cmp: ModelComparison | null;
  loading?: boolean;
}) {
  if (loading && !cmp) {
    return (
      <div
        style={{
          marginTop: 18,
          padding: "14px 16px",
          border: "1px dashed var(--line)",
          borderRadius: 12,
          font: `500 10.5px/1.6 ${MONO}`,
          color: "var(--faint)",
        }}
      >
        Comparing the baseline and extended models…
      </div>
    );
  }
  if (!cmp) return null;

  if (!cmp.available || !cmp.phase1 || !cmp.phase2 || !cmp.delta) {
    return (
      <div
        style={{
          marginTop: 18,
          padding: "14px 16px",
          border: "1px dashed var(--line)",
          borderRadius: 12,
          font: `500 10.5px/1.6 ${MONO}`,
          color: "var(--faint)",
        }}
      >
        {cmp.detail ?? "Model comparison unavailable."}
      </div>
    );
  }

  const { phase1: p1, phase2: p2, delta } = cmp;
  const dR2 = delta.r_squared ?? 0;

  const improved = dR2 > 0.005;
  const worsened = dR2 < -0.005;
  const verdictColor = improved
    ? "var(--green)"
    : worsened
      ? "var(--red)"
      : "var(--amber)";
  const verdictText = improved
    ? "External factors improved accuracy"
    : worsened
      ? "External factors did not improve accuracy"
      : "External factors made no measurable difference";

  const fmt = (v: number | null) => (v == null ? "n/a" : v.toFixed(4));
  const signed = (v: number | null) =>
    v == null ? "n/a" : `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(4)}`;

  const rows = [
    {
      label: "R²",
      a: p1.r_squared,
      b: p2.r_squared,
      d: delta.r_squared,
      lowerIsBetter: false,
    },
    { label: "MAE", a: p1.mae, b: p2.mae, d: delta.mae, lowerIsBetter: true },
    { label: "RMSE", a: p1.rmse, b: p2.rmse, d: delta.rmse, lowerIsBetter: true },
  ];

  const maxShap = Math.max(
    ...cmp.macro_drivers.map((d) => d.mean_abs_shap),
    0.0001
  );

  const th: React.CSSProperties = { padding: "6px 8px", fontWeight: 500 };
  const td: React.CSSProperties = { padding: "7px 8px", textAlign: "right" };

  return (
    <div style={{ marginTop: 22 }}>
      <h3 style={{ font: `600 13px ${MONO}`, color: "var(--text)", margin: "0 0 4px" }}>
        Do external factors improve the prediction?
      </h3>
      <div
        style={{
          font: `500 10px ${MONO}`,
          color: "var(--faint)",
          letterSpacing: "0.08em",
          marginBottom: 14,
        }}
      >
        PHASE 1 (BASELINE) VS PHASE 2 (BASELINE + EXTERNAL FACTORS) · 5-FOLD CV
      </div>

      <div
        style={{
          display: "inline-block",
          font: `700 11px ${MONO}`,
          color: verdictColor,
          border: `1px solid ${verdictColor}`,
          borderRadius: 999,
          padding: "5px 12px",
          marginBottom: 14,
        }}
      >
        {verdictText} ({signed(delta.r_squared)} R²)
      </div>

      <div style={{ overflowX: "auto" }}>
        <table
          style={{
            width: "100%",
            borderCollapse: "collapse",
            font: `500 11px ${MONO}`,
            minWidth: 420,
          }}
        >
          <thead>
            <tr style={{ color: "var(--faint)", textAlign: "right" }}>
              <th style={{ ...th, textAlign: "left" }}>METRIC</th>
              <th style={th}>PHASE 1</th>
              <th style={th}>PHASE 2</th>
              <th style={th}>Δ</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const neutral = r.d != null && Math.abs(r.d) < 0.0005;
              const better =
                r.d == null ? null : r.lowerIsBetter ? r.d < 0 : r.d > 0;
              return (
                <tr key={r.label} style={{ borderTop: "1px solid var(--line)" }}>
                  <td style={{ ...td, textAlign: "left", color: "var(--text)" }}>
                    {r.label}
                  </td>
                  <td style={{ ...td, color: "var(--text)" }}>{fmt(r.a)}</td>
                  <td style={{ ...td, color: "var(--text)" }}>{fmt(r.b)}</td>
                  <td
                    style={{
                      ...td,
                      color: neutral
                        ? "var(--faint)"
                        : better
                          ? "var(--green)"
                          : "var(--red)",
                    }}
                  >
                    {signed(r.d)}
                  </td>
                </tr>
              );
            })}
            <tr style={{ borderTop: "1px solid var(--line)" }}>
              <td style={{ ...td, textAlign: "left", color: "var(--faint)" }}>
                FEATURES
              </td>
              <td style={{ ...td, color: "var(--faint)" }}>{p1.n_features}</td>
              <td style={{ ...td, color: "var(--faint)" }}>{p2.n_features}</td>
              <td style={{ ...td, color: "var(--faint)" }}>
                +{p2.n_features - p1.n_features}
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      {cmp.macro_attribution_pct != null && (
        <div style={{ marginTop: 18 }}>
          <div
            style={{
              font: `500 10px ${MONO}`,
              color: "var(--faint)",
              letterSpacing: "0.08em",
              marginBottom: 9,
            }}
          >
            YET THEY CARRY{" "}
            <strong style={{ color: "var(--amber)" }}>
              {cmp.macro_attribution_pct.toFixed(1)}%
            </strong>{" "}
            OF PHASE 2 TOTAL ATTRIBUTION ({cmp.macro_feature_count} FEATURES)
          </div>

          {cmp.macro_drivers.slice(0, 6).map((d) => (
            <div
              key={d.feature}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                marginBottom: 5,
              }}
            >
              <div
                title={d.feature}
                style={{
                  width: 132,
                  flexShrink: 0,
                  font: `500 10.5px ${MONO}`,
                  color: "var(--text)",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
              >
                {d.feature}
              </div>
              <div
                style={{
                  flex: 1,
                  height: 6,
                  background: "var(--inputbg)",
                  borderRadius: 999,
                  overflow: "hidden",
                  minWidth: 40,
                }}
              >
                <div
                  style={{
                    width: `${Math.max(2, (d.mean_abs_shap / maxShap) * 100)}%`,
                    height: "100%",
                    background: "var(--amber)",
                    borderRadius: 999,
                  }}
                />
              </div>
              <div
                style={{
                  width: 58,
                  flexShrink: 0,
                  textAlign: "right",
                  font: `500 10px ${MONO}`,
                  color: "var(--faint)",
                }}
              >
                {d.mean_abs_shap.toFixed(4)}
              </div>
            </div>
          ))}
        </div>
      )}

      {cmp.interpretation && (
        <p
          style={{
            font: `500 10.5px/1.7 ${MONO}`,
            color: "var(--faint)",
            margin: "16px 0 0",
            paddingTop: 14,
            borderTop: "1px solid var(--line)",
          }}
        >
          {cmp.interpretation}
        </p>
      )}
    </div>
  );
}
