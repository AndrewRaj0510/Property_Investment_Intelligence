"use client";

import { LOAD_STEPS } from "@/lib/demo";

export function LoadingScreen({ loadStep }: { loadStep: number }) {
  const cur = LOAD_STEPS[Math.min(loadStep, LOAD_STEPS.length - 1)];

  return (
    <div
      style={{
        flex: 1,
        display: "grid",
        placeItems: "center",
        padding: "40px 24px",
      }}
    >
      <div
        style={{
          width: 520,
          maxWidth: "100%",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 30,
          animation: "fadeIn .4s both",
        }}
      >
        <div
          style={{
            width: 56,
            height: 56,
            borderRadius: "50%",
            border: "2.5px solid var(--line)",
            borderTopColor: "var(--gold)",
            animation: "spin 0.9s linear infinite",
          }}
        />
        <div style={{ textAlign: "center" }}>
          <div
            style={{
              font: "600 18px 'DM Sans',sans-serif",
              marginBottom: 8,
            }}
          >
            {cur.label}…
          </div>
          <div
            style={{
              font: "500 12.5px 'JetBrains Mono',monospace",
              color: "var(--muted)",
            }}
          >
            {cur.tag}
          </div>
        </div>
        <div
          style={{
            width: "100%",
            display: "flex",
            flexDirection: "column",
            gap: 9,
          }}
        >
          {LOAD_STEPS.map((st, i) => {
            const mark = i < loadStep ? "✓" : i === loadStep ? "●" : "○";
            const markColor =
              i < loadStep
                ? "var(--green)"
                : i === loadStep
                ? "var(--gold)"
                : "var(--faint)";
            return (
              <div
                key={st.label}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 12,
                  padding: "12px 16px",
                  border: "1px solid var(--line)",
                  borderRadius: 11,
                  background: "var(--card)",
                  opacity: i <= loadStep ? 1 : 0.38,
                  transition: "opacity .4s",
                }}
              >
                <span
                  style={{
                    font: "700 12px 'JetBrains Mono',monospace",
                    color: markColor,
                    width: 16,
                    textAlign: "center",
                  }}
                >
                  {mark}
                </span>
                <span
                  style={{
                    font: "500 13px 'DM Sans',sans-serif",
                    color: "var(--text)",
                  }}
                >
                  {st.label}
                </span>
                <span
                  style={{
                    marginLeft: "auto",
                    font: "500 10.5px 'JetBrains Mono',monospace",
                    color: "var(--faint)",
                  }}
                >
                  {st.tag}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
