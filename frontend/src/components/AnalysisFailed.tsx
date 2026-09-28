"use client";


export function AnalysisFailed({
  url,
  reason,
  onReset,
}: {
  url: string;
  reason: string;
  onReset: () => void;
}) {
  return (
    <main
      style={{
        maxWidth: 640,
        margin: "0 auto",
        padding: "72px 20px 96px",
      }}
    >
      <div
        style={{
          font: "500 10px 'JetBrains Mono',monospace",
          letterSpacing: "0.12em",
          color: "var(--amber)",
          marginBottom: 14,
        }}
      >
        NOT SCORED
      </div>

      <h1
        style={{
          font: "700 26px/1.25 'DM Sans',sans-serif",
          letterSpacing: "-0.02em",
          margin: "0 0 12px",
          textWrap: "balance",
        }}
      >
        This listing could not be analysed
      </h1>

      <p
        style={{
          font: "400 14px/1.65 'DM Sans',sans-serif",
          textAlign: "justify",
          color: "var(--muted)",
          margin: "0 0 20px",
          textWrap: "pretty",
        }}
      >
        {reason}
      </p>

      <p
        style={{
          font: "400 13px/1.6 'DM Sans',sans-serif",
          textAlign: "justify",
          color: "var(--faint)",
          margin: "0 0 24px",
        }}
      >
        No figures are shown because none were produced - nothing on this page
        is estimated or filled in on the listing&rsquo;s behalf.
      </p>

      {url && (
        <div
          style={{
            font: "500 11px/1.5 'JetBrains Mono',monospace",
            color: "var(--faint)",
            background: "var(--inputbg)",
            border: "1px solid var(--line)",
            borderRadius: 10,
            padding: "10px 12px",
            marginBottom: 26,
            wordBreak: "break-all",
          }}
        >
          {url}
        </div>
      )}

      <button
        onClick={onReset}
        style={{
          font: "600 13px 'DM Sans',sans-serif",
          color: "var(--bg)",
          background: "var(--text)",
          border: "none",
          borderRadius: 999,
          padding: "11px 22px",
          cursor: "pointer",
        }}
      >
        Try another listing
      </button>
    </main>
  );
}
