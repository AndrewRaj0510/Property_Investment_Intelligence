"use client";

export function Logo({
  onClick,
  variant = "header",
}: {
  onClick?: () => void;
  variant?: "header" | "login";
}) {
  if (variant === "login") {
    return (
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 10,
          marginBottom: 34,
        }}
      >
        <div
          style={{
            width: 52,
            height: 52,
            transform: "rotate(45deg)",
            border: "1.5px solid var(--gold)",
            display: "grid",
            placeItems: "center",
            borderRadius: 10,
            background: "var(--goldsoft)",
          }}
        >
          <div
            style={{
              width: 22,
              height: 22,
              borderRadius: 5,
              background: "var(--gold)",
              opacity: 0.9,
            }}
          />
        </div>
        <div
          style={{
            font: "700 26px/1 'DM Sans',sans-serif",
            letterSpacing: "-0.02em",
            marginTop: 6,
          }}
        >
          Yield<span style={{ color: "var(--gold)" }}>Lens</span>
        </div>
        <div
          style={{
            font: "500 11px 'JetBrains Mono',monospace",
            letterSpacing: "0.22em",
            textTransform: "uppercase",
            color: "var(--muted)",
          }}
        >
          Dubai Property Intelligence
        </div>
      </div>
    );
  }

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 11,
        cursor: onClick ? "pointer" : "default",
      }}
      onClick={onClick}
    >
      <div
        style={{
          width: 30,
          height: 30,
          transform: "rotate(45deg)",
          border: "1.5px solid var(--gold)",
          display: "grid",
          placeItems: "center",
          borderRadius: 7,
          background: "var(--goldsoft)",
        }}
      >
        <div
          style={{
            width: 12,
            height: 12,
            borderRadius: 3,
            background: "var(--gold)",
          }}
        />
      </div>
      <div
        style={{
          font: "700 17px 'DM Sans',sans-serif",
          letterSpacing: "-0.02em",
        }}
      >
        Yield<span style={{ color: "var(--gold)" }}>Lens</span>
      </div>
    </div>
  );
}
