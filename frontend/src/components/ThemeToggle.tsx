"use client";

import { CSSProperties } from "react";
import { HoverButton } from "./HoverButton";

export function ThemeToggle({
  label,
  onToggle,
  style,
}: {
  label: string;
  onToggle: () => void;
  style?: CSSProperties;
}) {
  return (
    <HoverButton
      onClick={onToggle}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        background: "var(--card)",
        border: "1px solid var(--line)",
        color: "var(--muted)",
        borderRadius: 999,
        padding: "8px 14px",
        font: "500 12px 'DM Sans',sans-serif",
        cursor: "pointer",
        transition: "all .25s",
        ...style,
      }}
      hoverStyle={{ borderColor: "var(--linegold)", color: "var(--text)" }}
    >
      <span
        style={{
          width: 8,
          height: 8,
          borderRadius: "50%",
          background: "var(--gold)",
          display: "inline-block",
        }}
      />
      {label}
    </HoverButton>
  );
}
