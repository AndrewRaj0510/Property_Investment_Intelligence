"use client";

import type { MacroSnapshot } from "@/lib/types";
import { Logo } from "./Logo";
import { ThemeToggle } from "./ThemeToggle";

function tempLabel(c?: number | null): string | undefined {
  if (c == null) return undefined;
  if (c >= 40) return "HOT";
  if (c >= 33) return "WARM";
  if (c >= 24) return "MILD";
  return "COOL";
}

function Pill({
  label,
  value,
  arrow,
  arrowColor,
  unavailableReason,
}: {
  label: string;
  value: string | null;
  arrow?: string;
  arrowColor?: string;
  unavailableReason?: string;
}) {
  const unavailable = value == null;
  return (
    <div
      style={{
        display: "flex",
        alignItems: "baseline",
        gap: 7,
        padding: "7px 12px",
        border: "1px solid var(--line)",
        borderRadius: 9,
        background: "var(--card)",
      }}
    >
      <span
        style={{
          font: "600 9.5px 'JetBrains Mono',monospace",
          letterSpacing: "0.1em",
          color: "var(--faint)",
        }}
      >
        {label}
      </span>
      <span
        style={{
          font: "600 13px 'JetBrains Mono',monospace",
          color: unavailable ? "var(--faint)" : "var(--text)",
        }}
        title={unavailable ? unavailableReason : undefined}
      >
        {unavailable ? "n/a" : value}
      </span>
      {!unavailable && arrow && (
        <span
          style={{
            font: "600 10px 'JetBrains Mono',monospace",
            color: arrowColor,
          }}
        >
          {arrow}
        </span>
      )}
    </div>
  );
}

export function AppHeader({
  macro,
  themeLabel,
  onToggleTheme,
  onReset,
  onOpenMacro,
}: {
  macro: MacroSnapshot | null;
  themeLabel: string;
  onToggleTheme: () => void;
  onReset: () => void;
  onOpenMacro?: () => void;
}) {
  const eibor =
    macro?.eibor_3m != null ? `${macro.eibor_3m.toFixed(2)}%` : null;
  const eiborUp =
    macro?.eibor_3m != null && macro.eibor_12m_avg != null
      ? macro.eibor_3m >= macro.eibor_12m_avg
      : true;

  const brent =
    macro?.brent_crude_usd != null
      ? `$${macro.brent_crude_usd.toFixed(1)}`
      : null;
  const brentUp =
    macro?.brent_crude_usd != null && macro.brent_12m_avg != null
      ? macro.brent_crude_usd >= macro.brent_12m_avg
      : true;

  const temp =
    macro?.temp_max_c != null ? `${Math.round(macro.temp_max_c)}°C` : null;

  const thermal =
    macro?.thermal_index != null ? `${macro.thermal_index.toFixed(1)}` : null;

  const asOf = (m?: string | null) => (m ? ` (as of ${m.slice(0, 7)})` : "");

  return (
    <header
      style={{
        display: "flex",
        alignItems: "center",
        gap: 24,
        padding: "0 28px",
        height: 64,
        borderBottom: "1px solid var(--line)",
        background: "var(--surface)",
        position: "sticky",
        top: 0,
        zIndex: 50,
        transition: "background-color .35s ease",
      }}
    >
      <Logo onClick={onReset} />

      <div
        style={{
          marginLeft: "auto",
          display: "flex",
          gap: 8,
          cursor: onOpenMacro ? "pointer" : "default",
        }}
        onClick={onOpenMacro}
        title={onOpenMacro ? "Open the macro & seasonal overlay" : undefined}
      >
        <Pill
          label={`EIBOR 3M${asOf(macro?.eibor_as_of)}`}
          value={eibor}
          unavailableReason="No EIBOR data ingested."
          arrow={eiborUp ? "▲" : "▼"}
          arrowColor={eiborUp ? "var(--red)" : "var(--green)"}
        />
        <Pill
          label={`BRENT${asOf(macro?.brent_as_of)}`}
          value={brent}
          unavailableReason="No Brent data ingested."
          arrow={brentUp ? "▲" : "▼"}
          arrowColor={brentUp ? "var(--green)" : "var(--red)"}
        />
        <Pill
          label="TEMP"
          value={temp}
          unavailableReason="Weather data not ingested (Objective 3.2)."
          arrow={tempLabel(macro?.temp_max_c)}
          arrowColor="var(--amber)"
        />
        <Pill
          label="THERMAL"
          value={thermal}
          unavailableReason="Weather data not ingested (Objective 3.2)."
          arrowColor="var(--amber)"
        />
      </div>

      <ThemeToggle label={themeLabel} onToggle={onToggleTheme} />
      <div
        style={{
          width: 34,
          height: 34,
          borderRadius: "50%",
          background: "var(--goldsoft)",
          border: "1px solid var(--linegold)",
          display: "grid",
          placeItems: "center",
          font: "700 12px 'JetBrains Mono',monospace",
          color: "var(--gold)",
        }}
      >
        AR
      </div>
    </header>
  );
}
