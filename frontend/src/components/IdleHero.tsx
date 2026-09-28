"use client";

import { useState } from "react";
import { HoverButton } from "./HoverButton";
import {
  checkUrl,
  LISTING_SOURCES,
  PLACEHOLDERS,
  type ListingSource,
} from "@/lib/listingSource";

const steps = [
  {
    n: "01",
    title: "Fetch the listing",
    body: "Apify actor scrapes price, size, view & finish from the URL.",
    delay: ".15s",
  },
  {
    n: "02",
    title: "Locate & benchmark",
    body: "Pin it on the map and compare against its community median.",
    delay: ".25s",
  },
  {
    n: "03",
    title: "Explain the yield",
    body: "SHAP drivers + macro context show exactly why, not just what.",
    delay: ".35s",
  },
];

export function IdleHero({
  url,
  onUrl,
  onAnalyze,
  onSample,
  source,
  onSource,
}: {
  url: string;
  onUrl: (v: string) => void;
  onAnalyze: () => void;
  onSample: () => void;
  source: ListingSource;
  onSource: (s: ListingSource) => void;
}) {
  const [focus, setFocus] = useState(false);

  const check = checkUrl(url, source);
  const showError = url.trim().length > 0 && !check.ok;
  const canAnalyze = check.ok;

  const submit = () => {
    if (canAnalyze) onAnalyze();
  };

  return (
    <div
      style={{
        flex: 1,
        display: "grid",
        placeItems: "center",
        position: "relative",
        overflow: "hidden",
        padding: "40px 24px",
      }}
    >
      <div
        style={{
          position: "absolute",
          inset: 0,
          backgroundImage:
            "radial-gradient(ellipse 55% 40% at 50% 0%, var(--goldsoft), transparent 70%)",
          pointerEvents: "none",
        }}
      />
      <div
        style={{
          position: "relative",
          width: 760,
          maxWidth: "100%",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          textAlign: "center",
          animation: "fadeUp .7s cubic-bezier(.22,1,.36,1) both",
        }}
      >
        <div
          style={{
            font: "600 11px 'JetBrains Mono',monospace",
            letterSpacing: "0.24em",
            textTransform: "uppercase",
            color: "var(--gold)",
            padding: "7px 14px",
            border: "1px solid var(--linegold)",
            borderRadius: 999,
            background: "var(--goldsoft)",
            marginBottom: 26,
          }}
        >
          Investment Intelligence
        </div>
        <h1
          style={{
            margin: "0 0 14px",
            font: "700 52px/1.08 'DM Sans',sans-serif",
            letterSpacing: "-0.03em",
          }}
        >
          Paste a listing.
          <br />
          Know if it&apos;s <span style={{ color: "var(--gold)" }}>worth buying.</span>
        </h1>
        <p
          style={{
            margin: "0 0 38px",
            font: "400 17px/1.6 'DM Sans',sans-serif",
            textAlign: "justify",
            color: "var(--muted)",
            maxWidth: 520,
            textWrap: "pretty",
          }}
        >
          Drop any Dubai property listing URL. We fetch it, locate it, and run the
          yield model with a plain-language explanation of every number.
        </p>

        <div
          role="radiogroup"
          aria-label="Listing source"
          style={{
            display: "inline-flex",
            gap: 4,
            padding: 4,
            marginBottom: 14,
            background: "var(--surface)",
            border: "1px solid var(--line)",
            borderRadius: 999,
          }}
        >
          {LISTING_SOURCES.map((s) => {
            const active = s.id === source;
            return (
              <button
                key={s.id}
                role="radio"
                aria-checked={active}
                onClick={() => onSource(s.id)}
                style={{
                  appearance: "none",
                  border: "none",
                  cursor: active ? "default" : "pointer",
                  borderRadius: 999,
                  minWidth: 116,
                  textAlign: "center",
                  padding: "8px 16px",
                  font: "600 12.5px 'JetBrains Mono',monospace",
                  letterSpacing: "0.06em",
                  color: active ? "#0A1628" : "var(--muted)",
                  background: active ? "var(--gold)" : "transparent",
                  boxShadow: active ? "0 2px 10px rgba(201,168,76,0.28)" : "none",
                  transition: "background .25s, color .25s, box-shadow .25s",
                }}
              >
                {s.label}
              </button>
            );
          })}
        </div>

        <div
          style={{
            width: "100%",
            display: "flex",
            gap: 10,
            background: "var(--card)",
            border: `1px solid ${
              showError ? "var(--danger, #C2523F)" : focus ? "var(--gold)" : "var(--line)"
            }`,
            borderRadius: 16,
            padding: 10,
            boxShadow: "var(--shadow)",
            transition: "border-color .3s",
          }}
        >
          <div
            style={{
              display: "grid",
              placeItems: "center",
              padding: "0 4px 0 12px",
              color: "var(--faint)",
              font: "500 15px 'JetBrains Mono',monospace",
            }}
          >
            ⌕
          </div>
          <input
            value={url}
            onChange={(e) => onUrl(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") submit();
            }}
            onFocus={() => setFocus(true)}
            onBlur={() => setFocus(false)}
            aria-invalid={showError}
            placeholder={PLACEHOLDERS[source]}
            style={{
              flex: 1,
              background: "transparent",
              border: "none",
              outline: "none",
              font: "500 15px 'JetBrains Mono',monospace",
              color: "var(--text)",
              minWidth: 0,
            }}
          />
          <HoverButton
            onClick={submit}
            disabled={!canAnalyze}
            title={canAnalyze ? undefined : `Paste a ${source === "bayut" ? "Bayut" : "Dubizzle"} listing URL`}
            style={{
              background: canAnalyze ? "var(--gold)" : "var(--line)",
              color: canAnalyze ? "#0A1628" : "var(--faint)",
              border: "none",
              borderRadius: 11,
              padding: "14px 26px",
              font: "700 14px 'DM Sans',sans-serif",
              cursor: canAnalyze ? "pointer" : "not-allowed",
              whiteSpace: "nowrap",
              transition: "transform .2s, box-shadow .2s, background .25s, color .25s",
              boxShadow: canAnalyze ? "0 4px 18px rgba(201,168,76,0.25)" : "none",
            }}
            hoverStyle={
              canAnalyze
                ? {
                    transform: "translateY(-1px)",
                    boxShadow: "0 8px 26px rgba(201,168,76,0.4)",
                  }
                : {}
            }
          >
            Analyze →
          </HoverButton>
        </div>

        {showError && (
          <div
            role="alert"
            style={{
              marginTop: 10,
              display: "flex",
              alignItems: "center",
              gap: 8,
              font: "500 12.5px 'DM Sans',sans-serif",
              color: "var(--danger, #C2523F)",
            }}
          >
            <span>{check.error}</span>
            {check.suggest && (
              <button
                onClick={() => onSource(check.suggest!)}
                style={{
                  appearance: "none",
                  background: "transparent",
                  border: "1px solid currentColor",
                  color: "inherit",
                  borderRadius: 999,
                  padding: "3px 11px",
                  font: "600 11.5px 'JetBrains Mono',monospace",
                  cursor: "pointer",
                }}
              >
                Switch to {check.suggest === "bayut" ? "Bayut" : "Dubizzle"}
              </button>
            )}
          </div>
        )}

        <div
          style={{
            marginTop: 18,
            display: "flex",
            alignItems: "center",
            gap: 10,
            font: "400 12.5px 'DM Sans',sans-serif",
            color: "var(--faint)",
          }}
        >
          <span>No URL handy?</span>
          <HoverButton
            onClick={onSample}
            style={{
              background: "transparent",
              border: "1px dashed var(--linegold)",
              color: "var(--gold)",
              borderRadius: 999,
              padding: "6px 14px",
              font: "500 12px 'JetBrains Mono',monospace",
              cursor: "pointer",
              transition: "background .25s",
            }}
            hoverStyle={{ background: "var(--goldsoft)" }}
          >
            Try a sample {source === "bayut" ? "Bayut" : "Dubizzle"} listing
          </HoverButton>
        </div>

        <div
          style={{
            marginTop: 64,
            display: "grid",
            gridTemplateColumns: "repeat(3,1fr)",
            gap: 14,
            width: "100%",
          }}
        >
          {steps.map((s) => (
            <div
              key={s.n}
              style={{
                textAlign: "left",
                padding: "18px 20px",
                border: "1px solid var(--line)",
                borderRadius: 14,
                background: "var(--surface)",
                animation: `fadeUp .7s ${s.delay} cubic-bezier(.22,1,.36,1) both`,
              }}
            >
              <div
                style={{
                  font: "700 20px 'JetBrains Mono',monospace",
                  color: "var(--gold)",
                  marginBottom: 5,
                }}
              >
                {s.n}
              </div>
              <div
                style={{
                  font: "600 14px 'DM Sans',sans-serif",
                  marginBottom: 3,
                }}
              >
                {s.title}
              </div>
              <div
                style={{
                  font: "400 12.5px/1.5 'DM Sans',sans-serif",
                  textAlign: "justify",
                  color: "var(--muted)",
                }}
              >
                {s.body}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
