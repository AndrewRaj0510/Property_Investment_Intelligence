"use client";

import { useState } from "react";
import { HoverButton } from "./HoverButton";


const ZOOMS = [14, 16, 18];

function mapSrc(lat: number, lng: number, zoom: number): string {
  const q = `${lat.toFixed(6)},${lng.toFixed(6)}`;
  return `https://maps.google.com/maps?q=${encodeURIComponent(q)}&hl=en&z=${zoom}&output=embed`;
}

export function ListingMap({
  lat,
  lng,
  community,
  metro,
}: {
  lat?: number | null;
  lng?: number | null;
  community?: string | null;
  metro?: string | null;
}) {
  const [zoom, setZoom] = useState(1);

  const hasCoords =
    typeof lat === "number" &&
    typeof lng === "number" &&
    Number.isFinite(lat) &&
    Number.isFinite(lng);

  if (!hasCoords) {
    return (
      <Shell community={community} metro={metro} lat={null} lng={null}>
        <div
          style={{
            padding: "34px 24px",
            font: "400 12.5px/1.65 'DM Sans',sans-serif",
            textAlign: "justify",
            color: "var(--faint)",
            textWrap: "pretty",
          }}
        >
          The portal did not publish coordinates for this listing, so there is no
          map to show. {community ? `It is advertised in ${community}.` : ""}
        </div>
      </Shell>
    );
  }

  const src = mapSrc(lat!, lng!, ZOOMS[zoom]);
  const openUrl = `https://www.google.com/maps/search/?api=1&query=${lat!.toFixed(6)},${lng!.toFixed(6)}&hl=en`;

  return (
    <Shell
      community={community}
      metro={metro}
      lat={lat!}
      lng={lng!}
      controls={
        <div style={{ display: "flex", gap: 6 }}>
          <HoverButton
            onClick={() => setZoom((z) => Math.max(0, z - 1))}
            style={zoomBtn}
            hoverStyle={{ borderColor: "var(--gold)" }}
          >
            −
          </HoverButton>
          <HoverButton
            onClick={() => setZoom((z) => Math.min(ZOOMS.length - 1, z + 1))}
            style={zoomBtn}
            hoverStyle={{ borderColor: "var(--gold)" }}
          >
            +
          </HoverButton>
        </div>
      }
    >
      <iframe
        key={src}
        className="listing-map-embed"
        src={src}
        title={`Map of ${community ?? "the listing"}`}
        referrerPolicy="no-referrer-when-downgrade"
        style={{
          display: "block",
          width: "100%",
          height: 300,
          border: 0,
          background: "var(--card2)",
        }}
      />
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "flex-end",
          gap: 10,
          flexWrap: "wrap",
          padding: "10px 20px",
          borderTop: "1px solid var(--line)",
        }}
      >
        <a
          href={openUrl}
          target="_blank"
          rel="noopener noreferrer"
          style={{
            font: "500 11px 'JetBrains Mono',monospace",
            color: "var(--gold)",
            textDecoration: "none",
          }}
        >
          Open full map ↗
        </a>
      </div>
    </Shell>
  );
}

function Shell({
  children,
  controls,
  community,
  metro,
  lat,
  lng,
}: {
  children: React.ReactNode;
  controls?: React.ReactNode;
  community?: string | null;
  metro?: string | null;
  lat: number | null;
  lng: number | null;
}) {
  return (
    <div
      style={{
        background: "var(--card)",
        border: "1px solid var(--line)",
        borderRadius: 16,
        overflow: "hidden",
        boxShadow: "var(--shadow)",
        animation: "fadeUp .6s .15s cubic-bezier(.22,1,.36,1) both",
        minWidth: 0,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 10,
          padding: "14px 20px",
          borderBottom: "1px solid var(--line)",
        }}
      >
        <div style={{ display: "flex", flexDirection: "column", gap: 2, minWidth: 0 }}>
          <div style={{ font: "600 13px 'DM Sans',sans-serif" }}>
            {community ? `Where it is: ${community}` : "Listing location"}
          </div>
          {lat != null && lng != null && (
            <div
              style={{
                font: "500 10px 'JetBrains Mono',monospace",
                color: "var(--faint)",
                letterSpacing: "0.04em",
              }}
            >
              {lat.toFixed(4)}, {lng.toFixed(4)}
            </div>
          )}
        </div>
        {controls}
      </div>
      {children}
      {metro && (
        <div
          style={{
            padding: "10px 20px 14px",
            borderTop: "1px solid var(--line)",
            font: "400 11.5px/1.5 'DM Sans',sans-serif",
            textAlign: "justify",
            color: "var(--faint)",
            textWrap: "pretty",
          }}
        >
          The model grouped this listing with {metro} stock. That is the station
          most common in the comparable Dubai Land Department records, not a
          measured distance from this building.
        </div>
      )}
    </div>
  );
}

const zoomBtn: React.CSSProperties = {
  width: 30,
  height: 30,
  border: "1px solid var(--line)",
  background: "var(--card2)",
  color: "var(--text)",
  borderRadius: 8,
  font: "600 15px 'JetBrains Mono',monospace",
  cursor: "pointer",
  transition: "border-color .2s",
};
