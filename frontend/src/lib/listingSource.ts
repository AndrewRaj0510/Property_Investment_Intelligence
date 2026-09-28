
export type ListingSource = "bayut" | "dubizzle";

export const LISTING_SOURCES: {
  id: ListingSource;
  label: string;
  domains: string[];
}[] = [
  { id: "bayut", label: "Bayut", domains: ["bayut.com"] },
  { id: "dubizzle", label: "Dubizzle", domains: ["dubizzle.com"] },
];

function hostMatches(host: string, domain: string): boolean {
  return host === domain || host.endsWith(`.${domain}`);
}

export const SAMPLE_URLS: Record<ListingSource, string> = {
  bayut: "https://www.bayut.com/property/details-8214402.html",
  dubizzle: "https://dubai.dubizzle.com/property-for-sale/residential/apartment/2024/8/12/example-listing-1234567/",
};

export const PLACEHOLDERS: Record<ListingSource, string> = {
  bayut: "https://www.bayut.com/property/details-000000.html",
  dubizzle: "https://dubai.dubizzle.com/property-for-sale/residential/apartment/...",
};

export function detectSource(raw: string): ListingSource | null {
  const trimmed = (raw || "").trim();
  if (!trimmed) return null;
  let host: string;
  try {
    const withScheme = /^https?:\/\//i.test(trimmed) ? trimmed : `https://${trimmed}`;
    host = new URL(withScheme).hostname.toLowerCase();
  } catch {
    return null;
  }
  const match = LISTING_SOURCES.find((s) =>
    s.domains.some((d) => hostMatches(host, d))
  );
  return match ? match.id : null;
}

export interface UrlCheck {
  ok: boolean;
  error: string;
  suggest?: ListingSource;
}

const LABEL: Record<ListingSource, string> = { bayut: "Bayut", dubizzle: "Dubizzle" };

export function checkUrl(raw: string, source: ListingSource): UrlCheck {
  const trimmed = (raw || "").trim();
  if (!trimmed) return { ok: false, error: "" };

  const detected = detectSource(trimmed);
  if (detected === source) return { ok: true, error: "" };

  if (detected) {
    return {
      ok: false,
      error: `That's a ${LABEL[detected]} link, but ${LABEL[source]} is selected.`,
      suggest: detected,
    };
  }
  return {
    ok: false,
    error: `Enter a ${LABEL[source]} listing URL (a single property page).`,
  };
}
