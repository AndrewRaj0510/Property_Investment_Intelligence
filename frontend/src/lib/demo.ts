
export const SCN = {
  community: {
    "Dubai Marina": 5.5,
    JVC: 6.8,
    Downtown: 4.9,
    "Business Bay": 5.7,
    JLT: 6.1,
    "Palm Jumeirah": 4.6,
  } as Record<string, number>,
  type: { Apartment: 0, Villa: -0.3, Townhouse: -0.15 } as Record<string, number>,
  beds: { Studio: 0.6, "1": 0.3, "2": 0, "3": -0.2, "4+": -0.4 } as Record<
    string,
    number
  >,
  finish: { Basic: -0.3, Mid: 0, High: 0.3, Luxury: 0.45 } as Record<
    string,
    number
  >,
  view: {
    None: 0,
    Community: 0.1,
    Pool: 0.15,
    Skyline: 0.3,
    Sea: 0.8,
  } as Record<string, number>,
};

export interface ScenarioState {
  scnCommunity: string;
  scnType: string;
  scnBeds: string;
  scnFinish: string;
  scnView: string;
  scnArea: number;
  scnBudget: number;
}

export interface ScnPart {
  name: string;
  v: number;
  base?: boolean;
}

export function scnModel(s: ScenarioState): { parts: ScnPart[]; total: number } {
  const base = SCN.community[s.scnCommunity];
  const parts: ScnPart[] = [
    { name: "Community baseline", v: base, base: true },
    { name: s.scnType, v: SCN.type[s.scnType] },
    {
      name: s.scnBeds === "Studio" ? "Studio" : s.scnBeds + " bed",
      v: SCN.beds[s.scnBeds],
    },
    { name: s.scnFinish + " finish", v: SCN.finish[s.scnFinish] },
    { name: s.scnView + " view", v: SCN.view[s.scnView] },
    { name: "Floor area", v: (-Math.abs(s.scnArea - 1200) / 1000) * 0.3 },
  ];
  const total = parts.reduce((a, p) => a + p.v, 0);
  return { parts, total };
}

export const DEMO_LISTING = {
  type: "APARTMENT",
  community: "DUBAI MARINA",
  title: "Sample 2-Bed Apartment, Marina Tower X",
  subtitle: "Placeholder listing - replace with live Apify scrape",
  price: "AED 9,999,999",
  pricePerSqft: "AED 999 / sqft",
  beds: "2",
  baths: "2",
  sqft: "1,234",
  view: "Sea",
  finish: "High",
};

export const DEMO_SHAP = [
  {
    name: "Sea view",
    value: "+0.8%",
    width: "92%",
    color: "var(--green)",
    delay: ".1s",
    note: "Sea-view units in this tower rent at a premium year-round.",
  },
  {
    name: "Metro proximity",
    value: "+0.4%",
    width: "52%",
    color: "var(--green)",
    delay: ".2s",
    note: "6 min walk to DMCC station widens the tenant pool.",
  },
  {
    name: "High finish tier",
    value: "+0.3%",
    width: "40%",
    color: "var(--green)",
    delay: ".3s",
    note: "Upgraded finish supports higher asking rent per sqft.",
  },
  {
    name: "Floor area",
    value: "+0.1%",
    width: "16%",
    color: "var(--gold)",
    delay: ".4s",
    note: "1,234 sqft is near the sweet spot for 2-bed demand.",
  },
  {
    name: "Summer seasonality",
    value: "−0.4%",
    width: "48%",
    color: "var(--red)",
    delay: ".5s",
    note: "July heat suppresses short-term occupancy 15-20%.",
  },
];

export const DEMO_MACRO = {
  eibor: "5.10%",
  brent: "$78.4",
  temp: "41°C",
  tempLabel: "HOT",
  tourists: "1.4M",
  touristsDelta: "+6%",
};

export const LOAD_STEPS = [
  { label: "Contacting Apify actor", tag: "apify/bayut-scraper" },
  { label: "Scraping listing attributes", tag: "price · size · view · finish" },
  { label: "Geocoding location", tag: "lat 25.08 · lng 55.14" },
  { label: "Running yield model", tag: "XGBoost + SHAP" },
];
