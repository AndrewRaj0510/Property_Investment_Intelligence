import type { ListingSource } from "@/lib/listingSource";

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

// Tags describe the step, never a result: the steps animate on a timer while
// /score runs, so no real value exists yet to show. (A fixed "apify/bayut-scraper"
// tag and Dubai Marina coordinates used to be shown for every listing.)
export function loadSteps(source: ListingSource | null) {
  return [
    {
      label: "Contacting Apify actor",
      tag: source ? `apify · ${source} scraper` : "apify",
    },
    { label: "Scraping listing attributes", tag: "price · size · view · finish" },
    { label: "Locating the community", tag: "portal area → DLD area" },
    { label: "Running yield model", tag: "XGBoost + SHAP" },
  ];
}

export const LOAD_STEP_COUNT = loadSteps(null).length;
