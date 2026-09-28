
const DRIVER_LABELS: Record<string, string> = {
  price_per_sqm: "Sale price / m²",
  rent_per_sqm: "Rent / m²",
  floor_area_median_sqm: "Floor area",
  bedrooms: "Bedrooms",
  parking_rate: "Parking availability",
  year: "Market year",
  n_sales: "Sales-sample depth",
  n_rents: "Rent-sample depth",
};

export function driverLabel(feature: string): string {
  if (DRIVER_LABELS[feature]) return DRIVER_LABELS[feature];
  if (feature.startsWith("community_")) return feature.slice("community_".length);
  if (feature.startsWith("nearest_metro_"))
    return `Near ${feature.slice("nearest_metro_".length)}`;
  if (feature.startsWith("property_type_"))
    return feature.slice("property_type_".length).replace(/^\w/, (c) => c.toUpperCase());
  return feature.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}
