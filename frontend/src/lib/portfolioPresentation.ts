import type { RunDetail } from "@/types/api";
import { record } from "./runPresentation";


export function parseWeightSummary(summary: unknown) {
  const weights: Record<string, unknown> = {};
  let incomplete = typeof summary !== "string";
  if (typeof summary === "string") for (const entry of summary.split(",")) {
    const match = entry.trim().match(/^(.+):\s*(-?\d+(?:\.\d+)?)%$/);
    if (match) weights[match[1].trim()] = Number(match[2]) / 100;
    else if (entry.trim()) incomplete = true;
  }
  return {weights, incomplete};
}

export function portfolioPeriods(run: RunDetail) {
  const result = record(run.result);
  if (Object.hasOwn(result, "optimal_weights")) return [{ label: "Selected portfolio", weights: record(result.optimal_weights), rounded: false, incomplete: result.optimal_weights == null || typeof result.optimal_weights !== "object" || Array.isArray(result.optimal_weights) }];
  const frame = record(result.results_df);
  const summaries = Array.isArray(frame.selected_assets_summary) ? frame.selected_assets_summary : [];
  const dates = Array.isArray(frame.period_end) ? frame.period_end : Array.isArray(frame.date) ? frame.date : [];
  return summaries.map((summary, index) => {
    const {weights, incomplete} = parseWeightSummary(summary);
    return { label: typeof dates[index] === "string" ? String(dates[index]) : `Period ${index + 1} (date unavailable)`, weights, rounded: true, incomplete };
  });
}
