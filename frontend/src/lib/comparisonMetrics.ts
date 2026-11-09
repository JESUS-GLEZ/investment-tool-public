import { comparisonMetricLabel, formatMetricValue } from "./metrics";
import { canonical, finite, type ComparisonFamily, type SavedCandidate } from "./savedComparison";

const KEYS: Record<ComparisonFamily, string[]> = {
  backtests: ["initial_value", "final_value", "total_return", "annualized_return", "annualized_volatility", "max_drawdown", "sharpe_ratio", "sortino_ratio", "total_periods", "average_turnover"],
  predictions: ["mae", "rmse", "bias", "directional_accuracy", "information_coefficient", "hit_rate_top_k", "coverage", "scored_pairs", "eligible_pairs", "scored_periods", "failed_periods", "wall_time"],
  solvers: ["hypervolume", "spacing", "spread", "cardinality", "best_sharpe", "median_sharpe", "wall_time", "actual_evaluations", "requested_evaluations", "seed_generation_seconds", "total_time_with_seeds_s"],
};
const LABELS: Record<string, string> = { mae: "Mean absolute error", rmse: "Root mean squared error", bias: "Forecast bias", information_coefficient: "Information coefficient", hit_rate_top_k: "Top-k hit rate", cardinality: "Pareto solutions", best_sharpe: "Best expected Sharpe-like score", median_sharpe: "Median expected Sharpe-like score", hypervolume: "Hypervolume (MV projection)", wall_time: "Execution time (seconds)", seed_generation_seconds: "Shared seed time (seconds)", total_time_with_seeds_s: "Time including shared seeds (seconds)" };
const PERCENT = new Set(["mae", "rmse", "bias", "directional_accuracy", "hit_rate_top_k", "coverage"]);
const COUNTS = new Set(["cardinality", "actual_evaluations", "requested_evaluations", "scored_pairs", "eligible_pairs", "scored_periods", "failed_periods"]);
export function metricRows(entries: SavedCandidate[]) {
  return entries.length ? [...new Set([...KEYS[entries[0].family], ...entries.flatMap(e => Object.keys(e.metrics))])].filter(key => entries.some(e => finite(e.metrics[key]))).map(key => ({ key, label: LABELS[key] ?? comparisonMetricLabel(key, entries.map(e => e.definitions)), reason: metricRestriction(key, entries) })) : [];
}
export function metricRestriction(key: string, entries: SavedCandidate[]): string | null {
  if (["annualized_return", "sharpe_ratio", "sortino_ratio"].includes(key)) {
    const field = key === "annualized_return" ? "annualized_return" : key === "sharpe_ratio" ? "sharpe" : "sortino";
    if (entries.some(e => e.definitions.version !== 2 || typeof e.definitions[field] !== "string")) return "Metric definition unavailable.";
    const signatures = entries.map(e => canonical({ definition: e.definitions[field], basis: key === "annualized_return" ? e.definitions.annualization_basis : undefined, frequency: e.definitions.periods_per_year, rf: key !== "annualized_return" ? e.definitions.annual_risk_free_rate : undefined }));
    if (new Set(signatures).size > 1) return "Metric definitions or risk-free assumptions differ.";
    if (key === "annualized_return" && entries.some(e => !["cagr", "geometric_period_return"].includes(String(e.definitions.annualized_return)) || !["calendar_act_365_25", "period_count"].includes(String(e.definitions.annualization_basis)))) return "Annualization basis unavailable.";
    if (key !== "annualized_return" && entries.some(e => !finite(e.definitions.annual_risk_free_rate) || !finite(e.definitions.periods_per_year))) return "Risk-adjustment metadata unavailable.";
  }
  if (key === "hit_rate_top_k") {
    const values = entries.map(e => e.run.request_payload.top_k);
    if (values.some(v => !finite(v)) || new Set(values).size > 1) return "Top-k settings differ or are not recorded.";
  }
  return null;
}
export function comparisonValue(family: ComparisonFamily, key: string, value: unknown): string {
  if (!finite(value)) return "—";
  if (family === "backtests") return formatMetricValue(key, value);
  if (PERCENT.has(key)) return `${(value * 100).toFixed(2)}%`;
  if (COUNTS.has(key)) return Number.isInteger(value) ? value.toLocaleString() : "—";
  return value.toLocaleString(undefined, { maximumSignificantDigits: 6 });
}
