/** Shared display units for backend financial metrics. */
const CURRENCY_KEYS = new Set(["initial_value", "final_value", "total_transaction_costs"]);
const COUNT_KEYS = new Set(["total_periods", "max_drawdown_duration_periods"]);
const PERCENTAGE_KEYS = new Set([
  "annualized_return",
  "total_return",
  "annualized_volatility",
  "max_drawdown",
  "win_rate",
  "best_period",
  "worst_period",
  "average_return",
  "std_return",
  "average_turnover",
  "total_turnover",
  "final_top3_weight",
]);
const RATIO_KEYS = new Set(["profit_factor", "final_hhi", "sharpe_ratio", "sortino_ratio", "calmar_ratio"]);
const NUMERIC_KEYS = new Set([...CURRENCY_KEYS, ...COUNT_KEYS, ...PERCENTAGE_KEYS, ...RATIO_KEYS]);

export function metricLabel(key: string, definitions?: unknown): string {
  if (key === "annualized_return") {
    const d = definitions && typeof definitions === "object" ? definitions as Record<string, unknown> : undefined;
    if (d?.version === 2 && d.annualized_return === "cagr") return "CAGR";
    if (d?.version === 2 && d.annualized_return === "geometric_period_return") return "Annualized Compound Return";
    return "Annualized Return (definition unavailable)";
  }
  if (key === "final_hhi") return "Final Capital HHI (0–1)";
  if (key === "average_turnover") return "Average Gross Turnover";
  if (key === "total_turnover") return "Total Gross Turnover";
  return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export function comparisonMetricLabel(key: string, definitions: unknown[]): string {
  if (key !== "annualized_return" || definitions.length === 0) return metricLabel(key);
  const labels = definitions.map((d) => metricLabel(key, d));
  return new Set(labels).size === 1 ? labels[0] : "Annualized Return (mixed definitions)";
}

export function metricDescription(key: string): string | undefined {
  if (key === "total_transaction_costs") return "Sum of recorded fees on actual buys and sells, including initial allocation. Already deducted from portfolio values and returns. Unavailable when fee records are incomplete.";
  if (key === "final_hhi") return "Sum of squared risky-asset capital weights relative to total portfolio value; cash is excluded. This differs from optimizer risk-contribution HHI.";
  if (key === "average_turnover") return "Mean gross value of buys plus sells divided by post-rebalance portfolio value, including recorded zero-trade periods and initial allocation. Not halved or annualized.";
  if (key === "total_turnover") return "Sum of the recorded gross turnover fractions. Can exceed 100%; not an annualized rate.";
  if (key === "annualized_return") return "CAGR requires recognized calendar-growth metadata; compound return requires recognized observation-count metadata. Unavailable or mixed definitions are labelled explicitly, with saved values unchanged.";
  return undefined;
}

export function formatMetricValue(key: string, value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value !== "number") return NUMERIC_KEYS.has(key) ? "—" : String(value);
  if (!Number.isFinite(value)) return "—";
  if (CURRENCY_KEYS.has(key)) {
    return `$${value.toLocaleString(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    })}`;
  }
  if (COUNT_KEYS.has(key)) return Number.isInteger(value) ? value.toLocaleString() : "—";
  if (PERCENTAGE_KEYS.has(key)) return `${(value * 100).toFixed(2)}%`;
  // Profit factor, HHI and risk-adjusted ratios are dimensionless.
  return value.toFixed(2);
}
