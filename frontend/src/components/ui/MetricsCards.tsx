/**
 * Metrics cards — renders key-value pairs from a result metrics object,
 * grouped into semantic sections.
 */

import { formatMetricValue, metricLabel, metricDescription } from "@/lib/metrics";

interface Props {
  metrics: Record<string, unknown>;
}

// ── Section definitions ─────────────────────────────────────────────────────

type Section = { title: string; keys: string[] };

const SECTIONS: Section[] = [
  {
    title: "Summary",
    keys: ["initial_value", "final_value", "total_periods", "win_rate", "profit_factor"],
  },
  {
    title: "Performance",
    keys: ["annualized_return", "total_return", "average_return", "best_period", "worst_period", "best_period_date", "worst_period_date"],
  },
  {
    title: "Risk",
    keys: ["annualized_volatility", "max_drawdown", "max_drawdown_duration_periods", "std_return"],
  },
  {
    title: "Risk-Adjusted",
    keys: ["sharpe_ratio", "sortino_ratio", "calmar_ratio"],
  },
  {
    title: "Portfolio",
    keys: ["final_hhi", "final_top3_weight", "average_turnover", "total_turnover", "total_transaction_costs"],
  },
];

// ── Card ────────────────────────────────────────────────────────────────────

function Card({ metricKey, value, displayLabel }: { metricKey: string; value: unknown; displayLabel?: string }) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white px-4 py-3">
      <p className="text-xs font-medium uppercase tracking-wide text-gray-500" title={metricDescription(metricKey)}>
        {displayLabel ?? metricLabel(metricKey)}
      </p>
      <p className="mt-1 text-base font-semibold text-gray-900 truncate">
        {formatMetricValue(metricKey, value)}
      </p>
    </div>
  );
}

// ── Main component ──────────────────────────────────────────────────────────

export default function MetricsCards({ metrics }: Props) {
  const annualReturnLabel = metricLabel("annualized_return", metrics.metric_definitions);
  // Collect keys that belong to at least one section
  const assignedKeys = new Set(SECTIONS.flatMap((s) => s.keys));
  // Remaining scalar metrics not in any section
  const otherEntries = Object.entries(metrics).filter(
    ([k, v]) => !assignedKeys.has(k) && (typeof v !== "object" || v === null),
  );

  const renderedSections = SECTIONS.map((sec) => {
    const entries = sec.keys
      .filter((k) => k in metrics)
      .map((k) => [k, metrics[k]] as [string, unknown]);
    return { ...sec, entries };
  }).filter((sec) => sec.entries.length > 0);

  if (renderedSections.length === 0 && otherEntries.length === 0) return null;

  return (
    <div className="space-y-5">
      {renderedSections.map((sec) => (
        <div key={sec.title}>
          <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-400">
            {sec.title}
          </h2>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
            {sec.entries.map(([k, v]) => (
              <Card key={k} metricKey={k} value={v} displayLabel={k === "annualized_return" ? annualReturnLabel : undefined} />
            ))}
          </div>
        </div>
      ))}

      {otherEntries.length > 0 && (
        <div>
          <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-400">
            Other
          </h2>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
            {otherEntries.map(([k, v]) => (
              <Card key={k} metricKey={k} value={v} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
