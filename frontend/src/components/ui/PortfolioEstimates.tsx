import type { RunDetail } from "@/types/api";
import { formatMetricValue } from "@/lib/metrics";
import { record } from "@/lib/runPresentation";

const finite = (value: unknown): number | undefined => typeof value === "number" && Number.isFinite(value) ? value : undefined;

/** Forward estimates of the selected weights; never substitute frontier extrema. */
export default function PortfolioEstimates({ run }: { run: RunDetail }) {
  const result = record(run.result);
  const diagnostics = record(result.optimization_diagnostics);
  const gross = finite(diagnostics.selected_gross_expected_return);
  const net = finite(diagnostics.selected_net_expected_return);
  const expected = net ?? gross;
  const savedRisk = finite(diagnostics.selected_volatility);
  const volatility = savedRisk !== undefined && savedRisk >= 0 ? savedRisk : undefined;
  const sharpe = expected !== undefined && volatility !== undefined && volatility > 0 ? finite(expected / volatility) : undefined;
  const frequency = result.frequency ?? record(result.data_freshness).frequency ?? run.request_payload?.frequency;
  const period = frequency === "monthly" ? "month" : frequency === "weekly" ? "week" : frequency === "daily" ? "day" : "observation period (frequency unavailable)";
  return <section aria-label="Selected portfolio estimates" className="space-y-2">
    <h3 className="font-medium">Selected portfolio estimates</h3>
    <dl className="grid gap-3 sm:grid-cols-3">
      <div className="rounded-lg border border-gray-200 p-3">
        <dt className="text-xs text-gray-500">Expected return ({net !== undefined ? "after estimated rebalance costs" : "before costs"})</dt>
        <dd className="mt-1 text-lg font-semibold tabular-nums">{formatMetricValue("average_return", expected)}</dd>
        {net !== undefined && gross !== undefined && <dd className="mt-1 text-xs text-gray-500">Before costs: {formatMetricValue("average_return", gross)}</dd>}
      </div>
      <div className="rounded-lg border border-gray-200 p-3">
        <dt className="text-xs text-gray-500">Estimated risk (volatility)</dt>
        <dd className="mt-1 text-lg font-semibold tabular-nums">{formatMetricValue("std_return", volatility)}</dd>
      </div>
      <div className="rounded-lg border border-gray-200 p-3">
        <dt className="text-xs text-gray-500">Expected Sharpe (0% risk-free)</dt>
        <dd className="mt-1 text-lg font-semibold tabular-nums">{formatMetricValue("sharpe_ratio", sharpe)}</dd>
      </div>
    </dl>
    <p className="text-xs text-gray-500">Per-{period} model estimates. Sharpe is expected return divided by volatility, using a zero risk-free baseline, and is not annualized. — means unavailable; Sharpe is undefined at zero volatility.</p>
  </section>;
}
