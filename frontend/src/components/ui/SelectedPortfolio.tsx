import InputFreshness from "./InputFreshness";
import RebalancePlan from "./RebalancePlan";
import PortfolioEstimates from "./PortfolioEstimates";
import { useId, useState } from "react";
import type { RunDetail } from "@/types/api";
import { record } from "@/lib/runPresentation";
import { portfolioPeriods } from "@/lib/portfolioPresentation";

/** Saved target weights only. Never normalise or substitute broker holdings. */
export default function SelectedPortfolio({ run }: { run: RunDetail }) {
  const id = useId();
  const periods = portfolioPeriods(run);
  const [periodIndex, setPeriodIndex] = useState<number | null>(null);
  const initial = run.request_payload?.initial_cash;
  const [capital, setCapital] = useState(String(typeof initial === "number" && initial > 0 ? initial : 1000));
  const selected = periods[Math.min(periodIndex ?? periods.length - 1, periods.length - 1)];
  const rows = Object.entries(selected?.weights ?? {}).filter((entry): entry is [string, number] => typeof entry[1] === "number" && Number.isFinite(entry[1]) && entry[1] !== 0).sort((a, b) => b[1] - a[1]);
  const invalid = selected?.incomplete || Object.values(selected?.weights ?? {}).some(v => typeof v !== "number" || !Number.isFinite(v));
  const total = rows.reduce((sum, [, value]) => sum + value, 0);
  const amount = capital.trim() === "" ? NaN : Number(capital);
  if (!run.result) return null;
  const investing = periods.length === 1 && !selected?.rounded;
  const plan = record(record(run.result).rebalance_plan);
  if (plan.currency === "USD" && Array.isArray(plan.rows)) return <section aria-label="Selected portfolio" className="space-y-4 rounded-lg border border-gray-200 bg-white p-5">
    <h2 className="text-lg font-semibold">Investment recommendation</h2>
    <PortfolioEstimates run={run} />
    <InputFreshness run={run} />
    <RebalancePlan plan={plan} />
  </section>;
  return <section aria-label="Selected portfolio" className="space-y-4 rounded-lg border border-gray-200 bg-white p-5">
    <div>
      <h2 className="text-lg font-semibold">{investing ? "Selected portfolio" : "Recorded portfolio allocations"}</h2>
      <p className="mt-1 text-sm text-gray-500">{investing ? "Saved model target weights. These are not broker holdings or confirmation of orders." : "Historical selections from the saved backtest. Dates retain their recorded labels."}</p>
    </div>
    {(run.request_payload?.execution_mode === "investing" || "optimal_weights" in record(run.result)) && <><PortfolioEstimates run={run} /><InputFreshness run={run} /></>}
    {periods.length > 1 && <label className="block text-sm" htmlFor={`${id}-period`}>Recorded period<select id={`${id}-period`} className="mt-1 block w-full rounded border border-gray-300 p-2" value={periodIndex ?? periods.length - 1} onChange={e => setPeriodIndex(Number(e.target.value))}>{periods.map((p, index) =>
      <option key={index} value={index}>{p.label}</option>)}</select>
    </label>}
    {periods.length === 1 && selected?.rounded && <p className="text-sm">Recorded period: {selected.label}</p>}
    {selected?.rounded && <p className="text-xs text-amber-800">This older result stores rounded allocation summaries; positions below 1% may be omitted. Values are shown as saved.</p>}
    {invalid && <p role="status" className="text-sm text-amber-800">Some saved weights are unavailable. The allocation shown may be incomplete.</p>}
    {!rows.length ? <p className="text-sm text-gray-500">No selected asset weights were saved for this result{selected ? " period" : ""}. See output files for any older allocation records.</p> : <>
      <label className="block text-sm" htmlFor={`${id}-capital`}>Illustrative capital ($)<input id={`${id}-capital`} type="number" min="0" step="any" className="mt-1 block w-full max-w-48 rounded border border-gray-300 p-2" value={capital} onChange={e => setCapital(e.target.value)} />
      </label>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-left text-xs uppercase text-gray-500">
              <th className="py-2">Asset</th>
              <th className="py-2 text-right">Target weight</th>
              <th className="py-2 text-right">Illustrative amount</th>
            </tr>
          </thead>
          <tbody>{rows.map(([symbol, weight]) =>
            <tr key={symbol} className="border-b border-gray-100">
              <th scope="row" className="py-3 text-left font-medium">{symbol}<div aria-hidden="true" className="mt-1 h-1 max-w-40 rounded bg-gray-100">
                <div className="h-1 rounded bg-blue-500" style={{ width: `${Math.min(100, Math.max(0, weight * 100))}%` }} />
              </div>
              </th>
              <td className="text-right tabular-nums" title={`${weight * 100}%`}>{Math.abs(weight * 100) < 0.01 ? `${weight < 0 ? "−" : ""}<0.01%` : `${(weight * 100).toFixed(2)}%`}</td>
              <td className="text-right tabular-nums">{Number.isFinite(amount) && amount >= 0 ? new Intl.NumberFormat(undefined, { style: "currency", currency: "USD" }).format(amount * weight) : "—"}</td>
            </tr>)}</tbody>
        </table>
      </div>
      <p className="text-xs text-gray-500">{rows.length} recorded assets · displayed weights total {(total * 100).toFixed(2)}%. Weights are not rescaled. Amounts are illustrative and exclude execution prices and costs.</p>
    </>}
  </section>;
}
