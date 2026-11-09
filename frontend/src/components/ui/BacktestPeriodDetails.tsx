import { lazy, Suspense, useId, useState } from "react";
import { Link } from "react-router-dom";
import { comparisonCompatibility, finite, type SavedCandidate } from "@/lib/savedComparison";
import { candidateDescription, executionLabel, resultAlias, savedMetric } from "@/lib/compareRuns";
import { MATRIX_METRICS } from "@/lib/compareMatrix";
import { metricRestriction } from "@/lib/comparisonMetrics";
import { formatMetricValue } from "@/lib/metrics";

const PlotlyChart = lazy(() => import("./PlotlyChart"));
const cell = "px-3 py-2 text-left align-top";
export default function BacktestPeriodDetails({entries, manifests, comparisonSearch}: {
  entries: SavedCandidate[]; manifests: Record<string, unknown>; comparisonSearch: string;
}) {
  const id = useId();
  const [curve, setCurve] = useState("none");
  const [returnsOpen, setReturnsOpen] = useState(false);
  const context = comparisonCompatibility(entries, manifests);
  const canDraw = (entries.length === 1 ? !!entries[0].periods : context.state === "matching" && context.chartAllowed)
    && entries.every(e => e.points.length && finite(e.metrics.initial_value) && e.metrics.initial_value > 0);
  const figure = canDraw && curve !== "none" ? {
    data: entries.map((entry, i) => {
      const initial = entry.metrics.initial_value as number;
      let peak = initial;
      return {
        type: "scatter", mode: "lines+markers", name: `${resultAlias(i)} · ${entry.name}`,
        x: entry.points.map(p => p.date),
        y: entry.points.map(p => {peak = Math.max(peak, p.value); return curve === "drawdown" ? (p.value / peak - 1) * 100 : p.value / initial * 100;}),
      };
    }),
    layout: {
      title: curve === "drawdown" ? "End-of-period drawdown" : "Portfolio growth",
      xaxis: {title: {text: "Recorded period end"}},
      yaxis: {title: {text: curve === "drawdown" ? "Drawdown (%)" : "Initial capital = 100"}},
      legend: {orientation: "h", y: -0.3},
    },
  } : null;
  const intervals = [...new Set(entries.flatMap(e => e.observations.map(o => `${o.start} → ${o.end}`)))].sort();
  return <div className="space-y-4">
    <p className="text-xs text-gray-600">Every execution is retained. Open its full results for portfolio weights and allocations.</p>
    <div className="overflow-x-auto"><table className="w-full text-sm">
      <thead><tr><th className={cell}>Execution</th>{MATRIX_METRICS.map(([key, label]) => <th key={key} className={cell}>{label}</th>)}</tr></thead>
      <tbody>{entries.map((entry, i) => <tr key={entry.id} className="border-t border-gray-100">
        <th scope="row" className={cell}>
          <Link className="text-blue-700 underline" to={`/runs/${entry.run.id}`} state={{comparisonSearch}}>{resultAlias(i)} · {entry.name}</Link>
          <span className="block text-xs font-normal text-gray-500">{candidateDescription(entry)} · {executionLabel(entry.run)} · {entry.run.id}</span>
        </th>
        {MATRIX_METRICS.map(([key]) => <td key={key} className={cell} title={metricRestriction(key, [entry]) ?? undefined}>
          {savedMetric(entry, key)}{metricRestriction(key, [entry]) && <span className="block text-xs text-amber-800">Reported only</span>}
        </td>)}
      </tr>)}</tbody>
    </table></div>
    <details className="rounded border border-gray-200 p-3">
      <summary className="cursor-pointer text-sm font-medium">Growth and drawdown curves</summary>
      <p className="my-2 text-xs text-gray-500">Saved valuations only. Drawdown includes initial capital in the running peak and is sampled at period ends, excluding intraperiod losses.</p>
      {canDraw ? <>
        <label htmlFor={`${id}-curve`} className="block text-sm">Performance curve</label>
        <select id={`${id}-curve`} className="mt-1 rounded border border-gray-300 p-2 text-sm" value={curve} onChange={event => setCurve(event.target.value)}>
          <option value="none">No curve</option><option value="growth">Growth (initial capital = 100)</option><option value="drawdown">Drawdown (%)</option>
        </select>
        {figure && <Suspense fallback={<p role="status">Loading curve…</p>}><PlotlyChart figure={figure} /></Suspense>}
      </> : <p className="text-sm text-amber-800">Curves require known matching coverage and context, and saved valuations.</p>}
    </details>
    {intervals.length > 0 && <details onToggle={event => setReturnsOpen(event.currentTarget.open)} className="rounded border border-gray-200 p-3">
      <summary className="cursor-pointer text-sm font-medium">Saved period returns</summary>
      <p className="mt-2 text-xs text-gray-500">Exact recorded boundaries. Missing returns stay blank; values are not reconstructed from valuations.</p>
      {returnsOpen && <div className="mt-2 overflow-x-auto"><table className="w-full text-sm">
        <thead><tr><th className={cell}>Recorded period</th>{entries.map((e, i) => <th key={e.id} className={cell}>{resultAlias(i)} · {e.name}</th>)}</tr></thead>
        <tbody>{intervals.map(interval => <tr key={interval} className="border-t border-gray-100">
          <th scope="row" className={cell}>{interval}</th>
          {entries.map(e => <td key={e.id} className={cell}>{formatMetricValue("total_return", e.observations.find(o => `${o.start} → ${o.end}` === interval)?.value)}</td>)}
        </tr>)}</tbody>
      </table></div>}
    </details>}
  </div>;
}
