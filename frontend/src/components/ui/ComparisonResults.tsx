import { lazy, Suspense, useId, useState } from "react";
import { comparisonCompatibility, configurationDifferences, finite, type SavedCandidate } from "@/lib/savedComparison";
import { comparisonValue, metricRows } from "@/lib/comparisonMetrics";
import { metricLabel } from "@/lib/metrics";
const PlotlyChart = lazy(() => import("@/components/ui/PlotlyChart"));
const field = "rounded-md border border-gray-300 bg-white px-3 py-2 text-sm";
export function ComparisonTable({ entries, rows, showRunIds = true }: { entries: SavedCandidate[]; rows: {key: string; label: string; display: string[]; note?: string; titles?: string[]}[]; showRunIds?: boolean }) {
  return <div className="overflow-x-auto rounded-lg border border-gray-200 bg-white"><table className="w-full text-left text-sm"><thead className="bg-gray-50"><tr><th className="min-w-44 px-4 py-3">Measure / setting</th>{entries.map(e => <th key={e.id} className="min-w-44 max-w-72 break-words px-4 py-3">{e.name}{showRunIds && <span className="block text-xs font-normal text-gray-500">{e.run.id}</span>}</th>)}</tr></thead><tbody className="divide-y divide-gray-100">{rows.map(row => <tr key={row.key}><th scope="row" className="px-4 py-3 font-medium">{row.label}{row.note && <span className="block text-xs font-normal text-amber-800">{row.note}</span>}</th>{row.display.map((value, i) => <td key={entries[i].id} title={row.titles?.[i]} className="max-w-72 px-4 py-3 align-top [overflow-wrap:anywhere]">{value}</td>)}</tr>)}</tbody></table></div>;
}
export function CompatibilityNotice({ entries, manifests = {} }: {entries: SavedCandidate[]; manifests?: Record<string, unknown>}) {
  const compatibility = comparisonCompatibility(entries, manifests);
  return <section aria-label="Comparison compatibility" className={`rounded-lg border p-4 ${compatibility.state === "matching" ? "border-green-200 bg-green-50" : "border-amber-200 bg-amber-50"}`}><h2 className="font-medium">{{matching: "Comparable recorded context", varied: "Comparison with recorded differences", unverified: "Compatibility unverified", blocked: "Results cannot be compared"}[compatibility.state]}</h2><p className="mt-1 text-sm">{compatibility.reasons[0]}</p>{compatibility.reasons.length > 1 && <details className="mt-2 text-sm"><summary className="cursor-pointer">{compatibility.reasons.length - 1} additional context checks</summary><ul className="mt-2 list-disc space-y-1 pl-5">{compatibility.reasons.slice(1).map(reason => <li key={reason}>{reason}</li>)}</ul></details>}</section>;
}
function Holdings({entries}: {entries: SavedCandidate[]}) {
  const id = useId();
  const [period, setPeriod] = useState("");
  const periods = [...new Set(entries.flatMap(e => e.holdings.map(h => h.label)))];
  const selected = periods.includes(period) ? period : periods.at(-1);
  if (!periods.length) return <p className="text-sm text-gray-500">Portfolio weights were not recorded for these results.</p>;
  return <div className="space-y-4"><label htmlFor={`${id}-period`} className="block space-y-1 text-sm font-medium">Recorded period</label><select id={`${id}-period`} className={`${field} block w-full`} value={selected ?? ""} onChange={e => setPeriod(e.target.value)}>{periods.map(p => <option key={p}>{p}</option>)}</select><p className="text-xs text-gray-500">Saved target-weight summaries; positions below 1% may be omitted. Weights are shown without rescaling. These are not end-of-period holdings or broker positions.</p><div className={`grid gap-4 ${entries.length > 1 ? "lg:grid-cols-2" : ""}`}>{entries.map(entry => {
    const saved = entry.holdings.find(h => h.label === selected);
    const values = Object.entries(saved?.weights ?? {});
    const weights = values.filter((pair): pair is [string, number] => finite(pair[1])).sort((a, b) => b[1] - a[1]);
    return <article key={entry.id} className="min-w-0 rounded-lg border border-gray-200 p-4"><h3 className="font-semibold">{entry.name}</h3>{saved?.executionTime && <p className="mt-1 text-xs text-gray-500">Execution time: {saved.executionTime}</p>}{weights.length ? <><table className="mt-3 w-full text-sm"><thead><tr className="border-b text-left"><th className="py-2">Asset</th><th className="text-right">Target weight</th></tr></thead><tbody>{weights.map(([symbol, weight]) => <tr key={symbol} className="border-b border-gray-100"><th scope="row" className="py-2 text-left font-medium">{symbol}</th><td className="text-right tabular-nums" title={`${weight * 100}%`}>{(weight * 100).toFixed(2)}%</td></tr>)}</tbody></table><p className="mt-2 text-xs text-gray-500">{weights.length} recorded assets · total {(weights.reduce((sum, [, w]) => sum + w, 0) * 100).toFixed(2)}%</p></> : <p className="mt-3 text-sm text-gray-500">{saved?.recorded && !saved.incomplete ? "No positions in the saved summary; positions below 1% may be omitted." : "Weights not recorded for this period."}</p>}{(saved?.incomplete || values.length > weights.length) && <p role="status" className="mt-2 text-sm text-amber-800">Some saved weights are invalid; the displayed allocation is incomplete.</p>}</article>;
  })}</div></div>;
}
/** One display for stored results. No performance recomputation or experiment launch. */
export default function ComparisonResults({entries, manifests = {}, reportedLegacy = false, hideCharts = false}: {entries: SavedCandidate[]; manifests?: Record<string, unknown>; reportedLegacy?: boolean; hideCharts?: boolean}) {
  const id = useId();
  const [tab, setTab] = useState("overview");
  const [allSettings, setAllSettings] = useState(false);
  const [allMetrics, setAllMetrics] = useState(false);
  const [chart, setChart] = useState("");
  if (!entries.length) return null;
  const family = entries[0].family;
  const compatibility = comparisonCompatibility(entries, manifests);
  const blocked = compatibility.state === "blocked";
  const single = entries.length === 1;
  const canDraw = single ? entries[0].periods !== null || family === "solvers" : compatibility.chartAllowed;
  const metrics = metricRows(entries);
  const differences = configurationDifferences(entries, manifests);
  const paths = family === "backtests" && entries.every(e => e.points.length && finite(e.metrics.initial_value) && e.metrics.initial_value > 0);
  const defaultChart = paths ? "portfolio" : metrics.find(m => !m.reason)?.key ?? "";
  const currentChart = chart === "none" ? "none" : chart === "portfolio" && paths || metrics.some(m => m.key === chart && !m.reason) ? chart : defaultChart;
  const chartMetric = metrics.find(m => m.key === currentChart && !m.reason);
  const figure = paths && currentChart === "portfolio" ? {
    data: entries.map(e => ({type: "scatter", mode: "lines+markers", name: `${e.name} · ${e.run.id}`, x: e.points.map(p => p.date), y: e.points.map(p => 100 * p.value / (e.metrics.initial_value as number))})),
    layout: {title: "Portfolio growth (initial capital = 100)", xaxis: {title: {text: "Recorded period end"}}, yaxis: {title: {text: "Initial capital = 100"}}, legend: {orientation: "h", y: -0.3}},
  } : chartMetric ? {
    data: [{type: "bar", x: entries.map(e => `${e.name} · ${e.run.id}`), y: entries.map(e => finite(e.metrics[currentChart]) ? e.metrics[currentChart] : null), text: entries.map(e => comparisonValue(family, currentChart, e.metrics[currentChart])), textposition: "auto", hovertemplate: "%{x}<br>%{text}<extra></extra>"}],
    layout: {title: chartMetric.label, xaxis: {automargin: true}, yaxis: {title: {text: "Saved value (percentages stored as fractions)"}}},
  } : null;
  const displayRows = metrics.map(row => ({...row, note: row.reason ? reportedLegacy ? `Reported values only. ${row.reason}` : row.reason : undefined, display: entries.map(e => row.reason && !reportedLegacy ? "Not comparable" : comparisonValue(family, row.key, e.metrics[row.key])), titles: row.key === "annualized_return" ? entries.map(e => metricLabel(row.key, e.definitions)) : undefined}));
  return <div className="space-y-4">
    <div role="tablist" aria-label="Result sections" className="flex gap-1 overflow-x-auto border-b border-gray-200">{[["overview", "Overview"], ...(family === "backtests" ? [["holdings", "Holdings"]] : []), ["configuration", "Configuration"]].map(([value, label]) => <button key={value} role="tab" id={`${id}-${value}`} aria-controls={`${id}-panel`} aria-selected={tab === value} tabIndex={tab === value ? 0 : -1} onKeyDown={event => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const buttons = [...event.currentTarget.parentElement!.querySelectorAll<HTMLButtonElement>("button")];
      const index = buttons.indexOf(event.currentTarget);
      const next = event.key === "Home" ? 0 : event.key === "End" ? buttons.length - 1 : (index + (event.key === "ArrowRight" ? 1 : -1) + buttons.length) % buttons.length;
      buttons[next].click(); buttons[next].focus();
    }} className={`border-b-2 px-4 py-3 text-sm font-medium ${tab === value ? "border-blue-600 text-blue-700" : "border-transparent text-gray-500"}`} onClick={() => setTab(value)}>{label}</button>)}</div>
    <div role="tabpanel" id={`${id}-panel`} aria-labelledby={`${id}-${tab}`}>
    {tab === "overview" && !blocked && <section aria-label="Saved metrics" className="space-y-4"><h2 className="text-lg font-semibold">Saved results</h2><p className="text-sm text-gray-600">{!canDraw && !single ? "Saved values only: compatibility is unverified. These values do not establish a ranking." : "Values retain their recorded definitions and evaluation coverage. No automatic ranking is applied."}</p>
      <ComparisonTable entries={entries} rows={allMetrics ? displayRows : displayRows.slice(0, 8)} />
      {displayRows.length > 8 && <button className="text-sm text-blue-700 underline" onClick={() => setAllMetrics(v => !v)}>{allMetrics ? "Show key metrics" : `Show all ${displayRows.length} metrics`}</button>}
      {!hideCharts && canDraw && <div className="max-w-xl"><label htmlFor={`${id}-chart`} className="block text-sm font-medium">Chart</label><select id={`${id}-chart`} className={`${field} block w-full`} value={currentChart} onChange={e => setChart(e.target.value)}><option value="none">No chart</option>{paths && <option value="portfolio">Portfolio growth (initial capital = 100)</option>}{metrics.filter(m => !m.reason).map(m => <option key={m.key} value={m.key}>{m.label}</option>)}</select></div>}
      {!hideCharts && canDraw && figure && currentChart !== "none" && <Suspense fallback={<p role="status">Loading chart…</p>}><PlotlyChart figure={figure} /></Suspense>}
    </section>}
    {tab === "overview" && blocked && <p className="text-sm text-amber-800">Results are not comparable. Inspect Configuration or open the individual runs.</p>}
    {tab === "holdings" && <section aria-label="Recorded holdings"><Holdings key={entries.map(e => e.id).join("|")} entries={entries} /></section>}
    {tab === "configuration" && <section aria-label="Configuration differences" className="space-y-3"><div className="flex flex-wrap items-center justify-between gap-3"><h2 className="text-lg font-semibold">Configuration differences</h2><label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={allSettings} onChange={e => setAllSettings(e.target.checked)} />Show shared settings too</label></div>{differences.some(r => r.different) || allSettings || single ? <ComparisonTable entries={entries} rows={differences.filter(r => allSettings || single || r.different)} /> : <p className="text-sm text-gray-600">No differences in the recorded settings. Enable shared settings to inspect the saved context.</p>}</section>}
    </div>
  </div>;
}
