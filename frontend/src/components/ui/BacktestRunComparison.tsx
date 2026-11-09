import { useId, useState } from "react";
import { configurationDifferences, type SavedCandidate } from "@/lib/savedComparison";
import { candidateDescription, coverageLabel, executionLabel, metricDelta, periodSpan, recordedChanges, savedBenchmark, savedMetric } from "@/lib/compareRuns";
import { configurationColumns, configurationKey, consistency, MATRIX_METRICS, pairedResult, periodRows, type PeriodRow } from "@/lib/compareMatrix";
import { metricRestriction } from "@/lib/comparisonMetrics";
import { libraryUniverse } from "@/lib/runLibrary";
import { ComparisonTable } from "./ComparisonResults";
import BacktestPeriodDetails from "./BacktestPeriodDetails";

const field = "rounded-md border border-gray-300 bg-white px-3 py-2 text-sm";
const cell = "px-3 py-3 text-left align-top";
const difference = (value: number | null) => value == null ? "—" : `${value > 0 ? "+" : ""}${(100 * value).toFixed(2)} pp`;

function MatrixCell({executions, row, referenceKey, manifests, metric, display}: {
  executions: SavedCandidate[]; row: PeriodRow; referenceKey: string;
  manifests: Record<string, unknown>; metric: string; display: string;
}) {
  return <td className={`${cell} tabular-nums`}>
    {!executions.length ? <span title="No saved result for this configuration in this context" className="text-gray-400">—</span> : <>
      {executions.length > 1 && <span className="mb-2 block text-xs text-amber-800">{executions.length} executions; shown separately</span>}
      {executions.map(entry => {
        const pair = pairedResult(row, entry, referenceKey, manifests, metric);
        const restriction = metricRestriction(metric, [entry]);
        return <div key={entry.id} className="mb-2">
          <span title={display === "differences" ? pair.reason ?? "Result minus reference in this period" : restriction ?? undefined}>
            {display === "values" ? savedMetric(entry, metric) : pair.reason ? "Not comparable" : metricDelta(metric, entry, pair.reference!, true)}
          </span>
          {display === "differences" && pair.reason && <span className="mt-1 block max-w-56 text-xs text-amber-800">{pair.reason}</span>}
          {display === "values" && restriction && <span className="block text-xs text-amber-800">Reported only: {restriction}</span>}
          {executions.length > 1 && <span className="block text-xs text-gray-500">{executionLabel(entry.run)} · {entry.run.id} · {entry.name}</span>}
        </div>;
      })}
    </>}
  </td>;
}

export default function BacktestRunComparison({entries, candidates, manifests, referenceId, onReferenceChange, comparisonSearch}: {
  entries: SavedCandidate[]; candidates: SavedCandidate[]; manifests: Record<string, unknown>;
  referenceId: string; onReferenceChange: (id: string) => void; comparisonSearch: string;
}) {
  const id = useId();
  const [metric, setMetric] = useState("total_return");
  const [display, setDisplay] = useState("values");
  const [expanded, setExpanded] = useState<string[]>([]);
  const [shared, setShared] = useState(false);
  const columns = configurationColumns(entries);
  const rows = periodRows(entries, manifests);
  const reference = entries.find(e => e.id === referenceId) ?? entries[0];
  const referenceKey = configurationKey(reference);
  const representatives = columns.map(c => ({...c.entries[0], name: `${c.label} · ${c.entries[0].name}`}));
  const strategyRows = configurationDifferences(representatives, {}).filter(row => row.key === "method" || row.key.startsWith("configuration:"));
  const repeatedName = columns.some(c => columns.some(other => other.key !== c.key && other.entries[0].name === c.entries[0].name));

  return <div className="space-y-5">
    <section aria-label="Configuration differences" className="rounded-lg border border-gray-200 bg-white p-4">
      <h2 className="font-semibold">What changed between configurations?</h2>
      <p className="mt-1 text-xs text-gray-500">Columns match recorded strategy parameters, including optimizer budgets and random seeds. Names, periods and datasets do not establish a match. Missing settings are never filled from current defaults.</p>
      {repeatedName && <p className="mt-2 text-sm text-amber-800">Some names refer to different recorded settings. Use the C labels and parameter differences to distinguish them.</p>}
      {(shared || strategyRows.some(row => row.different)) && <div className="mt-3 overflow-x-auto"><ComparisonTable entries={representatives} rows={strategyRows.filter(row => shared || row.different)} showRunIds={false} /></div>}
      {!strategyRows.some(row => row.different) && <p className="mt-2 text-sm text-gray-500">No differences in the recorded strategy settings. This does not establish that unrecorded settings match.</p>}
      <details className="mt-3 text-sm">
        <summary className="cursor-pointer">Configuration identities and shared settings</summary>
        <label className="my-3 flex items-center gap-2"><input type="checkbox" checked={shared} onChange={event => setShared(event.target.checked)} />Include identical settings above</label>
        <ul className="space-y-3">{columns.map(c => <li key={c.key}>
          <strong>{c.label} · {c.entries[0].name}</strong><span className="block text-xs text-gray-600">{candidateDescription(c.entries[0])}</span>
          <span className="block text-xs text-gray-500">Saved names: {[...new Set(c.entries.map(e => e.name))].join(" · ")} · {c.entries.length} executions</span>
        </li>)}</ul>
      </details>
    </section>

    <section aria-label="Backtest results" className="space-y-3">
      <h2 className="text-lg font-semibold">Configurations by evaluation period</h2>
      <p className="text-sm text-gray-600">Different coverage, data, history or engine contexts have separate rows. Differences use a reference from the same row only. Independent runs are never chained or averaged.</p>
      <div className="flex flex-wrap items-end gap-4">
        <div><label htmlFor={`${id}-metric`} className="mb-1 block text-sm font-medium">Metric</label>
          <select id={`${id}-metric`} className={field} value={metric} onChange={event => setMetric(event.target.value)}>{MATRIX_METRICS.map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select>
        </div>
        <div><label htmlFor={`${id}-reference`} className="mb-1 block text-sm font-medium">Reference configuration</label>
          <select id={`${id}-reference`} className={field} value={referenceKey} onChange={event => onReferenceChange(columns.find(c => c.key === event.target.value)!.entries[0].id)}>{columns.map(c => <option key={c.key} value={c.key}>{c.label} · {c.entries[0].name}</option>)}</select>
        </div>
        <div><label htmlFor={`${id}-display`} className="mb-1 block text-sm font-medium">Display</label>
          <select id={`${id}-display`} className={field} value={display} onChange={event => setDisplay(event.target.value)}><option value="values">Saved values</option><option value="differences">Differences from reference</option></select>
        </div>
      </div>
      <div role="region" aria-label="Results matrix" tabIndex={0} className="overflow-x-auto rounded-lg border border-gray-200 bg-white">
        <table className="w-full text-sm">
          <thead className="bg-gray-50"><tr>
            <th className={`${cell} sticky left-0 z-10 min-w-60 bg-gray-50`}>Period / run context</th><th className={`${cell} min-w-40`}>Saved benchmark</th>
            {columns.map(c => <th key={c.key} className={`${cell} min-w-40`}>
              <span className="block">{c.label} · {c.entries[0].name}</span><span className="mt-1 block max-w-56 text-xs font-normal text-gray-500">{candidateDescription(c.entries[0])}</span>
              {c.key === referenceKey && <span className="block text-xs text-blue-700">Reference</span>}
            </th>)}
          </tr></thead>
          <tbody>{rows.map(row => {
            const first = row.entries[0], opened = expanded.includes(row.key);
            const benchmarks = [...new Map(row.entries.flatMap(e => {const b = savedBenchmark(e, candidates); return b ? [[b.id, b] as const] : [];})).values()];
            return <tr key={row.key} className="border-t border-gray-100">
              <th scope="row" className={`${cell} sticky left-0 bg-white font-normal`}>
                <strong className="block">{periodSpan(first)}</strong><span className="mt-1 block text-xs text-gray-500">{coverageLabel(first)}</span>
                <span className="mt-1 block text-xs text-gray-600">{libraryUniverse(first.run).name}</span>
                <span className="mt-1 block text-xs text-gray-500">Historical input start: {String(first.run.request_payload.historical_start_date ?? "Not recorded")}</span>
                {recordedChanges(first, reference, manifests).filter(note => !["Strategy parameters differ from reference", "Strategy parameters not recorded"].includes(note)).map(note => <span key={note} className="mt-1 block text-xs text-gray-500">{note}</span>)}
                <button className="mt-2 text-xs text-blue-700 underline" aria-expanded={opened} onClick={() => setExpanded(opened ? expanded.filter(key => key !== row.key) : [...expanded, row.key])}>{opened ? "Hide" : "Inspect"} period details</button>
                <span className="mt-1 block text-[10px] text-gray-500">{[...new Set(row.entries.map(e => e.run.id))].join(" · ")}</span>
              </th>
              <td className={cell}>{benchmarks.length ? benchmarks.map(b => <div key={b.id} className="mb-2">
                <span>{savedMetric(b, metric)}</span><span className="block text-xs text-gray-500">{b.name} · {String(b.configuration.settings.asset_symbol ?? "Asset not recorded")} · {b.run.id}</span>
                {metricRestriction(metric, [b]) && <span className="block text-xs text-amber-800">Reported only</span>}
              </div>) : <span className="text-gray-500">Not recorded</span>}</td>
              {columns.map(c => <MatrixCell key={c.key} executions={row.entries.filter(e => configurationKey(e) === c.key)} row={row} referenceKey={referenceKey} manifests={manifests} metric={metric} display={display} />)}
            </tr>;
          })}</tbody>
        </table>
      </div>
      <p className="text-xs text-gray-500">Percentage differences use percentage points (pp); ratios use absolute differences. Benchmark values are retained observations, not an inferred market index. Their identity must be checked in the source run.</p>
    </section>

    {rows.filter(row => expanded.includes(row.key)).map(row => <section key={row.key} aria-label={`Period details ${periodSpan(row.entries[0])}`} className="space-y-3 rounded-lg border border-gray-200 bg-white p-4">
      <h2 className="font-semibold">{periodSpan(row.entries[0])} · period details</h2>
      <BacktestPeriodDetails entries={row.entries} manifests={manifests} comparisonSearch={comparisonSearch} />
      <details><summary className="cursor-pointer text-sm">Recorded run and data context</summary><div className="mt-3 overflow-x-auto"><ComparisonTable entries={row.entries} rows={configurationDifferences(row.entries, manifests)} /></div></details>
    </section>)}

    {columns.length > 1 && <section aria-label="Consistency summary" className="rounded-lg border border-gray-200 bg-white p-4">
      <h2 className="font-semibold">Consistency against {columns.find(c => c.key === referenceKey)!.label}</h2>
      <p className="my-2 text-xs text-gray-500">Descriptive paired observations, not statistical evidence or a strategy ranking. Only unique executions with matching context count. Overlapping windows, missing metrics and absent references are excluded. Partial periods retain their original coverage in the matrix; there is no combined return.</p>
      <div className="overflow-x-auto"><table className="w-full text-sm">
        <thead><tr>{["Configuration", "Matched periods", "Higher return", "Median return difference", "Worse drawdown"].map(label => <th key={label} className={cell}>{label}</th>)}</tr></thead>
        <tbody>{columns.filter(c => c.key !== referenceKey).map(c => {
          const result = consistency(rows, c.key, referenceKey, manifests);
          return <tr key={c.key} className="border-t border-gray-100">
            <th scope="row" className={cell}>{c.label} · {c.entries[0].name}</th>
            <td className={cell}>{result.matched} / {rows.length}<span className="block text-xs text-gray-500">{result.excluded} excluded</span></td>
            <td className={cell}>{result.matched ? `${result.higher} / ${result.matched}` : "—"}</td>
            <td className={cell}>{difference(result.median)}</td>
            <td className={cell}>{result.drawdownPairs ? `${result.worseDrawdown} / ${result.drawdownPairs}` : "Not recorded"}</td>
          </tr>;
        })}</tbody>
      </table></div>
    </section>}
  </div>;
}
