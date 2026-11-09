import { record } from "@/lib/runPresentation";
import { lastCompleteDate } from "@/lib/periodDates";
import type { RunDetail } from "@/types/api";

const text = (value: unknown) => typeof value === "string" ? value : "Not recorded";
export default function InputFreshness({run}: {run: RunDetail}) {
  const result = record(run.result);
  const data = record(result.data_freshness);
  const assets = Array.isArray(data.assets) ? data.assets.map(record) : [];
  const excluded = Object.entries(record(data.excluded_assets));
  const lagging = assets.filter(a => a.status === "behind_other_assets");
  const unknown = assets.filter(a => a.status === "timing_unavailable" || a.status === "no_observable_history");
  const historical = (data.frequency === "monthly" || data.frequency === "weekly") && typeof data.newest_asset_period_end === "string" && data.newest_asset_period_end < lastCompleteDate(data.frequency);
  return <div aria-label="Input data freshness" className="space-y-3 rounded border border-gray-200 bg-gray-50 p-4 text-sm">
    <h3 className="font-medium">Input data freshness</h3>
    {!Object.keys(data).length ? <p className="text-amber-800">Detailed timing was not recorded for this older result. The requested cutoff does not establish actual data freshness.{typeof result.data_asof === "string" ? ` Saved data-as-of label: ${result.data_asof}.` : ""}</p> : <>
      <dl className="grid gap-2 sm:grid-cols-2">
        <div><dt className="text-gray-500">Decision time</dt><dd>{text(data.decision_time)}</dd></div>
        <div><dt className="text-gray-500">Frequency</dt><dd>{text(data.frequency)}</dd></div>
        <div><dt className="text-gray-500">Requested historical cutoff</dt><dd>{text(data.requested_end_date)}</dd></div>
        <div><dt className="text-gray-500">Latest input availability (UTC)</dt><dd>{text(data.latest_available_at)}</dd></div>
        <div><dt className="text-gray-500">Latest observed period ends across assets</dt><dd>{text(data.oldest_asset_period_end)} → {text(data.newest_asset_period_end)}</dd></div>
      </dl>
      <p className="text-xs text-gray-500">Completed price histories visible to the strategy. Model-specific estimation or news filters may narrow these inputs further. These are observation and availability times, not confirmed execution prices or fills.</p>
      {historical && <p role="status" className="text-amber-800">The newest observed period predates the latest completed calendar period today. Check the saved decision date, historical cutoff and data coverage before using this recommendation.</p>}
      {(lagging.length > 0 || unknown.length > 0 || excluded.length > 0) && <p role="status" className="text-amber-800">{lagging.length} assets end earlier than their peers · {unknown.length} have unavailable timing/history · {excluded.length} excluded before strategy input. Details below.</p>}
      <details><summary className="cursor-pointer font-medium">Asset coverage and exclusions</summary>
        <div className="mt-2 max-h-72 overflow-auto"><table className="w-full text-xs"><thead><tr className="text-left"><th className="p-2">Asset</th><th className="p-2">Last observed period end</th><th className="p-2">Latest availability (UTC)</th><th className="p-2">Coverage</th></tr></thead><tbody>
          {assets.map(a => <tr key={text(a.symbol)}><th className="p-2 text-left">{text(a.symbol)}</th><td className="p-2">{text(a.last_period_end)}</td><td className="p-2">{text(a.latest_available_at)}</td><td className="p-2">{a.status === "observed" ? "Aligned with latest observed period" : a.status === "behind_other_assets" ? "Earlier than peers" : "History/timing unavailable"}</td></tr>)}
          {excluded.map(([symbol, reason]) => <tr key={`excluded-${symbol}`}><th className="p-2 text-left">{symbol}</th><td className="p-2" colSpan={3}>Excluded: {text(reason)}</td></tr>)}
        </tbody></table></div>
        <p className="mt-2 text-xs text-gray-500">Earlier-than-peer coverage is not a market-calendar completeness test. See the saved validation report for additional validation issues.</p>
      </details>
    </>}
  </div>;
}
