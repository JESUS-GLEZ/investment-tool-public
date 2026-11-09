/** A compact library of saved experiments. Results live on each run's detail page. */
import { useQuery } from "@tanstack/react-query";
import { Link, useSearchParams } from "react-router-dom";
import { listCampaigns, listRuns } from "@/api/endpoints";
import { CampaignPanel, RunCampaignAssignment } from "@/components/ui/Campaigns";
import WorkerHealth from "@/components/ui/WorkerHealth";
import type { RunSummary } from "@/types/api";
import { statusColor, fmtDate } from "@/lib/utils";
import { configurationSettings, experimentRows, recordedConfigurations } from "@/lib/runPresentation";
import { libraryPeriod, libraryRunType, libraryUniverse, searchLibrary } from "@/lib/runLibrary";

const PAGE_SIZE = 10;
const TYPES = { all: "All types", execution: "Single-strategy runs", strategy_comparison: "Strategy comparisons", solver_comparison: "Solver benchmarks", prediction_comparison: "Prediction evaluations" };
const STATUSES = ["all", "queued", "running", "completed", "failed", "cancelled"];
const GRID = "md:grid md:grid-cols-[1fr_1.25fr_1.15fr_1.35fr_5.5rem_7rem] md:items-center md:gap-4";
const CONTROL = "w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-sm";

function SavedSummary({ run }: { run: RunSummary }) {
  const configurations = recordedConfigurations(run);
  const universe = libraryUniverse(run);
  const rows = experimentRows(run).filter(([key]) => ["Data", "Historical window", "Requested benchmark", "Evaluation", "Sentiment analysis", "Multi-Greedy seeds", "Convex MV reference"].includes(key));
  return <div className="space-y-4 border-t border-gray-200 bg-gray-50 px-4 py-4">
    <dl className="grid gap-x-6 gap-y-3 text-sm sm:grid-cols-2 lg:grid-cols-3">
      {[["Run ID", run.id], ["Created", fmtDate(run.created_at)], ...rows, ...(universe.file ? [["Source file", universe.file]] : [])].map(([key, value]) => <div key={key} className="min-w-0">
        <dt className="text-xs font-medium text-gray-500">{key}</dt>
        <dd className="mt-1 break-words [overflow-wrap:anywhere]">{value}</dd>
      </div>)}
    </dl>
    <div>
      <h3 className="text-sm font-semibold text-gray-800">Configurations ({configurations.length})</h3>
      <ul className="mt-2 grid gap-3 sm:grid-cols-2">{configurations.map((config, index) =>
        <li key={index} className="min-w-0 text-sm">
          <p className="break-words font-medium">{config.name}{config.name !== config.method && <span className="font-normal text-gray-500"> · {config.method}</span>}</p>
          <p className="mt-1 text-xs leading-relaxed text-gray-600">{configurationSettings(config, true)}</p>
        </li>)}</ul>
      {!configurations.length && <p className="mt-2 text-sm text-gray-500">No configurations recorded.</p>}
    </div>
    <RunCampaignAssignment run={run} />
    <WorkerHealth run={run} />
    <p className="text-xs text-gray-500">Submitted settings only. Open details for full parameters, actual coverage and results.</p>
  </div>;
}

export default function DashboardPage() {
  const { data: runs, isLoading, error, isFetching, refetch } = useQuery({
    queryKey: ["runs"], queryFn: listRuns, staleTime: 0,
    refetchInterval: query => query.state.data?.some(run => run.status === "queued" || run.status === "running") ? 5000 : false,
  });
  const campaigns = useQuery({queryKey: ["campaigns"], queryFn: listCampaigns});
  const campaignNames = new Map((campaigns.data ?? []).map(item => [item.id, item.name]));
  const [params, setParams] = useSearchParams();
  const query = params.get("q") ?? "";
  const type = Object.hasOwn(TYPES, params.get("type") ?? "") ? params.get("type")! : "all";
  const status = STATUSES.includes(params.get("status") ?? "") ? params.get("status")! : "all";
  const sort = params.get("sort") === "oldest" ? "oldest" : "newest";
  const expanded = new Set((params.get("expanded") ?? "").split(",").filter(Boolean));
  const campaign = params.get("campaign") ?? "all";
  const campaignRuns = (runs ?? []).filter(run => campaign === "all" || (campaign === "ungrouped" ? !run.campaign_id : run.campaign_id === campaign));
  const filtered = searchLibrary(campaignRuns, query, type, status, sort);
  const pageCount = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const requestedPage = Number(params.get("page") ?? "1");
  const page = Number.isSafeInteger(requestedPage) && requestedPage > 0 ? Math.min(pageCount, requestedPage) : 1;
  const visible = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);
  function update(values: Record<string, string>, resetPage = true) {
    const next = new URLSearchParams(params);
    for (const [key, value] of Object.entries(values)) {
      if (!value || value === "all" || key === "sort" && value === "newest") next.delete(key);
      else next.set(key, value);
    }
    if (resetPage) next.delete("page");
    setParams(next, { replace: true });
  }
  function toggleSummary(id: string) {
    const next = new Set(expanded);
    if (next.has(id)) next.delete(id); else next.add(id);
    update({ expanded: [...next].join(",") }, false);
  }
  const hasFilters = campaign !== "all" || !!query || type !== "all" || status !== "all" || sort !== "newest";
  return <div className="space-y-5">
    <div className="flex flex-wrap items-center justify-between gap-3">
      <div>
        <h1 className="text-xl font-semibold text-gray-900">Dashboard</h1>
        <p className="mt-1 text-sm text-gray-500">Find a run and open its details.</p>
      </div>
      <div className="flex flex-wrap gap-2">
        <Link to="/runs/new" className="rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700">New Run</Link>
      </div>
    </div>
    <CampaignPanel value={campaign} onChange={campaign => update({campaign})} />
    <section aria-label="Find runs" className="grid gap-3 rounded-lg border border-gray-200 bg-white p-4 sm:grid-cols-2 md:grid-cols-[2fr_1fr_1fr_1fr]">
      <label className="text-xs font-medium text-gray-600">Search runs<input type="search" value={query} placeholder="ID, configuration, universe or period" onChange={event => update({ q: event.target.value })} className={`mt-1 ${CONTROL}`} />
      </label>
      <label className="text-xs font-medium text-gray-600">Type<select value={type} onChange={event => update({ type: event.target.value })} className={`mt-1 ${CONTROL}`}>{Object.entries(TYPES).map(([value, label]) =>
        <option key={value} value={value}>{label}</option>)}</select>
      </label>
      <label className="text-xs font-medium text-gray-600">Status<select value={status} onChange={event => update({ status: event.target.value })} className={`mt-1 ${CONTROL}`}>{STATUSES.map(value =>
        <option key={value} value={value}>{value === "all" ? "All statuses" : value[0].toUpperCase() + value.slice(1)}</option>)}</select>
      </label>
      <label className="text-xs font-medium text-gray-600">Sort<select value={sort} onChange={event => update({ sort: event.target.value })} className={`mt-1 ${CONTROL}`}>
        <option value="newest">Newest first</option>
        <option value="oldest">Oldest first</option>
      </select>
      </label>
    </section>
    {isLoading && <p role="status" className="text-sm text-gray-500">Loading runs…</p>}
    {error && <div role="alert" className="rounded border border-red-200 bg-red-50 p-3 text-sm text-red-700">Could not refresh runs.{runs && " Showing the last loaded list."} <button type="button" disabled={isFetching} onClick={() => void refetch()} className="underline">Retry</button>
    </div>}
    {runs && <>
      <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
        <p role="status" className="text-gray-500">{filtered.length ? `${(page - 1) * PAGE_SIZE + 1}–${Math.min(page * PAGE_SIZE, filtered.length)} of ${filtered.length}` : "0"} runs{hasFilters ? ` · ${runs.length} total` : ""}</p>
        <div className="flex gap-4">
          <button type="button" disabled={isFetching} onClick={() => void refetch()} className="text-gray-600 underline disabled:opacity-40" aria-label="Refresh runs">{isFetching ? "Refreshing…" : "Refresh"}</button>{hasFilters && <button type="button" onClick={() => update({ q: "", type: "all", status: "all", sort: "newest", campaign: "all" })} className="text-blue-700 underline">Clear filters</button>}</div>
      </div>
      {!runs.length ? <p className="rounded-lg border border-gray-200 bg-white p-6 text-sm text-gray-500">No runs yet. Create your first run to get started.</p> : !filtered.length ? <p className="rounded-lg border border-gray-200 bg-white p-6 text-sm text-gray-500">No runs match these filters.</p> : <section aria-label="Runs" className="overflow-hidden rounded-lg border border-gray-200 bg-white">
        <div aria-hidden="true" className={`hidden border-b border-gray-200 bg-gray-50 px-4 py-3 text-xs font-medium text-gray-500 ${GRID}`}>
          <span>Type</span>
          <span>Campaign</span>
          <span>Dates</span>
          <span>Asset universe / dataset</span>
          <span>Status</span>
          <span className="text-right">Details</span>
        </div>
        <ul className="divide-y divide-gray-200">{visible.map(run => {
          const period = libraryPeriod(run);
          const universe = libraryUniverse(run);
          const open = expanded.has(run.id);
          const strategy = run.run_type === "execution" ? recordedConfigurations(run)[0] : undefined;
          const strategyTitle = strategy && (strategy.name === strategy.method ? strategy.method : `${strategy.name} · ${strategy.method}`);
          return <li key={run.id}>
            <article aria-label={`${libraryRunType(run)} ${run.id}`}>
              <div className={`space-y-3 px-4 py-4 md:space-y-0 ${GRID}`}>
                <div className="min-w-0">
                  <p className="text-sm font-semibold text-gray-900">{libraryRunType(run)}</p>
                  {strategyTitle && <p className="mt-1 line-clamp-2 break-words text-xs font-medium text-gray-700 [overflow-wrap:anywhere]" title={strategyTitle}>{strategyTitle}</p>}
                  <p className="mt-1 text-xs text-gray-500">{run.id}</p>
                </div>
                <div className="min-w-0">
                  <p className="mb-1 text-xs text-gray-500 md:hidden">Campaign</p>
                  {run.campaign_id && campaignNames.has(run.campaign_id) ? <button type="button" title={campaignNames.get(run.campaign_id)} aria-label={`Show campaign ${campaignNames.get(run.campaign_id)}`} onClick={() => update({campaign: run.campaign_id!})} className="line-clamp-2 break-words text-left text-xs font-medium text-blue-700 hover:underline focus-visible:outline-2 focus-visible:outline-blue-600 [overflow-wrap:anywhere]">{campaignNames.get(run.campaign_id)}</button>
                    : <p className="text-xs text-gray-500">{run.campaign_id ? campaigns.isPending ? "Loading campaign…" : "Unavailable campaign" : "Ungrouped"}</p>}
                </div>
                <div className="min-w-0">
                  <p className="text-xs text-gray-500">{period.label}</p>
                  <p className="mt-1 text-sm tabular-nums">{period.value}</p>
                </div>
                <div className="min-w-0">
                  <p className="text-xs text-gray-500">{universe.label}</p>
                  <p className="mt-1 line-clamp-2 break-words text-sm [overflow-wrap:anywhere]" title={universe.file}>{universe.name}</p>
                </div>
                <div>
                  <span className={`inline-block rounded-full px-2 py-0.5 text-xs ${statusColor(run.status)}`}>{run.status}</span>{run.status === "running" && run.worker_health === "unresponsive" && <p className="mt-1 text-xs text-amber-800">No heartbeat</p>}</div>
                <div className="flex gap-4 md:flex-col md:items-end md:gap-2">
                  <Link to={`/runs/${run.id}`} state={{ dashboardSearch: params.toString() ? `?${params.toString()}` : "" }} aria-label={`Open details for ${run.id}`} className="text-sm font-medium text-blue-700 hover:underline">Open details</Link>
                  <button type="button" aria-label={`${open ? "Hide" : "Show"} summary for ${run.id}`} aria-expanded={open} aria-controls={`summary-${run.id}`} onClick={() => toggleSummary(run.id)} className="text-xs text-gray-600 hover:text-gray-900 hover:underline">{open ? "Hide summary" : "Summary"} <span aria-hidden="true">{open ? "▴" : "▾"}</span>
                  </button>
                </div>
              </div>
              {open && <div id={`summary-${run.id}`}>
                <SavedSummary run={run} />
              </div>}
            </article>
          </li>;
        })}</ul>
      </section>}
      {filtered.length > PAGE_SIZE && <nav aria-label="Run pages" className="flex items-center justify-between gap-3 text-sm">
        <button type="button" disabled={page === 1} onClick={() => update({ page: String(page - 1) }, false)} className="rounded-md border border-gray-300 px-3 py-1.5 disabled:opacity-40">Previous</button>
        <span className="text-gray-500">Page {page} of {pageCount}</span>
        <button type="button" disabled={page === pageCount} onClick={() => update({ page: String(page + 1) }, false)} className="rounded-md border border-gray-300 px-3 py-1.5 disabled:opacity-40">Next</button>
      </nav>}
    </>}
  </div>;
}
