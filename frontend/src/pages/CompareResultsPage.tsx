import { CampaignSelect } from "@/components/ui/Campaigns";
import { useId, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useQueries, useQuery } from "@tanstack/react-query";
import { getRun, getRunManifest, listRuns } from "@/api/endpoints";
import { libraryPeriod, libraryUniverse, libraryRunType, searchLibrary } from "@/lib/runLibrary";
import { configurationSettings, recordedConfigurations } from "@/lib/runPresentation";
import { comparisonFamily, comparisonCompatibility, FAMILIES, savedCandidates, type ComparisonFamily } from "@/lib/savedComparison";
import { candidateDescription, executionLabel, periodSpan, resultAlias } from "@/lib/compareRuns";
import { configurationColumns } from "@/lib/compareMatrix";
import ComparisonResults, { CompatibilityNotice } from "@/components/ui/ComparisonResults";
import BacktestRunComparison from "@/components/ui/BacktestRunComparison";

const field = "w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-sm";
const button = "rounded-md border border-gray-300 px-3 py-2 text-sm hover:bg-gray-50 disabled:opacity-40";

export default function CompareResultsPage() {
  const id = useId();
  const [params, setParams] = useSearchParams();
  const requestedFamily = params.get("family");
  const family: ComparisonFamily = requestedFamily && Object.hasOwn(FAMILIES, requestedFamily) ? requestedFamily as ComparisonFamily : "backtests";
  const limit = family === "backtests" ? 120 : 3;
  const selected = [...new Set(params.getAll("item"))];
  const [selectionOpen, setSelectionOpen] = useState(selected.length < 2);
  const [search, setSearch] = useState("");
  const [pickerRun, setPickerRun] = useState("");
  const [pickerCandidate, setPickerCandidate] = useState("");
  const [checkedRuns, setCheckedRuns] = useState<string[]>([]);
  const [poolRuns, setPoolRuns] = useState<string[]>([]);
  const [checkedConfigurations, setCheckedConfigurations] = useState<string[]>([]);
  const runs = useQuery({queryKey: ["runs"], queryFn: listRuns});
  const eligible = (runs.data ?? []).filter(run => run.status === "completed" && comparisonFamily(run) === family);
  const campaign = params.get("campaign") ?? "all";
  const visible = searchLibrary(eligible.filter(run => campaign === "all" || (campaign === "ungrouped" ? !run.campaign_id : run.campaign_id === campaign)), search, "all", "all", "newest");
  // Resolve URL selections against the signed-in account before fetching details.
  const ids = [...new Set([...selected.map(key => key.split(":")[0]), pickerRun, ...poolRuns])].filter(key => eligible.some(run => run.id === key));
  const details = useQueries({queries: ids.map(key => ({queryKey: ["run", key], queryFn: () => getRun(key)}))});
  const detailById = new Map(ids.map((key, index) => [key, details[index]]));
  const crossRun = new Set(selected.map(key => key.split(":")[0])).size > 1;
  const manifestIds = crossRun ? ids : [];
  const manifestQueries = useQueries({queries: manifestIds.map(key => ({queryKey: ["runManifest", key], queryFn: () => getRunManifest(key)}))});
  const manifests = Object.fromEntries(manifestIds.map((key, index) => [key, manifestQueries[index].data]));
  const candidates = new Map(details.flatMap(query => query.data ? savedCandidates(query.data) : []).map(candidate => [candidate.id, candidate]));
  const entries = selected.flatMap(key => candidates.get(key) ? [candidates.get(key)!] : []);
  const reference = entries.find(entry => entry.id === params.get("reference")) ?? entries[0];
  const picker = detailById.get(pickerRun);
  const options = picker?.data ? savedCandidates(picker.data) : [];
  const choice = options.find(candidate => candidate.id === pickerCandidate) ?? options[0];
  const bulkColumns = configurationColumns([...candidates.values()].filter(candidate => poolRuns.includes(candidate.run.id) && !(candidate.name === "buy_and_hold" && candidate.configuration.settings.strategy_type === "buy_and_hold")));
  const bulkLoading = poolRuns.some(key => detailById.get(key)?.isPending);
  const bulkErrors = poolRuns.filter(key => detailById.get(key)?.isError);
  const bulkItems = bulkColumns.filter(column => checkedConfigurations.includes(column.key)).flatMap(column => column.entries.map(entry => entry.id));
  const addedItems = [...new Set([...selected, ...bulkItems])];
  const loading = runs.isPending || selected.some(key => detailById.get(key.split(":")[0])?.isPending);
  const selectionResolved = entries.length === selected.length;
  const ready = selectionResolved && !loading && entries.length >= (family === "backtests" ? 1 : 2);
  const proposed = family !== "backtests" && choice && !selected.includes(choice.id) ? comparisonCompatibility([...entries, choice], manifests) : null;
  const proposedLoading = !!proposed && entries.length > 0 && manifestQueries.some(query => query.isPending);
  const comparisonSearch = `?${params.toString()}`;
  const openState = {comparisonSearch};

  function updateSelection(items: string[], nextFamily = family) {
    const next = {...(campaign !== "all" ? {campaign} : {}), family: nextFamily, item: items, ...(reference && items.includes(reference.id) ? {reference: reference.id} : {})};
    setParams(next);
    setSelectionOpen(items.length < 2);
  }
  function setReference(key: string) {
    const next = new URLSearchParams(params);
    next.set("reference", key);
    setParams(next);
    setPickerCandidate("");
  }

  return <div className="space-y-5">
    <div className="flex flex-wrap items-start justify-between gap-3">
      <div><h1 className="text-2xl font-semibold text-gray-900">Compare Runs</h1><p className="mt-1 text-sm text-gray-600">Check what changed, whether it helped, and whether the difference persisted across periods.</p></div>
      <button className={button} disabled={runs.isFetching} onClick={() => void runs.refetch()}>Refresh runs</button>
    </div>

    <section aria-label="Comparison purpose" className="grid gap-4 rounded-lg border border-gray-200 bg-white p-4 md:grid-cols-2">
      <div><label htmlFor={`${id}-family`} className="mb-1 block text-sm font-medium">Result family</label><select id={`${id}-family`} className={field} value={family} onChange={event => {setPickerRun(""); setPickerCandidate(""); setSearch(""); setCheckedRuns([]); setPoolRuns([]); setCheckedConfigurations([]); updateSelection([], event.target.value as ComparisonFamily);}}>{Object.entries(FAMILIES).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></div>
      <p className="self-center text-sm text-gray-600">{family === "backtests" ? "Choose completed runs, then the configurations to inspect. Periods and recorded context determine valid pairings automatically." : family === "predictions" ? "Predictors retain matching target dates and scored observations." : "Solvers must belong to the same saved benchmark problem."}</p>
    </section>

    <section aria-label="Choose saved results" className="space-y-4 rounded-lg border border-gray-200 bg-white p-4">
      <div className="flex items-center justify-between gap-3"><h2 className="text-sm font-semibold">Choose runs and configurations</h2><button className={button} aria-expanded={selectionOpen} onClick={() => setSelectionOpen(open => !open)}>{selectionOpen ? "Hide selection" : "Change selection"}</button></div>
      {selectionOpen && <div className="space-y-4">
        <CampaignSelect includeAll label="Filter runs by campaign" value={campaign} onChange={value => {const next = new URLSearchParams(params); if (value === "all") next.delete("campaign"); else next.set("campaign", value); setParams(next); setPickerRun(""); setPickerCandidate(""); setCheckedRuns([]); setPoolRuns([]); setCheckedConfigurations([]);}} />
        <div><label htmlFor={`${id}-search`} className="mb-1 block text-sm font-medium">Search runs</label><input id={`${id}-search`} className={field} value={search} onChange={event => {setSearch(event.target.value); setPickerRun(""); setPickerCandidate("");}} placeholder="Strategy, universe, dates or run ID" /></div>
        {runs.isPending ? <p role="status">Loading runs…</p> : runs.isError ? <p role="alert">Could not load runs. <button className={button} onClick={() => void runs.refetch()}>Retry runs</button></p> : <>
          <p className="text-xs text-gray-500">{visible.length} completed runs · Full results load only for runs you inspect or explicitly load.</p>
          <div role="region" aria-label="Available runs" tabIndex={0} className="max-h-80 overflow-auto rounded-md border border-gray-200"><table className="w-full text-left text-sm"><thead className="sticky top-0 bg-gray-50"><tr>{["Run / configurations", "Requested period", "Dataset / universe", "Executed (UTC)", ""].map((label, index) => <th key={index} className="px-3 py-2 font-medium">{label}</th>)}</tr></thead><tbody className="divide-y divide-gray-100">{visible.map(run => {
            const configurations = recordedConfigurations(run);
            return <tr key={run.id} className={run.id === pickerRun ? "bg-blue-50" : ""}>
              <td className="min-w-52 px-3 py-3">{family === "backtests" && <input type="checkbox" className="mr-2" aria-label={`Select run ${run.id}`} checked={checkedRuns.includes(run.id)} disabled={!checkedRuns.includes(run.id) && checkedRuns.length >= 20} onChange={event => setCheckedRuns(event.target.checked ? [...checkedRuns, run.id] : checkedRuns.filter(key => key !== run.id))} />}<span className="font-medium">{libraryRunType(run)}</span><span className="mt-1 block text-xs text-gray-500">{configurations.slice(0, 2).map(c => c.name).join(" · ")}{configurations.length > 2 ? ` · +${configurations.length - 2}` : ""}</span><span className="mt-1 block font-mono text-[10px] text-gray-400">{run.id}</span></td>
              <td className="min-w-40 px-3 py-3 text-xs">{libraryPeriod(run).value}</td>
              <td className="min-w-44 max-w-64 px-3 py-3 text-xs">{libraryUniverse(run).name}</td>
              <td className="whitespace-nowrap px-3 py-3 text-xs">{executionLabel(run)}</td>
              <td className="px-3 py-3"><button className={button} aria-label={`Inspect ${run.id}`} aria-pressed={pickerRun === run.id} onClick={() => {setPickerRun(run.id); setPickerCandidate("");}}>Inspect</button></td>
            </tr>;
          })}</tbody></table></div>
          {!visible.length && <p className="text-sm text-gray-500">No completed runs match this selection.</p>}
        </>}
        {family === "backtests" && <section aria-label="Bulk configuration selection" className="space-y-3">
          <div className="flex flex-wrap items-center gap-3"><button className={button} disabled={!checkedRuns.length} onClick={() => {setPoolRuns(checkedRuns.filter(key => eligible.some(run => run.id === key))); setCheckedConfigurations([]);}}>Load configurations from {checkedRuns.length} selected runs</button><button className={button} disabled={!checkedRuns.length} onClick={() => setCheckedRuns([])}>Clear run selection</button><span className="text-xs text-gray-500">Up to 20 runs per load; hidden search selections remain checked.</span></div>
          {bulkLoading && <p role="status">Loading configurations from selected runs…</p>}
          {bulkErrors.length > 0 && <p role="alert">Some selected runs could not be loaded. Retry before adding configurations. <button className={button} onClick={() => bulkErrors.forEach(key => void detailById.get(key)?.refetch())}>Retry selected runs</button></p>}
          {!bulkLoading && poolRuns.length > 0 && <div className="space-y-3 rounded-md border border-gray-200 p-4">
            <p className="text-sm font-medium">Choose configurations across the loaded runs</p>
            <p className="text-xs text-gray-500">Matching recorded settings share a choice. Different budgets, seeds or missing asset identities stay separate. Every matching execution is added. Saved automatic benchmarks appear alongside the matrix without selecting them here.</p>
            <div className="max-h-72 space-y-3 overflow-auto">{bulkColumns.map(column => <div key={column.key}><label className="flex items-start gap-2 text-sm"><input type="checkbox" className="mt-1" checked={checkedConfigurations.includes(column.key)} onChange={event => setCheckedConfigurations(event.target.checked ? [...checkedConfigurations, column.key] : checkedConfigurations.filter(key => key !== column.key))} /><span><span className="font-medium">{column.label} · {column.entries[0].name}</span><span className="block text-xs text-gray-600">{candidateDescription(column.entries[0])}</span><span className="block text-xs text-gray-500">{column.entries.length} saved executions in {new Set(column.entries.map(entry => entry.run.id)).size} runs · {[...new Set(column.entries.map(entry => entry.name))].join(" · ")}</span></span></label><details className="ml-6 mt-1 text-xs text-gray-500"><summary className="cursor-pointer">Recorded settings</summary><p className="mt-1">{configurationSettings(column.entries[0].configuration)}</p></details></div>)}</div>
            {!bulkColumns.length && <p className="text-sm text-amber-800">No complete numeric results were retained in these runs.</p>}
            <button className={button} disabled={bulkLoading || bulkErrors.length > 0 || addedItems.length === selected.length || addedItems.length > limit} onClick={() => updateSelection(addedItems)}>Add selected configurations ({addedItems.length - selected.length} results)</button>
            {addedItems.length > limit && <p role="alert" className="text-sm text-amber-800">This selection exceeds {limit} results. Choose fewer configurations; nothing has been dropped.</p>}
          </div>}
        </section>}
        {pickerRun && picker?.isPending && <p role="status">Loading saved configurations…</p>}
        {pickerRun && picker?.isError && <p role="alert">Could not load {pickerRun}. <button className={button} onClick={() => void picker.refetch()}>Retry result</button></p>}
        {picker?.data && <section aria-label="Inspected run" className="space-y-3 rounded-md border border-blue-200 bg-blue-50 p-4">
          <p className="text-sm font-medium">{libraryRunType(picker.data)} · {executionLabel(picker.data)}</p>
          <div className="flex flex-wrap items-end gap-3"><div className="min-w-0 flex-1"><label htmlFor={`${id}-configuration`} className="mb-1 block text-sm font-medium">Configuration</label><select id={`${id}-configuration`} className={field} value={choice?.id ?? ""} onChange={event => setPickerCandidate(event.target.value)}>{options.map(candidate => <option key={candidate.id} value={candidate.id}>{candidate.name} — {candidateDescription(candidate)}</option>)}</select></div><button className={button} disabled={!choice || selected.includes(choice.id) || selected.length >= limit || proposedLoading || proposed?.state === "blocked"} onClick={() => choice && updateSelection([...selected, choice.id])}>Add result</button></div>
          {choice && <><p className="text-sm text-gray-600">{candidateDescription(choice)}</p><p className="text-xs text-gray-500">Actual evaluation: {periodSpan(choice)}</p></>}
          {!options.length && <p className="text-sm text-amber-800">No complete numeric results were retained.</p>}
          {proposed && entries.length > 0 && <p role="status" className="text-sm text-amber-900">{proposedLoading ? "Checking candidate context…" : proposed.reasons[0]}</p>}
        </section>}
        <p className="text-xs text-gray-500">Completed results only. No runs are started. {family === "backtests" ? "Add up to 120 saved results. Multiple executions remain visible; none is chosen automatically." : "Up to three configurations from compatible evaluations."}</p>
      </div>}
    </section>

    {selected.length > 0 && <details open={selected.length <= 2} className="rounded-lg border border-gray-200 bg-white p-4"><summary className="cursor-pointer text-sm font-medium">Selected result identities ({selected.length})</summary><div className="mt-3 space-y-3">{selected.map((key, index) => {
      const entry = candidates.get(key), query = detailById.get(key.split(":")[0]);
      return <article key={key} className="flex flex-wrap items-start justify-between gap-3 border-t border-gray-100 pt-3">
        <div className="min-w-0 flex-1"><h2 className="text-sm font-semibold">{resultAlias(index)} · {entry?.name ?? key}</h2>{entry ? <><p className="mt-1 text-sm text-gray-700">{candidateDescription(entry)}</p><p className="mt-1 text-xs text-gray-500">{periodSpan(entry)} · Executed {executionLabel(entry.run)}</p><p className="mt-1 text-xs text-gray-500">{libraryUniverse(entry.run).name}</p><details className="mt-1 text-xs text-gray-500"><summary className="cursor-pointer">Saved run ID</summary><span className="font-mono">{entry.run.id}</span></details></> : <p className="mt-1 text-sm text-amber-800">{query?.isPending || runs.isPending ? "Loading…" : query?.isError ? "Saved result could not be loaded." : "Configuration unavailable in the current run library."}</p>}</div>
        <div className="flex flex-wrap items-center gap-3">{entry && <Link className="text-sm text-blue-700 underline" to={`/runs/${entry.run.id}`} state={openState}>Open full results</Link>}{query?.isError && <button className={button} onClick={() => void query.refetch()}>Retry {key.split(":")[0]}</button>}<button className={button} onClick={() => updateSelection(selected.filter(item => item !== key))}>Remove {entry?.name ?? key}</button></div>
      </article>;
    })}</div></details>}
    {!ready && selected.length < 2 && <p className="text-sm text-gray-600">{family === "backtests" ? "Select runs and configurations to build the results matrix, or inspect a run to add an individual result." : "Choose at least two results to compare."}</p>}
    {selected.length > 0 && loading && <p role="status">Loading selected results…</p>}
    {!loading && !selectionResolved && <p role="alert">Resolve unavailable selections before comparing.</p>}
    {manifestQueries.some(query => query.isError) && <p className="text-sm text-amber-800">Some manifests could not be read. <button className={button} onClick={() => manifestQueries.filter(query => query.isError).forEach(query => void query.refetch())}>Retry manifests</button></p>}
    {ready && (family === "backtests" ? <BacktestRunComparison entries={entries} candidates={[...candidates.values()]} manifests={manifests} referenceId={reference!.id} onReferenceChange={setReference} comparisonSearch={comparisonSearch} /> : <div className="space-y-4"><CompatibilityNotice entries={entries} manifests={manifests} /><ComparisonResults key={family} entries={entries.map((entry, index) => ({...entry, name: `${resultAlias(index)} · ${entry.name}`}))} manifests={manifests} hideCharts /></div>)}
  </div>;
}
