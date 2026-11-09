import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { getLocalDataInfo, listLocalDataFiles, listRuns } from "@/api/endpoints";
import { datasetFeature, datasetTitle } from "@/lib/localDatasets";
import { getErrorMessage } from "@/lib/utils";

const field = "w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-sm";
const featureLabels = { prices: "Price records", execution_quotes: "Execution quotes", news: "News records", cached_sentiment: "Cached sentiment" } as const;
export default function DataLibraryPage() {
  const [params, setParams] = useSearchParams();
  const [search, setSearch] = useState("");
  const client = useQueryClient();
  const files = useQuery({queryKey: ["localDataFiles"], queryFn: listLocalDataFiles});
  const runs = useQuery({queryKey: ["runs"], queryFn: listRuns});
  const file = params.get("file");
  const selected = files.data?.find(f => f.file === file);
  const info = useQuery({queryKey: ["localDataInfo", file], queryFn: () => getLocalDataInfo(file!), enabled: !!selected, retry: false});
  useEffect(() => {
    if (info.dataUpdatedAt) void client.invalidateQueries({queryKey: ["localDataFiles"]});
  }, [info.dataUpdatedAt, client]);
  const visible = (files.data ?? []).filter(f => [f.title, f.file, f.asset_config_name, f.start_date, f.end_date].join(" ").toLowerCase().includes(search.toLowerCase().trim()));
  const references = (runs.data ?? []).filter(r => r.request_payload?.input_file === file);
  const data = info.data;
  return <div className="space-y-6">
    <div><h1 className="text-2xl font-semibold text-gray-900">Data library</h1><p className="mt-1 text-sm text-gray-600">Browse saved market data before configuring a run. Selecting a dataset reads its metadata; it does not launch a process.</p></div>
    {files.isPending && <p role="status">Loading datasets…</p>}
    {files.isError && <p role="alert">{getErrorMessage(files.error, "Could not load datasets.")} <button className="underline" onClick={() => void files.refetch()}>Retry datasets</button></p>}
    <div className="grid items-start gap-5 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
      <section aria-label="Saved datasets" className="min-w-0 space-y-3 rounded-lg border border-gray-200 bg-white p-4">
        <label className="block space-y-1 text-sm font-medium">Search datasets<input className={field} value={search} onChange={e => setSearch(e.target.value)} placeholder="Name, universe, year or filename" /></label>
        <p className="text-xs text-gray-500">{visible.length} of {files.data?.length ?? 0} datasets · metadata is inspected on selection</p>
        <div className="max-h-[36rem] space-y-2 overflow-y-auto">{visible.map(f => <button key={f.file} aria-pressed={file === f.file} onClick={() => setParams({file: f.file})} className={`block w-full rounded-md border p-3 text-left ${file === f.file ? "border-blue-500 bg-blue-50" : "border-gray-200 hover:bg-gray-50"}`}>
          <span className="block text-sm font-medium">{f.title ?? datasetTitle(f.file)}</span>
          <span className="mt-1 block text-xs text-gray-500">{f.inspected_at ? `${f.stock_count ?? "?"} stocks · ${f.crypto_count ?? "?"} crypto · ${f.frequency ?? "Unknown frequency"}` : "Not inspected · label inferred from filename"}</span>
          {f.asset_config_name && <span className="mt-1 block text-xs text-gray-500">Linked universe: {f.asset_config_name}</span>}
          {f.inspected_at && <span className="mt-1 block text-xs text-gray-500">{f.features?.execution_quotes ? "Execution quotes · " : ""}{f.features?.news ? "News records" : "No news records"}</span>}
          <span className="mt-1 block break-all font-mono text-[10px] text-gray-400">{f.file}</span>
        </button>)}</div>
        {!visible.length && !files.isPending && !files.isError && <p className="text-sm text-gray-500">No datasets match.</p>}
      </section>
      <section aria-label="Dataset details" className="min-w-0 space-y-4 rounded-lg border border-gray-200 bg-white p-5">
        {!file && <p className="text-sm text-gray-600">Select a dataset to inspect its contents and related runs.</p>}
        {file && files.data && !selected && <p role="alert" className="text-sm text-amber-800">This dataset is unavailable. Select another saved dataset.</p>}
        {selected && info.isPending && <p role="status">Inspecting dataset…</p>}
        {selected && info.isError && <p role="alert">{getErrorMessage(info.error, "Could not inspect this dataset.")} <button className="underline" onClick={() => void info.refetch()}>Retry inspection</button></p>}
        {selected && data && <>
          <div><h2 className="text-lg font-semibold">{data.title ?? datasetTitle(data.file)}</h2><p className="mt-2 text-sm text-gray-600">{data.description ?? "No description recorded."}</p></div>
          <dl className="grid gap-4 text-sm sm:grid-cols-2">{[
            ["Overall date span", `${data.start_date ?? "Unknown"} → ${data.end_date ?? "Unknown"}`],
            ["Frequency", data.frequency ?? "Unknown"], ["Assets stored", `${data.stock_count} stocks · ${data.crypto_count} crypto`],
            ["Provider", data.provider ?? "Unknown"], ["Intended use", data.intended_use ?? "Not recorded"],
            ["Retrieved at", data.retrieved_at ?? "Unknown"], ["Linked universe", data.asset_config_name ?? data.asset_config_file ?? "Not recorded"],
          ].map(([label, value]) => <div key={label}><dt className="text-gray-500">{label}</dt><dd className="mt-1 break-words font-medium">{value}</dd></div>)}</dl>
          <p className="text-xs text-gray-500">The date span covers the whole snapshot; individual assets may have shorter histories. A linked universe describes the current config file, not an immutable record of past membership.</p>
          <div className="grid grid-cols-2 gap-2">{Object.entries(featureLabels).map(([key, label]) => <div key={key} className="rounded-md bg-gray-50 p-3 text-sm"><span className="block text-gray-500">{label}</span><span className="font-medium">{datasetFeature(data.features?.[key as keyof typeof featureLabels])}</span></div>)}</div>
          <p className="text-xs text-gray-500">Presence describes stored records, not data-quality validation or complete coverage. Sentiment is unknown unless explicitly recorded; inspection never calculates it.</p>
          <details><summary className="cursor-pointer text-sm font-medium">Assets ({data.stock_count + data.crypto_count})</summary><div className="mt-2 max-h-48 overflow-y-auto rounded-md bg-gray-50 p-3 text-sm leading-7">{data.stocks.join(", ")}{data.crypto.length > 0 && <p className="text-purple-700">Crypto: {data.crypto.join(", ")}</p>}</div></details>
          <details><summary className="cursor-pointer text-sm font-medium">Technical details</summary><dl className="mt-2 space-y-2 text-xs text-gray-500"><div><dt>Filename</dt><dd className="break-all font-mono">{data.file}</dd></div><div><dt>File size</dt><dd>{data.size_bytes === undefined ? "Unknown" : `${(data.size_bytes / 1024 ** 2).toFixed(1)} MiB`}</dd></div><div><dt>Time contract</dt><dd>{typeof data.time_contract_version === "number" ? data.time_contract_version : "Not recorded (legacy)"}</dd></div><div><dt>Metadata inspected</dt><dd>{data.inspected_at ?? "Not recorded"}</dd></div></dl></details>
          <div className="border-t pt-4"><h3 className="text-sm font-semibold">Runs referencing this filename</h3><p className="mt-1 text-xs text-gray-500">A filename reference does not verify that the file has remained unchanged.</p>
            {runs.isPending ? <p role="status">Loading runs…</p> : runs.isError ? <p role="alert">Could not load related runs. <button className="underline" onClick={() => void runs.refetch()}>Retry runs</button></p> : references.length ? <ul className="mt-2 max-h-48 space-y-2 overflow-y-auto text-sm">{references.map(r => <li key={r.id}><Link className="text-blue-700 underline" to={`/runs/${r.id}`}>{r.id}</Link> · {r.run_type} · {r.status}</li>)}</ul> : <p className="mt-2 text-sm text-gray-500">No runs reference this filename.</p>}
          </div>
          <Link to="/runs/new" className="inline-block text-sm text-blue-700 underline">Configure a new run</Link>
        </>}
      </section>
    </div>
  </div>;
}
