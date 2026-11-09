/** Read-only reference settings; editable run parameters and presets live in New Run. */
import ParameterHelp from "@/components/ui/ParameterHelp";
import { useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { getDefaults, getSymbols, listAssetConfigs } from "@/api/endpoints";
import { getErrorMessage } from "@/lib/utils";
import { createInitialExperimentData } from "@/lib/experimentData";
import { record, settingLabel, settingValue } from "@/lib/runPresentation";

const button = "text-sm underline disabled:opacity-50";
const field = "mt-1 block w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-sm";
const groups: Record<string, string> = { portfolio: "Portfolio and backtest engine", strategy_selection: "Engine strategy selection", construction: "Portfolio constraints", optimizer: "Standalone optimizer", genetic: "Genetic operators", gentle_variator: "Weight variation", decomposition: "Decomposition search", algorithm: "Algorithm controls", markowitz: "Mean–variance reference", prediction: "Prediction engine", momentum: "Momentum strategy", validation: "Data validation", covariance: "Covariance estimation", infra: "Execution infrastructure" };
function label(key: string) {
  return ({ pop_size: "Population size", seed_time_fraction: "Seed time allocation", initial_cash: "Initial capital", transaction_cost_rate: "Trading cost rate", risk_free_rate: "Annual risk-free rate", data_source: "Market data source", asset_config_file: "Starting universe file", historical_start_date: "Historical start date", historical_end_date: "Historical end date", validation_level: "Data validation", output_mode: "Output detail", frequency: "Price frequency" } as Record<string, string>)[key] ?? settingLabel(key).replace(/^./, c => c.toUpperCase());
}
function value(key: string, v: unknown) {
  if (key === "initial_cash" && typeof v === "number") return `$${v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  if (["seed_time_fraction", "start_index_percentage"].includes(key) && typeof v === "number") return `${v * 100}%`;
  return typeof v === "string" ? ({ alpaca: "Alpaca", local: "Local dataset", monthly: "Monthly", weekly: "Weekly", daily: "Daily", warn: "Warn", strict: "Strict", skip: "Skip", standard: "Standard", diagnostic: "Diagnostic" } as Record<string, string>)[v] ?? settingValue(key, v) : settingValue(key, v);
}
function Parameters({ parameters, technical = false, group }: { parameters: Record<string, unknown>; technical?: boolean; group?: string }) {
  return <dl className="grid gap-x-8 gap-y-4 text-sm sm:grid-cols-2">{Object.entries(parameters).map(([key, v]) =>
    <div key={key} className="min-w-0">
      <dt className="flex items-center gap-1 font-medium text-gray-600"><span>{label(key)}</span><ParameterHelp name={key} label={label(key)} group={group} />{technical && <span className="ml-2 font-mono text-xs font-normal text-gray-400">{key}</span>}</dt>
      <dd className="mt-1 text-gray-900 [overflow-wrap:anywhere]">{value(key, v)}</dd>
    </div>)}</dl>;
}
export default function SettingsPage() {
  const initialExperimentData = createInitialExperimentData();
  const defaults = useQuery({ queryKey: ["defaults"], queryFn: getDefaults });
  const catalog = useQuery({ queryKey: ["asset-configs"], queryFn: listAssetConfigs });
  const [file, setFile] = useState("");
  const [search, setSearch] = useState("");
  const options = catalog.data ?? [];
  const selected = options.find(c => c.file === file) ?? (file ? undefined : options.find(c => c.file === initialExperimentData.asset_config_file) ?? options[0]);
  const symbols = useQuery({ queryKey: ["symbols", selected?.file], queryFn: () => getSymbols(selected!.file), enabled: !!selected });
  const stockSymbols = (symbols.data?.stocks ?? []).filter(s => s.toLowerCase().includes(search.toLowerCase().trim()));
  const cryptoSymbols = (symbols.data?.crypto ?? []).filter(s => s.toLowerCase().includes(search.toLowerCase().trim()));
  const matches = [...stockSymbols.map(symbol => ({ symbol, kind: "Stock / ETF" })), ...cryptoSymbols.map(symbol => ({ symbol, kind: "Crypto" }))];
  return <div className="mx-auto max-w-5xl space-y-6">
    <div>
      <h1 className="text-2xl font-semibold text-gray-900">Settings</h1>
      <p className="mt-1 text-sm text-gray-600">Read-only reference values and asset universes. Configure each run and manage your saved presets in <Link className="text-blue-700 underline" to="/runs/new">New Run</Link>.</p>
    </div>
    <section aria-label="Run starting values" className="space-y-4 rounded-lg border border-gray-200 bg-white p-5">
      <h2 className="text-lg font-semibold">New Run starting values</h2>
      <p className="text-sm text-gray-500">These are the shared input values shown when opening a fresh run form. Loading a preset or editing a run can replace them. They are separate from the engine reference defaults below.</p>
      <Parameters parameters={Object.fromEntries(["data_source", "frequency", "historical_start_date", "historical_end_date", "initial_cash", "transaction_cost_rate", "risk_free_rate", "asset_config_file", "validation_level", "output_mode"].map(key => [key, initialExperimentData[key as keyof typeof initialExperimentData]]))} />
    </section>
    <section aria-label="Strategy search defaults" className="space-y-4 rounded-lg border border-gray-200 bg-white p-5">
      <h2 className="text-lg font-semibold">Strategy optimization defaults</h2>
      <p className="text-sm text-gray-500">Starting search settings for new strategy configurations. Change budgets and selection rules inside each strategy’s configuration in New Run. Evaluation budgets count objective evaluations, not generations.</p>
      {defaults.isPending && <p role="status">Loading default parameters…</p>}
      {defaults.isError && <div role="alert" className="space-y-2 text-sm text-red-700">
        <p>{getErrorMessage(defaults.error, "Could not load default parameters.")}</p>
        <button className={button} disabled={defaults.isFetching} onClick={() => void defaults.refetch()}>Retry defaults</button>
      </div>}
      {defaults.data && (defaults.data.strategy_optimizer ? <Parameters parameters={record(defaults.data.strategy_optimizer)} group="strategy_optimizer" /> : <p className="text-sm text-amber-800">Strategy search defaults are not provided by this server.</p>)}
    </section>
    <section aria-label="Asset universe browser" className="space-y-4 rounded-lg border border-gray-200 bg-white p-5">
      <h2 className="text-lg font-semibold">Asset universes</h2>
      <p className="text-sm text-gray-500">Inspect the candidates in a universe file. This selection is for browsing only; it does not change a run or the application default. Local datasets contain their own assets. Listed candidates are not a guarantee of provider coverage or current tradability.</p>
      {catalog.isPending && <p role="status">Loading asset universes…</p>}
      {catalog.isError && <div role="alert" className="space-y-2 text-sm text-red-700">
        <p>{getErrorMessage(catalog.error, "Could not load asset universes.")}</p>
        <button className={button} disabled={catalog.isFetching} onClick={() => void catalog.refetch()}>Retry asset universes</button>
      </div>}
      {!!options.length && <label className="block text-sm font-medium">Universe to inspect<select className={field} value={selected?.file ?? ""} onChange={e => { setFile(e.target.value); setSearch(""); }}>{!selected && <option value="">Select an available universe</option>}{options.map(config =>
        <option key={config.file} value={config.file}>{config.name} — {config.stock_count} stocks / ETFs, {config.crypto_count} crypto</option>)}</select>
      </label>}
      {catalog.isSuccess && !options.length && <p className="text-sm text-gray-500">No asset universe files are available.</p>}
      {selected && <>
        <p className="text-xs text-gray-500 [overflow-wrap:anywhere]">{selected.file}{selected.membership_mode === "dated" ? ` · Dated membership; up to ${selected.stock_limit ?? "the configured limit of"} equities per snapshot. Membership can change by period.` : " · Fixed candidate list."}</p>
        {symbols.isPending && <p role="status">Loading available symbols…</p>}
        {symbols.isError && <div role="alert" className="space-y-2 text-sm text-red-700">
          <p>{getErrorMessage(symbols.error, "Could not load available symbols.")}</p>
          <button className={button} disabled={symbols.isFetching} onClick={() => void symbols.refetch()}>Retry symbols</button>
        </div>}
        {symbols.data && <>
          <p className="text-sm font-medium">{symbols.data.stocks.length} stocks / ETFs · {symbols.data.crypto.length} crypto</p>
          <label className="block text-sm font-medium">Search symbols<input type="search" className={field} value={search} onChange={e => setSearch(e.target.value)} placeholder="Ticker symbol" />
          </label>
          <p className="text-xs text-gray-500">{matches.length} matching symbols. Scroll to browse the full list, or search for a ticker.</p>
          <ul className="flex max-h-72 flex-wrap content-start gap-2 overflow-y-auto overscroll-contain rounded-md border border-gray-200 p-3 focus-visible:outline-2 focus-visible:outline-blue-600" tabIndex={0} aria-label="Universe symbols">{matches.map(({ symbol, kind }) =>
            <li key={`${kind}:${symbol}`} className="rounded bg-gray-100 px-2 py-1 text-sm">
              <span className="font-mono">{symbol}</span>
              <span className="ml-2 text-xs text-gray-500">{kind}</span>
            </li>)}</ul>{!matches.length && <p className="text-sm text-gray-500">No symbols match this search.</p>}</>}
      </>}
    </section>
    <section aria-label="Engine reference defaults" className="space-y-3">
      <details className="rounded-lg border border-gray-200 bg-white p-5">
        <summary className="cursor-pointer text-lg font-semibold">Advanced engine reference</summary>
        <p className="my-4 text-sm text-gray-500">Server defaults for individual engine components. These are not account preferences or the effective configuration of every run. Run requests, presets and strategy-specific settings may override them; inspect a run’s saved parameters for the values it used.</p>
        {defaults.data && <div className="space-y-3">{Object.entries(defaults.data).filter(([group]) => group !== "strategy_optimizer").map(([group, parameters]) =>
          <details key={group} className="rounded border border-gray-200 p-4">
            <summary className="cursor-pointer text-sm font-medium">{groups[group] ?? label(group)}</summary>
            <div className="mt-4">
              <Parameters parameters={record(parameters)} group={group} technical />
            </div>
          </details>)}</div>}
        {!defaults.data && <p className="text-sm text-gray-500">Engine reference is available when server defaults load.</p>}
      </details>
    </section>
    <section className="rounded-lg border border-gray-200 bg-white p-5">
      <h2 className="text-lg font-semibold">Account and provider credentials</h2>
      <p className="mt-2 text-sm text-gray-500">Your sign-in credentials are separate from market-data API credentials. Provider credentials are configured on the server and are not displayed or editable on this page. Saved run presets belong to your account.</p>
    </section>
  </div>;
}
