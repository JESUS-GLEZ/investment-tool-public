import { useExperimentData } from "@/lib/useExperimentData";
import { containedPeriodWindow, periodStart } from "@/lib/periodDates";
/** One experiment editor for single runs and independent strategy comparisons. */
import { startTransition, useId, useRef, useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { createRun, createStrategyComparison, getDefaults, getSymbols } from "@/api/endpoints";
import type { DefaultsResponse, HoldingInput, ExecutionMode, ExecutionRequest, LocalDataInfo, StrategyComparisonRequest, StrategyType, StrategyVariant } from "@/types/api";
import { createStrategyInstance, strategiesFromPreset, strategyCatalogue, strategyError, strategyLabel, strategySummary, uniqueStrategyName, usesOptimizer } from "@/lib/strategyVariants";
import { portfolioNumberErrors } from "@/lib/portfolioNumbers";
import { currentHoldingsError } from "@/lib/currentHoldings";
import { getErrorMessage } from "@/lib/utils";
import { createInitialExperimentData, initialExperimentData, effectiveExperimentData, experimentDateError, localDatasetError, type ExperimentData } from "@/lib/experimentData";
import ExperimentInputSummary from "./ExperimentInputSummary";
import CurrentHoldingsFields from "./CurrentHoldingsFields";
import ExperimentDataFields from "./ExperimentDataFields";
import PresetControls from "./PresetControls";
import LegacyBacktestPresetImport from "./LegacyBacktestPresetImport";
import StrategyConfigurationDrawer from "./StrategyConfigurationDrawer";

type Mode = "execution" | "strategy_comparison";
const cls = "mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 text-sm";
const button = "rounded border border-gray-300 px-3 py-1.5 text-sm disabled:opacity-50";

type Props = {campaignId?: string | null; campaignName?: string; mode: Mode; active?: boolean; executionPurpose?: ExecutionMode; onExecutionPurposeChange?: (value: ExecutionMode) => void};
export default function StrategyExperimentForm({mode, active = true, executionPurpose, onExecutionPurposeChange, campaignId = null, campaignName = "Ungrouped"}: Props) {
  const defaults = useQuery({queryKey: ["defaults"], queryFn: getDefaults});
  if (!defaults.data?.strategy_optimizer) return active ? <div className="space-y-3 rounded border p-4">
    {defaults.isPending ? <p role="status">Loading strategy defaults…</p> : <p role="alert">Strategy defaults could not be loaded. <button type="button" className="underline" disabled={defaults.isFetching} onClick={() => {void defaults.refetch();}}>Reload defaults</button></p>}
  </div> : null;
  return <ExperimentEditor campaignId={campaignId} campaignName={campaignName} mode={mode} defaults={defaults.data} active={active} executionPurpose={executionPurpose} onExecutionPurposeChange={onExecutionPurposeChange} />;
}

function ExperimentEditor({mode, defaults, active, executionPurpose, onExecutionPurposeChange, campaignId = null, campaignName = "Ungrouped"}: Props & {defaults: DefaultsResponse; active: boolean}) {
  const id = useId();
  const navigate = useNavigate();
  const client = useQueryClient();
  const [presetImport, setPresetImport] = useState(0);
  const [setup, updateSetup, loadSetup] = useExperimentData();
  const [localExecutionMode, setLocalExecutionMode] = useState<ExecutionMode>("backtesting");
  const executionMode = executionPurpose ?? localExecutionMode;
  const setExecutionMode = (value: ExecutionMode) => {setLocalExecutionMode(value); onExecutionPurposeChange?.(value);};
  const [entries, setEntries] = useState<StrategyVariant[]>(() => (mode === "execution" ? ["combined"] : ["combined", "optimization"]).map(type => createStrategyInstance(type as StrategyType, `${strategyLabel(type as StrategyType)} 1`, defaults)));
  const [editor, setEditor] = useState<{value: StrategyVariant; index: number | null} | null>(null);
  const [catalogueOpen, setCatalogueOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [holdings, setHoldings] = useState<HoldingInput[]>([]);
  const [localInfo, setLocalInfo] = useState<LocalDataInfo | null>(null);
  const [benchmark, setBenchmark] = useState("SPY");
  const [includeBenchmark, setIncludeBenchmark] = useState(true);
  const [error, setError] = useState<string>();
  const [reviewOpen, setReviewing] = useState(false);
  const [reviewedPurpose, setReviewedPurpose] = useState<ExecutionMode>();
  const [reviewedCampaign, setReviewedCampaign] = useState<string | null>();
  const reviewing = reviewOpen && reviewedPurpose === executionMode && reviewedCampaign === campaignId;
  const review = useRef<HTMLElement>(null);
  const inFlight = useRef(false);
  const backtesting = mode === "strategy_comparison" || executionMode === "backtesting";
  const matchingBenchmark = includeBenchmark ? entries.find(v => v.strategy_type === "buy_and_hold" && (v.asset_symbol ?? "SPY") === benchmark) : undefined;
  const effectiveSetup = effectiveExperimentData(setup, localInfo);
  const holdingSymbols = useQuery({queryKey: ["symbols", effectiveSetup.asset_config_file], queryFn: () => getSymbols(effectiveSetup.asset_config_file), enabled: active && !backtesting});
  const universeSymbols = [...(holdingSymbols.data?.stocks ?? []), ...(holdingSymbols.data?.crypto ?? [])];
  const symbols = setup.data_source === "local" ? universeSymbols.filter(symbol => localInfo?.file === setup.input_file && [...localInfo.stocks, ...localInfo.crypto].includes(symbol)) : universeSymbols;
  const holdingError = !backtesting && holdings.length > 0 ? holdingSymbols.isPending ? "Waiting for holding symbols." : holdingSymbols.isError ? "Could not load holding symbols." : currentHoldingsError(holdings, symbols) : undefined;
  const parameters = mode === "execution" ? {
    ...effectiveSetup, backtest_start_date: backtesting ? setup.backtest_start_date : null,
    ...(!backtesting ? {current_holdings: holdings} : {}),
    execution_mode: executionMode, strategy_type: entries[0]?.strategy_type,
    strategy_configuration: entries[0] ?? null,
  } satisfies ExecutionRequest : {
    ...effectiveSetup, strategy_types: [], strategy_variants: entries,
    benchmark_asset: benchmark, include_benchmark: includeBenchmark,
  } satisfies StrategyComparisonRequest;
  const mutation = useMutation({mutationFn: async () => {
    return mode === "execution" ? createRun({...parameters, ...(campaignId ? {campaign_id: campaignId} : {})} as ExecutionRequest) : createStrategyComparison({...parameters, ...(campaignId ? {campaign_id: campaignId} : {})} as StrategyComparisonRequest);
  }, onSuccess: run => {
    void client.invalidateQueries({queryKey: ["runs"]});
    void client.invalidateQueries({queryKey: ["campaigns"]});
    startTransition(() => navigate(`/runs/${run.id}`));
  }, onSettled: () => {inFlight.current = false;}});
  const clear = () => {setError(undefined); setReviewing(false); if (!inFlight.current) mutation.reset();};
  const set = <K extends keyof ExperimentData>(key: K, value: ExperimentData[K]) => {clear(); updateSetup(key, value);};
  const entryError = entries.length === 0 ? "Add at least one strategy configuration." : entries.map((v, i) => {
    const issue = strategyError(v, entries.filter((_, index) => index !== i), effectiveSetup.frequency);
    return issue ? `${v.name}: ${issue}` : undefined;
  }).find(Boolean);
  const portfolioError = Object.values(portfolioNumberErrors(setup, !backtesting))[0];
  const benchmarkError = mode === "strategy_comparison" && includeBenchmark && !/^[A-Za-z0-9./_-]{1,30}$/.test(benchmark) ? "Choose a valid benchmark symbol." : undefined;
  const tradingWindow = containedPeriodWindow(setup.historical_start_date, setup.historical_end_date, effectiveSetup.frequency);
  const tradingStart = backtesting ? setup.backtest_start_date : null;
  const tradingStartError = tradingStart && tradingStart !== periodStart(tradingStart, effectiveSetup.frequency)
    ? "Backtest start must be a period boundary: Monday for weekly data or the first of the month for monthly data."
    : tradingStart && (!tradingWindow || tradingStart < tradingWindow.start || tradingStart > tradingWindow.end)
      ? "Backtest start must lie within the complete periods of the historical window." : undefined;
  const cashError = !backtesting && setup.initial_cash === 0 && holdings.length === 0 ? "Enter positive cash or at least one holding." : undefined;
  const dateError = experimentDateError(effectiveSetup, localInfo);
  const validationError = dateError ?? holdingError ?? cashError ?? portfolioError ?? entryError ?? benchmarkError ?? tradingStartError;
  const loadPreset = (p: ExecutionRequest | StrategyComparisonRequest) => {
    clear();
    loadSetup({...createInitialExperimentData(), ...Object.fromEntries(Object.keys(initialExperimentData).filter(key => key in p).map(key => [key, p[key as keyof typeof p]])),
      output_mode: p.output_mode ?? "standard", input_file: p.input_file ?? null,
      backtest_start_date: p.backtest_start_date ?? null,
    });
    setLocalInfo(info => info?.file === p.input_file ? info : null);
    setEntries(strategiesFromPreset(p, mode, defaults));
    if (mode === "execution") { setExecutionMode((p as ExecutionRequest).execution_mode ?? "backtesting"); setHoldings((p as ExecutionRequest).current_holdings ?? []); }
    else {
      const comparison = p as StrategyComparisonRequest;
      const legacyExplicitBenchmark = comparison.strategy_types?.includes("buy_and_hold") && !comparison.strategy_variants?.some(v => v.strategy_type === "buy_and_hold");
      setBenchmark(comparison.benchmark_asset ?? "SPY");
      // Legacy explicitly selected Buy & Hold was a benchmark even when auto-add was off.
      setIncludeBenchmark(legacyExplicitBenchmark || (comparison.include_benchmark ?? true));
    }
  };
  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (inFlight.current) return;
    let issue: string | undefined = validationError;
    if (!issue && setup.data_source === "local") {
      issue = localDatasetError(setup.input_file, localInfo);
    }
    if (issue) {setError(issue); setReviewing(false); return;}
    setError(undefined);
    if (!reviewing) {setReviewedCampaign(campaignId); setReviewedPurpose(executionMode); setReviewing(true); requestAnimationFrame(() => review.current?.focus()); return;}
    inFlight.current = true;
    mutation.mutate();
  };
  const openEditor = (value: StrategyVariant, index: number | null) => {clear(); setCatalogueOpen(false); setEditor({value, index});};
  if (!active) return null;
  return <>
    <form onSubmit={submit} className="space-y-6" aria-label={mode === "execution" ? "Execution setup" : "Strategy comparison setup"}>
      <fieldset disabled={mutation.isPending} className="space-y-6">
      <PresetControls<ExecutionRequest | StrategyComparisonRequest> key={presetImport} accept={executionPurpose === "investing" ? p => (p as ExecutionRequest).execution_mode === "investing" : undefined} runType={mode} value={parameters} validationError={validationError} onLoad={loadPreset} />
      {executionPurpose === "investing" && <p className="text-xs text-gray-500">Only investment-portfolio presets are listed here. Older single-strategy backtest presets can be imported under Backtest investment strategies.</p>}
      {mode === "strategy_comparison" && <LegacyBacktestPresetImport onImport={p => {
        loadPreset({...p, strategy_types: [], strategy_variants: strategiesFromPreset(p, "execution", defaults), include_benchmark: false});
        setPresetImport(value => value + 1);
      }} />}
      {(error || mutation.isError) && <p role="alert" className="rounded bg-red-50 p-3 text-sm text-red-700">{error ?? getErrorMessage(mutation.error, "Could not start run. Your configuration is preserved.")}</p>}
      <section aria-labelledby={`${id}-shared`} className="space-y-4 rounded-lg border border-gray-200 bg-white p-5">
        <h2 id={`${id}-shared`} className="text-lg font-semibold">1. Shared setup</h2>
        <p className="text-sm text-gray-500">Every strategy uses the same input data and evaluation assumptions.</p>
        {mode === "execution" && executionPurpose === undefined && <label className="block text-sm font-medium">Execution Mode
          <select className={cls} value={executionMode} onChange={e => {clear(); setExecutionMode(e.target.value as ExecutionMode);}}>
            <option value="backtesting">Historical backtest</option><option value="investing">Investment portfolio</option>
          </select>
        </label>}
        <ExperimentDataFields
          backtestStart={backtesting ? {value: setup.backtest_start_date ?? "", min: tradingWindow?.start, max: tradingWindow?.end, onChange: value => set("backtest_start_date", value || null)} : undefined}
          holdings={!backtesting}
          startDate={setup.historical_start_date} setStartDate={value => set("historical_start_date", value)}
          endDate={setup.historical_end_date} setEndDate={value => set("historical_end_date", value)}
          initialCash={setup.initial_cash} setInitialCash={value => set("initial_cash", value)}
          txCost={setup.transaction_cost_rate} setTxCost={value => set("transaction_cost_rate", value)}
          riskFree={setup.risk_free_rate} setRiskFree={value => set("risk_free_rate", value)}
          dataSource={setup.data_source} setDataSource={value => set("data_source", value)}
          assetConfigFile={setup.asset_config_file} setAssetConfigFile={value => set("asset_config_file", value)}
          frequency={effectiveSetup.frequency} setFrequency={value => set("frequency", value)}
          validationLevel={setup.validation_level} setValidationLevel={value => set("validation_level", value)}
          outputMode={setup.output_mode} setOutputMode={value => set("output_mode", value)}
          selectedFile={setup.input_file} onFileChange={value => set("input_file", value)}
          onLocalDataInfoChange={setLocalInfo} onAnyChange={clear} />
        {!backtesting && <CurrentHoldingsFields value={holdings} symbols={symbols} loading={holdingSymbols.isPending} failed={holdingSymbols.isError} onRetry={() => {void holdingSymbols.refetch();}} onChange={value => {clear(); setHoldings(value);}} />}
        {mode === "strategy_comparison" && <div className="space-y-2 border-t pt-4">
          <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={includeBenchmark} onChange={e => {clear(); setIncludeBenchmark(e.target.checked);}} />Include Buy & Hold benchmark</label>
          <p className="text-xs text-gray-500">A matching Buy & Hold configuration is reused as the benchmark. Otherwise, a separate benchmark runs. Without one, the first strategy is the comparison reference.</p>
          {includeBenchmark && <label className="block text-sm">Benchmark asset
            <input className={cls} required maxLength={30} value={benchmark} onChange={e => {clear(); setBenchmark(e.target.value);}} />
          </label>}
        </div>}
      </section>
      <section aria-labelledby={`${id}-strategies`} className="space-y-4 rounded-lg border border-gray-200 bg-white p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 id={`${id}-strategies`} className="text-lg font-semibold">2. Strategy configurations</h2>
          <button type="button" className={button} id={`${id}-catalogue-toggle`} aria-expanded={catalogueOpen} aria-controls={`${id}-catalogue`} disabled={mode === "strategy_comparison" && entries.length >= 20} onClick={() => setCatalogueOpen(value => !value)}>{mode === "execution" && entries.length ? "Replace strategy" : "Add strategy"}</button>
        </div>
        <p className="text-sm text-gray-500">{mode === "execution" ? "Configure one strategy for this run." : "Each row is an independent execution. Add the same strategy more than once to compare settings."} Asset counts and search settings belong to each configuration.</p>
        {catalogueOpen && <div id={`${id}-catalogue`} className="space-y-3 rounded border bg-gray-50 p-4">
          <label className="block text-sm">Search strategies<input className={cls} type="search" value={search} onChange={e => setSearch(e.target.value)} /></label>
          {strategyCatalogue.filter(s => `${s.label} ${s.description}`.toLowerCase().includes(search.toLowerCase())).map(s => <div key={s.type} className="flex items-center justify-between gap-3 rounded border bg-white p-3">
            <div><h3 className="text-sm font-medium">{s.label}</h3><p className="text-xs text-gray-500">{s.description}</p></div>
            <button type="button" className={button} aria-label={`Add ${s.label}`} onClick={() => openEditor(createStrategyInstance(s.type, uniqueStrategyName(s.type, entries), defaults), null)}>Add</button>
          </div>)}
          {!strategyCatalogue.some(s => `${s.label} ${s.description}`.toLowerCase().includes(search.toLowerCase())) && <p role="status" className="text-sm text-gray-500">No matching strategies.</p>}
        </div>}
        {entries.length === 0 && <p className="text-sm text-gray-500">No strategies added yet.</p>}
        <ol className="space-y-3">{entries.map((v, index) => <li key={index} className="flex flex-wrap items-start justify-between gap-3 rounded border p-4" aria-label={v.name}>
          <div className="min-w-0 flex-1"><h3 className="break-words font-medium">{v.name}</h3><p className="text-xs text-gray-500">{strategyLabel(v.strategy_type)}</p><p className="mt-1 text-sm text-gray-600">{strategySummary(v)}</p>
            {usesOptimizer(v.strategy_type) && <p className="mt-1 text-xs text-gray-500">Population {v.population_size} · {v.n_generations?.toLocaleString()} evaluations × {v.n_restarts} restarts · seed {v.random_seed} · {v.search_seconds === null ? "fixed work" : `${v.search_seconds}s search allowance`}</p>}
          </div>
          <div className="flex flex-wrap gap-2">
            <button type="button" className={button} aria-label={`Edit ${v.name}`} onClick={() => openEditor(v, index)}>Edit</button>
            {mode === "strategy_comparison" && <button type="button" className={button} disabled={entries.length >= 20} aria-label={`Duplicate ${v.name}`} onClick={() => openEditor({...v, name: uniqueStrategyName(v.strategy_type, entries)}, null)}>Duplicate</button>}
            <button type="button" className={`${button} text-red-700`} aria-label={`Remove ${v.name}`} onClick={() => {clear(); setEntries(current => current.filter((_, i) => i !== index));}}>Remove</button>
          </div>
        </li>)}</ol>
      </section>
      {reviewing && <section ref={review} tabIndex={-1} aria-labelledby={`${id}-review`} className="space-y-4 rounded-lg border border-blue-200 bg-blue-50 p-5">
        <h2 id={`${id}-review`} className="text-lg font-semibold">3. Review run</h2>
        <p className="text-sm"><strong>Campaign:</strong> {campaignName}</p>
        <ExperimentInputSummary setup={effectiveSetup} localInfo={localInfo} />
        {!backtesting && <div className="text-sm"><p className="font-medium">Starting portfolio (USD)</p><p>Available cash: ${setup.initial_cash}. {holdings.length ? "Current shares/units:" : "No current holdings; start from cash."}</p><ul className="max-h-48 overflow-auto">{holdings.map(h => <li key={h.symbol}>{h.symbol}: {h.quantity} shares/units</li>)}</ul></div>}
        <dl className="grid gap-x-6 gap-y-2 text-sm sm:grid-cols-2">
          <div><dt className="font-medium">Trading start</dt><dd>{backtesting ? setup.backtest_start_date || "Automatic warm-up; resolved by the engine" : "Latest completed data (investing mode)"}</dd></div>
          {mode === "strategy_comparison" && <div><dt className="font-medium">Benchmark</dt><dd>{includeBenchmark ? `${benchmark} Buy & Hold${matchingBenchmark ? ` (uses ${matchingBenchmark.name})` : ""}` : "First strategy (no added benchmark)"}</dd></div>}
        </dl>
        <p className="text-sm">{entries.length} strategy execution{entries.length === 1 ? "" : "s"}{mode === "strategy_comparison" && includeBenchmark && !matchingBenchmark ? " plus one benchmark" : ""}. {backtesting ? "Each backtest rebalances over its available trading periods." : "Produces a suggested portfolio for manual review; no orders are placed."}</p>
        {entries.filter(v => usesOptimizer(v.strategy_type)).map((v, index) => <div key={index} className="text-sm">
          <p className="font-medium">{v.name}: {strategySummary(v)}</p>
          <p>{((v.n_generations ?? 0) * v.n_restarts).toLocaleString()} NSGA evaluations across {v.n_restarts} restarts per rebalance, plus seeding (up to {v.seed_max_subsets.toLocaleString()} subsets) and refinement.</p>
          <p>{v.search_seconds === null ? "Fixed-work search has no time cutoff." : `Total configured allowance ≈ ${Math.round(v.search_seconds * ((defaults.strategy_optimizer?.seed_time_fraction ?? 0.4) + v.n_restarts * (1 - (defaults.strategy_optimizer?.seed_time_fraction ?? 0.4))))} seconds per rebalance. This is not a completion-time estimate.`} Selection-only configurations can reuse the same candidate front.</p>
        </div>)}
        {setup.transaction_cost_rate === 0 && <p className="text-sm text-amber-800">Trading costs are zero in this run.</p>}
        {setup.validation_level === "skip" && <p className="text-sm text-amber-800">Data validation is skipped.</p>}
        <details><summary className="cursor-pointer text-sm font-medium">Effective request (all settings)</summary><pre className="mt-2 max-h-96 overflow-auto rounded bg-white p-3 text-xs">{JSON.stringify(parameters, null, 2)}</pre></details>
        <button type="button" className={button} onClick={() => setReviewing(false)}>Back to editing</button>
      </section>}
      {(holdingError || cashError) && <p role="status" className="text-sm text-amber-800">{holdingError ?? cashError}</p>}
      {(entryError || benchmarkError) && <p role="status" className="text-sm text-amber-800">{entryError ?? benchmarkError}</p>}
      <button type="submit" disabled={mutation.isPending || !!validationError} className="w-full rounded-md bg-blue-600 px-4 py-3 text-sm font-medium text-white disabled:opacity-50">{mutation.isPending ? "Submitting…" : reviewing ? "Start run" : "Review run"}</button>
      </fieldset>
    </form>
    {editor && <StrategyConfigurationDrawer focusFallbackId={`${id}-catalogue-toggle`} value={editor.value} others={entries.filter((_, i) => i !== editor.index)} frequency={effectiveSetup.frequency}
      defaultSearchSeconds={defaults.strategy_optimizer!.search_seconds} onClose={() => setEditor(null)} onSave={value => {
        setEntries(current => mode === "execution" ? [value] : editor.index === null ? [...current, value] : current.map((v, i) => i === editor.index ? value : v));
        clear(); setEditor(null);
      }} />}
  </>;
}
