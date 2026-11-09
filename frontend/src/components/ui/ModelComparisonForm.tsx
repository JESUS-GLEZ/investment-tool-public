import { useExperimentData } from "@/lib/useExperimentData";
import ParameterHelp from "./ParameterHelp";
import { startTransition, useId, useRef, useState, type FormEvent } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { createPredictionComparison, createSolverComparison } from "@/api/endpoints";
import type { LocalDataInfo, PredictionComparisonRequest, SolverComparisonRequest, SolverVariant, PredictorVariant } from "@/types/api";
import { createInitialExperimentData, initialExperimentData, effectiveExperimentData, experimentDateError, localDatasetError, type ExperimentData } from "@/lib/experimentData";
import { portfolioNumberErrors } from "@/lib/portfolioNumbers";
import { getErrorMessage } from "@/lib/utils";
import ExperimentDataFields from "./ExperimentDataFields";
import ExperimentInputSummary from "./ExperimentInputSummary";
import PresetControls from "./PresetControls";
import ComparisonCatalogue from "./ComparisonCatalogue";
import ModelConfigurationDrawer from "./ModelConfigurationDrawer";
import { createModelVariant, hydrateModelVariant, uniqueModelName, modelKey, modelVariantError, modelVariantSummary, type ModelVariant } from "@/lib/modelVariants";

import { SOLVER_META, PREDICTOR_META, PREDICTOR_FAMILIES, predictorFamilyLabel, canonicalComparisonLabels } from "@/lib/comparisonCatalogue";

const cls = "mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 text-sm";
const section = "space-y-4 rounded-lg border border-gray-200 bg-white p-5";
const solverDefaults = {solvers: ["NSGA-II", "NSGA-III", "SPEA2"], population_size: 100, n_generations: 500, n_restarts: 3,
  min_assets: 2, num_assets: 5, use_multi_greedy_seeds: true, seed_max_subsets: 2000, seed_beam_width: 5, include_markowitz_reference: true};
const predictionDefaults = {predictors: ["RidgeVARX", "EWM-6"], n_eval_periods: 12, min_history: 12, top_k: 5, compute_sentiment: true};

/** Shared-data → selected pool → common settings → explicit review, for both benchmarks. */
export default function ModelComparisonForm({mode, active, campaignId = null, campaignName = "Ungrouped"}: {mode: "solver" | "prediction"; active: boolean; campaignId?: string | null; campaignName?: string}) {
  const id = useId();
  const helpPrefix = id;
  const navigate = useNavigate();
  const client = useQueryClient();
  const [setup, updateSetup, loadSetup] = useExperimentData();
  const [solver, setSolver] = useState(solverDefaults);
  const [prediction, setPrediction] = useState(predictionDefaults);
  const [localInfo, setLocalInfo] = useState<LocalDataInfo | null>(null);
  const [reviewOpen, setReviewing] = useState(false);
  const [reviewedCampaign, setReviewedCampaign] = useState<string | null>();
  const reviewing = reviewOpen && reviewedCampaign === campaignId;
  const [error, setError] = useState<string>();
  const review = useRef<HTMLElement>(null);
  const inFlight = useRef(false);
  const isSolver = mode === "solver";
  const [entries,setEntries] = useState<ModelVariant[]>(() => (mode === "solver" ? solverDefaults.solvers : predictionDefaults.predictors).map(key=>createModelVariant(key,[],mode === "solver")));
  const [editor,setEditor] = useState<{value:ModelVariant;index:number|null}|null>(null);
  const selected = entries.map(modelKey);
  const options = isSolver ? SOLVER_META : PREDICTOR_META;
  const effectiveSetup = {...effectiveExperimentData(setup, localInfo), backtest_start_date: null};
  const parameters = isSolver ? {...effectiveSetup,...solver,solvers:selected,solver_variants:entries as SolverVariant[]} : {...effectiveSetup,...prediction,predictors:selected,predictor_variants:entries as PredictorVariant[]};
  const entryError = entries.map((v,i)=>modelVariantError(v,entries.filter((_,index)=>index!==i),effectiveSetup.frequency)).find(Boolean);
  const validationError = experimentDateError(effectiveSetup, localInfo) ?? Object.values(portfolioNumberErrors(setup))[0] ?? (selected.length === 0 ? `Add at least one ${isSolver ? "solver" : "predictor"}.` : entryError ?? (isSolver && solver.min_assets > solver.num_assets ? "Min assets cannot be greater than max assets." : undefined));
  const mutation = useMutation({mutationFn: () => isSolver ? createSolverComparison({...parameters, ...(campaignId ? {campaign_id: campaignId} : {})}) : createPredictionComparison({...parameters, ...(campaignId ? {campaign_id: campaignId} : {})}),
    onSuccess: run => {void client.invalidateQueries({queryKey: ["campaigns"]}); void client.invalidateQueries({queryKey: ["runs"]}); startTransition(() => navigate(`/runs/${run.id}`));},
    onSettled: () => {inFlight.current = false;}});
  const clear = () => {setError(undefined); setReviewing(false); if (!inFlight.current) mutation.reset();};
  const set = <K extends keyof ExperimentData>(key: K, value: ExperimentData[K]) => {clear(); updateSetup(key, value);};
  const setS = <K extends keyof typeof solver>(key: K, value: typeof solver[K]) => {clear(); setSolver(current => ({...current, [key]: value}));};
  const setP = <K extends keyof typeof prediction>(key: K, value: typeof prediction[K]) => {clear(); setPrediction(current => ({...current, [key]: value}));};
  const load = (p: SolverComparisonRequest | PredictionComparisonRequest) => {
    clear();
    loadSetup({...createInitialExperimentData(), ...Object.fromEntries(Object.keys(initialExperimentData).filter(key => p[key as keyof typeof p] !== undefined).map(key => [key, p[key as keyof typeof p]])), input_file: p.input_file ?? null});
    if (isSolver) setSolver({...solverDefaults, ...Object.fromEntries(Object.keys(solverDefaults).filter(key => (p as SolverComparisonRequest)[key as keyof SolverComparisonRequest] !== undefined).map(key => [key, (p as SolverComparisonRequest)[key as keyof SolverComparisonRequest]]))});
    else setPrediction({...predictionDefaults, ...Object.fromEntries(Object.keys(predictionDefaults).filter(key => (p as PredictionComparisonRequest)[key as keyof PredictionComparisonRequest] !== undefined).map(key => [key, (p as PredictionComparisonRequest)[key as keyof PredictionComparisonRequest]]))});
    const variants = isSolver ? (p as SolverComparisonRequest).solver_variants : (p as PredictionComparisonRequest).predictor_variants;
    if (variants?.length) setEntries(variants.map(v=>hydrateModelVariant({...v, ...(isSolver ? {solver:canonicalComparisonLabels([(v as SolverVariant).solver],SOLVER_META)[0]} : {predictor:canonicalComparisonLabels([(v as PredictorVariant).predictor],PREDICTOR_META)[0]})})));
    else setEntries(canonicalComparisonLabels(isSolver ? (p as SolverComparisonRequest).solvers ?? solverDefaults.solvers : (p as PredictionComparisonRequest).predictors ?? predictionDefaults.predictors,options).map(key=>createModelVariant(key,[],isSolver,true)));
    setLocalInfo(info => info?.file === p.input_file ? info : null);
  };
  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (inFlight.current) return;
    const issue = validationError ?? (setup.data_source === "local" ? localDatasetError(setup.input_file, localInfo) : undefined);
    if (issue) {setError(issue); setReviewing(false); return;}
    setError(undefined);
    if (!reviewing) {setReviewedCampaign(campaignId); setReviewing(true); requestAnimationFrame(() => review.current?.focus()); return;}
    inFlight.current = true;
    mutation.mutate();
  };
  if (!active) return null;
  return <><form aria-label={`${isSolver ? "Solver" : "Prediction"} comparison setup`} onSubmit={submit} onInvalidCapture={event => {
    const details = (event.target as HTMLElement).closest("details"); if (details) details.open = true;
  }} className="space-y-6">
    <fieldset disabled={mutation.isPending} className="space-y-6">
      <PresetControls<SolverComparisonRequest | PredictionComparisonRequest> runType={isSolver ? "solver_comparison" : "prediction_comparison"} value={parameters} onLoad={load} validationError={validationError} />
      {(error || mutation.isError) && <p role="alert" className="rounded bg-red-50 p-3 text-sm text-red-700">{error ?? getErrorMessage(mutation.error, "Could not start run. Your configuration is preserved.")}</p>}
      <section aria-labelledby={`${id}-shared`} className={section}>
        <h2 id={`${id}-shared`} className="text-lg font-semibold">1. Shared setup</h2>
        <p className="text-sm text-gray-500">Every {isSolver ? "solver" : "predictor"} uses the same input data and evaluation assumptions.</p>
        <ExperimentDataFields startDate={setup.historical_start_date} setStartDate={v => set("historical_start_date", v)} endDate={setup.historical_end_date} setEndDate={v => set("historical_end_date", v)}
          initialCash={setup.initial_cash} setInitialCash={v => set("initial_cash", v)} txCost={setup.transaction_cost_rate} setTxCost={v => set("transaction_cost_rate", v)} riskFree={setup.risk_free_rate} setRiskFree={v => set("risk_free_rate", v)}
          outputMode={setup.output_mode} setOutputMode={v => set("output_mode", v)} dataSource={setup.data_source} setDataSource={v => set("data_source", v)}
          assetConfigFile={setup.asset_config_file} setAssetConfigFile={v => set("asset_config_file", v)} frequency={effectiveSetup.frequency} setFrequency={v => set("frequency", v)} validationLevel={setup.validation_level} setValidationLevel={v => set("validation_level", v)}
          selectedFile={setup.input_file} onFileChange={v => set("input_file", v)} onLocalDataInfoChange={setLocalInfo} onAnyChange={clear} />
      </section>
      <section aria-labelledby={`${id}-pool`} className={section}>
        <h2 id={`${id}-pool`} className="text-lg font-semibold">2. {isSolver ? "Solver" : "Predictor"} selection</h2>
        <ComparisonCatalogue kind={isSolver ? "solver" : "predictor"} options={isSolver ? SOLVER_META : PREDICTOR_FAMILIES} selected={selected} toggleId={`${id}-catalogue`} onAdd={key=>{clear();setEditor({value:createModelVariant(key,entries,isSolver),index:null});}}>
          <p className="text-sm text-gray-500">Add the same method more than once to compare named configurations. Save each configuration in its drawer.</p>
          <ol className="space-y-3">{entries.map((v,index)=><li key={index} aria-label={v.name} className="flex flex-wrap items-start justify-between gap-3 rounded border p-4">
            <div className="min-w-0 flex-1"><h3 className="break-words font-medium">{v.name}</h3><p className="text-xs text-gray-500">{isSolver ? options[modelKey(v)]?.label ?? modelKey(v) : predictorFamilyLabel(modelKey(v))}</p><p className="mt-1 break-words text-sm text-gray-600">{modelVariantSummary(v,solver)}</p></div>
            <div className="flex flex-wrap gap-2">{(["Edit","Duplicate","Remove"] as const).map(action=><button key={action} type="button" className="rounded border px-3 py-1.5 text-sm disabled:opacity-50" aria-label={`${action} ${v.name}`} disabled={action === "Duplicate" && entries.length >= 20} onClick={()=>{
              clear();
              if (action === "Remove") setEntries(current=>current.filter((_,i)=>i!==index));
              else setEditor({value:action === "Duplicate" ? {...v,name:uniqueModelName(modelKey(v),entries,isSolver)} : v,index:action === "Duplicate" ? null : index});
            }}>{action}</button>)}</div>
          </li>)}</ol>
        </ComparisonCatalogue>
      </section>
      <section aria-labelledby={`${id}-settings`} className={section}>
        <h2 id={`${id}-settings`} className="text-lg font-semibold">Comparison settings</h2>
        <p className="text-sm text-gray-500">{isSolver ? "Shared search defaults apply to entries without overrides. Portfolio constraints, starting seeds and reference are shared for a fair problem comparison." : "Evaluation settings apply to every predictor configuration. Model parameters belong to each entry above."}</p>
        {isSolver ? <>
          <div className="grid gap-4 sm:grid-cols-3">
            <div><div className="flex items-center gap-1"><label className="text-sm" htmlFor={`${id}-population`}>Population size</label><ParameterHelp name="population_size" label="Population size" /></div><input id={`${id}-population`} className={cls} type="number" required min={10} value={solver.population_size} onChange={e => setS("population_size", Number(e.target.value))} /></div>
            <div><div className="flex items-center gap-1"><label className="text-sm" htmlFor={`${id}-evaluations`}>Evaluations per restart</label><ParameterHelp name="n_generations" label="Evaluations per restart" /></div><input id={`${id}-evaluations`} className={cls} type="number" required min={1} value={solver.n_generations} onChange={e => setS("n_generations", Number(e.target.value))} /></div>
            <div><div className="flex items-center gap-1"><label className="text-sm" htmlFor={`${id}-restarts`}>Restarts</label><ParameterHelp name="n_restarts" label="Restarts" /></div><input id={`${id}-restarts`} className={cls} type="number" required min={1} value={solver.n_restarts} onChange={e => setS("n_restarts", Number(e.target.value))} /></div>
          </div>
          <p className="text-xs text-gray-500">The budget counts objective evaluations, not generations. Entries can inherit these defaults or override them in their configuration drawer.</p>
          <div className="grid gap-4 sm:grid-cols-2">
            <div><div className="flex items-center gap-1"><label className="text-sm" htmlFor={`${id}-min-assets`}>Min Assets</label><ParameterHelp name="min_assets" label="Min Assets" /></div><input id={`${id}-min-assets`} className={cls} type="number" required min={1} max={solver.num_assets} value={solver.min_assets} onChange={e => setS("min_assets", Number(e.target.value))} /></div>
            <div><div className="flex items-center gap-1"><label className="text-sm" htmlFor={`${id}-max-assets`}>Max Assets</label><ParameterHelp name="num_assets" label="Max Assets" /></div><input id={`${id}-max-assets`} className={cls} type="number" required min={solver.min_assets} value={solver.num_assets} onChange={e => setS("num_assets", Number(e.target.value))} /></div>
          </div>
          <details className="space-y-3 rounded border p-3"><summary className="cursor-pointer text-sm font-medium">Starting portfolios and reference</summary>
            <div className="flex items-center gap-1"><label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={solver.use_multi_greedy_seeds} onChange={e => setS("use_multi_greedy_seeds", e.target.checked)} />Use Multi-Greedy risk-parity seeds</label><ParameterHelp name="use_multi_greedy_seeds" label="Use Multi-Greedy risk-parity seeds" /></div>
            {solver.use_multi_greedy_seeds && <div className="grid gap-3 sm:grid-cols-2">
              <div><div className="flex items-center gap-1"><label className="text-sm" htmlFor={`${helpPrefix}-seed_max_subsets`}>Subset evaluation budget</label><ParameterHelp name="seed_max_subsets" label="Subset evaluation budget" /></div><input id={`${helpPrefix}-seed_max_subsets`} className={cls} type="number" required min={1} max={100000} value={solver.seed_max_subsets} onChange={e => setS("seed_max_subsets", Number(e.target.value))} /></div>
              <div><div className="flex items-center gap-1"><label className="text-sm" htmlFor={`${helpPrefix}-seed_beam_width`}>Candidates retained per risk preference</label><ParameterHelp name="seed_beam_width" label="Candidates retained per risk preference" /></div><input id={`${helpPrefix}-seed_beam_width`} className={cls} type="number" required min={1} max={100} value={solver.seed_beam_width} onChange={e => setS("seed_beam_width", Number(e.target.value))} /></div>
            </div>}
            <div className="flex items-center gap-1"><label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={solver.include_markowitz_reference} onChange={e => setS("include_markowitz_reference", e.target.checked)} />Include convex mean–variance reference (without BPRC constraints)</label><ParameterHelp name="include_markowitz_reference" label="Include convex mean–variance reference (without BPRC constraints)" /></div>
            <p className="text-xs text-gray-500">The reference is unconstrained by the balanced-risk and cardinality requirements. Seeding work is additional to the solver evaluation budget.</p>
          </details>
        </> : <>
          <div className="grid gap-4 sm:grid-cols-3">
            <div><div className="flex items-center gap-1"><label className="text-sm" htmlFor={`${id}-eval-periods`}>Eval Periods</label><ParameterHelp name="n_eval_periods" label="Eval Periods" /></div><input id={`${id}-eval-periods`} className={cls} type="number" required min={1} value={prediction.n_eval_periods} onChange={e => setP("n_eval_periods", Number(e.target.value))} /></div>
            <div><div className="flex items-center gap-1"><label className="text-sm" htmlFor={`${id}-min-history`}>Min History</label><ParameterHelp name="min_history" label="Min History" /></div><input id={`${id}-min-history`} className={cls} type="number" required min={1} value={prediction.min_history} onChange={e => setP("min_history", Number(e.target.value))} /></div>
            <div><div className="flex items-center gap-1"><label className="text-sm" htmlFor={`${id}-top-k`}>Top K</label><ParameterHelp name="top_k" label="Top K" /></div><input id={`${id}-top-k`} className={cls} type="number" required min={1} value={prediction.top_k} onChange={e => setP("top_k", Number(e.target.value))} /></div>
          </div>
          <p className="text-xs text-gray-500">Periods use the selected data frequency. Min History is the warm-up before forecasting; Top K controls the asset-ranking evaluation, not portfolio cardinality.</p>
          <div className="flex items-center gap-1"><label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={prediction.compute_sentiment} onChange={e => setP("compute_sentiment", e.target.checked)} />Compute sentiment for predictors that use news</label><ParameterHelp name="compute_sentiment" label="Compute sentiment for predictors that use news" /></div>
          {!prediction.compute_sentiment && selected.some(key => /^(RidgeVARX|SentimentEWM)/.test(key)) && <p className="text-sm text-amber-800">Selected news-based predictors will run without computed sentiment scores.</p>}
        </>}
      </section>
      {reviewing && <section ref={review} tabIndex={-1} aria-labelledby={`${id}-review`} className="space-y-4 rounded-lg border border-blue-200 bg-blue-50 p-5">
        <h2 id={`${id}-review`} className="text-lg font-semibold">3. Review run</h2>
        <p className="text-sm"><strong>Campaign:</strong> {campaignName}</p>
        <ExperimentInputSummary setup={effectiveSetup} localInfo={localInfo} />
        <p className="text-sm">{selected.length} {isSolver ? "solvers" : "predictors"}: {entries.map(v => v.name).join(", ")}</p>
        {isSolver ? <div className="space-y-1 text-sm">
          <p>Shared constraints: {solver.min_assets}–{solver.num_assets} assets.</p>
          {entries.map(v=><p key={v.name}><strong>{v.name}</strong>: {modelVariantSummary(v,solver)}</p>)}
          {new Set((entries as SolverVariant[]).map(v=>`${v.population_size ?? solver.population_size}/${v.n_generations ?? solver.n_generations}/${v.n_restarts ?? solver.n_restarts}`)).size > 1 && <p className="text-amber-800">Search budgets differ between configurations. Interpret this as a configuration comparison, rather than an equal-budget algorithm comparison.</p>}
          <p>Multi-Greedy seeds: {solver.use_multi_greedy_seeds ? `enabled (up to ${solver.seed_max_subsets} subsets, ${solver.seed_beam_width} candidates per preference)` : "disabled"}. Convex MV reference: {solver.include_markowitz_reference ? "included" : "excluded"}.</p>
          <p>Solver budget: up to {(entries as SolverVariant[]).reduce((total,v)=>total+(v.n_generations ?? solver.n_generations)*(v.n_restarts ?? solver.n_restarts),0).toLocaleString()} requested evaluations across the selected solvers, plus seeding. Actual counts can exceed the budget to finish a population. This is a frontier benchmark on the final estimation window, not a portfolio backtest.</p>
        </div> : <div className="space-y-1 text-sm">{entries.map(v=><p key={v.name}><strong>{v.name}</strong>: {modelVariantSummary(v,solver)}</p>)}<p>Evaluate up to {prediction.n_eval_periods} periods after at least {prediction.min_history} history periods · Top K {prediction.top_k} · sentiment {prediction.compute_sentiment ? "enabled when needed" : "disabled"}.</p><p>Walk-forward forecasts use only history available at each origin. Available history can shorten the evaluation window.</p></div>}
        <details><summary className="cursor-pointer text-sm font-medium">Effective request (all settings)</summary><pre className="mt-2 max-h-96 overflow-auto rounded bg-white p-3 text-xs">{JSON.stringify(parameters, null, 2)}</pre></details>
        <button type="button" className="rounded border px-3 py-1.5 text-sm" onClick={() => setReviewing(false)}>Back to editing</button>
      </section>}
      {validationError && <p role="status" className="text-sm text-amber-800">{validationError}</p>}
      <button type="submit" disabled={mutation.isPending || !!validationError} className="w-full rounded-md bg-blue-600 px-4 py-3 text-sm font-medium text-white disabled:opacity-50">{mutation.isPending ? "Submitting…" : reviewing ? "Start run" : "Review run"}</button>
    </fieldset>
  </form>
  {editor && <ModelConfigurationDrawer value={editor.value} others={entries.filter((_,i)=>i!==editor.index)} frequency={effectiveSetup.frequency} shared={solver} focusFallbackId={`${id}-catalogue`} onClose={()=>setEditor(null)} onSave={v=>{
    clear();setEntries(current=>editor.index === null ? [...current,v] : current.map((old,i)=>i===editor.index ? v : old));setEditor(null);
  }} />}
  </>;
}
