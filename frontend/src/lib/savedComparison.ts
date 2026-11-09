import { parseWeightSummary } from "./portfolioPresentation";
import type { RunDetail, RunSummary } from "@/types/api";
import { record, recordedConfigurations, configurationFields, settingLabel, settingValue, experimentRows, type RecordedConfiguration } from "./runPresentation";

export type ComparisonFamily = "backtests" | "predictions" | "solvers";
export type SavedCandidate = {
  id: string; name: string; run: RunDetail; family: ComparisonFamily;
  configuration: RecordedConfiguration; metrics: Record<string, unknown>; definitions: Record<string, unknown>;
  observations: {start: string; end: string; value: number | null}[];
  holdings: { label: string; weights: Record<string, unknown>; executionTime?: string; incomplete: boolean; recorded?: boolean }[];
  periods: string[] | null; points: { date: string; value: number }[]; actuals: string | null;
};
export const FAMILIES: Record<ComparisonFamily, string> = { backtests: "Investment backtests", predictions: "Prediction evaluations", solvers: "Solver benchmarks" };
export function comparisonFamily(run: RunSummary): ComparisonFamily | null {
  if (run.run_type === "strategy_comparison" || run.run_type === "execution" && run.request_payload?.execution_mode === "backtesting") return "backtests";
  return run.run_type === "prediction_comparison" ? "predictions" : run.run_type === "solver_comparison" ? "solvers" : null;
}
export const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);
export function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") return `{${Object.entries(record(value)).sort(([a], [b]) => a.localeCompare(b)).map(([key, v]) => `${JSON.stringify(key)}:${canonical(v)}`).join(",")}}`;
  return JSON.stringify(value) ?? "undefined";
}
function validDates(value: unknown): value is string[] {
  return Array.isArray(value) && value.length > 0 && value.every(v => typeof v === "string" && /^\d{4}-\d{2}-\d{2}/.test(v) && Number.isFinite(Date.parse(v)) && new Date(`${v.slice(0, 10)}T00:00:00Z`).toISOString().slice(0, 10) === v.slice(0, 10)) && new Set(value).size === value.length && value.every((v, i) => !i || v > value[i - 1]);
}
function frameRows(value: unknown): Record<string, unknown>[] {
  if (Array.isArray(value)) return value.map(record);
  const frame = record(value);
  const length = Array.isArray(frame.period_end) ? frame.period_end.length : 0;
  return Array.from({ length }, (_, i) => Object.fromEntries(Object.entries(frame).map(([key, v]) => [key, Array.isArray(v) ? v[i] : undefined])));
}
/** Adapt saved outputs once; no current defaults or recomputed performance metrics. */
export function savedCandidates(run: RunDetail): SavedCandidate[] {
  const family = comparisonFamily(run);
  const result = record(run.result);
  if (!family || run.status !== "completed" || result.partial === true) return [];
  const configurations = recordedConfigurations(run);
  const sharedMetrics = record(result.comparison_metrics);
  const outputs: [string, Record<string, unknown>][] = family === "backtests"
    ? run.run_type === "execution" ? [[configurations[0]?.name ?? "Strategy", { metrics: result.metrics, results: frameRows(result.results_df) }]]
      : [...new Set([...Object.keys(record(result.individual_results)), ...Object.keys(sharedMetrics)])].map(name => [name, record(record(result.individual_results)[name])])
    : Array.isArray(result.results) ? result.results.map(value => { const item = record(value); return [String(record(item.config).label ?? ""), item]; }) : [];
  const names = new Set<string>();
  return outputs.flatMap(([name, item]) => {
    const metrics = Object.keys(record(item.metrics)).length ? record(item.metrics) : record(sharedMetrics[name]);
    if (!name || names.has(name) || !Object.values(metrics).some(finite)) return [];
    names.add(name);
    const submitted = configurations.find(c => c.name === name);
    const effective = family === "backtests" ? record(item.configuration) : record(item.config);
    const algorithm = typeof effective.algorithm === "string" ? effective.algorithm : undefined;
    const settings = { ...run.request_payload, ...submitted?.settings, ...effective, ...record(effective.algorithm_kwargs), ...(family === "solvers" ? { solver: submitted?.settings.solver ?? algorithm ?? name } : family === "predictions" ? { predictor: submitted?.settings.predictor ?? name } : {}) };
    const method = submitted?.method ?? String(effective.strategy_type ?? algorithm ?? name);
    const configuration = { name, method, settings };
    const rows = frameRows(item.results);
    const starts = rows.map(row => row.period_start), ends = rows.map(row => row.period_end);
    const boundaries = rows.every(row => row.date_basis === "period_end_label") && validDates(starts) && validDates(ends) && starts.every((s, i) => s <= ends[i] && (!i || s > ends[i - 1]));
    const periods = family === "predictions" ? validDates(item.evaluation_dates) ? item.evaluation_dates : null
      : family === "backtests" && boundaries ? starts.map((start, i) => `${start} → ${ends[i]}`) : null;
    const points = periods && family === "backtests" && rows.every(row => finite(row.total_value)) ? rows.map(row => ({ date: String(row.period_end), value: row.total_value as number })) : [];
    const actuals = record(item.actuals_df);
    const actualsKnown = family === "predictions" && periods && Object.keys(actuals).length && Object.values(actuals).every(col => Array.isArray(col) && col.length === periods.length && col.every(v => v === null || finite(v)));
    const holdings = rows.map((row, i) => {
      const hasWeights = row.selected_assets !== null && typeof row.selected_assets === "object" && !Array.isArray(row.selected_assets);
      const summary = parseWeightSummary(row.selected_assets_summary);
      const recorded = hasWeights || typeof row.selected_assets_summary === "string";
      const malformed = Object.hasOwn(row, "selected_assets") && !hasWeights;
      return {
        label: typeof row.period_start === "string" && typeof row.period_end === "string" ? `${row.period_start} → ${row.period_end}` : typeof row.date === "string" ? row.date : `Period ${i + 1} (date unavailable)`,
        weights: hasWeights ? record(row.selected_assets) : summary.weights,
        incomplete: hasWeights ? false : malformed || recorded && summary.incomplete,
        recorded,
        executionTime: typeof row.execution_time === "string" ? row.execution_time : undefined,
      };
    });
    return [{
      holdings,
      observations: boundaries ? rows.map(row => ({start: String(row.period_start), end: String(row.period_end), value: finite(row.return) ? row.return : null})) : [],
      id: `${run.id}:${name}`, name, run, family, configuration, metrics,
      definitions: record(record(result.metric_definitions)[name] ?? metrics.metric_definitions), periods, points,
      actuals: actualsKnown ? canonical(actuals) : null
    }];
  });
}

const hash = (value: unknown) => typeof value === "string" && /^[a-f0-9]{64}$/i.test(value) ? value : undefined;
export function provenance(manifest: unknown) {
  const m = record(manifest);
  return m.format_version === 1 ? { dataset: hash(record(m.input_dataset).sha256), universe: hash(record(m.asset_universe).sha256), source: hash(m.source_sha256), packages: record(m.package_versions), model: record(m.model_configuration) } : { dataset: undefined, universe: undefined, source: undefined, packages: {}, model: {} };
}
export type Compatibility = { state: "matching" | "varied" | "unverified" | "blocked"; reasons: string[]; chartAllowed: boolean };
export function comparisonCompatibility(entries: SavedCandidate[], manifests: Record<string, unknown>): Compatibility {
  const blocked: string[] = [], warnings: string[] = [], unknown: string[] = [];
  if (entries.length < 2) return { state: "unverified", reasons: ["Choose at least two configurations."], chartAllowed: false };
  const first = entries[0];
  if (entries.some(e => e.family !== first.family)) blocked.push("These result families use different metrics.");
  if (entries.some(e => e.run.status !== "completed" || record(e.run.result).partial === true)) blocked.push("Only complete results can be compared.");
  const frequency = entries.map(e => record(e.run.result).frequency ?? e.run.request_payload.frequency);
  if (frequency.some(v => typeof v !== "string")) unknown.push("Data frequency is not recorded for every selection.");
  else if (new Set(frequency).size > 1) blocked.push("Data frequencies differ.");
  const sameRun = entries.every(e => e.run.id === first.run.id);
  if (first.family === "solvers" && !sameRun) blocked.push("Select solvers from the same benchmark: cross-run problem and hypervolume reference-point equivalence is not recorded.");
  if (first.family !== "solvers") {
    if (entries.some(e => !e.periods)) unknown.push("Actual evaluation periods are missing; requested dates alone do not establish comparable coverage.");
    else if (entries.some(e => canonical(e.periods) !== canonical(first.periods))) blocked.push("Actual evaluation periods differ. Separate periods cannot be ranked as the same evaluation.");
  }
  if (first.family === "predictions") {
    if (entries.some(e => !e.actuals)) unknown.push("Scored asset/period observations are not fully recorded.");
    else if (entries.some(e => e.actuals !== first.actuals)) blocked.push("Prediction metrics were scored on different observations or asset coverage.");
    if (entries.some(e => !finite(e.metrics.scored_pairs) || e.metrics.scored_pairs <= 0)) blocked.push("There are no verified scored observations for every predictor.");
    if (entries.some(e => finite(e.metrics.coverage) && e.metrics.coverage < 1 || finite(e.metrics.failed_periods) && e.metrics.failed_periods > 0)) warnings.push("Prediction coverage is incomplete; inspect coverage and failed-period counts alongside accuracy.");
  }
  if (!sameRun) {
    const identities = entries.map(e => provenance(manifests[e.run.id]));
    if (identities.some(p => !p.source || !p.dataset || !p.universe || !Object.keys(p.packages).length || Object.values(p.packages).some(v => typeof v !== "string" || v === "unavailable"))) unknown.push("Some saved manifests lack dataset, universe, engine or package identity. Cross-run equivalence is unverified.");
    if (new Set(identities.map(p => p.source).filter(Boolean)).size > 1) blocked.push("Engine versions differ. Inspect configurations or open each run separately.");
    if (new Set(identities.map(p => p.universe).filter(Boolean)).size > 1) warnings.push("Asset-universe contents differ. This compares different asset universes, even when filenames match.");
    if (new Set(identities.map(p => p.dataset).filter(Boolean)).size > 1) warnings.push("Input snapshots differ. Inspect data and universe differences before attributing results to a strategy.");
    if (new Set(identities.map(p => canonical(p.packages))).size > 1) warnings.push("Recorded package environments differ.");
    if (new Set(identities.map(p => canonical(p.model))).size > 1) warnings.push("Shared model settings differ.");
    for (const [key, label] of [...(first.family === "backtests" ? [["transaction_cost_rate", "Trading costs"], ["risk_free_rate", "Risk-free rates"], ["initial_cash", "Initial capital"]] : []), ["historical_start_date", "Training windows"], ["historical_end_date", "Historical cutoffs"]]) {
      const values = entries.map(e => e.run.request_payload[key]);
      if (values.some(v => v == null)) unknown.push(`${label} are not recorded for every run.`);
      else if (new Set(values.map(canonical)).size > 1) warnings.push(`${label} differ; interpret the saved values under their own assumptions.`);
    }
  }
  const reasons = [...new Set([...blocked, ...unknown, ...warnings])];
  return { state: blocked.length ? "blocked" : unknown.length ? "unverified" : warnings.length ? "varied" : "matching", reasons: reasons.length ? reasons : [sameRun ? "Same run and matching recorded evaluation coverage." : "Recorded evaluation periods, input identity and engine match."], chartAllowed: !blocked.length && !unknown.length };
}

export type DifferenceRow = { key: string; label: string; values: unknown[]; display: string[]; different: boolean };
export function configurationDifferences(entries: SavedCandidate[], manifests: Record<string, unknown>): DifferenceRow[] {
  const columns = entries.map(entry => {
    const rows: Record<string, { label: string; value: unknown }> = {};
    for (const [label, value] of experimentRows(entry.run)) rows[`experiment:${label}`] = { label, value };
    rows.method = { label: "Method", value: entry.configuration.method };
    if (entry.family !== "backtests") rows.identifier = { label: "Recorded model identifier", value: entry.configuration.settings[entry.family === "predictions" ? "predictor" : "solver"] };
    rows.actual_periods = { label: "Actual evaluation", value: entry.periods ? `${entry.periods[0]} … ${entry.periods.at(-1)} (${entry.periods.length} periods)` : entry.family === "solvers" ? "One optimization problem" : "Not recorded" };
    for (const [key, value] of configurationFields(entry.configuration)) {
      const canonicalKey = key.replace(/^optimization_/, "");
      rows[`configuration:${canonicalKey}`] = { label: settingLabel(canonicalKey), value };
    }
    const p = provenance(manifests[entry.run.id]);
    rows.dataset_hash = { label: "Input snapshot SHA-256", value: p.dataset };
    rows.universe_hash = { label: "Asset universe SHA-256", value: p.universe };
    rows.source_hash = { label: "Engine SHA-256", value: p.source };
    rows.packages = { label: "Recorded package versions", value: Object.keys(p.packages).length ? p.packages : undefined };
    for (const [key, value] of Object.entries(p.model)) rows[`model:${key}`] = { label: `Shared model: ${settingLabel(key)}`, value };
    return rows;
  });
  return [...new Set(columns.flatMap(Object.keys))].map(key => {
    const values = columns.map((col, i) => {
      if (col[key]) return col[key].value;
      const settings = entries[i].configuration.settings;
      const selection = settings.selection ?? settings.optimization_selection;
      if (key.startsWith("configuration:") && settings.strategy_type === "buy_and_hold" && key !== "configuration:asset_symbol") return "Not applicable";
      if (key.startsWith("configuration:") && settings.strategy_type === "momentum_risk_parity" && !["top_n", "lookback", "vol_window", "trend_window", "min_weight"].includes(key.slice("configuration:".length))) return "Not applicable";
      if ((["configuration:news_top_assets", "configuration:news_min_articles"].includes(key) && settings.news_shortlist === false)
        || (key === "configuration:return_target_fraction" && typeof selection === "string" && selection !== "return_target")
        || (key === "configuration:volatility_increase" && typeof selection === "string" && selection !== "return_with_risk_budget")) return "Not applicable";
      return undefined;
    });
    return {
      key, label: columns.find(col => col[key])![key].label, values,
      display: values.map(v => key.endsWith("_hash") && typeof v === "string" ? `${v.slice(0, 12)}…` : settingValue(key.replace(/^configuration:/, ""), v)),
      different: new Set(values.map(canonical)).size > 1
    };
  });
}
