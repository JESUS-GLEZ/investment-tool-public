/** Presentation and comparison rules for retained backtests. Never run the engine. */
import type { RunSummary } from "@/types/api";
import { configurationFields, record, settingValue } from "./runPresentation";
import { libraryPeriod, libraryRunType } from "./runLibrary";
import { canonical, finite, provenance, type SavedCandidate } from "./savedComparison";
import { comparisonValue, metricRestriction } from "./comparisonMetrics";

export function resultAlias(index: number): string {
  let label = "";
  for (let n = index + 1; n > 0; n = Math.floor((n - 1) / 26)) label = String.fromCharCode(65 + (n - 1) % 26) + label;
  return label;
}
export function executionLabel(run: RunSummary): string {
  const timestamp = Date.parse(run.created_at);
  return Number.isFinite(timestamp) ? `${new Date(timestamp).toISOString().slice(0, 16).replace("T", " ")} UTC` : "Execution date not recorded";
}
export function periodSpan(entry: SavedCandidate): string {
  if (!entry.periods?.length) return libraryPeriod(entry.run).value;
  const first = entry.periods[0].split(" → ")[0].slice(0, 10);
  const last = entry.periods.at(-1)!.split(" → ").at(-1)!.slice(0, 10);
  return `${first} → ${last}`;
}
export function coverageLabel(entry: SavedCandidate): string {
  if (!entry.periods) return "Actual coverage not recorded";
  const [start, end] = periodSpan(entry).split(" → ");
  const year = start.slice(0, 4) === end.slice(0, 4);
  const span = year ? start.endsWith("01-01") && end.endsWith("12-31") ? "Calendar-year span" : "Partial year" : "Multiple years";
  return `${span} · ${entry.periods.length} recorded periods`;
}
export function runChoiceLabel(run: RunSummary): string {
  return `${libraryRunType(run)} · ${libraryPeriod(run).value} · ${executionLabel(run)}`;
}
export function candidateDescription(entry: SavedCandidate): string {
  const fields = Object.fromEntries(configurationFields(entry.configuration));
  const s = fields;
  const selection = s.selection ?? s.optimization_selection;
  const parts = [entry.configuration.settings.strategy_type === "buy_and_hold" ? "Buy and hold" : entry.configuration.method];
  if (s.balance != null) parts.push(settingValue("balance", s.balance));
  if (s.return_estimator != null) parts.push(settingValue("return_estimator", s.return_estimator));
  if (selection != null) {
    let label = settingValue("selection", selection);
    const target = s.return_target_fraction ?? s.optimization_return_target_fraction;
    const budget = s.volatility_increase ?? s.optimization_volatility_increase;
    if (selection === "return_target" && finite(target)) label += ` ${settingValue("return_target_fraction", target)}`;
    if (selection === "return_with_risk_budget" && finite(budget)) label += ` +${settingValue("volatility_increase", budget)}`;
    parts.push(label);
  }
  if (s.news_shortlist === false) parts.push("No news filter");
  if (s.news_shortlist === true) parts.push(`News top ${finite(s.news_top_assets) ? s.news_top_assets : "?"}`);
  if (typeof s.asset_symbol === "string") parts.push(s.asset_symbol);
  return [...new Set(parts)].join(" · ");
}
/** Ignore names, dates and dataset filenames. Do not fill absent settings from defaults. */
export function strategySignature(entry: SavedCandidate): string {
  const fields = Object.fromEntries(configurationFields(entry.configuration).map(([key, value]) => [key.replace(/^optimization_/, "").replace(/^num_assets$/, "max_assets"), value]));
  return canonical({method: entry.configuration.settings.strategy_type ?? entry.configuration.method, fields});
}
export function recordedChanges(entry: SavedCandidate, reference: SavedCandidate, manifests: Record<string, unknown>): string[] {
  const notes: string[] = [];
  if (strategySignature(entry) !== strategySignature(reference)) notes.push("Strategy parameters differ from reference");
  if (!configurationFields(entry.configuration).length) notes.push("Strategy parameters not recorded");
  const a = provenance(manifests[entry.run.id]), b = provenance(manifests[reference.run.id]);
  if (entry.run.id !== reference.run.id) {
    if (!a.source || !b.source) notes.push("Engine identity unverified");
    else if (a.source !== b.source) notes.push("Engine changed");
    if (!a.dataset || !a.universe || !b.dataset || !b.universe) notes.push("Data/universe identity unverified");
    else if (a.universe !== b.universe) notes.push("Universe changed");
    if (a.dataset && b.dataset && a.dataset !== b.dataset) notes.push("Input snapshot changed");
    if (!Object.keys(a.packages).length || !Object.keys(b.packages).length || [...Object.values(a.packages), ...Object.values(b.packages)].some(value => typeof value !== "string" || value === "unavailable")) notes.push("Package identity unverified");
    else if (canonical(a.packages) !== canonical(b.packages)) notes.push("Package environment changed");
    if (canonical(a.model) !== canonical(b.model)) notes.push("Shared model settings changed");
  }
  for (const [key, label] of [["transaction_cost_rate", "Trading costs"], ["risk_free_rate", "Risk-free rate"]]) {
    const value = entry.run.request_payload[key], other = reference.run.request_payload[key];
    if (!finite(value) || !finite(other)) notes.push(`${label} not recorded`);
    else if (value !== other) notes.push(`${label} changed`);
  }
  if (!entry.periods) notes.push("Actual periods unverified");
  return notes;
}
const percentageMetrics = new Set(["total_return", "annualized_return", "annualized_volatility", "max_drawdown", "average_turnover"]);
export function metricDelta(key: string, entry: SavedCandidate, reference: SavedCandidate, allowed: boolean): string {
  if (entry.id === reference.id) return "Reference";
  if (!allowed || metricRestriction(key, [entry, reference])) return "Not comparable";
  const value = entry.metrics[key], base = reference.metrics[key];
  if (!finite(value) || !finite(base)) return "Not recorded";
  const delta = (value - base) * (percentageMetrics.has(key) ? 100 : 1);
  return `${delta > 0 ? "+" : ""}${delta.toFixed(2)}${percentageMetrics.has(key) ? " pp" : ""}`;
}
/** Only a retained, unambiguous benchmark identity; never assume a ticker from defaults. */
export function savedBenchmark(entry: SavedCandidate, candidates: SavedCandidate[]): SavedCandidate | undefined {
  const request = entry.run.request_payload;
  const explicitName = record(entry.run.result).benchmark_name;
  const sameRun = candidates.filter(c => c.run.id === entry.run.id && c.id !== entry.id);
  const named = typeof explicitName === "string" ? sameRun.find(c => c.name === explicitName) : undefined;
  const automatic = request.include_benchmark !== false ? sameRun.find(c => c.name === "buy_and_hold" && c.configuration.settings.strategy_type === "buy_and_hold") : undefined;
  const explicit = request.include_benchmark === true && typeof request.benchmark_asset === "string" ? sameRun.filter(c => c.configuration.settings.strategy_type === "buy_and_hold" && c.configuration.settings.asset_symbol === request.benchmark_asset) : [];
  const benchmark = named ?? automatic ?? (explicit.length === 1 ? explicit[0] : undefined);
  return benchmark && entry.periods && canonical(entry.periods) === canonical(benchmark.periods) && entry.run.request_payload.frequency === benchmark.run.request_payload.frequency ? benchmark : undefined;
}
export function benchmarkDifference(entry: SavedCandidate, benchmark?: SavedCandidate): string {
  if (!benchmark || !finite(entry.metrics.total_return) || !finite(benchmark.metrics.total_return)) return "Not recorded";
  const pp = 100 * (entry.metrics.total_return - benchmark.metrics.total_return);
  return `${pp > 0 ? "+" : ""}${pp.toFixed(2)} pp`;
}
export const savedMetric = (entry: SavedCandidate, key: string) => comparisonValue("backtests", key, entry.metrics[key]);
