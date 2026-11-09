/** Organize retained observations; never combine executions or reconstruct returns. */
import { configurationFields } from "./runPresentation";
import { canonical, comparisonCompatibility, finite, provenance, type SavedCandidate } from "./savedComparison";
import { record } from "./runPresentation";
import { metricRestriction } from "./comparisonMetrics";
import { strategySignature } from "./compareRuns";

export type ConfigurationColumn = {key: string; label: string; entries: SavedCandidate[]};
export type PeriodRow = {key: string; entries: SavedCandidate[]};
export const MATRIX_METRICS = [["total_return", "Return"], ["max_drawdown", "Max drawdown"], ["annualized_volatility", "Annualized volatility"], ["sharpe_ratio", "Sharpe"], ["average_turnover", "Average gross turnover"]] as const;

export function configurationKey(entry: SavedCandidate): string {
  // An unnamed benchmark asset or a model with no saved settings cannot establish a series.
  return configurationFields(entry.configuration).length ? strategySignature(entry) : `unverified:${entry.id}`;
}
export function configurationColumns(entries: SavedCandidate[]): ConfigurationColumn[] {
  const groups = new Map<string, SavedCandidate[]>();
  for (const entry of entries) {
    const key = configurationKey(entry);
    groups.set(key, [...(groups.get(key) ?? []), entry]);
  }
  return [...groups].map(([key, values], index) => ({key, label: `C${index + 1}`, entries: values}));
}
function contextKey(entry: SavedCandidate, manifests: Record<string, unknown>): string {
  const p = provenance(manifests[entry.run.id]);
  const keys = ["frequency", "historical_start_date", "historical_end_date", "initial_cash", "transaction_cost_rate", "risk_free_rate"];
  const assumptions = Object.fromEntries(keys.map(key => [key, entry.run.request_payload[key]]));
  assumptions.frequency = record(entry.run.result).frequency ?? assumptions.frequency;
  const known = p.dataset && p.universe && p.source && Object.keys(p.packages).length && Object.values(p.packages).every(v => typeof v === "string" && v !== "unavailable") && keys.every(key => assumptions[key] != null);
  return known ? canonical({dataset: p.dataset, universe: p.universe, source: p.source, packages: p.packages, model: p.model, assumptions}) : `run:${entry.run.id}`;
}
export function periodRows(entries: SavedCandidate[], manifests: Record<string, unknown>): PeriodRow[] {
  const groups = new Map<string, SavedCandidate[]>();
  for (const entry of entries) {
    const key = canonical({periods: entry.periods ?? entry.id, context: contextKey(entry, manifests)});
    groups.set(key, [...(groups.get(key) ?? []), entry]);
  }
  return [...groups].map(([key, values]) => ({key, entries: values})).sort((a, b) => (a.entries[0].periods?.[0] ?? "~").localeCompare(b.entries[0].periods?.[0] ?? "~") || a.entries[0].run.created_at.localeCompare(b.entries[0].run.created_at) || a.key.localeCompare(b.key));
}
export function pairedResult(row: PeriodRow, entry: SavedCandidate, referenceKey: string, manifests: Record<string, unknown>, metric: string): {reference?: SavedCandidate; reason: string | null} {
  const references = row.entries.filter(e => configurationKey(e) === referenceKey);
  const repeats = row.entries.filter(e => configurationKey(e) === configurationKey(entry));
  if (references.length !== 1) return {reason: references.length ? "Multiple reference executions; remove unwanted executions under Selected result identities." : "No reference in this period and context."};
  if (repeats.length !== 1) return {reason: "Multiple executions; no execution is selected or averaged automatically."};
  const reference = references[0];
  const context = comparisonCompatibility([reference, entry], manifests);
  if (!context.chartAllowed || context.state !== "matching") return {reference, reason: context.reasons.join(" ")};
  return {reference, reason: metricRestriction(metric, [reference, entry])};
}
export function consistency(rows: PeriodRow[], columnKey: string, referenceKey: string, manifests: Record<string, unknown>) {
  const eligible = rows.flatMap(row => {
    const entries = row.entries.filter(e => configurationKey(e) === columnKey);
    if (entries.length !== 1) return [];
    const entry = entries[0], pair = pairedResult(row, entry, referenceKey, manifests, "total_return");
    return pair.reason || !pair.reference || !finite(entry.metrics.total_return) || !finite(pair.reference.metrics.total_return) ? [] : [{entry, reference: pair.reference}];
  });
  // Overlapping windows are not independent periods. Exclude all members of an overlap.
  const independent = eligible.filter((pair, index) => eligible.every((other, otherIndex) => index === otherIndex || pair.entry.periods!.at(-1)!.split(" → ")[1] < other.entry.periods![0].split(" → ")[0] || other.entry.periods!.at(-1)!.split(" → ")[1] < pair.entry.periods![0].split(" → ")[0]));
  const deltas = independent.map(({entry, reference}) => (entry.metrics.total_return as number) - (reference.metrics.total_return as number)).sort((a, b) => a - b);
  const middle = Math.floor(deltas.length / 2);
  const drawdowns = independent.filter(({entry, reference}) => finite(entry.metrics.max_drawdown) && finite(reference.metrics.max_drawdown));
  return {matched: deltas.length, excluded: rows.length - deltas.length, higher: deltas.filter(v => v > 0).length, median: deltas.length ? deltas.length % 2 ? deltas[middle] : (deltas[middle - 1] + deltas[middle]) / 2 : null, drawdownPairs: drawdowns.length, worseDrawdown: drawdowns.filter(({entry, reference}) => (entry.metrics.max_drawdown as number) < (reference.metrics.max_drawdown as number)).length};
}
