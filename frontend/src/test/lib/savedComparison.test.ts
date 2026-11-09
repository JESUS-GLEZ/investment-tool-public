import { expect, it } from "vitest";
import type { RunDetail } from "@/types/api";
import { canonical, savedCandidates, comparisonCompatibility, configurationDifferences, comparisonFamily } from "@/lib/savedComparison";
import { metricRestriction, comparisonValue, metricRows } from "@/lib/comparisonMetrics";

import { backtest, definitions, metrics, rows } from "../fixtures/savedRuns";
const manifest = (dataset = "d", source = "e") => ({ format_version: 1, input_dataset: { sha256: dataset.repeat(64) }, asset_universe: { sha256: "a".repeat(64) }, source_sha256: source.repeat(64), package_versions: { numpy: "2.0" }, model_configuration: { min_assets: 2 } });
function prediction(id = "p"): RunDetail {
  return { ...backtest(id), run_type: "prediction_comparison", request_payload: { ...backtest(id).request_payload, top_k: 5, predictor_variants: [{ name: "Six", predictor: "EWM", span: 6 }, { name: "Twelve", predictor: "EWM", span: 12 }] }, result: { results: ["Six", "Twelve"].map(label => ({ config: { label }, evaluation_dates: ["2026-01-01T00:00:00"], actuals_df: { AAPL: [0.1], MSFT: [null] }, metrics: { mae: 0.05, rmse: 0.1, bias: 0, scored_pairs: 1, coverage: 0.5, failed_periods: 0, hit_rate_top_k: 0.5 } })) } };
}
it("adapts named configurations and retained automatic benchmarks without inventing their asset", () => {
  const candidates = savedCandidates(backtest());
  expect(candidates.map(c => c.name)).toEqual(["Sharpe", "Target", "Buy & Hold"]);
  expect(candidates[0].points).toHaveLength(2);
  expect(candidates[0].periods?.[0]).toBe("2026-01-01 → 2026-01-31");
  expect(candidates[1].metrics.final_value).toBe(1200);
  expect(candidates[2].configuration.settings.asset_symbol).toBeUndefined();
  expect(configurationDifferences(candidates.slice(0, 2), {}).find(r => r.key === "configuration:population_size")?.display).toEqual(["100", "200"]);
});
it("adapts a single backtest column frame, preserving zero and metric definitions", () => {
  const run = backtest(); run.run_type = "execution"; run.request_payload = { ...run.request_payload, execution_mode: "backtesting", strategy_type: "optimization" };
  run.result = { metrics: { ...metrics, final_value: 0, metric_definitions: definitions }, results_df: Object.fromEntries(Object.keys(rows[0]).map(key => [key, rows.map(row => row[key as keyof typeof row])])) };
  expect(savedCandidates(run)[0].metrics.final_value).toBe(0);
  expect(savedCandidates(run)[0].definitions).toEqual(definitions);
  expect(savedCandidates(run)[0].points).toHaveLength(2);
  expect(comparisonValue("backtests", "final_value", 0)).toBe("$0.00");
});
it("excludes investing, partial and failed outputs and leaves ambiguous periods unverified", () => {
  const run = backtest();
  expect(comparisonFamily({ ...run, run_type: "execution", request_payload: { execution_mode: "investing" } })).toBeNull();
  expect(savedCandidates({ ...run, status: "failed" })).toEqual([]);
  expect(savedCandidates({ ...run, result: { ...run.result, partial: true } })).toEqual([]);
  const malformed = structuredClone(run); (malformed.result!.individual_results as Record<string, unknown>).Sharpe = { metrics, results: rows.map(row => ({ ...row, date_basis: "legacy" })) };
  expect(comparisonCompatibility(savedCandidates(malformed).slice(0, 2), {}).state).toBe("unverified");
});
it("requires matching actual period boundaries and frequencies", () => {
  const a = savedCandidates(backtest())[0], b = structuredClone(savedCandidates(backtest())[1]);
  expect(comparisonCompatibility([a, b], {}).state).toBe("matching");
  b.periods = ["2025-01-01 → 2025-01-31"];
  expect(comparisonCompatibility([a, b], {}).state).toBe("blocked");
  b.periods = a.periods; b.run = { ...b.run, result: { ...b.run.result, frequency: "daily" } };
  expect(comparisonCompatibility([a, b], {}).state).toBe("blocked");
});
it("requires cross-run provenance, allows explicit universe experiments and blocks engine changes", () => {
  const a = savedCandidates(backtest("a"))[0], b = savedCandidates(backtest("b"))[1];
  expect(comparisonCompatibility([a, b], {}).state).toBe("unverified");
  expect(comparisonCompatibility([a, b], { a: manifest(), b: manifest() }).state).toBe("matching");
  expect(comparisonCompatibility([a, b], { a: manifest(), b: manifest("a") }).state).toBe("varied");
  expect(comparisonCompatibility([a, b], { a: manifest(), b: manifest("d", "f") }).state).toBe("blocked");
  const missing = manifest(); missing.package_versions.numpy = "unavailable";
  expect(comparisonCompatibility([a, b], { a: manifest(), b: missing }).chartAllowed).toBe(false);
});
it("compares prediction scoring masks, includes coverage, and guards top-k", () => {
  const entries = savedCandidates(prediction());
  expect(entries).toHaveLength(2);
  expect(comparisonCompatibility(entries, {}).state).toBe("varied");
  expect(metricRows(entries).map(r => r.key)).toContain("coverage");
  expect(comparisonValue("predictions", "bias", 0)).toBe("0.00%");
  expect(metricRestriction("hit_rate_top_k", entries)).toBeNull();
  entries[1] = { ...entries[1], actuals: canonical({ AAPL: [0.2] }) };
  expect(comparisonCompatibility(entries, {}).state).toBe("blocked");
  entries[1].run = prediction("q"); entries[1].run.request_payload.top_k = 10;
  expect(metricRestriction("hit_rate_top_k", entries)).toContain("Top-k");
});
it("keeps solver front size and expected Sharpe distinct from portfolio holdings and realized returns", () => {
  const run = { ...backtest(), run_type: "solver_comparison" as const, result: { results: ["NSGA-II", "SPEA2"].map(algorithm => ({ config: { label: algorithm, algorithm, population_size: 100 }, metrics: { hypervolume: 0.123, cardinality: 40, best_sharpe: 2, wall_time: 1 } })) } };
  const entries = savedCandidates(run);
  expect(metricRows(entries).find(r => r.key === "cardinality")?.label).toBe("Pareto solutions");
  expect(comparisonCompatibility(entries, {}).state).toBe("matching");
  expect(comparisonCompatibility([entries[0], { ...entries[1], run: { ...run, id: "b" } }], {}).state).toBe("blocked");
});
it("does not compare mixed or unknown metric definitions and never formats nonfinite values as results", () => {
  const entries = savedCandidates(backtest()).slice(0, 2);
  expect(metricRestriction("annualized_return", entries)).toBeNull();
  entries[1].definitions = { ...definitions, annualized_return: "geometric_period_return", annualization_basis: "period_count" };
  expect(metricRestriction("annualized_return", entries)).toContain("differ");
  entries[1].definitions = {};
  expect(metricRestriction("sharpe_ratio", entries)).toContain("unavailable");
  for (const value of [NaN, Infinity, null, "0.1"]) { expect(comparisonValue("predictions", "rmse", value)).toBe("—"); }
});
it("canonicalizes object order while retaining masks, values and array order", () => {
  expect(canonical({ b: [1, null], a: 0 })).toBe('{"a":0,"b":[1,null]}');
  expect(canonical({ a: 0, b: [1, null] })).toBe(canonical({ b: [1, null], a: 0 }));
});

it("compares universe contents even when dataset and universe filenames are identical", () => {
  const entries = [savedCandidates(backtest("a"))[0], savedCandidates(backtest("b"))[0]];
  const changed = manifest(); changed.asset_universe.sha256 = "b".repeat(64);
  const manifests = { a: manifest(), b: changed };
  expect(comparisonCompatibility(entries, manifests).state).toBe("varied");
  expect(comparisonCompatibility(entries, manifests).reasons.join(" ")).toContain("Asset-universe contents differ");
  expect(configurationDifferences(entries, manifests).find(row => row.key === "universe_hash")?.different).toBe(true);
  expect(comparisonCompatibility(entries, { a: manifest(), b: { ...manifest(), asset_universe: null } }).state).toBe("unverified");
});
it("distinguishes saved legacy predictor identifiers without inferring today's parameter defaults", () => {
  const run = prediction(); run.request_payload = { ...run.request_payload, predictor_variants: [], predictors: ["EWM-6", "EWM-12"] };
  (run.result!.results as Record<string, unknown>[]).forEach((value, i) => { value.config = { label: i ? "EWM-12" : "EWM-6" }; });
  const entries = savedCandidates(run);
  expect(configurationDifferences(entries, {}).find(row => row.key === "identifier")?.display).toEqual(["EWM-6", "EWM-12"]);
});

it("uses saved solver identifiers when algorithm constructors were serialized as objects", () => {
  const run = { ...backtest(), run_type: "solver_comparison" as const, request_payload: { solvers: ["NSGA-II"] }, result: { results: [{ config: { label: "NSGA-II", algorithm: { __type__: "ABCMeta", __module__: "platypus.algorithms" } }, metrics: { hypervolume: 0.01, cardinality: 40 } }] } };
  const entry = savedCandidates(run)[0];
  expect(entry.configuration.settings.solver).toBe("NSGA-II");
  expect(entry.configuration.method).toBe("NSGA-II");
  expect(configurationDifferences([entry], {}).find(row => row.key === "identifier")?.display).toEqual(["NSGA-II"]);
});

it("distinguishes a disabled news filter from missing recorded configuration", () => {
  const entries = savedCandidates(backtest()).slice(0, 2);
  entries[0].configuration.settings.news_shortlist = false;
  entries[1].configuration.settings.news_shortlist = true;
  entries[1].configuration.settings.news_top_assets = 50;
  expect(configurationDifferences(entries, {}).find(r => r.key === "configuration:news_top_assets")?.display).toEqual(["Not applicable", "50"]);
  delete entries[0].configuration.settings.news_shortlist;
  expect(configurationDifferences(entries, {}).find(r => r.key === "configuration:news_top_assets")?.display[0]).toBe("Not recorded");
});
it("keeps saved allocation summaries and period labels without renormalizing them", () => {
  const run = backtest();
  const output = (run.result!.individual_results as Record<string, Record<string, unknown>>).Sharpe;
  output.results = [{...rows[0], selected_assets: {AAA: 0.7, BBB: 0.29}}, {...rows[1], selected_assets_summary: "CCC: 60.00%, DDD: 39.00%"}];
  const entry = savedCandidates(run)[0];
  expect(entry.holdings[0].weights).toEqual({AAA: 0.7, BBB: 0.29});
  expect(entry.holdings[1].weights).toEqual({CCC: 0.6, DDD: 0.39});
  expect(entry.holdings[1].label).toBe("2026-02-01 → 2026-02-28");
});
it("distinguishes empty recorded summaries from missing or malformed weight records", () => {
  const run = backtest();
  const output = (run.result!.individual_results as Record<string, Record<string, unknown>>).Sharpe;
  output.results = [{...rows[0], selected_assets: {}}, {...rows[1]}, {...rows[1], selected_assets: []}];
  const holdings = savedCandidates(run)[0].holdings;
  expect(holdings[0]).toMatchObject({weights: {}, recorded: true, incomplete: false});
  expect(holdings[1]).toMatchObject({recorded: false, incomplete: false});
  expect(holdings[2]).toMatchObject({recorded: false, incomplete: true});
});
