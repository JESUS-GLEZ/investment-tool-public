import { expect, it } from "vitest";
import { benchmarkDifference, candidateDescription, coverageLabel, executionLabel, metricDelta, periodSpan, recordedChanges, savedBenchmark, strategySignature } from "@/lib/compareRuns";
import { comparisonCompatibility, savedCandidates } from "@/lib/savedComparison";
import { backtest } from "../fixtures/savedRuns";
const manifest = (source = "e") => ({format_version: 1, source_sha256: source.repeat(64), input_dataset: {sha256: "d".repeat(64)}, asset_universe: {sha256: "a".repeat(64)}, package_versions: {numpy: "2"}, model_configuration: {}});
it("uses recorded strategy parameters for identity, ignoring saved names and date/universe labels", () => {
  const a = savedCandidates(backtest())[0], b = structuredClone(a);
  b.name = "Different display name"; b.configuration.name = b.name;
  b.configuration.settings.input_file = "other.pkl";
  b.configuration.settings.historical_start_date = "2019-01-01";
  expect(strategySignature(a)).toBe(strategySignature(b));
  b.configuration.settings.population_size = 777;
  expect(strategySignature(a)).not.toBe(strategySignature(b));
  const target = savedCandidates(backtest())[1];
  expect(candidateDescription(target)).toContain("75%");
});
it("keeps the existing strict engine identity guard", () => {
  const entries = [savedCandidates(backtest("a"))[0], savedCandidates(backtest("b"))[0]];
  expect(comparisonCompatibility(entries, {a: manifest(), b: manifest("f")}).state).toBe("blocked");
});
it("expresses reference differences in percentage points or ratio units and withholds unverifiable comparisons", () => {
  const [a, b] = savedCandidates(backtest());
  expect(metricDelta("total_return", b, a, true)).toBe("+10.00 pp");
  b.metrics.sharpe_ratio = 1.5;
  expect(metricDelta("sharpe_ratio", b, a, true)).toBe("+0.30");
  expect(metricDelta("total_return", b, a, false)).toBe("Not comparable");
  b.metrics.total_return = 0;
  expect(metricDelta("total_return", b, a, true)).toBe("-10.00 pp");
  b.definitions = {};
  expect(metricDelta("sharpe_ratio", b, a, true)).toBe("Not comparable");
});
it("labels partial years from actual observations, never the requested end date", () => {
  const entry = savedCandidates(backtest())[0];
  expect(coverageLabel(entry)).toBe("Partial year · 2 recorded periods");
  expect(periodSpan(entry)).toBe("2026-01-01 → 2026-02-28");
  entry.periods = null;
  expect(coverageLabel(entry)).toBe("Actual coverage not recorded");
  expect(executionLabel(entry.run)).toContain("2026-10-01 00:00 UTC");
});
it("uses only an identifiable retained benchmark with matching actual coverage, preserving zero returns", () => {
  const run = structuredClone(backtest()); run.request_payload.include_benchmark = true; run.request_payload.benchmark_asset = "SPY";
  const output = run.result!.individual_results as Record<string, Record<string, unknown>>;
  output["Buy & Hold"].configuration = {strategy_type: "buy_and_hold", asset_symbol: "SPY"};
  const entries = savedCandidates(run);
  const benchmark = savedBenchmark(entries[0], entries);
  expect(benchmark?.name).toBe("Buy & Hold");
  expect(benchmarkDifference(entries[1], benchmark)).toBe("+10.00 pp");
  benchmark!.metrics.total_return = 0;
  expect(benchmarkDifference(entries[1], benchmark)).toBe("+20.00 pp");
  benchmark!.periods = ["2025-01-01 → 2025-12-31"];
  expect(savedBenchmark(entries[0], entries)).toBeUndefined();
  expect(benchmarkDifference(entries[1])).toBe("Not recorded");
});
it("flags parameter, engine and assumption changes without pretending annual universes are identical", () => {
  const [a, b] = [savedCandidates(backtest("a"))[0], savedCandidates(backtest("b"))[0]];
  b.configuration.settings.selection = "return_target";
  b.run.request_payload.transaction_cost_rate = 0.01;
  const notes = recordedChanges(b, a, {a: manifest(), b: manifest("f")});
  expect(notes).toContain("Strategy parameters differ from reference");
  expect(notes).toContain("Engine changed");
  expect(notes).toContain("Trading costs changed");
  expect(recordedChanges(b, a, {})).toContain("Data/universe identity unverified");
});
it("retains saved period returns without inferring them from valuation movements", () => {
  const run = backtest();
  const output = (run.result!.individual_results as Record<string, Record<string, unknown>>).Sharpe;
  output.results = [{period_start: "2026-01-01", period_end: "2026-01-31", date_basis: "period_end_label", total_value: 9000, return: 0}, {period_start: "2026-02-01", period_end: "2026-02-28", date_basis: "period_end_label", total_value: 10000}];
  expect(savedCandidates(run)[0].observations.map(o => o.value)).toEqual([0, null]);
});
it("does not describe a buy-and-hold benchmark with inherited optimizer parameters", () => {
  const benchmark = savedCandidates(backtest())[2];
  benchmark.configuration.settings.optimization_selection = "return_target";
  benchmark.configuration.settings.balance = "hybrid";
  benchmark.configuration.settings.return_target_fraction = 0.75;
  benchmark.configuration.settings.news_shortlist = true;
  expect(candidateDescription(benchmark)).toBe("Buy and hold");
});
it("does not report a known universe change when the reference identity is missing", () => {
  const a = savedCandidates(backtest("a"))[0], b = savedCandidates(backtest("b"))[0];
  const changes = recordedChanges(b, a, {b: manifest()});
  expect(changes).toContain("Data/universe identity unverified");
  expect(changes).not.toContain("Universe changed");
});
