import { expect, it } from "vitest";
import { configurationColumns, configurationKey, consistency, pairedResult, periodRows } from "@/lib/compareMatrix";
import { resultAlias } from "@/lib/compareRuns";
import { savedCandidates, type SavedCandidate } from "@/lib/savedComparison";
import { backtest } from "../fixtures/savedRuns";

const manifest = (dataset = "d") => ({format_version: 1, source_sha256: "e".repeat(64), input_dataset: {sha256: dataset.repeat(64)}, asset_universe: {sha256: "a".repeat(64)}, package_versions: {numpy: "2"}, model_configuration: {}});
function annual(id: string, year: number): SavedCandidate[] {
  const entries = savedCandidates(structuredClone(backtest(id))).slice(0, 2);
  entries.forEach(e => {e.periods = [`${year}-01-01 → ${year}-12-31`]; e.metrics.max_drawdown = e.name === "Sharpe" ? -0.1 : -0.2;});
  return entries;
}
it("groups renamed recorded configurations across periods but separates settings and seeds", () => {
  const a = annual("a", 2020)[0], b = annual("b", 2021)[0];
  b.name = "Renamed"; b.configuration.name = b.name;
  expect(configurationColumns([a, b])).toHaveLength(1);
  b.configuration.settings.random_seed = 1;
  expect(configurationColumns([a, b])).toHaveLength(2);
  b.configuration.settings.random_seed = undefined;
  b.configuration.settings.population_size = 500;
  expect(configurationColumns([a, b])).toHaveLength(2);
});
it("does not infer a series for benchmarks with no recorded asset", () => {
  const a = savedCandidates(backtest("a"))[2], b = savedCandidates(backtest("b"))[2];
  expect(configurationKey(a)).not.toBe(configurationKey(b));
  a.configuration.settings.asset_symbol = "SPY"; b.configuration.settings.asset_symbol = "SPY";
  expect(configurationKey(a)).toBe(configurationKey(b));
});
it("pairs only actual identical coverage and context, retaining gaps and partial windows", () => {
  const a = annual("a", 2020), b = annual("b", 2020);
  expect(periodRows([...a, ...b], {})).toHaveLength(2);
  expect(periodRows([...a, ...b], {a: manifest(), b: manifest()})).toHaveLength(1);
  expect(periodRows([...a, ...b], {a: manifest(), b: manifest("b")})).toHaveLength(2);
  b[0].periods = ["2020-01-01 → 2020-06-30"];
  expect(periodRows([...a, ...b], {a: manifest(), b: manifest()})).toHaveLength(2);
});
it("separates same-period history, engine and costs instead of silently combining experiments", () => {
  for (const key of ["historical_start_date", "historical_end_date", "initial_cash", "risk_free_rate", "transaction_cost_rate", "frequency"]) {
    const a = annual("a", 2020), b = annual("b", 2020);
    b.forEach(e => {e.run.request_payload[key] = "different";});
    if (key === "frequency") b.forEach(e => {e.run.result!.frequency = "different";});
    expect(periodRows([...a, ...b], {a: manifest(), b: manifest()})).toHaveLength(2);
  }
  const entries = [...annual("a", 2020), ...annual("b", 2020)];
  expect(periodRows(entries, {a: manifest(), b: {...manifest(), source_sha256: "f".repeat(64)}})).toHaveLength(2);
});
it("does not assign unknown actual coverage to a known-period row", () => {
  const a = annual("a", 2020);
  a[1].periods = null;
  expect(periodRows(a, {})).toHaveLength(2);
});
it("preserves repeated executions and withholds reference selection instead of choosing or averaging", () => {
  const entries = [...annual("a", 2020), ...annual("b", 2020)];
  const manifests = {a: manifest(), b: manifest()};
  const rows = periodRows(entries, manifests);
  expect(rows).toHaveLength(1);
  expect(rows[0].entries).toHaveLength(4);
  expect(pairedResult(rows[0], entries[1], configurationKey(entries[0]), manifests, "total_return").reason).toContain("Multiple reference executions");
  expect(consistency(rows, configurationKey(entries[1]), configurationKey(entries[0]), manifests).matched).toBe(0);
});
it("counts unique paired periods and computes median differences without compounding returns", () => {
  const entries = [...annual("a", 2020), ...annual("b", 2021)];
  entries[3].metrics.total_return = 0; // valid zero; -10 pp versus reference
  const rows = periodRows(entries, {});
  expect(consistency(rows, configurationKey(entries[1]), configurationKey(entries[0]), {})).toEqual({matched: 2, excluded: 0, higher: 1, median: 0, drawdownPairs: 2, worseDrawdown: 2});
});
it("excludes all overlapping windows, missing returns and absent references from consistency", () => {
  const a = annual("a", 2020), b = annual("b", 2020);
  const entries = [...a, ...b];
  const result = consistency(periodRows(entries, {}), configurationKey(a[1]), configurationKey(a[0]), {});
  expect(result.matched).toBe(0);
  expect(result.excluded).toBe(2);
  const c = annual("c", 2021); c[1].metrics.total_return = null;
  expect(consistency(periodRows(c, {}), configurationKey(c[1]), configurationKey(c[0]), {}).matched).toBe(0);
  expect(pairedResult({key: "", entries: [c[1]]}, c[1], configurationKey(c[0]), {}, "total_return").reason).toContain("No reference");
});
it("withholds incompatible metric definitions while retaining raw values", () => {
  const entries = annual("a", 2020); entries[1].definitions = {};
  const row = periodRows(entries, {})[0];
  expect(pairedResult(row, entries[1], configurationKey(entries[0]), {}, "sharpe_ratio").reason).toBe("Metric definition unavailable.");
  expect(pairedResult(row, entries[1], configurationKey(entries[0]), {}, "total_return").reason).toBeNull();
});
it("supports readable execution aliases beyond 26 results", () => {
  expect([0, 25, 26, 119].map(resultAlias)).toEqual(["A", "Z", "AA", "DP"]);
});
