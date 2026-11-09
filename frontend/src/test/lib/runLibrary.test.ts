import { expect, it } from "vitest";
import type { RunSummary } from "@/types/api";
import { libraryPeriod, libraryRunType, libraryUniverse, searchLibrary } from "@/lib/runLibrary";
const run = (payload: Record<string, unknown>, type: RunSummary["run_type"] = "execution"): RunSummary => ({ id: "test", run_type: type, status: "completed", created_at: "2026-10-01T12:00:00Z", completed_at: null, error_message: null, request_payload: payload });
it("labels requested dates without confusing training history with trading or investing", () => {
  const p = { historical_start_date: "2016-01-01", historical_end_date: "2026-09-30T23:00:00-04:00" };
  expect(libraryPeriod(run({ ...p, execution_mode: "investing" }))).toEqual({ label: "Requested data cutoff", value: "2026-09-30" });
  expect(libraryRunType(run({ execution_mode: "investing" }))).toBe("Investing recommendation");
  expect(libraryPeriod(run({ ...p, execution_mode: "backtesting", backtest_start_date: "2026-01-01" }))).toEqual({ label: "Requested backtest", value: "2026-01-01 → 2026-09-30" });
  expect(libraryPeriod(run({ ...p, backtest_start_date: null }, "strategy_comparison")).value).toBe("Automatic start → 2026-09-30");
  expect(libraryPeriod(run({}, "strategy_comparison")).value).toBe("Start not recorded → Not recorded");
  expect(libraryPeriod(run({ ...p, historical_end_date: "2026-02-30" }, "prediction_comparison"))).toEqual({ label: "Historical window", value: "2016-01-01 → Not recorded" });
});
it("identifies the saved local dataset instead of an ignored configured universe", () => {
  const fixture = run({ data_source: "local", input_file: "input_monthly_validation_2018_filtered_news_2016_2018.pkl", asset_config_file: "ignored.json" });
  expect(libraryUniverse(fixture)).toMatchObject({ label: "Local dataset", name: "Validation 2018 · filtered news · history 2016–2018", file: fixture.request_payload?.input_file });
  expect(libraryUniverse(run({ data_source: "alpaca", asset_config_file: "example_asset_universe.json" }))).toMatchObject({ label: "Alpaca universe", name: "example asset universe" });
  expect(libraryUniverse(run({})).name).toBe("Not recorded");
  expect(libraryUniverse(run({data_source:"local",input_file:"input_data_monthly_2017-01-01_to_2018-09-30_6stocks_2018-10-02_execution_enriched.pkl"})).name).toBe("Monthly snapshot 2017–2018 · saved 2018-10-02 · execution prices");
});
it("searches hidden configurations and source identities with combined filters, sorting without mutation", () => {
  const a = run({ strategy_configuration: { strategy_type: "optimization", name: "Risk target", selection: "return_target", return_target_fraction: 0.75 } });
  const b = { ...run({ data_source: "alpaca", asset_config_file: "example_asset_universe.json" }, "solver_comparison"), id: "b", created_at: "2026-10-02T00:00:00Z", status: "failed" as const };
  const input = [a, b];
  expect(searchLibrary(input, "RISK 75%", "all", "completed", "newest")).toEqual([a]);
  expect(searchLibrary(input, "example_asset_universe.json", "solver_comparison", "failed", "newest")).toEqual([b]);
  expect(searchLibrary(input, "", "execution", "failed", "newest")).toEqual([]);
  expect(searchLibrary(input, "", "all", "all", "newest")).toEqual([b, a]);
  expect(input).toEqual([a, b]);
});
