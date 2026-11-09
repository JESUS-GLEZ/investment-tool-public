import { expect, it } from "vitest";
import type { RunSummary } from "@/types/api";
import { recordedConfigurations, configurationSettings, experimentRows, settingValue } from "@/lib/runPresentation";
const run = (payload: Record<string, unknown>, type: RunSummary["run_type"] = "strategy_comparison") => ({ run_type: type, request_payload: payload }) as RunSummary;

it("resolves saved inheritance without current defaults and keeps repeated configurations", () => {
  const configs = recordedConfigurations(run({
    population_size: 200, min_assets: 2, max_assets: 5, strategy_types: ["optimization", "buy_and_hold"], strategy_variants: [
      { name: "Target", strategy_type: "optimization", selection: "return_target", return_target_fraction: 0.75, population_size: null, min_assets: null, max_assets: 4, random_seed: 0, search_seconds: null, news_shortlist: false },
      { name: "Sharpe", strategy_type: "optimization", selection: "sharpe", population_size: 100 },
    ]
  }));
  expect(configs.map(c => c.name)).toEqual(["Buy & Hold", "Target", "Sharpe"]);
  expect(configs[1].settings).toMatchObject({ population_size: 200, min_assets: 2, max_assets: 4, random_seed: 0, search_seconds: null });
  expect(configurationSettings(configs[1])).toContain("No wall-clock limit");
  expect(configurationSettings(configs[1])).toContain("Relative return target: 75%");
  expect(configurationSettings(configs[1])).toContain("News shortlist: Disabled");
  expect(configurationSettings(configs[0])).not.toContain("Population");
  expect(configurationSettings(recordedConfigurations(run({}, "execution"))[0])).toBe("No settings recorded");
});
it("shows local universes as embedded and keeps requested benchmark separate", () => {
  const fixture = run({ data_source: "local", input_file: "prices.pkl", asset_config_file: "ignored.json", strategy_types: ["optimization"], include_benchmark: true, benchmark_asset: "SPY" });
  expect(experimentRows(fixture)).toContainEqual(["Universe", "Embedded in local dataset"]);
  expect(experimentRows(fixture)).toContainEqual(["Requested benchmark", "SPY"]);
  expect(recordedConfigurations(fixture)).toHaveLength(1);
  expect(settingValue("initial_cash", 0)).toBe("0");
});
it("supports legacy registries and named predictor/solver overrides", () => {
  const legacy = recordedConfigurations(run({ predictors: ["ewm-6", "EWM-6", "EWM-12"] }, "prediction_comparison"));
  expect(legacy).toHaveLength(2);
  expect(legacy[0].name).toBe("EWM-6");
  const named = recordedConfigurations(run({ population_size: 300, n_restarts: 4, solver_variants: [{ name: "Fast", solver: "NSGA-II", population_size: 100, n_generations: null }], n_generations: 1000 }, "solver_comparison"));
  expect(named[0].settings).toMatchObject({ population_size: 100, n_restarts: 4, n_generations: 1000 });
  expect(configurationSettings(recordedConfigurations(run({ predictor_variants: [{ name: "Slow", predictor: "EWM", span: 12 }], population_size: 300 }, "prediction_comparison"))[0])).toBe("EWM span: 12");
});

it("exposes meaningful shared switches even when no method parameters differ", () => {
  for (const enabled of [true, false]) {
    expect(experimentRows(run({ compute_sentiment: enabled }, "prediction_comparison"))).toContainEqual(["Sentiment analysis", enabled ? "Enabled" : "Disabled"]);
    const rows = experimentRows(run({ use_multi_greedy_seeds: enabled, include_markowitz_reference: !enabled }, "solver_comparison"));
    expect(rows).toContainEqual(["Multi-Greedy seeds", enabled ? "Enabled" : "Disabled"]);
    expect(rows).toContainEqual(["Convex MV reference", enabled ? "Disabled" : "Enabled"]);
  }
  expect(experimentRows(run({}, "prediction_comparison"))).toContainEqual(["Sentiment analysis", "Not recorded"]);
});

it("records legacy Buy & Hold asset even when the additional benchmark is disabled", () => {
  for (const symbol of ["QQQ", "SPY"]) {
    const fixture = run({ strategy_types: ["buy_and_hold"], include_benchmark: false, benchmark_asset: symbol });
    expect(configurationSettings(recordedConfigurations(fixture)[0])).toBe(`Asset: ${symbol}`);
    expect(experimentRows(fixture)).toContainEqual(["Requested benchmark", "Disabled"]);
  }
});
