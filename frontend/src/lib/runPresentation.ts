import type { RunSummary } from "@/types/api";
import { strategyCatalogue } from "./strategyVariants";
import { PREDICTOR_META, SOLVER_META, predictorFamilyLabel } from "./comparisonCatalogue";

export const record = (value: unknown): Record<string, unknown> => value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
const records = (value: unknown) => Array.isArray(value) ? value.map(record) : [];
const strings = (value: unknown): string[] => Array.isArray(value) ? value.filter((s): s is string => typeof s === "string") : [];
const canonical = (id: string, options: Record<string, unknown>) => Object.keys(options).find(key => key.toLowerCase() === id.toLowerCase()) ?? id;
const strategyName = (id: unknown) => strategyCatalogue.find(s => s.type === id)?.label ?? String(id ?? "Not recorded");
export type RecordedConfiguration = { name: string; method: string; settings: Record<string, unknown> };

/** Resolve inheritance only from this run's saved shared settings, never today's defaults. */
export function recordedConfigurations(run: RunSummary): RecordedConfiguration[] {
  const p = run.request_payload ?? {};
  const resolve = (entry: Record<string, unknown>) => {
    const merged = { ...p, ...entry };
    for (const key of ["population_size", "n_generations", "n_restarts", "min_assets", "max_assets", "num_assets"]) {
      if (entry[key] == null && p[key] != null) merged[key] = p[key];
    }
    return merged;
  };
  if (run.run_type === "execution") {
    const entry = record(p.strategy_configuration);
    const method = strategyName(entry.strategy_type ?? p.strategy_type);
    return [{ name: String(entry.name ?? method), method, settings: resolve(entry) }];
  }
  if (run.run_type === "strategy_comparison") {
    const variants = records(p.strategy_variants);
    const configured = new Set(variants.map(v => v.strategy_type));
    return [
      ...[...new Set(strings(p.strategy_types))].filter(type => !configured.has(type)).map(type => ({ name: strategyName(type), method: strategyName(type), settings: { ...p, strategy_type: type, ...(type === "buy_and_hold" ? { asset_symbol: p.benchmark_asset } : {}) } })),
      ...variants.map(v => ({ name: String(v.name ?? v.strategy_type), method: strategyName(v.strategy_type), settings: resolve(v) })),
    ];
  }
  const solver = run.run_type === "solver_comparison";
  const variants = records(p[solver ? "solver_variants" : "predictor_variants"]);
  const key = solver ? "solver" : "predictor";
  const meta = solver ? SOLVER_META : PREDICTOR_META;
  const label = (id: string) => solver ? SOLVER_META[id]?.label ?? id : predictorFamilyLabel(id);
  return variants.length ? variants.map(v => ({ name: String(v.name ?? v[key]), method: label(canonical(String(v[key]), meta)), settings: resolve(v) }))
    : [...new Set(strings(p[solver ? "solvers" : "predictors"]).map(id => canonical(id, meta)))].map(id => ({ name: meta[id]?.label ?? id, method: label(id), settings: { ...p, [key]: id } }));
}

const LABELS: Record<string, string> = {
  min_assets: "Min assets", max_assets: "Max assets", num_assets: "Max assets", selection: "Selection", optimization_selection: "Selection",
  balance: "Balance", return_estimator: "Return estimator", return_target_fraction: "Relative return target", optimization_return_target_fraction: "Relative return target",
  volatility_increase: "Volatility increase", optimization_volatility_increase: "Volatility increase", population_size: "Population", n_generations: "Evaluations/restart", n_restarts: "Restarts", random_seed: "Random seed",
  search_seconds: "Search time limit (seconds)", seed_max_subsets: "Seed subsets", seed_beam_width: "Seed beam width", refinement_supports: "Refinement supports",
  span: "EWM span", lag_order: "Lag order", alpha: "Ridge alpha", momentum_weight: "EWM blend weight", min_train_history: "Training history",
  asset_symbol: "Asset", top_n: "Top assets", lookback: "Momentum lookback", vol_window: "Volatility window", trend_window: "Trend window", min_weight: "Min weight",
  news_shortlist: "News shortlist", news_top_assets: "News shortlist size", news_min_articles: "Minimum articles", mutation_prob: "Mutation probability", divisions_outer: "Reference divisions", epsilons: "Epsilon tolerances",
  partitions: "Fuzzy partitions", lags: "Return lags", sigma: "Gaussian width", beta: "Width adaptation", trend: "Trend correction", grid_points: "Centroid grid points",
};
export const settingLabel = (key: string) => LABELS[key] ?? key.replace(/_/g, " ");
const PERCENT = new Set(["return_target_fraction", "optimization_return_target_fraction", "volatility_increase", "optimization_volatility_increase", "momentum_weight", "mutation_prob", "min_weight", "transaction_cost_rate", "risk_free_rate"]);
export function settingValue(key: string, value: unknown): string {
  if (value == null) return key === "search_seconds" && value === null ? "No wall-clock limit" : "Not recorded";
  if (typeof value === "boolean") return value ? "Enabled" : "Disabled";
  if (typeof value === "number") return Number.isFinite(value) ? PERCENT.has(key) ? `${Number((value * 100).toFixed(4))}%` : value.toLocaleString() : "Not recorded";
  const labels: Record<string, string> = { investing: "Investing", backtesting: "Backtesting", sharpe: "Maximum Sharpe", return_target: "Relative return target", return_with_risk_budget: "Return with volatility budget", risk_contributions: "Risk contributions", capital_weights: "Capital weights", hybrid: "Capital and risk", ewm_6: "EWM (span 6)", ewm_12: "EWM (span 12)", ewm_12_mean_50: "EWM-12 / mean (50/50)" };
  return typeof value === "string" ? labels[value] ?? value : JSON.stringify(value);
}
export function configurationFields(configuration: RecordedConfiguration, compact = false): [string, unknown][] {
  const keys = compact
    ? ["selection", "optimization_selection", "balance", "return_estimator", "return_target_fraction", "optimization_return_target_fraction", "news_shortlist", "span", "lag_order", "alpha", "momentum_weight", "asset_symbol", "top_n", "min_assets", "max_assets", "num_assets"]
    : Object.keys(LABELS);
  const settings = configuration.settings;
  const strategy = settings.strategy_type;
  const relevant = (key: string) => {
    if (strategy === "buy_and_hold") return key === "asset_symbol";
    if (strategy === "momentum_risk_parity") return ["top_n", "lookback", "vol_window", "trend_window", "min_weight"].includes(key);
    if (settings.predictor) return ["span", "lag_order", "alpha", "momentum_weight", "min_train_history", "partitions", "lags", "sigma", "beta", "trend", "grid_points"].includes(key);
    if (settings.solver) return !["selection", "optimization_selection", "balance", "return_estimator", "return_target_fraction", "optimization_return_target_fraction", "volatility_increase", "optimization_volatility_increase", "max_assets", "news_shortlist", "news_top_assets", "news_min_articles"].includes(key);
    if (key.startsWith("optimization_") && (settings.selection != null || strategy !== "optimization")) return false;
    if (["return_target_fraction", "optimization_return_target_fraction"].includes(key)) return (settings.selection ?? settings.optimization_selection) === "return_target";
    if (["volatility_increase", "optimization_volatility_increase"].includes(key)) return (settings.selection ?? settings.optimization_selection) === "return_with_risk_budget";
    if (["news_top_assets", "news_min_articles"].includes(key)) return settings.news_shortlist === true;
    return !["asset_symbol", "top_n", "lookback", "vol_window", "trend_window", "min_weight", "num_assets", "span", "lag_order", "alpha", "momentum_weight", "min_train_history", "partitions", "lags", "sigma", "beta", "trend", "grid_points", "mutation_prob", "divisions_outer", "epsilons"].includes(key);
  };
  return keys.filter(key => relevant(key) && (settings[key] != null || key === "search_seconds" && Object.hasOwn(settings, key))).map(key => [key, settings[key]]);
}
export function configurationSettings(configuration: RecordedConfiguration, compact = false): string {
  return configurationFields(configuration, compact).map(([key,value]) => `${settingLabel(key)}: ${settingValue(key,value)}`).join(" · ") || "No settings recorded";
}
export function experimentRows(run: RunSummary): [string, string][] {
  const p = run.request_payload ?? {};
  const val = (key: string) => settingValue(key, p[key]);
  return [
    ["Mode", run.run_type === "strategy_comparison" ? "Backtesting" : run.run_type === "execution" ? val("execution_mode") : "Benchmark"],
    ["Data", `${val("data_source")} · ${val("frequency")}`],
    ["Universe", p.data_source === "local" ? "Embedded in local dataset" : val("asset_config_file")],
    ...(p.data_source === "local" ? [["Dataset", val("input_file")] as [string, string]] : []),
    ["Historical window", `${val("historical_start_date")} → ${val("historical_end_date")}`],
    ["Trading start", p.backtest_start_date ? val("backtest_start_date") : "Not specified"],
    ["Initial capital", p.initial_cash == null ? "Not recorded" : `$${val("initial_cash")}`],
    ["Costs / risk-free rate", `${val("transaction_cost_rate")} / ${val("risk_free_rate")}`],
    ...(run.run_type === "strategy_comparison" ? [["Requested benchmark", p.include_benchmark === false ? "Disabled" : p.include_benchmark === true ? String(p.benchmark_asset ?? "Not recorded") : "Not recorded"] as [string, string]] : []),
    ...(run.run_type === "solver_comparison" ? [["Multi-Greedy seeds", val("use_multi_greedy_seeds")], ["Convex MV reference", val("include_markowitz_reference")]] as [string, string][] : []),
    ...(run.run_type === "prediction_comparison" ? [["Sentiment analysis", val("compute_sentiment")], ["Evaluation", `${val("n_eval_periods")} periods · history ${val("min_history")} · top ${val("top_k")}`]] as [string, string][] : []),
  ];
}
