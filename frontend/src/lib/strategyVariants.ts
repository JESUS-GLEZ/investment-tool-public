import type { DefaultsResponse, ExecutionRequest, StrategyComparisonRequest, StrategyType, StrategyVariant, StrategySearchDefaults } from "@/types/api";

export const strategyCatalogue: {type: StrategyType; label: string; description: string}[] = [
  {type: "optimization", label: "Optimization", description: "Expected returns and covariance with balanced risk contributions, capital weights, or both."},
  {type: "combined", label: "Combined", description: "Sentiment and Ridge VARX forecasts with portfolio optimisation."},
  {type: "sentiment_ewm", label: "Sentiment EWM", description: "Sentiment-adjusted EWM forecasts with portfolio optimisation."},
  {type: "momentum_risk_parity", label: "Momentum Risk Parity", description: "Trend-filtered momentum selection with inverse-volatility weights."},
  {type: "buy_and_hold", label: "Buy & Hold", description: "Hold one asset as a baseline. No portfolio optimiser."},
];
export const strategyLabel = (type: StrategyType) => strategyCatalogue.find(s => s.type === type)?.label ?? type;
export const usesOptimizer = (type: StrategyType) => ["optimization", "combined", "sentiment_ewm"].includes(type);

export function newStrategyVariant(name: string, defaults: StrategySearchDefaults, selection: StrategyVariant["selection"] = "sharpe"): StrategyVariant {
  return {
    name, strategy_type: "optimization", return_estimator: "ewm_6", balance: "risk_contributions",
    news_shortlist: false, news_top_assets: 50, news_min_articles: 3,
    selection, return_target_fraction: 0.75, volatility_increase: 0.25,
    population_size: defaults.pop_size, n_generations: defaults.n_generations,
    min_assets: 2, max_assets: 5,
    n_restarts: defaults.n_restarts, random_seed: defaults.random_seed,
    search_seconds: defaults.search_seconds, seed_max_subsets: defaults.seed_max_subsets,
    seed_beam_width: defaults.seed_beam_width, refinement_supports: defaults.refinement_supports ?? 80,
  };
}
function defaultNumber(defaults: DefaultsResponse, group: string, key: string, fallback: number) {
  const value = defaults[group]?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}
export function createStrategyInstance(type: StrategyType, name: string, defaults: DefaultsResponse): StrategyVariant {
  if (!defaults.strategy_optimizer) throw new Error("Strategy defaults are unavailable");
  return {
    ...newStrategyVariant(name, defaults.strategy_optimizer), strategy_type: type,
    return_estimator: type === "optimization" ? "ewm_6" : undefined,
    min_assets: defaultNumber(defaults, "construction", "min_assets", 2),
    max_assets: defaultNumber(defaults, "construction", "max_assets", 5),
    ...(type === "buy_and_hold" ? {asset_symbol: "SPY"} : {}),
    ...(type === "momentum_risk_parity" ? {
      top_n: defaultNumber(defaults, "momentum", "top_n", 10),
      lookback: defaultNumber(defaults, "momentum", "lookback", 6),
      vol_window: defaultNumber(defaults, "momentum", "vol_window", 6),
      trend_window: defaultNumber(defaults, "momentum", "trend_window", 10),
      min_weight: defaultNumber(defaults, "momentum", "min_weight", 0.02),
    } : {}),
  };
}
export function uniqueStrategyName(type: StrategyType, entries: StrategyVariant[]) {
  let i = 1;
  while (entries.some(v => v.name.trim().toLowerCase() === `${strategyLabel(type)} ${i}`.toLowerCase())) i++;
  return `${strategyLabel(type)} ${i}`;
}
/** Resolve legacy inherited values once. Subsequent edits are per instance. */
export function strategiesFromPreset(parameters: ExecutionRequest | StrategyComparisonRequest, mode: "execution" | "strategy_comparison", defaults: DefaultsResponse): StrategyVariant[] {
  const p = parameters as ExecutionRequest & StrategyComparisonRequest;
  const variants = mode === "execution" ? (p.strategy_configuration ? [p.strategy_configuration] : []) : p.strategy_variants ?? [];
  const overridden = new Set(variants.map(v => v.strategy_type));
  const legacyTypes = mode === "execution" ? [p.strategy_type ?? "combined"] : p.strategy_types ?? ["combined"];
  const legacy = [...new Set(legacyTypes)].filter(t => !overridden.has(t)).map(type => ({
    ...createStrategyInstance(type, uniqueStrategyName(type, variants), defaults),
    ...(type === "buy_and_hold" && mode === "strategy_comparison" ? {asset_symbol: p.benchmark_asset ?? "SPY"} : {}),
    selection: type === "optimization" ? p.optimization_selection ?? "sharpe" : "sharpe",
    return_target_fraction: p.optimization_return_target_fraction ?? 0.75,
    volatility_increase: p.optimization_volatility_increase ?? 0.25,
    population_size: p.population_size ?? defaults.strategy_optimizer!.pop_size,
    n_generations: p.n_generations ?? defaults.strategy_optimizer!.n_generations,
    min_assets: p.min_assets ?? 2, max_assets: p.max_assets ?? 5,
  } as StrategyVariant));
  return [...legacy, ...variants.map(v => ({
    ...createStrategyInstance(v.strategy_type, v.name, defaults), ...v,
    population_size: v.population_size ?? p.population_size ?? defaults.strategy_optimizer!.pop_size,
    n_generations: v.n_generations ?? p.n_generations ?? defaults.strategy_optimizer!.n_generations,
    min_assets: v.min_assets ?? p.min_assets ?? 2, max_assets: v.max_assets ?? p.max_assets ?? 5,
  }))];
}

export function strategyError(v: StrategyVariant, others: StrategyVariant[], frequency: string): string | undefined {
  if (!["daily", "weekly", "monthly"].includes(frequency)) return "Choose a supported data frequency.";
  if (!strategyCatalogue.some(s => s.type === v.strategy_type)) return "Choose a supported strategy from the catalogue.";
  const name = v.name.trim();
  if (!name || name.length > 80 || (Array.from(name).some(c => c.charCodeAt(0) < 32 || "/\\<>".includes(c))) || [".", "..", ...strategyCatalogue.map(s => s.type)].includes(name.toLowerCase())) return "Use a descriptive name without path separators, markup or control characters; built-in strategy names are reserved.";
  if (others.some(other => other.name.trim().toLowerCase() === name.toLowerCase())) return "Each configuration needs a unique name.";
  const number = (key: keyof StrategyVariant, min: number, max = Infinity, integer = true) => {
    const value = v[key];
    return typeof value === "number" && Number.isFinite(value) && value >= min && value <= max && (!integer || Number.isInteger(value));
  };
  if (v.strategy_type === "buy_and_hold") return /^[A-Za-z0-9./_-]{1,30}$/.test(v.asset_symbol ?? "") ? undefined : "Choose a valid asset symbol.";
  if (v.strategy_type === "momentum_risk_parity") {
    if (!number("top_n", 1) || !number("lookback", 1) || !number("vol_window", 2) || !number("trend_window", 1) || !number("min_weight", 0, 0.999999, false)) return "Check the momentum settings: use positive whole periods and a minimum weight between 0 and 1 (exclusive).";
    return;
  }
  if (!number("min_assets", 1) || !number("max_assets", 1) || v.min_assets! > v.max_assets!) return "Min assets cannot be greater than max assets; both must be positive whole numbers.";
  if (!number("population_size", 10) || !number("n_generations", 1) || !number("n_restarts", 1, 20) || !number("random_seed", 0, 4294967295) || !number("seed_max_subsets", 1, 360000) || !number("seed_beam_width", 1, 100) || !number("refinement_supports", 0, 500) || (v.search_seconds !== null && !number("search_seconds", Number.MIN_VALUE, Infinity, false))) return "Check the search settings and their allowed ranges.";
  if (!number("return_target_fraction", 0, 1, false) || !number("volatility_increase", 0, 1, false)) return "Portfolio selection percentages must be between 0 and 100.";
  if (v.news_shortlist && (!number("news_top_assets", v.max_assets!, 10000) || !number("news_min_articles", 1, 1000))) return "News shortlist size must be at least maximum assets; use a positive minimum article count.";
}
export function strategySummary(v: StrategyVariant) {
  if (v.strategy_type === "buy_and_hold") return `100% ${v.asset_symbol}`;
  if (v.strategy_type === "momentum_risk_parity") return `Up to ${v.top_n} assets · momentum ${v.lookback} / volatility ${v.vol_window} / trend ${v.trend_window} periods · minimum weight ${v.min_weight}`;
  const selection = v.selection === "return_target" ? `Return target ${Math.round(v.return_target_fraction * 100)}%` : v.selection === "return_with_risk_budget" ? `Volatility budget +${Math.round(v.volatility_increase * 100)}%` : "Maximum Sharpe";
  return `${v.min_assets}–${v.max_assets} assets · ${selection}${v.strategy_type === "optimization" ? ` · ${{ewm_6: "EWM-6", ewm_12: "EWM-12", ewm_12_mean_50: "EWM-12 / mean blend"}[v.return_estimator ?? "ewm_6"]} · ${{risk_contributions: "Balanced risk", capital_weights: "Balanced weights", hybrid: "Balanced weights and risk"}[v.balance ?? "risk_contributions"]}${v.news_shortlist ? ` · news shortlist ${v.news_top_assets}` : ""}` : ""}`;
}
