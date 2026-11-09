/* ---------- Enums (matching backend exactly) ---------- */

export type RunStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

export type RunType =
  | "execution"
  | "strategy_comparison"
  | "solver_comparison"
  | "prediction_comparison";

export type DataSource = "alpaca" | "local";

export type ExecutionMode = "investing" | "backtesting";

export type Frequency = "daily" | "weekly" | "monthly";

export type StrategyType =
  | "combined"
  | "optimization"
  | "momentum_risk_parity"
  | "buy_and_hold"
  | "sentiment_ewm";

export type OutputMode = "standard" | "diagnostic";

export type ValidationLevel = "strict" | "warn" | "skip";

/* ---------- Request payloads ---------- */

export interface HoldingInput { symbol: string; quantity: number; }

export interface ExecutionRequest {
  current_holdings?: HoldingInput[];
  campaign_id?: string | null;
  output_mode?: OutputMode;
  data_source?: DataSource;
  execution_mode?: ExecutionMode;
  frequency?: Frequency;
  strategy_type?: StrategyType;
  strategy_configuration?: StrategyVariant | null;
  backtest_start_date?: string | null;
  validation_level?: ValidationLevel;
  min_assets?: number;
  max_assets?: number;
  historical_start_date?: string;
  historical_end_date?: string;
  initial_cash?: number;
  transaction_cost_rate?: number;
  risk_free_rate?: number;
  input_file?: string | null;
  asset_config_file?: string;
}

export type PortfolioSelection = "sharpe" | "return_with_risk_budget" | "return_target";

export interface StrategyVariant {
  name: string;
  strategy_type: StrategyType;
  asset_symbol?: string | null;
  top_n?: number | null;
  lookback?: number | null;
  vol_window?: number | null;
  trend_window?: number | null;
  min_weight?: number | null;
  return_estimator?: "ewm_6" | "ewm_12" | "ewm_12_mean_50" | null;
  balance?: "risk_contributions" | "capital_weights" | "hybrid";
  news_shortlist?: boolean;
  news_top_assets?: number;
  news_min_articles?: number;
  selection: PortfolioSelection;
  return_target_fraction: number;
  volatility_increase: number;
  population_size?: number | null;
  n_generations?: number | null;
  min_assets?: number | null;
  max_assets?: number | null;
  n_restarts: number;
  random_seed: number;
  search_seconds: number | null;
  seed_max_subsets: number;
  seed_beam_width?: number;
  refinement_supports?: number;
}

export interface StrategyComparisonRequest {
  campaign_id?: string | null;
  output_mode?: OutputMode;
  strategy_types?: StrategyType[];
  data_source?: DataSource;
  frequency?: Frequency;
  validation_level?: ValidationLevel;
  benchmark_asset?: string;
  include_benchmark?: boolean;
  min_assets?: number;
  max_assets?: number;
  population_size?: number;
  n_generations?: number;
  strategy_variants?: StrategyVariant[];
  optimization_selection?: PortfolioSelection;
  optimization_volatility_increase?: number;
  optimization_return_target_fraction?: number;
  historical_start_date?: string;
  backtest_start_date?: string | null;
  historical_end_date?: string;
  initial_cash?: number;
  transaction_cost_rate?: number;
  risk_free_rate?: number;
  input_file?: string | null;
  asset_config_file?: string;
}

export interface SolverVariant {
  name: string; solver: string;
  population_size?: number | null; n_generations?: number | null; n_restarts?: number | null;
  random_seed?: number | null; mutation_prob?: number | null;
  divisions_outer?: number | null; epsilons?: number[] | null;
}
export interface PredictorVariant {
  name: string; predictor: string;
  span?: number | null; lag_order?: number | null; alpha?: number | null;
  momentum_weight?: number | null; min_train_history?: number | null;
  partitions?: number | null; lags?: number | null; sigma?: number | null;
  beta?: number | null; trend?: boolean | null; grid_points?: number | null;
}

export interface SolverComparisonRequest {
  campaign_id?: string | null;
  solver_variants?: SolverVariant[] | null;
  output_mode?: OutputMode;
  data_source?: DataSource;
  frequency?: Frequency;
  validation_level?: ValidationLevel;
  population_size?: number;
  n_generations?: number;
  n_restarts?: number;
  min_assets?: number;
  num_assets?: number;
  include_markowitz_reference?: boolean;
  use_multi_greedy_seeds?: boolean;
  seed_max_subsets?: number;
  seed_beam_width?: number;
  solvers?: string[];
  historical_start_date?: string;
  historical_end_date?: string;
  initial_cash?: number;
  transaction_cost_rate?: number;
  risk_free_rate?: number;
  input_file?: string | null;
  asset_config_file?: string;
}

export interface PredictionComparisonRequest {
  campaign_id?: string | null;
  predictor_variants?: PredictorVariant[] | null;
  output_mode?: OutputMode;
  data_source?: DataSource;
  frequency?: Frequency;
  validation_level?: ValidationLevel;
  n_eval_periods?: number;
  min_history?: number;
  top_k?: number;
  predictors?: string[];
  compute_sentiment?: boolean;
  historical_start_date?: string;
  historical_end_date?: string;
  initial_cash?: number;
  transaction_cost_rate?: number;
  risk_free_rate?: number;
  input_file?: string | null;
  asset_config_file?: string;
}

export interface AssetConfigSummary {
  file: string;
  name: string;
  stock_count: number;
  crypto_count: number;
  membership_mode?: "dated";
  stock_limit?: number;
}

/* ---------- Response payloads ---------- */

export interface RunSummary {
  campaign_id?: string | null;
  request_payload?: Record<string, unknown>;
  heartbeat_at?: string | null;
  worker_health?: "alive" | "unresponsive" | "unknown" | null;
  progress_fraction?: number;
  progress_message?: string;
  id: string;
  run_type: RunType;
  status: RunStatus;
  created_at: string;
  completed_at: string | null;
  error_message: string | null;
}

export interface RunDetail extends RunSummary {
  request_payload: Record<string, unknown>;
  result: Record<string, unknown> | null;
}

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
}

export interface ErrorResponse {
  error: string;
  message: string;
  details: Record<string, unknown>;
  status_code: number;
}

/* ---------- WebSocket progress ---------- */

export interface ProgressMessage {
  fraction: number;
  message: string;
}

/* ---------- Reference data ---------- */

export interface SymbolsResponse {
  stocks: string[];
  crypto: string[];
}

export type StrategySearchDefaults = {
  pop_size: number;
  n_generations: number;
  n_restarts: number;
  random_seed: number;
  seed_max_subsets: number;
  seed_beam_width: number;
  search_seconds: number | null;
  seed_time_fraction: number;
  refinement_supports?: number;
};

export type DefaultsResponse = Record<string, Record<string, unknown>> & {
  strategy_optimizer?: StrategySearchDefaults;
};

export interface DatasetMetadata {
  title?: string;
  description?: string;
  intended_use?: string | null;
  provider?: string | null;
  retrieved_at?: string | null;
  inspected_at?: string;
  size_bytes?: number;
  features?: { prices?: boolean | null; execution_quotes?: boolean | null; news?: boolean | null; cached_sentiment?: boolean | null };
}

export interface LocalDataInfo extends DatasetMetadata {
  frequency?: Frequency | null;
  asset_config_file?: string | null;
  asset_config_name?: string | null;
  asset_config_available?: boolean | null;
  time_contract_version?: number | null;
  file: string;
  stocks: string[];
  crypto: string[];
  start_date: string | null;
  end_date: string | null;
  stock_count: number;
  crypto_count: number;
}

export interface LocalDataFileSummary extends DatasetMetadata {
  crypto_count?: number;
  asset_config_name?: string | null;
  file: string;
  frequency: string | null;
  start_date: string | null;
  end_date: string | null;
  stock_count: number | null;
}

/** Organisation metadata, independent of model configuration and results. */
export interface CampaignFields {
  name: string;
  hypothesis: string;
  notes: string;
}
export interface Campaign extends CampaignFields {
  id: string;
  created_at: string;
  updated_at: string;
  run_count: number;
}
