import type { RunDetail } from "@/types/api";
export const definitions = { version: 2, annualized_return: "cagr", annualization_basis: "calendar_act_365_25", sharpe: "arithmetic_excess_sample_std_sqrt_periods_per_year", sortino: "sample_downside_all_observations_sqrt_frequency_approximation", annual_risk_free_rate: 0, periods_per_year: 12 };
export const metrics = { initial_value: 1000, final_value: 1100, total_return: 0.1, annualized_return: 0.12, sharpe_ratio: 1.2 };
export const rows = [{ period_start: "2026-01-01", period_end: "2026-01-31", date_basis: "period_end_label", total_value: 1050 }, { period_start: "2026-02-01", period_end: "2026-02-28", date_basis: "period_end_label", total_value: 1100 }];
export function backtest(id = "a"): RunDetail {
  return {
id, run_type: "strategy_comparison", status: "completed", created_at: "2026-10-01T00:00:00Z", completed_at: null, error_message: null,
    request_payload: { frequency: "monthly", data_source: "local", input_file: "prices.pkl", historical_start_date: "2024-01-01", historical_end_date: "2026-03-01", backtest_start_date: "2026-01-01", initial_cash: 1000, transaction_cost_rate: 0, risk_free_rate: 0, strategy_variants: [{ name: "Sharpe", strategy_type: "optimization", selection: "sharpe", population_size: 100 }, { name: "Target", strategy_type: "optimization", selection: "return_target", return_target_fraction: 0.75, population_size: 200 }] },
    result: { frequency: "monthly", comparison_metrics: { Sharpe: metrics, Target: { ...metrics, final_value: 1200, total_return: 0.2 }, "Buy & Hold": metrics }, metric_definitions: { Sharpe: definitions, Target: definitions, "Buy & Hold": definitions }, individual_results: { Sharpe: { metrics, results: rows }, Target: { results: rows }, "Buy & Hold": { results: rows, configuration: { strategy_type: "buy_and_hold" } } } }
};
}
