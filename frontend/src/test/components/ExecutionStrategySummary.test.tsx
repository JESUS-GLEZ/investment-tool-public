import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import ExecutionStrategySummary from "@/components/ui/ExecutionStrategySummary";
import type { RunSummary } from "@/types/api";

const run = (payload: Record<string, unknown>) => ({ id: "run", run_type: "execution", status: "queued", created_at: "2026-11-01T12:00:00Z", completed_at: null, error_message: null, request_payload: payload }) satisfies RunSummary;
afterEach(cleanup);

it("identifies the named strategy and its saved model settings before results exist", () => {
  render(<ExecutionStrategySummary run={run({strategy_type: "combined", strategy_configuration: {name: "Monthly hybrid", strategy_type: "optimization", balance: "hybrid", selection: "return_target", return_target_fraction: 0.75, return_estimator: "ewm_6", min_assets: 2, max_assets: 5}})} />);
  expect(screen.getByRole("heading", {name: "Monthly hybrid"})).toBeInTheDocument();
  expect(screen.getByText("Method: Optimization")).toBeInTheDocument();
  expect(screen.getByText("Capital and risk")).toBeInTheDocument();
  expect(screen.getByText("Relative return target", {selector: "dd"})).toBeInTheDocument();
  expect(screen.getByText("75%")).toBeInTheDocument();
  expect(screen.getByText("EWM (span 6)")).toBeInTheDocument();
  expect(screen.queryByText("Combined")).not.toBeInTheDocument();
});

it("explains legacy execution selections without inventing missing model defaults", () => {
  render(<ExecutionStrategySummary run={run({strategy_type: "optimization", optimization_selection: "return_target", optimization_return_target_fraction: 0.6})} />);
  expect(screen.getByRole("heading", {name: "Optimization"})).toBeInTheDocument();
  expect(screen.getByText("60%")).toBeInTheDocument();
  expect(screen.queryByText("Risk contributions")).not.toBeInTheDocument();
  expect(screen.queryByText("EWM (span 6)")).not.toBeInTheDocument();
});

it("shows a benchmark's actual saved asset without irrelevant optimizer settings", () => {
  render(<ExecutionStrategySummary run={run({population_size: 300, strategy_configuration: {name: "Reference", strategy_type: "buy_and_hold", asset_symbol: "GLD", balance: "hybrid", selection: "sharpe"}})} />);
  expect(screen.getByText("Method: Buy & Hold")).toBeInTheDocument();
  expect(screen.getByText("GLD")).toBeInTheDocument();
  expect(screen.queryByText("Capital and risk")).not.toBeInTheDocument();
  expect(screen.queryByText("Maximum Sharpe")).not.toBeInTheDocument();
});

it("makes missing legacy strategy metadata explicit", () => {
  render(<ExecutionStrategySummary run={run({})} />);
  expect(screen.getByRole("heading", {name: "Not recorded"})).toBeInTheDocument();
  expect(screen.queryByText("Optimization")).not.toBeInTheDocument();
});
