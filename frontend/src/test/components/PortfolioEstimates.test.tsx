import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import PortfolioEstimates from "@/components/ui/PortfolioEstimates";
import type { RunDetail } from "@/types/api";

const run = (diagnostics: Record<string, unknown>, frequency: unknown = "monthly") => ({id: "run", run_type: "execution", status: "completed", created_at: "2026-11-01", completed_at: null, error_message: null, request_payload: {frequency: "monthly", risk_free_rate: 0.05}, result: {frequency, optimization_diagnostics: diagnostics}}) satisfies RunDetail;
afterEach(cleanup);

it("shows selected net and gross forecasts, volatility and period Sharpe with explicit units", () => {
  render(<PortfolioEstimates run={run({selected_gross_expected_return: 0.02, selected_net_expected_return: 0.015, selected_volatility: 0.03})} />);
  expect(screen.getByText("1.50%")).toBeInTheDocument();
  expect(screen.getByText("Before costs: 2.00%")).toBeInTheDocument();
  expect(screen.getByText("3.00%")).toBeInTheDocument();
  expect(screen.getByText("0.50")).toBeInTheDocument();
  expect(screen.getByText(/Per-month model estimates/)).toHaveTextContent("not annualized");
  expect(screen.getByText("Expected Sharpe (0% risk-free)")).toBeInTheDocument();
});

it.each([['weekly', 'week'], ['daily', 'day']])("uses the saved %s frequency without annualizing", (frequency, period) => {
  render(<PortfolioEstimates run={run({selected_net_expected_return: 0.02, selected_volatility: 0.04}, frequency)} />);
  expect(screen.getByText(new RegExp(`Per-${period} model estimates`))).toBeInTheDocument();
  expect(screen.getByText("0.50")).toBeInTheDocument();
});

it("preserves zero estimates and treats zero-volatility Sharpe as undefined", () => {
  render(<PortfolioEstimates run={run({selected_net_expected_return: 0, selected_volatility: 0})} />);
  expect(screen.getAllByText("0.00%")).toHaveLength(2);
  expect(screen.getByText("—", {selector: 'dd'})).toBeInTheDocument();
});

it("never borrows a different frontier portfolio's estimates or coerces malformed values", () => {
  render(<PortfolioEstimates run={run({selected_net_expected_return: "0.3", selected_volatility: -1, final_front: {max_expected_return: 0.4, min_variance: 0.001, sharpe_objectives: [0.001, -0.2]}})} />);
  expect(screen.getAllByText("—", {selector: 'dd'})).toHaveLength(3);
  expect(screen.queryByText("40.00%")).not.toBeInTheDocument();
});

it("shows negative return and Sharpe without suppressing losses", () => {
  render(<PortfolioEstimates run={run({selected_net_expected_return: -0.01, selected_volatility: 0.02})} />);
  expect(screen.getByText("-1.00%")).toBeInTheDocument();
  expect(screen.getByText("-0.50")).toBeInTheDocument();
});

it("labels gross-only older estimates before costs and leaves missing risk unavailable", () => {
  render(<PortfolioEstimates run={run({selected_gross_expected_return: 0.03}, null)} />);
  expect(screen.getByText("Expected return (before costs)")).toBeInTheDocument();
  expect(screen.getByText("3.00%")).toBeInTheDocument();
  expect(screen.getAllByText("—", {selector: 'dd'})).toHaveLength(2);
});
