import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import SelectedPortfolio from "@/components/ui/SelectedPortfolio";
import type { RunDetail } from "@/types/api";
const run = (result: Record<string, unknown>) => ({ id: "run", run_type: "execution", status: "completed", created_at: "2026-10-01T12:00:00Z", completed_at: null, error_message: null, request_payload: { initial_cash: 1000 }, result }) satisfies RunDetail;
afterEach(cleanup);
it("shows actual holdings, cost-funded estimates and dated prices without rescaling capital", () => {
  render(<SelectedPortfolio run={run({optimal_weights:{SPY:0.5,GLD:0.5},rebalance_plan:{currency:"USD",current_cash:100,current_value:300,estimated_cash:0,estimated_value:298,estimated_cost:2,rows:[{symbol:"SPY",current_quantity:2,current_weight:2/3,reference_price:100,price_period_end:"2024-02-29",price_available_at:"2024-03-01T05:00:00Z",target_weight:0.5,estimated_quantity:1.5,trade_quantity:-0.5,trade_value:-50,estimated_cost:0.5,action:"sell"}]}})} />);
  expect(screen.getByRole("heading",{name:"Investment recommendation"})).toBeInTheDocument();
  expect(screen.getByText("sell 0.5",{exact:false})).toBeInTheDocument();
  expect(screen.getByText(/period end 2024-02-29/)).toBeInTheDocument();
  expect(screen.getByText("$300.00 / $100.00")).toBeInTheDocument();
  expect(screen.getByText("$298.00 / $0.00")).toBeInTheDocument();
  expect(screen.queryByLabelText("Illustrative capital ($)")).not.toBeInTheDocument();
  expect(screen.getByText(/not live quotes or executed orders/)).toBeInTheDocument();
});
it("shows exact saved targets, keeps tiny positions, excludes zero and never rescales", () => {
  render(<SelectedPortfolio run={run({ optimal_weights: { AAPL: 0.6, "BTC/USD": 0.2, TINY: 0.000001, ZERO: 0, BAD: null } })} />);
  expect(screen.getByText("AAPL")).toBeInTheDocument();
  expect(screen.getByText("BTC/USD")).toBeInTheDocument();
  expect(screen.getByText("TINY")).toBeInTheDocument();
  expect(screen.queryByText("ZERO")).not.toBeInTheDocument();
  expect(screen.getByText("60.00%")).toBeInTheDocument();
  expect(screen.getByText(/weights total 80.00%/)).toBeInTheDocument();
  expect(screen.getByRole("status")).toHaveTextContent("may be incomplete");
  expect(screen.getByText("$600.00")).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("Illustrative capital ($)"), { target: { value: "2000" } });
  expect(screen.getByText("$1,200.00")).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("Illustrative capital ($)"), { target: { value: "" } });
  expect(screen.queryByText("$1,200.00")).not.toBeInTheDocument();
});
it("shows historical periods without inventing dates or hiding summary limitations", () => {
  render(<SelectedPortfolio run={run({ results_df: { selected_assets_summary: ["AAPL:40.00%, BTC/USD:60.00%", "GLD:100.00%, invalid"], period_end: ["2020-01-31T23:00:00Z"] } })} />);
  expect(screen.getByLabelText("Recorded period")).toHaveValue("1");
  expect(screen.getByText(/below 1%/)).toBeInTheDocument();
  expect(screen.getByRole("status")).toHaveTextContent("incomplete");
  fireEvent.change(screen.getByLabelText("Recorded period"), { target: { value: "0" } });
  expect(screen.getByRole("option", { name: "2020-01-31T23:00:00Z" })).toBeInTheDocument();
  expect(screen.getByText("BTC/USD")).toBeInTheDocument();
  expect(screen.queryByRole("status")).not.toBeInTheDocument();
});
it("explains missing and malformed old results", () => {
  const { rerender } = render(<SelectedPortfolio run={run({ metrics: {} })} />);
  expect(screen.getByText(/No selected asset weights/)).toBeInTheDocument();
  rerender(<SelectedPortfolio run={run({ optimal_weights: null })} />);
  expect(screen.getByRole("status")).toHaveTextContent("incomplete");
});
it("separates requested cutoffs from observations and shows lagging assets", () => {
  render(<SelectedPortfolio run={run({optimal_weights: {ABC: 1}, data_freshness: {
    decision_time: "2024-04-01T04:00:00Z", frequency: "monthly", requested_end_date: "2024-12-31",
    latest_available_at: "2024-03-01T05:00:00Z", oldest_asset_period_end: "2024-01-31", newest_asset_period_end: "2024-02-29",
    assets: [{symbol: "ABC", status: "observed", last_period_end: "2024-02-29", latest_available_at: "2024-03-01T05:00:00Z"}, {symbol: "OLD", status: "behind_other_assets", last_period_end: "2024-01-31"}], excluded_assets: {BAD: "Invalid prices"},
  }})} />);
  expect(screen.getByText("2024-12-31")).toBeInTheDocument();
  expect(screen.getByText("2024-01-31 → 2024-02-29")).toBeInTheDocument();
  expect(screen.getByText(/1 assets end earlier/)).toBeInTheDocument();
  expect(screen.getByText("Excluded: Invalid prices")).toBeInTheDocument();
  expect(screen.getByText(/not confirmed execution/)).toBeInTheDocument();
});
it("does not infer actual freshness from an older request", () => {
  render(<SelectedPortfolio run={run({optimal_weights: {ABC: 1}})} />);
  expect(screen.getByText(/Detailed timing was not recorded/)).toBeInTheDocument();
});
