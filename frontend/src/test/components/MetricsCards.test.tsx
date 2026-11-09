import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import MetricsCards from "@/components/ui/MetricsCards";

describe("MetricsCards", () => {
  it("labels corrected calendar growth as CAGR and preserves legacy labels", () => {
    const { rerender } = render(<MetricsCards metrics={{ annualized_return: 0.12, metric_definitions: { version: 2, annualized_return: "cagr" } }} />);
    expect(screen.getByText("CAGR")).toBeInTheDocument();
    rerender(<MetricsCards metrics={{ annualized_return: 0.12 }} />);
    expect(screen.getByText("Annualized Return (definition unavailable)")).toBeInTheDocument();
    expect(screen.queryByText("CAGR")).not.toBeInTheDocument();
  });

  it("renders nothing when metrics is empty", () => {
    const { container } = render(<MetricsCards metrics={{}} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders Summary section with initial_value formatted as currency", () => {
    render(<MetricsCards metrics={{ initial_value: 10000 }} />);
    expect(screen.getByText("Summary")).toBeInTheDocument();
    expect(screen.getByText("$10,000.00")).toBeInTheDocument();
  });

  it("renders Performance section for annualized_return as percentage", () => {
    render(<MetricsCards metrics={{ annualized_return: 0.12 }} />);
    expect(screen.getByText("Performance")).toBeInTheDocument();
    expect(screen.getByText("12.00%")).toBeInTheDocument();
  });

  it("renders Risk section with max_drawdown as percentage", () => {
    render(<MetricsCards metrics={{ max_drawdown: -0.25 }} />);
    expect(screen.getByText("Risk")).toBeInTheDocument();
    expect(screen.getByText("-25.00%")).toBeInTheDocument();
  });

  it("renders Risk-Adjusted section with sharpe_ratio as decimal", () => {
    render(<MetricsCards metrics={{ sharpe_ratio: 1.75 }} />);
    expect(screen.getByText("Risk-Adjusted")).toBeInTheDocument();
    expect(screen.getByText("1.75")).toBeInTheDocument();
  });

  it("shows profit factor and portfolio HHI as ratios, not percentages", () => {
    render(<MetricsCards metrics={{ profit_factor: 1.5, final_hhi: 0.25, final_top3_weight: 0.8 }} />);
    expect(screen.getByText("1.50")).toBeInTheDocument();
    expect(screen.getByText("0.25")).toBeInTheDocument();
    expect(screen.getByText("80.00%")).toBeInTheDocument();
    expect(screen.queryByText("150.00%")).not.toBeInTheDocument();
    expect(screen.queryByText("25.00%")).not.toBeInTheDocument();
    expect(screen.getByText("Final Capital HHI (0–1)")).toHaveAttribute("title", expect.stringContaining("cash is excluded"));
  });

  it("labels gross turnover and its non-annualized definition", () => {
    render(<MetricsCards metrics={{ average_turnover: 0.15, total_turnover: 1.8 }} />);
    expect(screen.getByText("Average Gross Turnover")).toHaveAttribute("title", expect.stringContaining("Not halved or annualized"));
    expect(screen.getByText("Total Gross Turnover")).toBeInTheDocument();
    expect(screen.getByText("180.00%")).toBeInTheDocument();
  });

  it("renders unknown metric in Other section", () => {
    render(<MetricsCards metrics={{ custom_metric: 42.5 } as Record<string, unknown>} />);
    expect(screen.getByText("Other")).toBeInTheDocument();
    expect(screen.getByText("42.50")).toBeInTheDocument();
  });

  it("shows recorded fees as currency and missing evidence as unavailable", () => {
    const {rerender} = render(<MetricsCards metrics={{total_transaction_costs: 12.34}} />);
    expect(screen.getByText("Total Transaction Costs")).toHaveAttribute("title", expect.stringContaining("Already deducted"));
    expect(screen.getByText("$12.34")).toBeInTheDocument();
    rerender(<MetricsCards metrics={{total_transaction_costs: null}} />);
    expect(screen.getByText("—")).toBeInTheDocument();
  });

  it("formats null values as em-dash", () => {
    render(<MetricsCards metrics={{ sharpe_ratio: null }} />);
    expect(screen.getByText("—")).toBeInTheDocument();
  });

  it("formats total_periods as plain integer", () => {
    render(<MetricsCards metrics={{ total_periods: 252 } as Record<string, unknown>} />);
    expect(screen.getByText("252")).toBeInTheDocument();
  });

  it("renders multiple sections when multiple metric groups present", () => {
    render(
      <MetricsCards
        metrics={{
          initial_value: 1000,
          sharpe_ratio: 1.2,
          annualized_return: 0.08,
        }}
      />
    );
    expect(screen.getByText("Summary")).toBeInTheDocument();
    expect(screen.getByText("Performance")).toBeInTheDocument();
    expect(screen.getByText("Risk-Adjusted")).toBeInTheDocument();
  });

  it("skips object values (non-scalar)", () => {
    render(
      <MetricsCards
        metrics={{ sharpe_ratio: 1.5, nested: { foo: "bar" } } as Record<string, unknown>}
      />
    );
    // nested should not appear as a card
    expect(screen.queryByText("Nested")).not.toBeInTheDocument();
  });
});
