import { describe, expect, it } from "vitest";
import { formatMetricValue, metricLabel, comparisonMetricLabel } from "@/lib/metrics";

describe("financial metric display units", () => {
  it.each([
    ["profit_factor", 1.5, "1.50"],
    ["final_hhi", 0.25, "0.25"],
    ["sharpe_ratio", -0.5, "-0.50"],
    ["total_return", 0.12, "12.00%"],
    ["max_drawdown", -0.25, "-25.00%"],
    ["average_turnover", 0.1, "10.00%"],
    ["final_top3_weight", 0.8, "80.00%"],
    ["total_periods", 12, "12"],
    ["max_drawdown_duration_periods", 3, "3"],
    ["best_period_date", "2026-01-31", "2026-01-31"],
  ])("formats %s with its own units", (key, value, expected) => {
    expect(formatMetricValue(key, value)).toBe(expected);
  });

  it("preserves cents in currency values", () => {
    const amount = 1234.56;
    const expected = `$${amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    expect(formatMetricValue("initial_value", amount)).toBe(expected);
    expect(formatMetricValue("final_value", amount)).toBe(expected);
    expect(formatMetricValue("total_transaction_costs", amount)).toBe(expected);
  });

  it.each([null, undefined, NaN, Infinity, -Infinity])(
    "shows unavailable values as a dash for every numeric unit (%s)",
    (value) => {
      for (const key of ["initial_value", "total_periods", "total_return", "profit_factor", "final_hhi"]) {
        expect(formatMetricValue(key, value)).toBe("—");
      }
    },
  );
  it.each(["1.5", "Infinity", true, {}, []])("rejects nonnumeric saved values for numeric fields (%s)", (value) => {
    for (const key of ["initial_value", "total_periods", "total_return", "profit_factor", "final_hhi"]) {
      expect(formatMetricValue(key, value)).toBe("—");
    }
  });

  it("does not round invalid fractional counts into an apparent count", () => {
    expect(formatMetricValue("total_periods", 1.25)).toBe("—");
  });

  it("requires recognized metadata before assigning growth semantics", () => {
    const calendar = { version: 2, annualized_return: "cagr" };
    const compound = { version: 2, annualized_return: "geometric_period_return" };
    expect(metricLabel("annualized_return", calendar)).toBe("CAGR");
    expect(metricLabel("annualized_return", compound)).toBe("Annualized Compound Return");
    for (const d of [undefined, null, "legacy", { annualized_return: "cagr" }, { version: 2, annualized_return: "unknown" }, { version: 3, annualized_return: "cagr" }]) {
      expect(metricLabel("annualized_return", d)).toBe("Annualized Return (definition unavailable)");
    }
    expect(comparisonMetricLabel("annualized_return", [calendar, calendar])).toBe("CAGR");
    expect(comparisonMetricLabel("annualized_return", [calendar, compound])).toBe("Annualized Return (mixed definitions)");
    expect(comparisonMetricLabel("annualized_return", [calendar, undefined])).toBe("Annualized Return (mixed definitions)");
  });
});
