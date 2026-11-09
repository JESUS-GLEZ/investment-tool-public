import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import PredictionComparisonCharts from "@/components/ui/PredictionComparisonCharts";

const { plot, purge } = vi.hoisted(() => ({ plot: vi.fn(), purge: vi.fn() }));
vi.mock("plotly.js-dist-min", () => ({ default: { newPlot: plot, purge } }));

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("prediction comparison coverage", () => {
  it("renders unavailable metrics and reports a zero common sample", () => {
    render(<PredictionComparisonCharts results={[{
      config: { label: "EWM-6" },
      metrics: { mae: null, information_coefficient: null, scored_pairs: 0, eligible_pairs: 12, scored_periods: 0, evaluation_periods: 3, failed_periods: 1 },
      predictions_df: { A: [null, null, null] },
      actuals_df: { A: [null, null, null] },
      per_asset_metrics: { mae: [null], directional_accuracy: [null] },
    }]} />);
    expect(screen.getByText("Evaluation coverage")).toBeInTheDocument();
    expect(screen.getByText(/EWM-6: 0\/12 pairs scored/)).toHaveTextContent("0/3 periods; 1 failed periods");
    for (const value of ["metrics", "scatter", "mae"]) fireEvent.change(screen.getByLabelText("Prediction diagnostic"), {target: {value}});
    for (const call of plot.mock.calls) {
      expect(call[3].showSendToCloud).toBe(false);
    }
    expect(plot.mock.calls[1][2].xaxis.title).toEqual({ text: "Predicted return" });
    expect(plot.mock.calls[2][1][0].colorbar.title).toEqual({ text: "MAE" });
    const bars = plot.mock.calls[0][1];
    expect(bars[0].y).toEqual([]);
    const scatter = plot.mock.calls[1][1];
    expect(scatter[1].x.every(Number.isFinite)).toBe(true);
    const heatmap = plot.mock.calls[2][1];
    expect(heatmap[0].z).toEqual([[null]]);
  });

  it("keeps older stored comparisons usable without new coverage fields", () => {
    render(<PredictionComparisonCharts results={[{
      config: { label: "EWM-6" }, metrics: { mae: 0.1 },
      predictions_df: { A: [0.1] }, actuals_df: { A: [0.2] },
      per_asset_metrics: { mae: [0.1], directional_accuracy: [1] },
    }]} />);
    expect(screen.queryByText("Evaluation coverage")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Prediction diagnostic")).toBeInTheDocument();
  });
});
