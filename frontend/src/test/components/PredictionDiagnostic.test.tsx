import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import PredictionComparisonCharts from "@/components/ui/PredictionComparisonCharts";
import Plotly from "plotly.js-dist-min";
vi.mock("plotly.js-dist-min", () => ({default: {newPlot: vi.fn(), purge: vi.fn()}}));
afterEach(() => {cleanup(); vi.clearAllMocks();});
it("mounts only the chosen fallback diagnostic and purges the preceding chart", () => {
  render(<PredictionComparisonCharts results={[{config: {label: "EWM"}, metrics: {mae: 0.1}, predictions_df: {AAA: [0.1]}, actuals_df: {AAA: [0.2]}, per_asset_metrics: {mae: [0.1], directional_accuracy: [1]}}]} />);
  expect(Plotly.newPlot).not.toHaveBeenCalled();
  fireEvent.change(screen.getByLabelText("Prediction diagnostic"), {target: {value: "mae"}});
  expect(Plotly.newPlot).toHaveBeenCalledTimes(1);
  fireEvent.change(screen.getByLabelText("Prediction diagnostic"), {target: {value: "direction"}});
  expect(Plotly.purge).toHaveBeenCalledTimes(1);
  expect(Plotly.newPlot).toHaveBeenCalledTimes(2);
  expect(screen.queryByText("Per-Asset MAE by Predictor")).not.toBeInTheDocument();
  expect(screen.getByText("Per-Asset Directional Accuracy by Predictor")).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("Prediction diagnostic"), {target: {value: ""}});
  expect(Plotly.purge).toHaveBeenCalledTimes(2);
});
