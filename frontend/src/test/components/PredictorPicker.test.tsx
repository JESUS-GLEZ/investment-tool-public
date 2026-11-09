import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import PredictorPicker from "@/components/ui/PredictorPicker";

afterEach(cleanup);

it("exposes research variants without selecting them by default", () => {
  const onChange = vi.fn();
  render(<PredictorPicker selected={["EWM-6"]} onChange={onChange} />);
  fireEvent.click(screen.getByRole("button", {name: "Add predictor"}));
  for (const label of ["EWM12-Mean-25", "EWM12-Mean-50", "EWM12-Mean-75", "MomentumRidge", "Momentum12-2Ridge", "ZeroReturn", "HistoricalMean", "FuzzyThesis-fixed", "FuzzyThesis-adaptive", "FuzzyThesis-no-trend", "PooledRidge", "FuzzyRidge-fixed", "FuzzyRidge-adaptive"]) {
    const option = screen.getByRole("button", { name: `Add ${label}` });
    expect(option).toBeEnabled();
    fireEvent.click(option);
    expect(onChange).toHaveBeenLastCalledWith(["EWM-6", label]);
  }
});
