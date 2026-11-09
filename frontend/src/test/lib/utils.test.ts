import { describe, it, expect } from "vitest";
import { statusColor, runTypeLabel, fmtDate, getErrorMessage } from "@/lib/utils";

describe("statusColor", () => {
  it("returns yellow classes for queued", () => {
    expect(statusColor("queued")).toContain("yellow");
  });

  it("returns blue classes for running", () => {
    expect(statusColor("running")).toContain("blue");
  });

  it("returns green classes for completed", () => {
    expect(statusColor("completed")).toContain("green");
  });

  it("returns red classes for failed", () => {
    expect(statusColor("failed")).toContain("red");
  });
});

describe("runTypeLabel", () => {
  it("returns human-readable label for execution", () => {
    expect(runTypeLabel("execution")).toBe("Execution");
  });

  it("returns human-readable label for strategy_comparison", () => {
    expect(runTypeLabel("strategy_comparison")).toBe("Strategy Comparison");
  });

  it("returns human-readable label for solver_comparison", () => {
    expect(runTypeLabel("solver_comparison")).toBe("Solver Comparison");
  });

  it("returns human-readable label for prediction_comparison", () => {
    expect(runTypeLabel("prediction_comparison")).toBe("Prediction Comparison");
  });

  it("returns raw string for unknown run type", () => {
    expect(runTypeLabel("unknown_type")).toBe("unknown_type");
  });
});

describe("fmtDate", () => {
  it("returns em-dash for null", () => {
    expect(fmtDate(null)).toBe("—");
  });

  it("returns a non-empty string for a valid ISO date", () => {
    const result = fmtDate("2024-01-15T10:30:00Z");
    expect(result).toBeTruthy();
    expect(result).not.toBe("—");
  });

  it("includes year for a valid date", () => {
    const result = fmtDate("2024-06-01T00:00:00Z");
    expect(result).toContain("2024");
  });
});

describe("getErrorMessage", () => {
  it("shows validation reasons without input values or context", () => {
    const message = getErrorMessage({response: {data: {detail: [{loc: ["body"], msg: "Value error, Minimum assets cannot exceed maximum assets", input: {password: "owned-secret-sentinel"}, ctx: {error: "private-context-sentinel"}}]}}}, "Failed to save");
    expect(message).toBe("Minimum assets cannot exceed maximum assets");
    expect(message).not.toContain("sentinel");
  });
  it("identifies fields and one-based configuration positions and deduplicates", () => {
    const detail = [
      {loc: ["body", "initial_cash"], msg: "Input should be a finite number"},
      {loc: ["body", "parameters", "strategy_variants", 0, "max_assets"], msg: "Input should be greater than or equal to 1"},
      {loc: ["body", "initial_cash"], msg: "Input should be a finite number"},
    ];
    expect(getErrorMessage({response: {data: {detail}}}, "Failed")).toBe("Initial cash: Input should be a finite number; Configuration / 1 / max assets: Input should be greater than or equal to 1");
  });
  it("ignores malformed entries and retains normal fallback messages", () => {
    expect(getErrorMessage({response: {data: {detail: [null, "invalid", {msg: 7}, {msg: ""}, {msg: "Required field", loc: {bad: true}}]}}}, "Failed")).toBe("Required field");
    expect(getErrorMessage({response: {data: {detail: [null, {msg: 7}], message: "Please try again"}}}, "Failed")).toBe("Please try again");
    expect(getErrorMessage({response: {data: {detail: [null]}}}, "Failed")).toBe("Failed");
  });
  it.each([
    ["Plain API error", "Plain API error"],
    [{detail: "Expired session"}, "Expired session"],
    [{message: "Try again"}, "Try again"],
    [{error: "Unavailable"}, "Unavailable"],
  ])("preserves existing non-validation API errors", (data, expected) => {
    expect(getErrorMessage({response: {data}}, "Failed")).toBe(expected);
  });
  it("preserves network errors and fallback behavior", () => {
    expect(getErrorMessage(new Error("Offline"), "Failed")).toBe("Offline");
    expect(getErrorMessage(null, "Failed")).toBe("Failed");
  });
});
