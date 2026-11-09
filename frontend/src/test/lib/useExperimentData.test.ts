import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { useExperimentData } from "@/lib/useExperimentData";
afterEach(() => {cleanup(); vi.useRealTimers();});
it("updates untouched defaults by frequency but preserves explicit dates", () => {
  vi.useFakeTimers(); vi.setSystemTime(new Date("2026-10-27T12:00:00Z"));
  const {result} = renderHook(useExperimentData);
  expect(result.current[0].historical_end_date).toBe("2026-09-30");
  act(() => result.current[1]("frequency", "weekly"));
  expect(result.current[0]).toMatchObject({historical_start_date: "2019-12-30", historical_end_date: "2026-10-25"});
  act(() => result.current[1]("historical_end_date", "2024-08-01"));
  act(() => result.current[1]("frequency", "daily"));
  expect(result.current[0].historical_end_date).toBe("2024-08-01");
});
it("preserves loaded presets and local-dataset windows across frequency changes", () => {
  const {result} = renderHook(useExperimentData);
  act(() => result.current[2]({...result.current[0], historical_start_date: "2024-01-15", historical_end_date: "2024-08-01"}));
  act(() => result.current[1]("frequency", "weekly"));
  expect(result.current[0]).toMatchObject({historical_start_date: "2024-01-15", historical_end_date: "2024-08-01"});
  const {result: local} = renderHook(useExperimentData);
  const before = local.current[0].historical_end_date;
  act(() => local.current[1]("data_source", "local"));
  act(() => local.current[1]("frequency", "daily"));
  expect(local.current[0].historical_end_date).toBe(before);
});
