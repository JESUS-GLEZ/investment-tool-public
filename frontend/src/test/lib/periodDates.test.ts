import { describe, expect, it } from "vitest";
import { createInitialExperimentData, datasetPeriodWindow } from "@/lib/experimentData";
import { inputDate, lastCompleteDate, periodInputValue, todayDate } from "@/lib/periodDates";

describe("calendar period controls", () => {
  it("initializes on form opening across calendar boundaries", () => {
    expect(createInitialExperimentData(new Date("2026-10-27T12:00:00Z")).historical_end_date).toBe("2026-09-30");
    expect(createInitialExperimentData(new Date("2027-01-01T00:00:00Z")).historical_end_date).toBe("2026-12-31");
    expect(lastCompleteDate("monthly", new Date("2024-03-01T00:00:00Z"))).toBe("2024-02-29");
  });
  it("round-trips ISO weeks across week-year and DST boundaries", () => {
    for (const [week, start, end] of [["2020-W53", "2020-12-28", "2021-01-03"], ["2025-W01", "2024-12-30", "2025-01-05"], ["2024-W13", "2024-03-25", "2024-03-31"]]) {
      expect(inputDate(week, "weekly")).toBe(start);
      expect(inputDate(week, "weekly", true)).toBe(end);
      expect(periodInputValue(start, "weekly")).toBe(week);
      expect(periodInputValue(end, "weekly")).toBe(week);
    }
    expect(lastCompleteDate("weekly", new Date("2026-10-25T20:00:00Z"))).toBe("2026-10-18");
  });
});

it("rejects invalid native-control fallback text without rolling weeks or dates", () => {
  expect(inputDate("2021-W53","weekly")).toBe("");
  expect(inputDate("not-a-week","weekly")).toBe("");
  expect(inputDate("2024-13","monthly")).toBe("");
  expect(inputDate("2024-02-30","daily")).toBe("");
  expect(inputDate("2020-W53","weekly",true)).toBe("2021-01-03");
});

it("never lets the dataset-range shortcut include an unfinished month or week", () => {
  const info = {file:"current.pkl",time_contract_version:1,start_date:"2020-01-01",end_date:"2099-12-31",stocks:[],crypto:[],stock_count:0,crypto_count:0};
  expect(datasetPeriodWindow(info,"monthly")?.end).toBe(lastCompleteDate("monthly"));
  expect(datasetPeriodWindow(info,"weekly")?.end).toBe(lastCompleteDate("weekly"));
  expect(datasetPeriodWindow(info,"daily")?.end).toBe(todayDate());
  expect(datasetPeriodWindow({...info,start_date:"2099-01-01"},"monthly")).toBeNull();
});
