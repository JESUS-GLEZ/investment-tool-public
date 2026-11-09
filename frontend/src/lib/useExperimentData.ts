import { useRef, useState } from "react";
import { createInitialExperimentData, type ExperimentData } from "./experimentData";
import { lastCompleteDate, periodStart } from "./periodDates";
import type { Frequency } from "@/types/api";

/** Refresh only untouched defaults on frequency changes; never rewrite saved windows. */
export function useExperimentData() {
  const [setup, setSetup] = useState<ExperimentData>(() => createInitialExperimentData());
  const automatic = useRef({start: true, end: true});
  const update = <K extends keyof ExperimentData>(key: K, value: ExperimentData[K]) => {
    if (key === "historical_start_date") automatic.current.start = false;
    if (key === "historical_end_date") automatic.current.end = false;
    const defaults = {...automatic.current};
    setSetup(previous => ({...previous, [key]: value, ...(key === "frequency" && previous.data_source === "alpaca" ? {
      historical_start_date: defaults.start ? periodStart(createInitialExperimentData().historical_start_date, value as Frequency) : previous.historical_start_date,
      historical_end_date: defaults.end ? lastCompleteDate(value as Frequency) : previous.historical_end_date,
    } : {})}));
  };
  const load = (saved: ExperimentData) => {
    automatic.current = {start: false, end: false};
    setSetup(saved);
  };
  return [setup, update, load] as const;
}
