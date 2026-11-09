import { useId } from "react";
import type { Frequency } from "@/types/api";
import { historicalWindowError, lastCompleteDate, todayDate } from "@/lib/periodDates";
import PeriodDateField from "./PeriodDateField";

export default function HistoricalWindowFields({start, end, frequency, onStart, onEnd, bounds}: {
  bounds?: {start: string; end: string} | null;
  start: string; end: string; frequency: Frequency; onStart: (value: string) => void; onEnd: (value: string) => void;
}) {
  const id = useId();
  const error = historicalWindowError(start, end, frequency, bounds);
  const completed = frequency === "daily" ? todayDate() : lastCompleteDate(frequency);
  const max = bounds && bounds.end < completed ? bounds.end : completed;
  const min = bounds?.start;
  return <div className="space-y-3">
    <p className="text-xs text-gray-500">{frequency === "monthly" ? "Choose the first and last complete month to include." : frequency === "weekly" ? "Choose complete ISO weeks, Monday through Sunday." : "Choose any calendar dates."} Availability is checked by the backend using each asset’s market calendar.</p>
    {bounds && <p className="text-xs text-gray-500">Dataset coverage: {bounds.start} → {bounds.end}. Only complete periods within this range are selectable.</p>}
    <div className="grid gap-4 sm:grid-cols-2">
      <PeriodDateField id={`${id}-start`} label="Start Date" helpName="historical_start_date" value={start} frequency={frequency} min={min} max={end && end < max ? end : max} onChange={onStart} />
      <PeriodDateField id={`${id}-end`} label="End Date" helpName="historical_end_date" value={end} frequency={frequency} min={start && (!min || start > min) ? start : min} max={max} end onChange={onEnd} />
    </div>
    {!error && <p className="text-xs text-gray-500">Requested window: {start || "—"} → {end || "—"} (inclusive market-local dates).</p>}
    {error && <p role="status" className="text-sm text-amber-800">{error} Select the periods again if you changed frequency or loaded incompatible dates.</p>}
  </div>;
}
