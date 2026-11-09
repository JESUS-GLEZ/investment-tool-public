import { useState } from "react";
import type { Frequency } from "@/types/api";
import { inputDate, periodFieldError, periodInputType, periodInputValue } from "@/lib/periodDates";
import ParameterHelp from "./ParameterHelp";

/** Native controls follow browser locale; API dates remain inclusive ISO boundaries. */
export default function PeriodDateField({id, label, value, frequency, onChange, end = false, min, max, optional = false, helpName}: {
  id: string; label: string; value: string; frequency: Frequency; onChange: (value: string) => void;
  end?: boolean; min?: string; max?: string; optional?: boolean; helpName?: string;
}) {
  const period = periodInputValue(value, frequency);
  const canonical = !value || inputDate(period, frequency, end) === value;
  const [draft, setDraft] = useState({value, frequency, text: canonical ? period : ""});
  const text = draft.value === value && draft.frequency === frequency ? draft.text : canonical ? period : "";
  const incomplete = !!text && !inputDate(text, frequency, end);
  const error = incomplete ? `Choose a valid ${frequency === "weekly" ? "ISO week" : frequency === "monthly" ? "month" : "date"}.`
    : periodFieldError(value, frequency, {end, min, max, optional});
  return <div className="space-y-1">
    <div className="flex items-center gap-1"><label htmlFor={id} className="text-sm font-medium">{label}</label>{helpName && <ParameterHelp name={helpName} label={label} />}</div>
    <input id={id} placeholder={frequency === "weekly" ? "YYYY-Www" : frequency === "monthly" ? "YYYY-MM" : "YYYY-MM-DD"} type={periodInputType(frequency)} required={!optional || !!text}
      ref={node => {node?.setCustomValidity(error ?? "");}}
      className="block min-w-0 w-full rounded border border-gray-300 bg-white p-2 text-sm"
      value={text} min={periodInputValue(min ?? "", frequency)} max={periodInputValue(max ?? "", frequency)}
      aria-invalid={!!error} aria-describedby={error ? `${id}-error` : undefined}
      onChange={e => {
        const text = e.target.value;
        const date = inputDate(text, frequency, end);
        setDraft({value:date,frequency,text}); onChange(date);
      }} />
    {error && <p id={`${id}-error`} className="text-xs text-amber-800">{error}</p>}
    {optional && (value || text) && <button type="button" className="text-xs text-blue-700 underline" onClick={() => {setDraft({value:"",frequency,text:""}); onChange("");}}>Use automatic warm-up</button>}
  </div>;
}
