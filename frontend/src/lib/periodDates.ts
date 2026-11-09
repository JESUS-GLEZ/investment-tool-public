import type { Frequency } from "@/types/api";

// Calendar arithmetic uses UTC solely to avoid browser DST shifting date labels.
const day = 86_400_000;
const date = (value: string) => new Date(`${value}T00:00:00Z`);
const iso = (value: Date) => value.toISOString().slice(0, 10);
export const todayDate = (now = new Date()) => iso(now);
export function periodStart(value: string, frequency: Frequency): string {
  const d = date(value);
  if (frequency === "monthly") d.setUTCDate(1);
  if (frequency === "weekly") d.setUTCDate(d.getUTCDate() - (d.getUTCDay() + 6) % 7);
  return iso(d);
}
export function periodEnd(value: string, frequency: Frequency): string {
  const d = date(periodStart(value, frequency));
  if (frequency === "monthly") d.setUTCMonth(d.getUTCMonth() + 1);
  else d.setUTCDate(d.getUTCDate() + (frequency === "weekly" ? 7 : 1));
  return iso(new Date(d.getTime() - day));
}
export function lastCompleteDate(frequency: Frequency, now = new Date()): string {
  return iso(new Date(date(periodStart(todayDate(now), frequency)).getTime() - day));
}
export function periodInputType(frequency: Frequency): "date" | "week" | "month" {
  return frequency === "monthly" ? "month" : frequency === "weekly" ? "week" : "date";
}
export function periodInputValue(value: string, frequency: Frequency): string {
  if (!value) return "";
  if (frequency === "monthly") return value.slice(0, 7);
  if (frequency === "daily") return value;
  const thursday = date(periodStart(value, frequency));
  thursday.setUTCDate(thursday.getUTCDate() + 3);
  const year = thursday.getUTCFullYear();
  const week = Math.floor((thursday.getTime() - date(`${year}-01-01`).getTime()) / (7 * day)) + 1;
  return `${String(year).padStart(4, "0")}-W${String(week).padStart(2, "0")}`;
}
export function inputDate(value: string, frequency: Frequency, end = false): string {
  if (!value) return "";
  // Also validate text fallback controls in browsers without native month/week support.
  const pattern = frequency === "monthly" ? /^\d{4}-(0[1-9]|1[0-2])$/ : frequency === "weekly" ? /^\d{4}-W(0[1-9]|[1-4]\d|5[0-3])$/ : /^\d{4}-\d{2}-\d{2}$/;
  if (!pattern.test(value) || Number(value.slice(0,4)) < 1) return "";
  let label = value;
  if (frequency === "monthly") label += "-01";
  if (frequency === "weekly") {
    const [year, week] = value.split("-W");
    const first = date(periodStart(`${year}-01-04`, "weekly"));
    first.setUTCDate(first.getUTCDate() + (Number(week) - 1) * 7);
    label = iso(first);
  }
  if (!Number.isFinite(date(label).getTime()) || iso(date(label)) !== label || periodInputValue(label, frequency) !== value) return "";
  const result = end ? periodEnd(label, frequency) : label;
  return /^\d{4}-\d{2}-\d{2}$/.test(result) ? result : "";
}

/** Whole calendar periods contained in an exact inclusive window. */
export function containedPeriodWindow(start: string, end: string, frequency: Frequency): {start: string; end: string} | null {
  if (!start || !end || start > end) return null;
  const first = start === periodStart(start, frequency) ? start : iso(new Date(date(periodEnd(start, frequency)).getTime() + day));
  const last = end === periodEnd(end, frequency) ? end : iso(new Date(date(periodStart(end, frequency)).getTime() - day));
  return first <= last ? {start: first, end: last} : null;
}

/** Forms accept only the period type selected by frequency. */
export function historicalWindowError(start: string, end: string, frequency: Frequency, bounds?: {start: string; end: string} | null): string | undefined {
  if (!start || !end) return "Choose both ends of the historical window.";
  if (start > end) return "Historical start must be on or before the end.";
  if (frequency !== "daily" && (start !== periodStart(start, frequency) || end !== periodEnd(end, frequency))) return `Choose whole ${frequency === "monthly" ? "months" : "ISO weeks"} for this frequency.`;
  if (bounds && (start < bounds.start || end > bounds.end)) return `Choose dates within dataset coverage: ${bounds.start} to ${bounds.end}.`;
  if (end > (frequency === "daily" ? todayDate() : lastCompleteDate(frequency))) return "Choose a completed historical period.";
}

/** Field-level guidance complements native browser validity without changing date semantics. */
export function periodFieldError(value: string, frequency: Frequency, {end = false, min, max, optional = false}: {end?: boolean; min?: string; max?: string; optional?: boolean} = {}): string | undefined {
  const kind = frequency === "monthly" ? "month" : frequency === "weekly" ? "ISO week" : "date";
  if (!value) return optional ? undefined : `Choose a valid ${kind}.`;
  if (inputDate(periodInputValue(value, frequency), frequency, end) !== value) return `Choose a complete ${kind} for this frequency.`;
  if (min && value < min) return `Choose a ${kind} on or after ${min}.`;
  if (max && value > max) return `Choose a ${kind} on or before ${max}.`;
}
