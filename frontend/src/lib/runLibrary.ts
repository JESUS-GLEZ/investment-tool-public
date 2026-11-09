import type { RunSummary } from "@/types/api";
import { configurationSettings, recordedConfigurations } from "./runPresentation";
import { datasetTitle } from "./localDatasets";

export function libraryRunType(run: RunSummary): string {
  switch (run.run_type) {
    case "execution":
      return run.request_payload?.execution_mode === "investing" ? "Investing recommendation"
        : run.request_payload?.execution_mode === "backtesting" ? "Backtest" : "Execution";
    case "strategy_comparison": return "Strategy comparison";
    case "solver_comparison": return "Solver benchmark";
    case "prediction_comparison": return "Prediction evaluation";
  }
}

// Keep saved calendar dates intact. Never shift them through the browser timezone.
function savedDate(value: unknown): string | undefined {
  if (typeof value !== "string") return;
  const date = value.slice(0, 10);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) return;
  const parsed = new Date(`${date}T00:00:00Z`);
  return Number.isFinite(parsed.valueOf()) && parsed.toISOString().slice(0, 10) === date ? date : undefined;
}

export function libraryPeriod(run: RunSummary): { label: string; value: string } {
  const p = run.request_payload ?? {};
  const start = savedDate(p.historical_start_date);
  const end = savedDate(p.historical_end_date);
  if (run.run_type === "execution" && p.execution_mode === "investing") {
    return { label: "Requested data cutoff", value: end ?? "Not recorded" };
  }
  if (run.run_type === "strategy_comparison" || run.run_type === "execution" && p.execution_mode === "backtesting") {
    return { label: "Requested backtest", value: `${savedDate(p.backtest_start_date) ?? (Object.hasOwn(p, "backtest_start_date") && p.backtest_start_date == null ? "Automatic start" : "Start not recorded")} → ${end ?? "Not recorded"}` };
  }
  return { label: "Historical window", value: `${start ?? "Not recorded"} → ${end ?? "Not recorded"}` };
}

export function libraryUniverse(run: RunSummary): { label: string; name: string; file?: string } {
  const p = run.request_payload ?? {};
  // A local snapshot owns its universe. The request's asset_config_file may be ignored.
  const local = p.data_source === "local";
  const file = local ? p.input_file : p.asset_config_file;
  return {
    label: local ? "Local dataset" : p.data_source === "alpaca" ? "Alpaca universe" : "Universe",
    name: typeof file === "string" && file.trim()
      ? local ? datasetTitle(file) : file.replace(/\.json$/i, "").replace(/_/g, " ")
      : "Not recorded",
    file: typeof file === "string" ? file : undefined,
  };
}

/** Search the saved experiment, including configurations hidden behind its summary. */
export function searchLibrary(runs: RunSummary[], query: string, type: string, status: string, sort: string): RunSummary[] {
  const terms = query.toLowerCase().trim().split(/\s+/).filter(Boolean);
  return runs.filter(run => {
    if (type !== "all" && run.run_type !== type || status !== "all" && run.status !== status) return false;
    const universe = libraryUniverse(run);
    const haystack = [run.id, libraryRunType(run), run.status, libraryPeriod(run).value,
    universe.label, universe.name, universe.file,
    ...recordedConfigurations(run).flatMap(config => [config.name, config.method, configurationSettings(config, true)]),
    ].join(" ").toLowerCase();
    return terms.every(term => haystack.includes(term));
  }).sort((a, b) => {
    const time = (iso: string) => Number.isFinite(Date.parse(iso)) ? Date.parse(iso) : 0;
    const difference = time(a.created_at) - time(b.created_at) || a.id.localeCompare(b.id);
    return sort === "oldest" ? difference : -difference;
  });
}
