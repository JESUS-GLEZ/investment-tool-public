import type { ExecutionRequest, LocalDataInfo } from "@/types/api";
import { containedPeriodWindow, historicalWindowError, lastCompleteDate, todayDate } from "./periodDates";
import { DEFAULT_ASSET_CONFIG_FILE } from "@/components/ui/AssetUniverseSelect";

export type ExperimentData = Required<Pick<ExecutionRequest,
  "data_source" | "frequency" | "validation_level" | "output_mode" | "asset_config_file" |
  "historical_start_date" | "historical_end_date" | "initial_cash" | "transaction_cost_rate" | "risk_free_rate"
>> & {input_file: string | null; backtest_start_date: string | null};

export function createInitialExperimentData(now = new Date()): ExperimentData { return {
  data_source: "alpaca", frequency: "monthly", validation_level: "warn", output_mode: "standard",
  asset_config_file: DEFAULT_ASSET_CONFIG_FILE, historical_start_date: "2020-01-01", historical_end_date: lastCompleteDate("monthly", now),
  backtest_start_date: null, initial_cash: 1000, transaction_cost_rate: 0, risk_free_rate: 0, input_file: null,
}; }

// Compatibility for consumers of the non-date defaults. Forms initialize lazily.
export const initialExperimentData = createInitialExperimentData();

export function localDatasetError(file: string | null, info: LocalDataInfo | null): string | undefined {
  if (!file) return "Select a local dataset before launching.";
  if (!info || info.file !== file) return "Waiting for local dataset details. Try again in a moment.";
  if (info.time_contract_version === null) return "This dataset needs a timestamp upgrade. Fetch a new dataset or upgrade it before launching.";
  if (info.asset_config_available === false) return "The dataset’s recorded universe is unavailable. Restore it or choose another dataset.";
}

/** The file’s recorded inputs take precedence, including after same-file preset loads. */
export function effectiveExperimentData(setup: ExperimentData, info: LocalDataInfo | null): ExperimentData {
  const local = setup.data_source === "local" && info?.file === setup.input_file ? info : null;
  return {...setup, input_file: setup.data_source === "local" ? setup.input_file : null,
    asset_config_file: local?.asset_config_file ?? setup.asset_config_file,
    frequency: local?.frequency ?? setup.frequency};
}

/** Overall coverage bounds the UI; individual asset eligibility remains the engine's responsibility. */
export function datasetPeriodWindow(info: LocalDataInfo | null, frequency: ExecutionRequest["frequency"]) {
  if (!info?.start_date || !info.end_date || info.time_contract_version === null) return null;
  const effective = frequency ?? "monthly";
  const completed = effective === "daily" ? todayDate() : lastCompleteDate(effective);
  return containedPeriodWindow(info.start_date, info.end_date < completed ? info.end_date : completed, effective);
}

export function experimentDateError(setup: ExperimentData, info: LocalDataInfo | null): string | undefined {
  const local = setup.data_source === "local";
  const issue = local ? localDatasetError(setup.input_file, info) : undefined;
  if (!issue && local && info?.start_date && info.end_date && !datasetPeriodWindow(info, setup.frequency)) return "This dataset contains no complete periods at its frequency.";
  return issue ?? historicalWindowError(setup.historical_start_date, setup.historical_end_date, setup.frequency,
    local ? datasetPeriodWindow(info, setup.frequency) : null);
}
