import PeriodDateField from "./PeriodDateField";
import { datasetPeriodWindow } from "@/lib/experimentData";
import HistoricalWindowFields from "./HistoricalWindowFields";
import ParameterHelp from "./ParameterHelp";
import { useId, useState } from "react";
import type { DataSource, Frequency, ValidationLevel, OutputMode, LocalDataInfo } from "@/types/api";
import OutputModeSelect from "./OutputModeSelect";
import PortfolioNumberFields from "./PortfolioNumberFields";
import LocalDataPanel from "./LocalDataPanel";
import AssetUniverseSelect from "./AssetUniverseSelect";
const DATA_SOURCES: DataSource[] = ["alpaca", "local"];
const FREQUENCIES: Frequency[] = ["daily", "weekly", "monthly"];
const VALIDATION_LEVELS: ValidationLevel[] = ["strict", "warn", "skip"];
const selectCls = "mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none";
const inputCls = selectCls;
function SectionHeader({title}: {title: string}) { return <h2 className="pt-2 text-xs font-semibold uppercase tracking-wide text-gray-500">{title}</h2>; }
/** Shared data/date/portfolio fields for all comparison types. */
export default function ExperimentDataFields({
  startDate, setStartDate,
  endDate, setEndDate,
  initialCash, setInitialCash,
  txCost, setTxCost,
  riskFree, setRiskFree,
  dataSource, setDataSource,
  assetConfigFile, setAssetConfigFile,
  frequency, setFrequency,
  validationLevel, setValidationLevel,
  outputMode, setOutputMode,
  selectedFile, onFileChange,
  onLocalDataInfoChange,
  onAnyChange,
  holdings = false,
  backtestStart,
}: {
  startDate: string;
              setStartDate: (v: string) => void;
  endDate: string;
              setEndDate: (v: string) => void;
  initialCash: number;
              setInitialCash: (v: number) => void;
  txCost: number;
              setTxCost: (v: number) => void;
  riskFree: number;
              setRiskFree: (v: number) => void;
  dataSource: DataSource;
              setDataSource: (v: DataSource) => void;
  assetConfigFile: string;
              setAssetConfigFile: (v: string) => void;
  frequency: Frequency;
              setFrequency: (v: Frequency) => void;
  validationLevel: ValidationLevel;
              setValidationLevel: (v: ValidationLevel) => void;
  outputMode: OutputMode; setOutputMode: (v: OutputMode) => void;
  selectedFile: string | null; onFileChange: (f: string | null) => void;
  onLocalDataInfoChange?: (info: LocalDataInfo | null) => void;
  onAnyChange?: () => void;
  holdings?: boolean;
  backtestStart?: {value: string; min?: string; max?: string; onChange: (value: string) => void};
}) {
  const commonId = useId();
  const isLocal = dataSource === "local";
  const [localInfo, setLocalInfo] = useState<LocalDataInfo | null>(null);
  const linkedUniverse = localInfo?.file === selectedFile ? localInfo.asset_config_file : null;
  const datasetWindow = isLocal && localInfo?.file === selectedFile ? datasetPeriodWindow(localInfo, frequency) : null;
  const lockFrequency = isLocal && (!localInfo || !!localInfo.frequency);

  const handleDataSourceChange = (value: DataSource) => {
    onAnyChange?.();
    setLocalInfo(null);
    onLocalDataInfoChange?.(null);
    setDataSource(value);
  };

  const handleFrequencyChange = (value: Frequency) => {
    onAnyChange?.();
    setFrequency(value);
  };

  const handleValidationLevelChange = (value: ValidationLevel) => {
    onAnyChange?.();
    setValidationLevel(value);
  };

  return (
    <>
      <OutputModeSelect value={outputMode} onChange={(value) => { onAnyChange?.(); setOutputMode(value); }} />
      <SectionHeader title="Data" />
      <div className="grid grid-cols-2 gap-4">
        <div>
          <div className="flex items-center gap-1"><label htmlFor={`${commonId}-data-source-1`} className="block text-sm font-medium text-gray-700">Data Source</label><ParameterHelp name="data_source" label="Data Source" /></div>
          <select id={`${commonId}-data-source-1`}
            aria-label="Data Source"
            value={dataSource}
            onChange={(e) => handleDataSourceChange(e.target.value as DataSource)}
            className={selectCls}
          >
            {DATA_SOURCES.map((d) => <option key={d} value={d}>{d}</option>)}
          </select>

        </div>
        <div>
          <div className="flex items-center gap-1"><label htmlFor={`${commonId}-frequency-1`} className="block text-sm font-medium text-gray-700">Frequency</label><ParameterHelp name="frequency" label="Frequency" /></div>
          <select id={`${commonId}-frequency-1`}
            value={frequency}
            onChange={(e) => handleFrequencyChange(e.target.value as Frequency)}
            disabled={lockFrequency}
            className={lockFrequency ? `${selectCls} cursor-not-allowed opacity-60` : selectCls}
          >
            {FREQUENCIES.map((f) => <option key={f} value={f}>{f}</option>)}
          </select>

          {isLocal && (
            <p className="mt-1 text-xs text-gray-400">{lockFrequency ? "set by the selected dataset" : "Frequency was not recorded. Confirm it matches the dataset."}</p>
          )}
        </div>
      </div>
      <div>
        <div className="flex items-center gap-1"><label htmlFor={`${commonId}-validation-level-1`} className="block text-sm font-medium text-gray-700">Validation Level</label><ParameterHelp name="validation_level" label="Validation Level" /></div>
        <select id={`${commonId}-validation-level-1`}
          value={validationLevel}
          onChange={(e) => handleValidationLevelChange(e.target.value as ValidationLevel)}
          className={selectCls}
        >
          {VALIDATION_LEVELS.map((v) => <option key={v} value={v}>{v}</option>)}
        </select>

      </div>

      {!isLocal && <AssetUniverseSelect value={assetConfigFile} onChange={(file) => {
        onAnyChange?.();
        setAssetConfigFile(file);
      }} />}
      {isLocal && (
        <>
          <SectionHeader title="Local Dataset" />
          <LocalDataPanel
            selectedFile={selectedFile}
            onFileChange={(file) => {
              onAnyChange?.();
              setLocalInfo(null);
              onLocalDataInfoChange?.(null);
              onFileChange(file);
            }}
            onFrequencyChange={(value) => {
              onAnyChange?.();
              setFrequency(value);
            }}
            onInfoChange={(info) => {
              onAnyChange?.();
              setLocalInfo(info);
              if (info?.frequency) setFrequency(info.frequency);
              onLocalDataInfoChange?.(info);
            }}
          />
          {localInfo?.start_date && localInfo.end_date && <button type="button" className="rounded border border-gray-300 px-3 py-1.5 text-sm" onClick={() => {
            if (!datasetWindow) return;
            onAnyChange?.(); setStartDate(datasetWindow.start); setEndDate(datasetWindow.end);
          }} disabled={!datasetWindow} aria-describedby={!datasetWindow ? `${commonId}-dataset-range-error` : undefined}>Use full dataset date range</button>}
          {localInfo && !datasetWindow && <p id={`${commonId}-dataset-range-error`} className="text-xs text-amber-800">{localInfo.time_contract_version === null ? "Upgrade the dataset timestamps before selecting its range." : "No complete periods are available in the recorded dataset range."}</p>}
          {linkedUniverse ? <div className="rounded border border-gray-200 bg-gray-50 p-3 text-sm">
            <p className="font-medium">Dataset universe</p>
            <p>{localInfo?.asset_config_name ?? linkedUniverse}</p>
            <p className="break-all text-xs text-gray-500">{linkedUniverse}</p>
            <p className="mt-1 text-xs text-gray-500">Assets, eligibility and lifecycle rules use the dataset’s recorded universe. No additional assets are downloaded.</p>
            {localInfo?.asset_config_available === false && <p role="alert" className="mt-2 text-red-700">The recorded universe is unavailable. Restore its JSON file or choose another dataset before launching.</p>}
          </div> : localInfo && <div className="space-y-2 rounded border border-amber-200 bg-amber-50 p-3">
            <p className="text-sm text-amber-900">This dataset has no recorded universe. Choose its matching universe and lifecycle rules. Only matching assets already stored in the dataset are used; no missing assets are downloaded.</p>
            <AssetUniverseSelect value={assetConfigFile} onChange={file => {onAnyChange?.(); setAssetConfigFile(file);}} />
          </div>}
        </>
      )}
      <SectionHeader title="Historical Data Window" />
      <HistoricalWindowFields bounds={datasetWindow} start={startDate} end={endDate} frequency={frequency}
        onStart={value => {onAnyChange?.(); setStartDate(value);}}
        onEnd={value => {onAnyChange?.(); setEndDate(value);}} />

      {backtestStart && <div className="space-y-1">
        <PeriodDateField id={`${commonId}-trading-start`} label="Backtest start date (optional)" helpName="backtest_start_date" frequency={frequency} optional {...backtestStart} />
        <p className="text-xs text-gray-500">Earlier history remains available for estimation. Leave blank for automatic warm-up; choose a period to start simulated trading explicitly.</p>
      </div>}
      <SectionHeader title="Portfolio" />
      <PortfolioNumberFields holdings={holdings} values={{initial_cash: initialCash, transaction_cost_rate: txCost, risk_free_rate: riskFree}}
        onChange={(key, value) => {
          onAnyChange?.();
          const setters = {initial_cash: setInitialCash, transaction_cost_rate: setTxCost, risk_free_rate: setRiskFree};
          setters[key](value);
        }} className={inputCls} />
    </>
  );
}
