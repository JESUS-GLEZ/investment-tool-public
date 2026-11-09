import type { ExperimentData } from "@/lib/experimentData";
import type { LocalDataInfo } from "@/types/api";

export default function ExperimentInputSummary({setup, localInfo}: {setup: ExperimentData; localInfo: LocalDataInfo | null}) {
  return <dl className="grid gap-x-6 gap-y-2 text-sm sm:grid-cols-2">
    <div className="min-w-0"><dt className="font-medium">Input data</dt><dd className="break-words">{setup.data_source === "local" ? setup.input_file : "Alpaca"} · {setup.frequency}</dd>
      {setup.data_source === "local" && localInfo && <dd>{localInfo.stock_count} stocks · {localInfo.crypto_count} crypto stored</dd>}
    </div>
    <div className="min-w-0"><dt className="font-medium">Universe and lifecycle rules</dt><dd className="break-words">{setup.asset_config_file}{setup.data_source === "local" ? localInfo?.asset_config_file ? " (recorded in dataset)" : " (explicit configuration; dataset has no recorded universe)" : ""}</dd></div>
    <div><dt className="font-medium">Historical window</dt><dd>{setup.historical_start_date} → {setup.historical_end_date} · complete periods only</dd></div>
    <div><dt className="font-medium">Portfolio assumptions</dt><dd>Initial cash {setup.initial_cash} · transaction cost {setup.transaction_cost_rate} · risk-free rate {setup.risk_free_rate}</dd></div>
    <div><dt className="font-medium">Validation / output</dt><dd>{setup.validation_level} / {setup.output_mode}</dd></div>
  </dl>;
}
