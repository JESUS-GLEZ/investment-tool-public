import ParameterHelp from "./ParameterHelp";
import { useId } from "react";
import { portfolioNumberErrors, type PortfolioNumberKey, type PortfolioNumbers } from "@/lib/portfolioNumbers";

/** Shared monetary fields retain the existing form layout and numeric bounds. */
export default function PortfolioNumberFields({ values, onChange, className, percentExamples = false, holdings = false }: {
  values: PortfolioNumbers;
  onChange: (key: PortfolioNumberKey, value: number) => void;
  className: string;
  percentExamples?: boolean;
  holdings?: boolean;
}) {
  const id = useId();
  const errors = portfolioNumberErrors(values, holdings);
  const field = (key: PortfolioNumberKey, label: string, hint: string | null, min: number, max: number | undefined, step: number) => (
    <div>
      <div className="flex items-center gap-1"><label htmlFor={`${id}-${key}`} className="block text-sm font-medium text-gray-700">
        {label}{hint && <> <span className="text-gray-400">{hint}</span></>}
      </label><ParameterHelp name={key} label={label} /></div>
      <input id={`${id}-${key}`} type="number" required min={min} max={max} step={step}
        value={Number.isFinite(values[key]) ? values[key] : ""}
        onChange={(event) => onChange(key, event.currentTarget.valueAsNumber)}
        aria-invalid={!!errors[key]} aria-describedby={errors[key] ? `${id}-${key}-error` : undefined}
        className={className} />

      {errors[key] && <p id={`${id}-${key}-error`} className="mt-1 text-xs text-red-700">{errors[key]}</p>}
    </div>
  );
  return <>
    <div className="grid grid-cols-2 gap-4">
      {field("initial_cash", holdings ? "Available cash (USD)" : "Initial Cash ($)", null, holdings ? 0 : 1, undefined, holdings ? 0.01 : 1)}
      {field("transaction_cost_rate", "Transaction Cost", percentExamples ? "(e.g. 0.001 = 0.1%)" : "(e.g. 0.001)", 0, 0.1, 0.001)}
    </div>
    <p className="text-xs text-gray-500">Transaction costs apply to buys and sells. A nonzero rate also adjusts optimization return objectives using estimated turnover at decision-time prices.</p>
    {field("risk_free_rate", "Risk-free Rate", percentExamples ? "(annual, e.g. 0.04 = 4%)" : "(annual, e.g. 0.04)", 0, 1, 0.01)}
  </>;
}
