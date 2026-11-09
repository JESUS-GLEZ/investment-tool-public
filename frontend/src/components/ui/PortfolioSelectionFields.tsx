import ParameterHelp from "./ParameterHelp";
import type { PortfolioSelection } from "@/types/api";

const cls = "mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 text-sm";

export default function PortfolioSelectionFields({ prefix, method, fraction, increase, onMethod, onFraction, onIncrease }: {
  prefix: string; method: PortfolioSelection; fraction: number; increase: number;
  onMethod: (value: PortfolioSelection) => void; onFraction: (value: number) => void; onIncrease: (value: number) => void;
}) {
  return <div className="space-y-3">
    <div>
      <div className="flex items-center gap-1"><label htmlFor={`${prefix}-selection`} className="block text-sm font-medium text-gray-700">Portfolio selection</label><ParameterHelp name="selection" label="Portfolio selection" /></div>
      <select id={`${prefix}-selection`} className={cls} value={method} onChange={e => onMethod(e.target.value as PortfolioSelection)}>
        <option value="sharpe">Maximum Sharpe ratio</option>
        <option value="return_with_risk_budget">Higher return within a volatility budget</option>
        <option value="return_target">Minimum risk at a return target</option>
      </select>

    </div>
    {method === "return_with_risk_budget" && <div>
      <div className="flex items-center gap-1"><label htmlFor={`${prefix}-volatility-increase`} className="block text-sm font-medium text-gray-700">Maximum increase in estimated volatility (%)</label><ParameterHelp name="volatility_increase" label="Maximum increase in estimated volatility" /></div>
      <input id={`${prefix}-volatility-increase`} className={cls} type="number" min={0} max={100} required step="any" value={Number.isFinite(increase) ? increase * 100 : ""} onChange={e => onIncrease(e.target.valueAsNumber / 100)} />

      <p className="text-xs text-gray-500 mt-1">Relative to the Sharpe portfolio from the same front. This can still select low-return portfolios; it is not a limit on actual losses.</p>
    </div>}
    {method === "return_target" && <div>
      <div className="flex items-center gap-1"><label htmlFor={`${prefix}-return-target`} className="block text-sm font-medium text-gray-700">Position in feasible return range (%)</label><ParameterHelp name="return_target_fraction" label="Position in feasible return range" /></div>
      <input id={`${prefix}-return-target`} className={cls} type="number" min={0} max={100} required step="any" value={Number.isFinite(fraction) ? fraction * 100 : ""} onChange={e => onFraction(e.target.valueAsNumber / 100)} />

      <p className="text-xs text-gray-500 mt-1">0% uses the minimum-variance portfolio’s return; 100% uses the highest expected return found. Choose the lowest-variance candidate meeting that target. The configured portfolio constraints apply. These are estimates, not promised returns.</p>
    </div>}
  </div>;
}
