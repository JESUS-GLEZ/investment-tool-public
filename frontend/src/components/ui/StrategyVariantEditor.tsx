import ParameterHelp from "./ParameterHelp";
import { useId, useState } from "react";
import { usesOptimizer, strategyLabel } from "@/lib/strategyVariants";
import type { StrategyVariant } from "@/types/api";
import PortfolioSelectionFields from "./PortfolioSelectionFields";

const cls = "mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 text-sm";

export default function StrategyVariantEditor({ variant, onChange, defaultSearchSeconds }: {
  variant: StrategyVariant;
  onChange: (value: StrategyVariant) => void;
  defaultSearchSeconds: number | null;
}) {
  const prefix = useId();
  const [lastTimedAllowance, setLastTimedAllowance] = useState(variant.search_seconds ?? defaultSearchSeconds ?? 300);
  const update = (values: Partial<StrategyVariant>) => onChange({...variant, ...values});
  const numeric = (key: "population_size" | "n_generations" | "min_assets" | "max_assets" | "n_restarts" | "random_seed" | "seed_max_subsets" | "seed_beam_width" | "refinement_supports", label: string, min: number) => <div key={key}>
    <div className="flex items-center gap-1"><label htmlFor={`${prefix}-${key}`} className="text-sm text-gray-700">{label}</label><ParameterHelp name={key} label={label} /></div>
    <input id={`${prefix}-${key}`} className={cls} type="number" step={1} min={min} max={key === "n_restarts" ? 20 : key === "random_seed" ? 4294967295 : key === "seed_max_subsets" ? 360000 : key === "seed_beam_width" ? 100 : key === "refinement_supports" ? 500 : undefined} required value={typeof variant[key] === "number" && Number.isFinite(variant[key]) ? variant[key] : ""} onChange={e => update({[key]: e.target.valueAsNumber})} />

  </div>;
  return <fieldset className="space-y-4">
    <legend className="mb-4 text-sm text-gray-500">{strategyLabel(variant.strategy_type)}</legend>
    <div className="flex gap-3 items-end">
      <div className="flex-1"><label htmlFor={`${prefix}-name`} className="text-sm font-medium">Name</label>
        <input id={`${prefix}-name`} className={cls} maxLength={80} required value={variant.name} onChange={e => update({name: e.target.value})} /></div>
    </div>
    {variant.strategy_type === "optimization" && <div>
      <div className="flex items-center gap-1"><label htmlFor={`${prefix}-return-estimator`} className="text-sm font-medium">Return estimator</label><ParameterHelp name="return_estimator" label="Return estimator" /></div>
      <select id={`${prefix}-return-estimator`} className={cls} value={variant.return_estimator ?? "ewm_6"}
        onChange={e => update({return_estimator: e.target.value as StrategyVariant["return_estimator"]})}>
        <option value="ewm_6">EWM-6 (current default)</option>
        <option value="ewm_12">EWM-12</option>
        <option value="ewm_12_mean_50">50% EWM-12 + 50% historical mean</option>
      </select>

      <p className="text-xs text-gray-500 mt-1">Uses only history available at each decision. Different forecasts run separate optimization searches.</p>
    </div>}
    {variant.strategy_type === "optimization" && <div className="space-y-3">
      <div><div className="flex items-center gap-1"><label htmlFor={`${prefix}-balance`} className="text-sm font-medium">Portfolio balance</label><ParameterHelp name="balance" label="Portfolio balance" /></div>
        <select id={`${prefix}-balance`} className={cls} value={variant.balance ?? "risk_contributions"}
          onChange={e => update({balance: e.target.value as StrategyVariant["balance"]})}>
          <option value="risk_contributions">Balanced risk contributions (current default)</option>
          <option value="capital_weights">Balanced capital weights</option>
          <option value="hybrid">Balanced weights and risk contributions</option>
        </select>

        <p className="text-xs text-gray-500 mt-1">{variant.balance === "hybrid"
          ? "Bounds both capital weights and risk contributions per active asset, while minimizing risk-contribution concentration. Neither quantity must be exactly equal."
          : "Applies allocation bounds and a concentration objective to the selected quantity. Capital weights need not be equal."}</p>
      </div>
      <div className="flex items-center gap-1"><label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={variant.news_shortlist ?? false}
        onChange={e => update({news_shortlist: e.target.checked})} />News shortlist</label><ParameterHelp name="news_shortlist" label="News shortlist" /></div>

      {variant.news_shortlist && <div className="grid grid-cols-2 gap-3">
        <div><div className="flex items-center gap-1"><label htmlFor={`${prefix}-news-top`} className="text-sm">Top scored assets</label><ParameterHelp name="news_top_assets" label="Top scored assets" /></div>
          <input id={`${prefix}-news-top`} className={cls} type="number" min={variant.max_assets ?? 5} max={10000} step={1} required value={Number.isFinite(variant.news_top_assets) ? variant.news_top_assets! : ""}
            onChange={e => update({news_top_assets: e.target.valueAsNumber})} /></div>
        <div><div className="flex items-center gap-1"><label htmlFor={`${prefix}-news-min`} className="text-sm">Minimum distinct articles</label><ParameterHelp name="news_min_articles" label="Minimum distinct articles" /></div>
          <input id={`${prefix}-news-min`} className={cls} type="number" min={1} max={1000} step={1} required value={Number.isFinite(variant.news_min_articles) ? variant.news_min_articles! : ""}
            onChange={e => update({news_min_articles: e.target.valueAsNumber})} /></div>
        <p className="col-span-2 text-xs text-gray-500">Ranks news from the previous calendar day, week or month to match the data frequency, using only articles available at the decision cutoff. Assets with insufficient news or unavailable scores remain eligible and unscored. Shortlist size must be at least the configured maximum assets.</p>
      </div>}
    </div>}
    {variant.strategy_type === "buy_and_hold" && <div>
      <label htmlFor={`${prefix}-asset-symbol`} className="text-sm font-medium">Asset symbol</label>
      <input id={`${prefix}-asset-symbol`} className={cls} required maxLength={30} value={variant.asset_symbol ?? ""} onChange={e => update({asset_symbol: e.target.value})} />
      <p className="mt-1 text-xs text-gray-500">Must be available in the selected dataset or universe. This holds a single asset.</p>
    </div>}
    {variant.strategy_type === "momentum_risk_parity" && <div className="grid grid-cols-2 gap-3">
      {([['top_n', 'Maximum selected assets', 1], ['lookback', 'Momentum lookback (periods)', 1], ['vol_window', 'Volatility window (periods)', 2], ['trend_window', 'Trend window (periods)', 1]] as const).map(([key, label, min]) => <div key={key}>
        <div className="flex items-center gap-1"><label htmlFor={`${prefix}-${key}`} className="text-sm">{label}</label><ParameterHelp name={key} label={label} /></div>
        <input id={`${prefix}-${key}`} className={cls} type="number" required step={1} min={min} value={Number.isFinite(variant[key]) ? variant[key]! : ""} onChange={e => update({[key]: e.target.valueAsNumber})} />

      </div>)}
      <div><div className="flex items-center gap-1"><label htmlFor={`${prefix}-min-weight`} className="text-sm">Minimum weight (fraction)</label><ParameterHelp name="min_weight" label="Minimum weight" /></div>
        <input id={`${prefix}-min-weight`} className={cls} type="number" required step="any" min={0} max={0.999999} value={Number.isFinite(variant.min_weight) ? variant.min_weight! : ""} onChange={e => update({min_weight: e.target.valueAsNumber})} /></div>
    </div>}
    {usesOptimizer(variant.strategy_type) && <>
    <div className="grid grid-cols-2 gap-3">{numeric("min_assets", "Min Assets", 1)}{numeric("max_assets", "Max Assets", 1)}</div>
    <PortfolioSelectionFields prefix={prefix} method={variant.selection} fraction={variant.return_target_fraction} increase={variant.volatility_increase}
      onMethod={selection => update({selection})} onFraction={return_target_fraction => update({return_target_fraction})} onIncrease={volatility_increase => update({volatility_increase})} />
    <details><summary className="text-sm cursor-pointer">Advanced search settings</summary>
      <p className="text-xs text-gray-500 mt-2">These values belong to this configuration. Selection-only configurations share a candidate front; different search settings run separately.</p>
      <div className="grid grid-cols-2 gap-3 mt-3">
        {numeric("population_size", "Population size", 10)}{numeric("n_generations", "Evaluations per restart", 1)}
        {numeric("n_restarts", "Independent restarts", 1)}{numeric("random_seed", "Random seed", 0)}
        {numeric("seed_max_subsets", "Maximum seed subsets", 1)}
        {numeric("seed_beam_width", "Seed beam width", 1)}
        {numeric("refinement_supports", "Weight refinement subsets (0 disables)", 0)}
        {variant.search_seconds !== null && <div><div className="flex items-center gap-1"><label htmlFor={`${prefix}-seconds`} className="text-sm">Search allowance (seconds)</label><ParameterHelp name="search_seconds" label="Search allowance" /></div>
          <input id={`${prefix}-seconds`} className={cls} type="number" min={Number.MIN_VALUE} required step="any" value={Number.isFinite(variant.search_seconds) ? variant.search_seconds! : ""} onChange={e => {const seconds = e.target.valueAsNumber; if (Number.isFinite(seconds) && seconds > 0) setLastTimedAllowance(seconds); update({search_seconds: seconds});}} /></div>}
      </div>
      <label className="flex items-center gap-2 text-sm mt-3"><input type="checkbox" checked={variant.search_seconds === null} onChange={e => update({search_seconds: e.target.checked ? null : lastTimedAllowance})} />Fixed-work mode (no time cutoff)</label>
      <p className="text-xs text-gray-500 mt-1">With a time limit, 40% funds shared seeding and 60% funds each restart. At 300 seconds and 3 restarts the total allowance is 660 seconds. When weight refinement is enabled, 10% from each restart allowance is pooled for one final refinement. Fixed-work mode may take substantially longer.</p>
    </details></>}
  </fieldset>;
}
