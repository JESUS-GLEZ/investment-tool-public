import { record } from "@/lib/runPresentation";

const money = (value: unknown) => typeof value === "number" && Number.isFinite(value) ? new Intl.NumberFormat(undefined, {style: "currency", currency: "USD"}).format(value) : "—";
const number = (value: unknown) => typeof value === "number" && Number.isFinite(value) ? new Intl.NumberFormat(undefined, {maximumSignificantDigits: 10}).format(value) : "—";
const percent = (value: unknown) => typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : "—";

export default function RebalancePlan({plan}: {plan: Record<string, unknown>}) {
  const rows = Array.isArray(plan.rows) ? plan.rows.map(record) : [];
  return <div className="space-y-4" aria-label="Estimated rebalance">
    <h3 className="font-medium">Current portfolio and estimated trades</h3>
    <dl className="grid gap-3 text-sm sm:grid-cols-2">
      <div><dt className="text-gray-500">Current value / available cash</dt><dd>{money(plan.current_value)} / {money(plan.current_cash)}</dd></div>
      <div><dt className="text-gray-500">Estimated trading costs</dt><dd>{money(plan.estimated_cost)}</dd></div>
      <div><dt className="text-gray-500">Estimated value after costs / remaining cash</dt><dd>{money(plan.estimated_value)} / {money(plan.estimated_cash)}</dd></div>
    </dl>
    <div className="overflow-x-auto"><table className="w-full text-sm">
      <thead><tr className="border-b text-right"><th className="p-2 text-left">Asset / reference price</th><th className="p-2">Current units / weight</th><th className="p-2">Model target weight</th><th className="p-2">Estimated units after rebalance</th><th className="p-2">Suggested action / units</th><th className="p-2">Trade value / cost</th></tr></thead>
      <tbody>{rows.map(row => <tr key={String(row.symbol)} className="border-b text-right tabular-nums">
        <th className="p-2 text-left font-medium">{String(row.symbol)}<div className="text-xs font-normal text-gray-500">{money(row.reference_price)} · period end {String(row.price_period_end)}</div></th>
        <td className="p-2">{number(row.current_quantity)} / {percent(row.current_weight)}</td>
        <td className="p-2">{percent(row.target_weight)}</td>
        <td className="p-2">{number(row.estimated_quantity)}</td>
        <td className="p-2 capitalize">{String(row.action)} {number(typeof row.trade_quantity === "number" ? Math.abs(row.trade_quantity) : null)}</td>
        <td className="p-2">{money(typeof row.trade_value === "number" ? Math.abs(row.trade_value) : null)} / {money(row.estimated_cost)}</td>
      </tr>)}</tbody>
    </table></div>
    <p className="text-xs text-gray-500">Estimates use latest completed, available closing prices in USD. These are not live quotes or executed orders. Purchases are reduced to fund estimated fees, so resulting allocations can differ from model target weights. Confirm prices, fractional-unit support and actual fees before trading manually.</p>
    <details><summary className="cursor-pointer text-sm">Reference price availability</summary><ul className="mt-2 text-xs">{rows.map(row => <li key={String(row.symbol)}>{String(row.symbol)}: {String(row.price_available_at)}</li>)}</ul></details>
  </div>;
}
