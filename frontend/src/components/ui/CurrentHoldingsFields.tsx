import AssetSymbolPicker from "./AssetSymbolPicker";
import { useId } from "react";
import type { HoldingInput } from "@/types/api";

export default function CurrentHoldingsFields({value, symbols, loading, failed, onChange, onRetry}: {
  value: HoldingInput[]; symbols: string[]; loading: boolean; failed: boolean; onChange: (value: HoldingInput[]) => void; onRetry: () => void;
}) {
  const id = useId();
  return <fieldset className="space-y-3 border-t pt-4">
    <legend className="text-sm font-medium">Current holdings (optional)</legend>
    <p className="text-xs text-gray-500">Enter the shares or units you currently own, including fractional quantities. Cash and valuations are in USD. Leave empty to start from cash. Quantities should already reflect any splits or other past corporate actions.</p>
    {loading && <p role="status" className="text-sm">Loading holding symbols…</p>}
    {failed && <p role="alert" className="text-sm text-red-700">Holding symbols could not be loaded. <button type="button" className="underline" onClick={onRetry}>Retry holding symbols</button></p>}
    <div className="space-y-2">{value.map((holding, index) => <div key={index} className="grid grid-cols-[1fr_1fr_auto] items-end gap-2">
      <AssetSymbolPicker id={`${id}-symbol-${index}`} label={`Asset ${index + 1}`} value={holding.symbol}
        symbols={symbols.filter(symbol => symbol === holding.symbol || !value.some(item => item.symbol === symbol))}
        onChange={symbol => onChange(value.map((item,i) => i === index ? {...item,symbol} : item))} />
      <label className="text-sm" htmlFor={`${id}-quantity-${index}`}>Shares / units {index + 1}<input id={`${id}-quantity-${index}`} className="mt-1 block w-full rounded border p-2" type="number" min={Number.MIN_VALUE} step="any" required value={Number.isFinite(holding.quantity) ? holding.quantity : ""} onChange={e => onChange(value.map((item, i) => i === index ? {...item, quantity: e.target.valueAsNumber} : item))} /></label>
      <button type="button" className="rounded border px-3 py-2 text-sm text-red-700" aria-label={`Remove holding ${index + 1}`} onClick={() => onChange(value.filter((_, i) => i !== index))}>Remove</button>
    </div>)}</div>
    <button type="button" className="rounded border px-3 py-1.5 text-sm disabled:opacity-50" disabled={loading || failed || value.length >= symbols.length} onClick={() => onChange([...value, {symbol: "", quantity: NaN}])}>Add holding</button>
    <p className="text-xs text-gray-500">Only assets in the chosen universe are offered; for local data they must also be present in the dataset. Changing the universe preserves entries for you to review. Prices are dated completed-bar references, not live quotes.</p>
  </fieldset>;
}
