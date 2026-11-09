/** Copy an older execution preset into the shared backtest editor without rewriting it. */
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { listPresets } from "@/api/endpoints";
import type { ExecutionRequest } from "@/types/api";
import { getErrorMessage } from "@/lib/utils";

export default function LegacyBacktestPresetImport({ onImport }: { onImport: (parameters: ExecutionRequest) => void }) {
  const [open, setOpen] = useState(false);
  const [notice, setNotice] = useState("");
  const presets = useQuery({ queryKey: ["presets", "execution"], queryFn: () => listPresets<ExecutionRequest>("execution"), enabled: open });
  const options = (presets.data ?? []).filter(p => p.run_type === "execution" && (p.parameters.execution_mode ?? "backtesting") === "backtesting");
  return <section className="space-y-2 text-sm" aria-label="Import older backtest presets">
    <button type="button" className="text-gray-600 underline" aria-expanded={open} onClick={() => setOpen(value => !value)}>Import an older single-strategy backtest preset</button>
    {open && <div className="space-y-2 rounded border border-gray-200 p-3">
      <p className="text-xs text-gray-500">Copies the strategy and input settings into this backtest, with the additional benchmark disabled. Save as a new preset if needed; the original is not changed.</p>
      {presets.isPending && <p role="status">Loading older backtest presets…</p>}
      {presets.isError && <p role="alert">{getErrorMessage(presets.error, "Could not load older backtest presets.")} <button type="button" className="underline" disabled={presets.isFetching} onClick={() => void presets.refetch()}>Retry older backtest presets</button>
      </p>}
      {!!options.length && <label className="block">Older backtest preset<select className="mt-1 block w-full rounded border border-gray-300 bg-white p-2" value="" onChange={e => {
        const preset = options.find(p => String(p.id) === e.target.value);
        if (preset) { onImport(preset.parameters); setNotice(`Imported ${preset.name}. Review and save as a new backtest preset if needed.`); }
      }}>
        <option value="">Choose a preset to copy…</option>{options.map(p =>
          <option key={p.id} value={p.id}>{p.name}</option>)}</select>
      </label>}
      {presets.isSuccess && !options.length && <p className="text-gray-500">No older single-strategy backtest presets are available.</p>}
    </div>}
    {notice && <p role="status" className="text-gray-600">{notice}</p>}
  </section>;
}
