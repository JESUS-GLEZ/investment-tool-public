import ParameterHelp from "./ParameterHelp";
import { useId } from "react";
import { useQuery } from "@tanstack/react-query";
import { listAssetConfigs } from "@/api/endpoints";

export const DEFAULT_ASSET_CONFIG_FILE = "example_asset_universe.json";

export default function AssetUniverseSelect({ value, onChange }: {
  value: string;
  onChange: (file: string) => void;
}) {
  const selectId = useId();
  const configs = useQuery({ queryKey: ["asset-configs"], queryFn: listAssetConfigs });
  const options = configs.data ?? [];
  const selected = options.find((config) => config.file === value);

  return (
    <div>
      <div className="flex items-center gap-1"><label htmlFor={selectId} className="block text-sm font-medium text-gray-700">
        Asset universe
      </label><ParameterHelp name="asset_config_file" label="Asset universe" /></div>
      <select id={selectId} aria-label="Asset universe" value={value}
        onChange={(event) => onChange(event.target.value)}
        className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 text-sm">
        {!selected && <option value={value}>{value} (not in active catalog)</option>}
        {options.map((config) => (
          <option key={config.file} value={config.file}>
            {config.name} — {config.stock_count} stocks, {config.crypto_count} crypto
          </option>
        ))}
      </select>

      {configs.isPending && <p role="status" className="mt-1 text-sm text-gray-500">Loading asset universes…</p>}
      {configs.isError && <div role="alert" className="mt-1 text-sm text-red-700">Could not load asset universes.
        <button type="button" disabled={configs.isFetching} onClick={() => { void configs.refetch(); }} className="ml-2 underline disabled:opacity-50">Retry asset universes</button>
      </div>}
      {selected && <p className="mt-1 text-xs text-gray-500">
        {selected.file}
        {selected.membership_mode === "dated" && <> · Dated membership: up to {selected.stock_limit} equities per snapshot, plus fixed candidates. Historical snapshots are required for the trading period.</>}
      </p>}
    </div>
  );
}
