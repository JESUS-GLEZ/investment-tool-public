import ParameterHelp from "./ParameterHelp";
import { useId } from "react";
import type { OutputMode } from "@/types/api";

export default function OutputModeSelect({ value, onChange }: {
  value: OutputMode; onChange: (value: OutputMode) => void;
}) {
  const id = useId();
  return <div>
    <div className="flex items-center gap-1"><label htmlFor={id} className="block text-sm font-medium text-gray-700">Saved output</label><ParameterHelp name="output_mode" label="Saved output" /></div>
    <select id={id} value={value} onChange={(e) => onChange(e.target.value as OutputMode)}
      className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 text-sm">
      <option value="standard">Standard — compact audit records</option>
      <option value="diagnostic">Diagnostic — full plots and archives</option>
    </select>

    <p className="mt-1 text-xs text-gray-500">Standard keeps results, holdings, checkpoints and, for optimization, one final Pareto plot per rebalance. Interactive charts remain available. Diagnostic saves additional chart exports and intermediate archives and uses more disk space.</p>
  </div>;
}
