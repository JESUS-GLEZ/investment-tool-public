import type { RunSummary } from "@/types/api";
import { configurationFields, recordedConfigurations, settingLabel, settingValue } from "@/lib/runPresentation";

/** Identify the saved strategy without substituting today's model defaults. */
export default function ExecutionStrategySummary({ run }: { run: RunSummary }) {
  if (run.run_type !== "execution") return null;
  const configuration = recordedConfigurations(run)[0];
  const fields = configurationFields(configuration, true);
  return <section aria-label="Strategy used" className="space-y-3 rounded-lg border border-gray-200 bg-white p-5">
    <div>
      <p className="text-xs font-medium uppercase text-gray-500">Strategy used</p>
      <h2 className="mt-1 break-words text-lg font-semibold">{configuration.name}</h2>
      {configuration.name !== configuration.method && <p className="mt-1 text-sm text-gray-600">Method: {configuration.method}</p>}
    </div>
    {fields.length > 0 && <dl className="grid gap-x-6 gap-y-3 text-sm sm:grid-cols-2 lg:grid-cols-3">{fields.map(([key, value]) => <div key={key}>
      <dt className="text-xs text-gray-500">{settingLabel(key)}</dt>
      <dd className="mt-1 font-medium">{settingValue(key, value)}</dd>
    </div>)}</dl>}
    <p className="text-xs text-gray-500">Recorded configuration for this run.</p>
  </section>;
}
