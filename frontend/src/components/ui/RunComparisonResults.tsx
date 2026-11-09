import { useState } from "react";
import type { SavedCandidate } from "@/lib/savedComparison";
import ComparisonResults, { CompatibilityNotice } from "./ComparisonResults";
export default function RunComparisonResults({entries}: {entries: SavedCandidate[]}) {
  const [selection, setSelection] = useState<string[] | null>(null);
  const ids = selection ?? entries.slice(0, 3).map(e => e.id);
  const selected = entries.filter(e => ids.includes(e.id));
  return <section aria-label="Comparison Results" className="space-y-4"><h2 className="text-lg font-semibold">Comparison Results</h2>{entries.length > 1 && <details className="rounded-lg border border-gray-200 bg-white p-4"><summary className="cursor-pointer text-sm font-medium">Displayed configurations ({selected.length} of {entries.length})</summary><p className="mt-2 text-xs text-gray-500">Choose up to three to keep the results readable. All saved configurations remain available here.</p><div className="mt-3 grid gap-2 sm:grid-cols-2">{entries.map(e => <label key={e.id} className="flex items-center gap-2 text-sm"><input type="checkbox" checked={ids.includes(e.id)} disabled={!ids.includes(e.id) && ids.length >= 3} onChange={event => setSelection(event.target.checked ? [...ids, e.id] : ids.filter(id => id !== e.id))} />{e.name}</label>)}</div></details>}{selected.length > 1 && <CompatibilityNotice entries={selected} />}{selected.length ? <ComparisonResults entries={selected} reportedLegacy /> : <p className="text-sm text-gray-500">Choose a configuration to inspect.</p>}</section>;
}
