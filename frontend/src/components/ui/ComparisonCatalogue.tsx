import { useId, useState, type ReactNode } from "react";

export type ComparisonOptions = Record<string, {label: string; description: string}>;

/** Compact selected pool with a searchable, optional catalogue. */
export default function ComparisonCatalogue({kind, options, selected, onChange, onAdd, children, toggleId}: {
  kind: "solver" | "predictor"; options: ComparisonOptions; selected: string[]; onChange?: (next: string[]) => void; onAdd?: (key: string) => void; children?: ReactNode; toggleId?: string;
}) {
  const id = useId();
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const matches = Object.entries(options).filter(([key, value]) => `${key} ${value.label} ${value.description}`.toLowerCase().includes(search.toLowerCase()));
  const button = "rounded border border-gray-300 px-3 py-1.5 text-sm disabled:opacity-50";
  return <div className="space-y-4">
    <button type="button" id={toggleId} className={button} aria-expanded={open} aria-controls={`${id}-catalogue`} onClick={() => setOpen(value => !value)}>Add {kind}</button>
    {open && <div id={`${id}-catalogue`} className="space-y-3 rounded border bg-gray-50 p-4">
      <label htmlFor={`${id}-search`} className="block text-sm">Search {kind}s</label>
      <input id={`${id}-search`} type="search" value={search} onChange={e => setSearch(e.target.value)} className="block w-full rounded border border-gray-300 p-2 text-sm" />
      <div className="max-h-80 space-y-2 overflow-y-auto">{matches.map(([key, option]) => <div key={key} className="flex items-center justify-between gap-3 rounded border bg-white p-3">
        <div className="min-w-0"><p className="break-words text-sm font-medium">{option.label}</p><p className="text-xs text-gray-500">{option.description}</p></div>
        <button type="button" aria-label={`Add ${option.label}`} disabled={onAdd ? selected.length >= 20 : selected.includes(key)} className={button} onClick={() => onAdd ? onAdd(key) : onChange?.([...selected, key])}>{!onAdd && selected.includes(key) ? "Added" : "Add"}</button>
      </div>)}</div>
      {!matches.length && <p role="status" className="text-sm text-gray-500">No matching {kind}s.</p>}
    </div>}
    {children ?? <ol className="space-y-3">{selected.map(key => <li key={key} aria-label={options[key]?.label ?? key} className="flex flex-wrap items-start justify-between gap-3 rounded border p-4">
      <div className="min-w-0 flex-1"><h3 className="break-words font-medium">{options[key]?.label ?? key}</h3><p className="text-xs text-gray-500">{options[key]?.description ?? "Unavailable in the current catalogue. Remove this entry or update the preset."}</p></div>
      <button type="button" aria-label={`Remove ${options[key]?.label ?? key}`} className={`${button} text-red-700`} onClick={() => onChange?.(selected.filter(value => value !== key))}>Remove</button>
    </li>)}</ol>}
    {!selected.length && <p role="status" className="text-sm text-amber-800">Add at least one {kind}.</p>}
    <p className="text-xs text-gray-500">{selected.length} {kind}{selected.length === 1 ? "" : "s"} selected. All use the same comparison settings below.</p>
  </div>;
}
