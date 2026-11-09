import { useEffect, useId, useState } from "react";

/** Search is a draft until a result is chosen or the field is left. Escape cancels it. */
export default function AssetSymbolPicker({id, label, value, symbols, onChange}: {
  id: string; label: string; value: string; symbols: string[]; onChange: (value: string) => void;
}) {
  const listId = useId();
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState({value, query:value, editing:false});
  const query = open && draft.value === value ? draft.query : value;
  const editing = open && draft.value === value && draft.editing;
  const [active, setActive] = useState(0);
  const options = symbols.filter(symbol => symbol.includes((!editing && symbols.includes(value) ? "" : query).trim().toUpperCase())).sort((a,b) => a.localeCompare(b, "en"));
  const choose = (symbol: string) => {onChange(symbol); setDraft({value:symbol,query:symbol,editing:false}); setOpen(false); setActive(0);};
  const index = Math.min(active, Math.max(0, options.length - 1));
  const error = !open && !!value && !symbols.includes(value);
  useEffect(() => {if (open) document.getElementById(`${listId}-${index}`)?.scrollIntoView?.({block:"nearest"});}, [open,index,listId,query]);
  return <div className="relative">
    <label htmlFor={id} className="text-sm">{label}</label>
    <input id={id} role="combobox" aria-autocomplete="list" aria-expanded={open} aria-controls={listId}
      aria-activedescendant={open && options.length ? `${listId}-${index}` : undefined}
      aria-invalid={error} aria-describedby={error ? `${id}-error` : undefined}
      autoComplete="off" spellCheck={false} required placeholder="Search symbol…" value={query}
      className="mt-1 block w-full rounded border p-2 text-sm"
      onFocus={e => {setDraft({value,query:value,editing:false}); setOpen(true); setActive(Math.max(0,options.indexOf(value))); e.currentTarget.select();}}
      onClick={e => {
        if (!open) {setDraft({value,query:value,editing:false}); setActive(Math.max(0,options.indexOf(value)));}
        setOpen(true);
        if (query === value) e.currentTarget.select();
      }}
      onBlur={() => {if (open && query !== value) onChange(query.trim()); setOpen(false);}}
      onChange={e => {setDraft({value,query:e.target.value.toUpperCase(),editing:true}); setOpen(true); setActive(0);}}
      onKeyDown={e => {
        if (e.key === "Escape") {e.preventDefault(); setDraft({value,query:value,editing:false}); setOpen(false);}
        if (e.key === "ArrowDown" || e.key === "ArrowUp") {
          e.preventDefault(); setOpen(true);
          setActive(open ? Math.max(0, Math.min(options.length - 1, index + (e.key === "ArrowDown" ? 1 : -1))) : Math.max(0,options.indexOf(value)));
        }
        if (e.key === "Enter" && open) {e.preventDefault(); if (options.length) choose(options[index]);}
      }} />
    {open && <div className="absolute z-20 mt-1 w-full rounded border bg-white shadow-lg">
      <ul id={listId} role="listbox" aria-label={`${label} matches`} className="max-h-48 overflow-auto">
        {options.map((symbol, i) => <li id={`${listId}-${i}`} key={symbol} role="option" aria-selected={i === index}
          className={`cursor-pointer px-3 py-2 text-sm ${i === index ? "bg-blue-50" : "hover:bg-gray-50"}`}
          onMouseDown={e => e.preventDefault()} onClick={() => choose(symbol)}>{symbol}</li>)}
      </ul>
      <p role="status" className="border-t px-3 py-2 text-xs text-gray-500">{options.length ? `${options.length} alphabetical matches · scroll or type to narrow the list` : "No matching eligible symbols"}</p>
    </div>}
    {error && <p id={`${id}-error`} className="mt-1 text-xs text-amber-800">Choose an eligible symbol from this universe.</p>}
  </div>;
}
