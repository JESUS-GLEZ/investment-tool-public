import { useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { parameterHelp } from "@/lib/parameterHelp";

function HelpCard({label, explanation, onClose}: {label: string; explanation: string; onClose: () => void}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const title = useId();
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    const element = dialog.current!;
    if (element.showModal) element.showModal(); else element.setAttribute("open", "");
    element.querySelector("button")?.focus();
    return () => {element.close?.(); if (previous?.isConnected) previous.focus();};
  }, []);
  return createPortal(<dialog ref={dialog} aria-labelledby={title} aria-modal="true"
    onCancel={event => {event.preventDefault(); event.stopPropagation(); onClose();}}
    onClick={event => {if (event.target === event.currentTarget) onClose();}}
    className="fixed inset-0 m-auto max-h-[calc(100dvh-2rem)] w-[calc(100vw-2rem)] max-w-md overflow-y-auto rounded-xl border border-gray-200 bg-white p-0 text-gray-900 shadow-xl backdrop:bg-black/25">
    <div className="space-y-3 p-5">
      <div className="flex items-start justify-between gap-4"><h2 id={title} className="text-sm font-semibold">{label}</h2>
        <button type="button" aria-label="Close parameter help" onClick={onClose} className="shrink-0 rounded px-2 py-0.5 text-lg leading-none text-gray-500 hover:bg-gray-100 focus-visible:outline-2 focus-visible:outline-blue-600">×</button>
      </div>
      <p className="text-sm leading-relaxed text-gray-600">{explanation}</p>
    </div>
  </dialog>, document.body);
}

/** A compact, touch/keyboard-accessible information button; never submits a form. */
export default function ParameterHelp({name, label, group}: {name: string; label: string; group?: string}) {
  const [open, setOpen] = useState(false);
  const explanation = parameterHelp(name, group);
  if (!explanation) return null;
  return <span className="inline-flex shrink-0 align-middle">
    <button type="button" aria-label={`Explain ${label}`} aria-haspopup="dialog" aria-expanded={open} title={`About ${label}`}
      onClick={() => setOpen(true)} className="inline-flex h-7 w-7 items-center justify-center rounded-full text-gray-400 hover:bg-blue-50 hover:text-blue-700 focus-visible:outline-2 focus-visible:outline-blue-600">
      <svg aria-hidden="true" viewBox="0 0 20 20" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.5"><circle cx="10" cy="10" r="7.25" /><path d="M10 9v5" /><circle cx="10" cy="6.25" r="0.75" fill="currentColor" stroke="none" /></svg>
    </button>
    {open && <HelpCard label={label} explanation={explanation} onClose={() => setOpen(false)} />}
  </span>;
}
