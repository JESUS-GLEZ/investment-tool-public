import { useEffect, useId, useRef, type ReactNode } from "react";
import { createPortal } from "react-dom";

/** Shared native dialog: focus containment, Escape and return-to-trigger behavior. */
export default function ConfigurationDrawer({title, error, focusFallbackId, onClose, onSave, children}: {
  title: string; error?: string; focusFallbackId: string; onClose: () => void; onSave: () => void; children: ReactNode;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const titleId = useId();
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    const element = dialog.current!;
    if (element.showModal) element.showModal(); else element.setAttribute("open", "");
    element.querySelector("input")?.focus();
    return () => {(previous?.isConnected ? previous : document.getElementById(focusFallbackId))?.focus();};
  }, [focusFallbackId]);
  return createPortal(<dialog ref={dialog} aria-labelledby={titleId} aria-modal="true" onCancel={event => {event.preventDefault(); onClose();}}
    className="fixed inset-y-0 right-0 m-0 ml-auto h-dvh max-h-none w-full max-w-none overflow-y-auto border-0 bg-white p-0 text-gray-900 shadow-xl backdrop:bg-black/40 sm:max-w-xl">
    <form className="flex min-h-full flex-col" onInvalidCapture={event => {const details = (event.target as HTMLElement).closest("details"); if (details) details.open = true;}}
      onSubmit={event => {event.preventDefault(); onSave();}}>
      <div className="flex items-center justify-between border-b p-5"><h2 id={titleId} className="text-lg font-semibold">{title}</h2><button type="button" className="rounded border px-3 py-1 text-sm" onClick={onClose}>Cancel</button></div>
      <div className="flex-1 space-y-4 p-5">{error && <p role="alert" className="rounded bg-red-50 p-3 text-sm text-red-700">{error}</p>}{children}</div>
      <div className="sticky bottom-0 flex justify-end border-t bg-white p-5"><button type="submit" className="rounded bg-blue-600 px-4 py-2 text-sm font-medium text-white">Save configuration</button></div>
    </form>
  </dialog>, document.body);
}
