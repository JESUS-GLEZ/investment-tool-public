import { useId, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { listPresets, savePreset, deletePreset } from "@/api/endpoints";
import type { RunType } from "@/types/api";
import { getErrorMessage } from "@/lib/utils";

export default function PresetControls<T extends object>({ runType, value, onLoad, validationError, accept }: {
  runType: RunType;
  value: T;
  onLoad: (value: T) => void;
  validationError?: string;
  accept?: (parameters: T) => boolean;
}) {
  const validationId = useId();
  const client = useQueryClient();
  const [selected, setSelected] = useState<number | undefined>();
  const [name, setName] = useState("");
  const [notice, setNotice] = useState("");
  const [formError, setFormError] = useState("");
  const queryKey = ["presets", runType];
  const presets = useQuery({ queryKey, queryFn: () => listPresets<T>(runType) });
  const available = (presets.data ?? []).filter(preset => !accept || accept(preset.parameters));
  const mutation = useMutation({
    mutationFn: async (action: "create" | "update" | "delete") => {
      if (action === "delete") {
        await deletePreset(selected!);
        setSelected(undefined);
        setName("");
        setNotice("Preset deleted.");
      } else {
        if (validationError) return;
        const result = await savePreset({ name: name.trim(), run_type: runType, parameters: value }, action === "update" ? selected : undefined);
        setSelected(result.id);
        setName(result.name);
        setNotice("Preset saved. No run has been launched.");
      }
      await client.invalidateQueries({ queryKey });
    },
  });
  const save = (action: "create" | "update", form: HTMLFormElement | null) => {
    mutation.reset();
    setNotice("");
    setFormError("");
    if (form && !form.reportValidity()) {
      setFormError("Check the highlighted form fields before saving this preset.");
      return;
    }
    mutation.mutate(action);
  };
  const buttonClass = "rounded border border-gray-300 px-3 py-1 text-sm disabled:opacity-50";
  return (
    <section aria-label="Saved presets" className="space-y-2 rounded-md border border-gray-200 bg-gray-50 p-3">
      <label className="block text-sm font-medium">
        Saved preset
        <select aria-label="Saved preset" className="mt-1 block w-full rounded border border-gray-300 bg-white p-2"
          value={selected ?? ""} disabled={mutation.isPending || presets.isPending}
          onChange={(event) => {
            mutation.reset(); setNotice(""); setFormError("");
            const preset = available.find((item) => item.id === Number(event.target.value));
            setSelected(preset?.id); setName(preset?.name ?? "");
            if (preset) { onLoad(preset.parameters); setNotice("Preset loaded. Review the fields before launching."); }
          }}>
          <option value="">Choose a preset…</option>
          {available.map((preset) => <option key={preset.id} value={preset.id}>{preset.name}</option>)}
        </select>
      </label>
      <label className="block text-sm">
        Preset name
        <input aria-label="Preset name" maxLength={100} value={name} onChange={(event) => setName(event.target.value)}
          className="mt-1 block w-full rounded border border-gray-300 p-2" placeholder="e.g. Monthly solver comparison" />
      </label>
      <div className="flex flex-wrap gap-2">
        <button type="button" className={buttonClass} disabled={!name.trim() || mutation.isPending || !!validationError} aria-describedby={validationError ? validationId : undefined} onClick={(event) => save("create", event.currentTarget.form)}>Save as new</button>
        <button type="button" className={buttonClass} disabled={!selected || !name.trim() || mutation.isPending || !!validationError} aria-describedby={validationError ? validationId : undefined} onClick={(event) => save("update", event.currentTarget.form)}>Update selected</button>
        <button type="button" className={buttonClass} disabled={!selected || mutation.isPending} onClick={() => { setFormError(""); mutation.mutate("delete"); }}>Delete selected</button>
      </div>
      {validationError && <p id={validationId} className="text-xs text-red-700">Resolve the configuration issue before saving. {validationError}</p>}
      <p className="text-xs text-gray-500">Saved to your account. Edit the form below, then save or update a preset.</p>
      {notice && <p role="status" className="text-sm text-gray-600">{notice}</p>}
      {formError && <p role="alert" className="text-sm text-red-700">{formError}</p>}
      {presets.isPending && <p role="status" className="text-sm text-gray-500">Loading saved presets…</p>}
      {presets.error && <div role="alert" className="text-sm text-red-700">{getErrorMessage(presets.error, "Could not load saved presets.")}
        <button type="button" disabled={presets.isFetching || mutation.isPending} onClick={() => { void presets.refetch(); }} className="ml-2 underline disabled:opacity-50">Retry saved presets</button>
      </div>}
      {mutation.error && <p role="alert" className="text-sm text-red-700">{getErrorMessage(mutation.error, "Could not access presets.")}</p>}
    </section>
  );
}
