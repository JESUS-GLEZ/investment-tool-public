import { useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { assignRunCampaign, listCampaigns, saveCampaign } from "@/api/endpoints";
import type { Campaign, CampaignFields, RunSummary } from "@/types/api";
import { getErrorMessage } from "@/lib/utils";

const field = "mt-1 block w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-sm";
const button = "rounded border border-gray-300 px-3 py-1.5 text-sm disabled:opacity-50";
const useCampaigns = () => useQuery({queryKey: ["campaigns"], queryFn: listCampaigns});

export function CampaignSelect({value, onChange, includeAll = false, disabled = false, label = "Campaign"}: {
  value: string; onChange: (id: string) => void; includeAll?: boolean; disabled?: boolean; label?: string;
}) {
  const query = useCampaigns();
  const known = (query.data ?? []).some(c => c.id === value);
  return <div className="min-w-0">
    <label className="block text-sm font-medium">{label}
      <select className={field} value={value} disabled={disabled || query.isPending} onChange={e => onChange(e.target.value)}>
        {includeAll && <option value="all">All campaigns</option>}
        <option value={includeAll ? "ungrouped" : ""}>Ungrouped</option>
        {value && !known && !["all", "ungrouped"].includes(value) && <option value={value}>Unavailable campaign</option>}
        {(query.data ?? []).map(c => <option key={c.id} value={c.id}>{c.name}{includeAll ? ` (${c.run_count})` : ""}</option>)}
      </select>
    </label>
    {query.isError && <p role="alert" className="mt-1 text-xs text-red-700">Could not load campaigns. <button type="button" className="underline" onClick={() => void query.refetch()}>Retry campaigns</button></p>}
  </div>;
}

function CampaignEditor({campaign, onSaved, onCancel}: {campaign: Campaign | null; onSaved: (c: Campaign) => void; onCancel: () => void}) {
  const [draft, setDraft] = useState<CampaignFields>(() => campaign ? {name: campaign.name, hypothesis: campaign.hypothesis, notes: campaign.notes} : {name: "", hypothesis: "", notes: ""});
  const client = useQueryClient();
  const mutation = useMutation({mutationFn: () => saveCampaign(draft, campaign?.id), onSuccess: c => {
    void client.invalidateQueries({queryKey: ["campaigns"]}); onSaved(c);
  }});
  function submit(e: FormEvent) {e.preventDefault(); if (!mutation.isPending) mutation.mutate();}
  return <form onSubmit={submit} className="space-y-3 rounded border bg-gray-50 p-4" aria-label={campaign ? "Edit campaign" : "Create campaign"}>
    <fieldset disabled={mutation.isPending} className="space-y-3">
      <label className="block text-sm font-medium">Campaign name<input required maxLength={120} value={draft.name} className={field} onChange={e => setDraft({...draft, name: e.target.value})} /></label>
      <label className="block text-sm font-medium">Hypothesis or objective<textarea rows={2} maxLength={2000} value={draft.hypothesis} className={field} onChange={e => setDraft({...draft, hypothesis: e.target.value})} /></label>
      <label className="block text-sm font-medium">Notes and conclusions<textarea rows={3} maxLength={10000} value={draft.notes} className={field} onChange={e => setDraft({...draft, notes: e.target.value})} /></label>
      {mutation.isError && <p role="alert" className="text-sm text-red-700">{getErrorMessage(mutation.error, "Could not save campaign.")}</p>}
      <div className="flex gap-2"><button type="submit" className={button} disabled={!draft.name.trim() || mutation.isPending}>{mutation.isPending ? "Saving…" : "Save campaign"}</button><button type="button" className={button} onClick={onCancel}>Cancel</button></div>
    </fieldset>
  </form>;
}

export function CampaignPanel({value, onChange}: {value: string; onChange: (id: string) => void}) {
  const query = useCampaigns();
  const selected = query.data?.find(c => c.id === value);
  const [editor, setEditor] = useState<{campaign: Campaign | null} | null>(null);
  return <section aria-label="Campaigns" className="space-y-3 rounded-lg border border-gray-200 bg-white p-4">
    <div className="flex flex-wrap items-end gap-3">
      <div className="min-w-48 flex-1"><CampaignSelect value={value} onChange={onChange} includeAll label="Filter by campaign" /></div>
      <button type="button" className={button} onClick={() => setEditor({campaign: null})}>New campaign</button>
      {selected && <button type="button" className={button} onClick={() => setEditor({campaign: selected})}>Edit campaign</button>}
    </div>
    {selected && <>
      <details className="text-sm"><summary className="cursor-pointer font-medium">Hypothesis and notes</summary>
        <p className="mt-2 whitespace-pre-wrap break-words">{selected.hypothesis || "No hypothesis recorded."}</p>
        <p className="mt-2 whitespace-pre-wrap break-words text-gray-600">{selected.notes || "No notes yet."}</p>
      </details>
      <div className="flex flex-wrap gap-4 text-sm"><Link className="text-blue-700 underline" to={`/runs/new?campaign=${selected.id}`}>New Run in this campaign</Link><Link className="text-blue-700 underline" to={`/results/compare?campaign=${selected.id}`}>Compare campaign runs</Link></div>
    </>}
    {!selected && <p className="text-xs text-gray-500">Group related runs around a hypothesis. Grouping is optional and does not change their inputs or results.</p>}
    {editor && <CampaignEditor key={editor.campaign?.id ?? "new"} campaign={editor.campaign} onCancel={() => setEditor(null)} onSaved={c => {setEditor(null); onChange(c.id);}} />}
  </section>;
}

export function RunCampaignAssignment({run}: {run: RunSummary}) {
  const client = useQueryClient();
  const mutation = useMutation({mutationFn: (id: string) => assignRunCampaign(run.id, id || null), onSuccess: () => {
    void client.invalidateQueries({queryKey: ["runs"]});
    void client.invalidateQueries({queryKey: ["run", run.id]});
    void client.invalidateQueries({queryKey: ["campaigns"]});
  }});
  return <div className="max-w-lg space-y-1">
    <CampaignSelect value={run.campaign_id ?? ""} label={`Campaign for ${run.id}`} onChange={id => mutation.mutate(id)} disabled={mutation.isPending} />
    {mutation.isPending && <p role="status" className="text-xs text-gray-500">Updating campaign…</p>}
    {mutation.isError && <p role="alert" className="text-xs text-red-700">{getErrorMessage(mutation.error, "Could not change campaign. The previous assignment is preserved.")}</p>}
  </div>;
}
