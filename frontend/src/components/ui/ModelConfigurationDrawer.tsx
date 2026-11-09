import ParameterHelp from "./ParameterHelp";
import { useId, useState } from "react";
import type { PredictorVariant } from "@/types/api";
import { modelKey, hydrateModelVariant, modelVariantError, predictorFields, type ModelVariant } from "@/lib/modelVariants";
import ConfigurationDrawer from "./ConfigurationDrawer";
import { predictorFamilyLabel } from "@/lib/comparisonCatalogue";

const cls = "mt-1 block w-full rounded border border-gray-300 px-3 py-2 text-sm disabled:bg-gray-100";
export default function ModelConfigurationDrawer({value, others, frequency, shared, focusFallbackId, onSave, onClose}: {
  value: ModelVariant; others: ModelVariant[]; frequency: string; shared:{population_size:number;n_generations:number;n_restarts:number};
  focusFallbackId: string; onSave:(v:ModelVariant)=>void; onClose:()=>void;
}) {
  const helpPrefix = useId();
  const [draft,setDraft] = useState<ModelVariant>(()=>hydrateModelVariant({...value}));
  const [error,setError] = useState<string>();
  const change = (patch:Partial<ModelVariant>) => {setDraft(current => ({...current,...patch})); setError(undefined);};
  const solver = "solver" in draft;
  const inherited = solver && draft.population_size == null && draft.n_generations == null && draft.n_restarts == null;
  const num = (value:string) => value === "" ? NaN : Number(value);
  return <ConfigurationDrawer title={`Configure ${solver ? modelKey(draft) : predictorFamilyLabel(draft.predictor)}`} error={error} focusFallbackId={focusFallbackId} onClose={onClose} onSave={() => {
    const issue = modelVariantError(draft,others,frequency); setError(issue);
    if (!issue) onSave({...draft,name:draft.name.trim()});
  }}>
    <label className="block text-sm">Name<input required maxLength={80} className={cls} value={draft.name} onChange={e=>change({name:e.target.value})} /></label>
    {solver ? <>
      <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={inherited} onChange={e=>change(e.target.checked ? {population_size:null,n_generations:null,n_restarts:null} : {...shared})} />Use shared solver budgets</label>
      {([['population_size','Population size',10],['n_generations','Evaluations per restart',1],['n_restarts','Restarts',1]] as const).map(([field,label,min])=><div key={field}><div className="flex items-center gap-1"><label className="block text-sm" htmlFor={`${helpPrefix}-${field}`}>{label}</label><ParameterHelp name={field} label={label} /></div><input id={`${helpPrefix}-${field}`} className={cls} type="number" required min={min} disabled={inherited} value={draft[field] ?? shared[field]} onChange={e=>change({[field]:num(e.target.value)})} /></div>)}
      <p className="text-xs text-gray-500">Shared budgets stay linked to the comparison defaults. Override them to compare search configurations; unequal budgets will be highlighted in review.</p>
      <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={draft.random_seed != null} onChange={e=>change({random_seed:e.target.checked ? 0 : null})} />Use reproducible random seed</label>
      {draft.random_seed != null && <div><div className="flex items-center gap-1"><label className="block text-sm" htmlFor={`${helpPrefix}-random_seed`}>Random seed</label><ParameterHelp name="random_seed" label="Random seed" /></div><input id={`${helpPrefix}-random_seed`} className={cls} type="number" required min={0} max={4294967295} value={draft.random_seed} onChange={e=>change({random_seed:num(e.target.value)})} /></div>}
      <p className="text-xs text-gray-500">Each restart advances the seed by one. Duplicate settings with the same seed reproduce the same search; change the seed to measure run-to-run variation.</p>
      <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={draft.mutation_prob != null} onChange={e=>change({mutation_prob:e.target.checked ? 0.2 : null})} />Override mutation probability</label>
      {draft.mutation_prob != null && <div><div className="flex items-center gap-1"><label className="block text-sm" htmlFor={`${helpPrefix}-mutation_prob`}>Mutation probability</label><ParameterHelp name="mutation_prob" label="Mutation probability" /></div><input id={`${helpPrefix}-mutation_prob`} className={cls} type="number" required min={0} max={1} step={0.01} value={draft.mutation_prob} onChange={e=>change({mutation_prob:num(e.target.value)})} /></div>}
      {draft.solver === 'NSGA-III' && <>
        <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={draft.divisions_outer != null} onChange={e=>change({divisions_outer:e.target.checked ? Math.max(1,(draft.population_size ?? shared.population_size)-1) : null})} />Override reference divisions</label>
        {draft.divisions_outer != null && <div><div className="flex items-center gap-1"><label className="block text-sm" htmlFor={`${helpPrefix}-divisions_outer`}>Reference divisions</label><ParameterHelp name="divisions_outer" label="Reference divisions" /></div><input id={`${helpPrefix}-divisions_outer`} className={cls} type="number" required min={1} value={draft.divisions_outer} onChange={e=>change({divisions_outer:num(e.target.value)})} /></div>}
        <p className="text-xs text-gray-500">NSGA-III derives its actual population from reference directions and rounds it to a multiple of four.</p>
      </>}
      {draft.solver === 'EpsNSGAII' && <>
        <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={draft.epsilons != null} onChange={e=>change({epsilons:e.target.checked ? [0.001,0.001] : null})} />Override epsilon tolerances</label>
        {draft.epsilons?.map((value,index)=><div key={index}><div className="flex items-center gap-1"><label className="block text-sm" htmlFor={`${helpPrefix}-epsilons-${index}`}>{index === 0 ? 'Variance epsilon' : 'Negative-return epsilon'}</label><ParameterHelp name="epsilons" label={index === 0 ? "Variance epsilon" : "Negative-return epsilon"} /></div><input id={`${helpPrefix}-epsilons-${index}`} className={cls} type="number" required min={0.000000001} step="any" value={value} onChange={e=>change({epsilons:draft.epsilons!.map((x,i)=>i===index ? num(e.target.value) : x)})} /></div>)}
      </>}
    </> : <>
      {predictorFields(draft.predictor).map(field=><div key={field.key}><div className="flex items-center gap-1"><label className="block text-sm" htmlFor={`${helpPrefix}-${field.key}`}>{field.label}</label><ParameterHelp name={field.key} label={field.label} /></div><input id={`${helpPrefix}-${field.key}`} className={cls} type="number" required min={field.min} max={field.max} step={field.step === undefined ? 1 : "any"} value={draft[field.key] == null ? '' : Number(draft[field.key])} onChange={e=>change({[field.key]:num(e.target.value)} as Partial<PredictorVariant>)} /></div>)}
      {draft.predictor.startsWith('FuzzyThesis') && <div className="flex items-center gap-1"><label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={draft.trend ?? true} onChange={e=>change({trend:e.target.checked})} />Trend correction</label><ParameterHelp name="trend" label="Trend correction" /></div>}
      {predictorFields(draft.predictor).length === 0 && <p className="text-sm text-gray-500">This baseline has no tunable model parameters. You can give this execution a distinct name.</p>}
      <p className="text-xs text-gray-500">Only parameters supported by this implementation are exposed. Evaluation dates and scored asset/period pairs stay shared. Minimum training history controls model fitting, not the comparison window.</p>
    </>}
  </ConfigurationDrawer>;
}
