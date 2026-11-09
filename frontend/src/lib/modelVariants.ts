import type { PredictorVariant, SolverVariant } from "@/types/api";
import { SOLVER_META, PREDICTOR_META, predictorFamilyLabel } from "./comparisonCatalogue";

export type ModelVariant = SolverVariant | PredictorVariant;
export const modelKey = (v: ModelVariant) => "solver" in v ? v.solver : v.predictor;
export function uniqueModelName(key: string, entries: ModelVariant[], solver: boolean, legacy = false): string {
  const label = solver ? SOLVER_META[key]?.label ?? key : legacy ? PREDICTOR_META[key]?.label ?? key : predictorFamilyLabel(key);
  let name = label, suffix = 2;
  while (entries.some(v => v.name.trim().toLowerCase() === name.toLowerCase())) name = `${label} ${suffix++}`;
  return name;
}
export type PredictorField = {key: keyof PredictorVariant; label: string; min: number; max?: number; step?: number};
export function predictorFields(key: string): PredictorField[] {
  const fields: PredictorField[] = [];
  if (/^(EWM-|SentimentEWM-|EWM12-Mean-)/.test(key)) fields.push({key:"span",label:"EWM span (periods)",min:1});
  if (key.startsWith("EWM12-Mean-")) fields.push({key:"momentum_weight",label:"EWM blend weight (0–1)",min:0,max:1,step:0.05});
  if (key.startsWith("RidgeVARX")) fields.push({key:"lag_order",label:"Lag order",min:1});
  if (key.startsWith("RidgeVARX") || key.includes("Ridge")) fields.push({key:"alpha",label:"Ridge regularisation (alpha)",min:Number.MIN_VALUE,step:0.1});
  if (key.includes("Ridge") && !key.startsWith("RidgeVARX")) fields.push({key:"min_train_history",label:"Minimum training history (periods)",min:key.startsWith("Momentum") ? 12 : 1});
  if (key.startsWith("Fuzzy") || key === "PooledRidge") fields.push({key:"lags",label:"Return lags",min:1});
  if (key.startsWith("Fuzzy")) fields.push({key:"partitions",label:"Fuzzy partitions",min:2},{key:"sigma",label:"Gaussian width (sigma)",min:Number.MIN_VALUE,step:0.1},{key:"beta",label:"Width adaptation (beta)",min:0,step:0.1});
  if (key.startsWith("FuzzyThesis")) fields.push({key:"grid_points",label:"Centroid grid points",min:3});
  return fields;
}
export function createModelVariant(key: string, entries: ModelVariant[], solver: boolean, legacy = false): ModelVariant {
  const name = uniqueModelName(key, entries, solver, legacy);
  if (solver) return {name, solver:key, random_seed:legacy ? null : 0};
  const defaults: Partial<PredictorVariant> = {span:key.startsWith("EWM12-Mean-") ? 12 : Number(key.match(/-(\d+)$/)?.[1] ?? 6), lag_order:key.endsWith("lag5") ? 5 : 3, alpha:10,
    min_train_history:12, momentum_weight:Number(key.match(/Mean-(\d+)$/)?.[1] ?? 50)/100,
    lags:3, partitions:9, sigma:10, beta:key.endsWith("-fixed") ? 0 : 2, grid_points:513};
  const variant: PredictorVariant = {name,predictor:key};
  for (const {key:field} of predictorFields(key)) Object.assign(variant,{[field]:defaults[field]});
  if (key.startsWith("FuzzyThesis")) variant.trend = !key.endsWith("-no-trend");
  return variant;
}
/** Display/retain constructor defaults when named API presets omit optional overrides. */
export function hydrateModelVariant(v: ModelVariant): ModelVariant {
  if ("solver" in v) return v;
  return {...createModelVariant(v.predictor,[],false), ...Object.fromEntries(Object.entries(v).filter(([,value])=>value != null))} as PredictorVariant;
}
export function modelVariantError(v: ModelVariant, others: ModelVariant[], frequency: string): string | undefined {
  const name = v.name.trim();
  if (!name || name.length > 80 || /[\\/:*?"<>|]/.test(name) || [...name].some(c=>c.charCodeAt(0)<32) || [".",".."].includes(name)) return "Use a name without path characters (1–80 characters).";
  if (others.some(x => x.name.trim().toLowerCase() === name.toLowerCase())) return "Configuration names must be unique.";
  const key = modelKey(v);
  if (!("solver" in v ? SOLVER_META : PREDICTOR_META)[key]) return "This method is unavailable. Remove it or update the preset.";
  for (const [field,value] of Object.entries(v)) if (typeof value === "number" && !Number.isFinite(value)) return `${field} must be finite.`;
  if ("solver" in v) {
    for (const [field,min] of [["population_size",10],["n_generations",1],["n_restarts",1],["random_seed",0],["divisions_outer",1]] as const) {
      const value = v[field]; if (value != null && (!Number.isInteger(value) || value < min)) return `${field} must be an integer >= ${min}.`;
    }
    if (v.random_seed != null && v.random_seed > 4294967295) return "Random seed must fit an unsigned 32-bit integer.";
    if (v.mutation_prob != null && (v.mutation_prob < 0 || v.mutation_prob > 1)) return "Mutation probability must be between zero and one.";
    if (v.epsilons && (v.epsilons.length !== 2 || v.epsilons.some(x => !Number.isFinite(x) || x <= 0))) return "Provide two positive epsilon values.";
  } else {
    if (["MomentumRidge","Momentum12-2Ridge"].includes(key) && frequency !== "monthly") return "MomentumRidge and Momentum12-2Ridge require monthly data.";
    for (const field of predictorFields(key)) {
      const value = v[field.key];
      if (value != null && (typeof value !== "number" || value < field.min || (field.max !== undefined && value > field.max) || (field.step === undefined && !Number.isInteger(value)))) return `Check ${field.label}.`;
    }
    if ((key === "PooledRidge" || key.startsWith("FuzzyRidge")) && (v.min_train_history ?? 12) < (v.lags ?? 3)) return "Minimum training history must cover the return lags.";
  }
}
export function modelVariantSummary(value: ModelVariant, shared: {population_size:number;n_generations:number;n_restarts:number}): string {
  const v = hydrateModelVariant(value);
  if ("solver" in v) return `Population ${v.population_size ?? shared.population_size} · ${v.n_generations ?? shared.n_generations} evaluations × ${v.n_restarts ?? shared.n_restarts} restarts · seed ${v.random_seed ?? "unfixed"}${v.mutation_prob != null ? ` · mutation ${v.mutation_prob}` : ""}${v.divisions_outer != null ? ` · divisions ${v.divisions_outer}` : ""}${v.epsilons ? ` · epsilons ${v.epsilons.join(", ")}` : ""}`;
  return predictorFields(v.predictor).filter(field => v[field.key] != null).map(field => `${field.label}: ${v[field.key]}`).concat(v.trend == null ? [] : [`Trend: ${v.trend ? "on" : "off"}`]).join(" · ") || "Baseline with no tunable model parameters";
}
