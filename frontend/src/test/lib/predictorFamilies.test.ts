import { expect, it } from "vitest";
import { PREDICTOR_FAMILIES, PREDICTOR_META, predictorFamilyLabel } from "@/lib/comparisonCatalogue";
import { createModelVariant, hydrateModelVariant, modelVariantError } from "@/lib/modelVariants";
import type { PredictorVariant } from "@/types/api";

it("offers one template per family while recognising every saved registry identifier", () => {
  const supported = Object.entries(PREDICTOR_FAMILIES).flatMap(([key, family]) => [key, ...family.aliases]);
  expect([...supported].sort()).toEqual(Object.keys(PREDICTOR_META).sort());
  expect(new Set(supported).size).toBe(supported.length);
  expect(predictorFamilyLabel("EWM-12")).toBe("EWM");
  expect(predictorFamilyLabel("SentimentEWM-6")).toBe("Sentiment EWM");
});

it.each([
  ["EWM-3", {span: 3}], ["EWM-12", {span: 12}],
  ["SentimentEWM-6", {span: 6}],
  ["EWM12-Mean-25", {span: 12, momentum_weight: 0.25}],
  ["EWM12-Mean-75", {span: 12, momentum_weight: 0.75}],
  ["RidgeVARX-lag5", {lag_order: 5}],
  ["FuzzyThesis-fixed", {beta: 0, trend: true}],
  ["FuzzyThesis-no-trend", {beta: 2, trend: false}],
  ["FuzzyRidge-fixed", {beta: 0}],
] as const)("preserves legacy %s parameters and display name", (key, settings) => {
  const legacy = createModelVariant(key, [], false, true);
  expect(legacy).toMatchObject({predictor: key, name: PREDICTOR_META[key].label, ...settings});
  expect(hydrateModelVariant({name: "Saved experiment", predictor: key})).toMatchObject({name: "Saved experiment", predictor: key, ...settings});
  expect(modelVariantError(legacy, [], "monthly")).toBeUndefined();
});

it("creates independent family instances without changing explicit saved overrides", () => {
  const first = createModelVariant("EWM-6", [], false) as PredictorVariant;
  const second = createModelVariant("EWM-6", [first], false) as PredictorVariant;
  second.span = 18;
  expect(first).toMatchObject({name: "EWM", span: 6});
  expect(second).toMatchObject({name: "EWM 2", span: 18});
  expect(hydrateModelVariant({name: "Old custom", predictor: "EWM-3", span: 9})).toMatchObject({name: "Old custom", predictor: "EWM-3", span: 9});
});
