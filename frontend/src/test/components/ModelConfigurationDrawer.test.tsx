import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import ModelConfigurationDrawer from "@/components/ui/ModelConfigurationDrawer";
import { PREDICTOR_META } from "@/lib/comparisonCatalogue";
import { createModelVariant } from "@/lib/modelVariants";

afterEach(cleanup);
it.each(Object.keys(PREDICTOR_META))("%s default parameters pass native validation and save",key=>{
  const onSave=vi.fn();
  render(<ModelConfigurationDrawer value={createModelVariant(key,[],false)} others={[]} frequency="monthly" shared={{population_size:100,n_generations:500,n_restarts:3}} focusFallbackId="none" onClose={vi.fn()} onSave={onSave} />);
  const dialog=screen.getByRole("dialog");
  for (const input of dialog.querySelectorAll('input')) expect(input.checkValidity()).toBe(true);
  fireEvent.click(screen.getByRole("button",{name:"Save configuration"}));
  expect(onSave).toHaveBeenCalledTimes(1);
  if (key.startsWith("FuzzyRidge")) {
    expect(screen.queryByLabelText("Trend correction")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Centroid grid points")).not.toBeInTheDocument();
  }
});
