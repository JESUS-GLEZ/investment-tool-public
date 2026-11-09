import ComparisonCatalogue from "./ComparisonCatalogue";

import { SOLVER_META } from "@/lib/comparisonCatalogue";

export default function SolverPicker({selected, onChange}: {selected: string[]; onChange: (next: string[]) => void}) {
  return <ComparisonCatalogue kind="solver" options={SOLVER_META} selected={selected} onChange={onChange} />;
}
