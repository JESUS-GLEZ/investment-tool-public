import ComparisonCatalogue from "./ComparisonCatalogue";

import { PREDICTOR_META } from "@/lib/comparisonCatalogue";

export default function PredictorPicker({selected, onChange}: {selected: string[]; onChange: (next: string[]) => void}) {
  return <ComparisonCatalogue kind="predictor" options={PREDICTOR_META} selected={selected} onChange={onChange} />;
}
