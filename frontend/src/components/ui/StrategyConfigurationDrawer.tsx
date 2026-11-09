import { useState } from "react";
import type { StrategyVariant } from "@/types/api";
import { strategyError, strategyLabel } from "@/lib/strategyVariants";
import StrategyVariantEditor from "./StrategyVariantEditor";
import ConfigurationDrawer from "./ConfigurationDrawer";

export default function StrategyConfigurationDrawer({value, others, frequency, defaultSearchSeconds, onSave, onClose, focusFallbackId}: {
  focusFallbackId: string; value: StrategyVariant; others: StrategyVariant[]; frequency: string; defaultSearchSeconds: number | null;
  onSave: (v: StrategyVariant) => void; onClose: () => void;
}) {
  const [draft, setDraft] = useState({...value});
  const [error, setError] = useState<string>();
  return <ConfigurationDrawer title={`Configure ${strategyLabel(draft.strategy_type)}`} error={error} focusFallbackId={focusFallbackId} onClose={onClose} onSave={() => {
    const issue = strategyError(draft, others, frequency); setError(issue);
    if (!issue) onSave({...draft, name: draft.name.trim()});
  }}><StrategyVariantEditor variant={draft} defaultSearchSeconds={defaultSearchSeconds} onChange={value => {setDraft(value); setError(undefined);}} /></ConfigurationDrawer>;
}
