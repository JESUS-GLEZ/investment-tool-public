/**
 * StrategyPicker — checkbox-card grid for selecting one or more strategies.
 *
 * Each strategy is a clickable card; selected ones are highlighted in blue.
 * Replaces the awkward <select multiple> with Ctrl+click UX.
 */
import type { StrategyType } from "@/types/api";

const STRATEGY_META: Record<StrategyType, { label: string; description: string }> = {
  combined: {
    label: "Combined",
    description: "Blends momentum, risk parity, and optimization signals.",
  },
  optimization: {
    label: "Optimization",
    description: "Multi-objective genetic / Pareto-front portfolio optimizer.",
  },
  momentum_risk_parity: {
    label: "Momentum Risk Parity",
    description: "Weights assets by inverse volatility with momentum tilt.",
  },
  buy_and_hold: {
    label: "Buy & Hold",
    description: "Equal-weight static allocation; benchmark baseline.",
  },
  sentiment_ewm: {
    label: "Sentiment EWM",
    description: "Exponentially weighted momentum driven by news sentiment.",
  },
};

interface Props {
  selected: StrategyType[];
  onChange: (next: StrategyType[]) => void;
  /** If true only one strategy can be selected (NewRunPage) */
  single?: boolean;
}

export default function StrategyPicker({ selected, onChange, single = false }: Props) {
  const toggle = (s: StrategyType) => {
    if (single) {
      onChange([s]);
      return;
    }
    if (selected.includes(s)) {
      // keep at least one selected
      if (selected.length === 1) return;
      onChange(selected.filter((x) => x !== s));
    } else {
      onChange([...selected, s]);
    }
  };

  return (
    <div className="space-y-2">
      {(Object.keys(STRATEGY_META) as StrategyType[]).map((s) => {
        const isSelected = selected.includes(s);
        return (
          <button
            key={s}
            type="button"
            aria-pressed={isSelected}
            onClick={() => toggle(s)}
            className={`w-full text-left rounded-lg border px-4 py-3 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-400 ${
              isSelected
                ? "border-blue-500 bg-blue-50 ring-1 ring-blue-400"
                : "border-gray-200 bg-white hover:border-gray-300 hover:bg-gray-50"
            }`}
          >
            <div className="flex items-center gap-3">
              <span
                className={`flex h-4 w-4 shrink-0 items-center justify-center rounded ${
                  single
                    ? `rounded-full border-2 ${isSelected ? "border-blue-500 bg-blue-500" : "border-gray-300"}`
                    : `rounded border-2 ${isSelected ? "border-blue-500 bg-blue-500" : "border-gray-300"}`
                }`}
              >
                {isSelected && (
                  <svg className="h-2.5 w-2.5 text-white" viewBox="0 0 10 10" fill="none">
                    {single ? (
                      <circle cx="5" cy="5" r="2.5" fill="currentColor" />
                    ) : (
                      <path
                        d="M1.5 5.5L4 8L8.5 2.5"
                        stroke="currentColor"
                        strokeWidth="1.5"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    )}
                  </svg>
                )}
              </span>
              <div>
                <p className={`text-sm font-medium ${isSelected ? "text-blue-800" : "text-gray-800"}`}>
                  {STRATEGY_META[s].label}
                </p>
                <p className="text-xs text-gray-500 mt-0.5">{STRATEGY_META[s].description}</p>
              </div>
            </div>
          </button>
        );
      })}
      {!single && (
        <p className="text-xs text-gray-400 pt-1">
          {selected.length} of {Object.keys(STRATEGY_META).length} selected — click to toggle
        </p>
      )}
    </div>
  );
}
