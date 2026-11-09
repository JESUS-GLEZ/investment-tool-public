export const SOLVER_META: Record<string, { label: string; description: string }> = {
  "NSGA-II": {
    label: "NSGA-II",
    description: "Non-dominated Sorting Genetic Algorithm II. Fast, widely used Pareto-front solver.",
  },
  "NSGA-III": {
    label: "NSGA-III",
    description: "Reference-point based many-objective extension of NSGA-II.",
  },
  SPEA2: {
    label: "SPEA2",
    description: "Strength Pareto Evolutionary Algorithm 2. Archive-based multi-objective solver.",
  },
  EpsNSGAII: {
    label: "ε-NSGA-II",
    description: "Epsilon-dominance NSGA-II variant. Better diversity on complex fronts.",
  },
  GWASFGA: {
    label: "GWASFGA",
    description: "Goal-based Weighted Aggregation Scalarizing Function GA. Custom multi-objective solver.",
  },
};


export const PREDICTOR_META: Record<string, { label: string; description: string }> = {
  "ZeroReturn": {
    label: "ZeroReturn",
    description: "Unchanged-price baseline. Has no asset ranking signal.",
  },
  "HistoricalMean": {
    label: "HistoricalMean",
    description: "Expanding arithmetic mean of available historical returns.",
  },
  "EWM12-Mean-25": {
    label: "EWM12-Mean-25",
    description: "Research: 25% EWM-12 and 75% expanding historical mean. Fixed weights.",
  },
  "EWM12-Mean-50": {
    label: "EWM12-Mean-50",
    description: "Research: 50% EWM-12 and 50% expanding historical mean. Fixed weights.",
  },
  "EWM12-Mean-75": {
    label: "EWM12-Mean-75",
    description: "Research: 75% EWM-12 and 25% expanding historical mean. Fixed weights.",
  },
  "MomentumRidge": {
    label: "MomentumRidge",
    description: "Research, monthly only: pooled ridge calibrating EWM-6, EWM-12, 12-2 momentum and volatility.",
  },
  "Momentum12-2Ridge": {
    label: "Momentum12-2Ridge",
    description: "Research, monthly only: pooled ridge calibrating 12-2 momentum into next-month return.",
  },
  "FuzzyThesis-fixed": {
    label: "FuzzyThesis-fixed",
    description: "Research: thesis centroid and trend correction, fixed Gaussian widths.",
  },
  "FuzzyThesis-adaptive": {
    label: "FuzzyThesis-adaptive",
    description: "Research: thesis centroid and trend correction, frequency-adaptive widths.",
  },
  "FuzzyThesis-no-trend": {
    label: "FuzzyThesis-no-trend",
    description: "Research: adaptive fuzzy centroid without the trend correction.",
  },
  "PooledRidge": {
    label: "PooledRidge",
    description: "Research control: pooled ridge using recent returns, historical mean and volatility.",
  },
  "FuzzyRidge-fixed": {
    label: "FuzzyRidge-fixed",
    description: "Research: pooled next-return ridge using fixed-width fuzzy memberships.",
  },
  "FuzzyRidge-adaptive": {
    label: "FuzzyRidge-adaptive",
    description: "Research: pooled next-return ridge using adaptive fuzzy memberships.",
  },
  RidgeVARX: {
    label: "RidgeVARX",
    description: "Ridge-regularised Vector AutoRegression with eXogenous inputs. Uses sentiment signals.",
  },
  "RidgeVARX-lag5": {
    label: "RidgeVARX (lag 5)",
    description: "RidgeVARX with 5-period lag window. Captures slower-moving momentum.",
  },
  "EWM-3": {
    label: "EWM-3",
    description: "Exponentially Weighted Mean over 3 periods. Fast momentum baseline.",
  },
  "EWM-6": {
    label: "EWM-6",
    description: "Exponentially Weighted Mean over 6 periods. Medium-term momentum.",
  },
  "EWM-12": {
    label: "EWM-12",
    description: "Exponentially Weighted Mean over 12 periods. Slow, trend-following signal.",
  },
  "SentimentEWM-3": {
    label: "Sentiment EWM-3",
    description: "EWM-3 weighted by news sentiment scores. Requires sentiment data.",
  },
  "SentimentEWM-6": {
    label: "Sentiment EWM-6",
    description: "EWM-6 weighted by news sentiment scores. Requires sentiment data.",
  },
};



/** One addable template per implementation; registry aliases remain valid in saved runs. */
export const PREDICTOR_FAMILIES = {
  ZeroReturn: { ...PREDICTOR_META.ZeroReturn, aliases: [] },
  HistoricalMean: { ...PREDICTOR_META.HistoricalMean, aliases: [] },
  "EWM-6": {
    label: "EWM", description: "Exponentially weighted mean. Configure the span in periods; add multiple instances to compare spans.",
    aliases: ["EWM-3", "EWM-12"],
  },
  "SentimentEWM-3": {
    label: "Sentiment EWM", description: "Sentiment-weighted EWM with a configurable span. Requires sentiment data.",
    aliases: ["SentimentEWM-6"],
  },
  "EWM12-Mean-50": {
    label: "EWM + historical mean", description: "Research: blend EWM with the expanding historical mean. Configure the span and EWM blend weight.",
    aliases: ["EWM12-Mean-25", "EWM12-Mean-75"],
  },
  RidgeVARX: {
    ...PREDICTOR_META.RidgeVARX, description: "Ridge autoregression with sentiment inputs. Configure lag order and regularisation.",
    aliases: ["RidgeVARX-lag5"],
  },
  MomentumRidge: { ...PREDICTOR_META.MomentumRidge, aliases: [] },
  "Momentum12-2Ridge": { ...PREDICTOR_META["Momentum12-2Ridge"], aliases: [] },
  "FuzzyThesis-adaptive": {
    label: "FuzzyThesis", description: "Research: fuzzy centroid with configurable widths and optional trend correction. Set beta to zero for fixed widths.",
    aliases: ["FuzzyThesis-fixed", "FuzzyThesis-no-trend"],
  },
  PooledRidge: { ...PREDICTOR_META.PooledRidge, aliases: [] },
  "FuzzyRidge-adaptive": {
    label: "FuzzyRidge", description: "Research: pooled ridge with configurable fuzzy memberships. Set beta to zero for fixed widths.",
    aliases: ["FuzzyRidge-fixed"],
  },
};

export function predictorFamilyLabel(key: string): string {
  return Object.entries(PREDICTOR_FAMILIES).find(([id, family]) => id === key || (family.aliases as readonly string[]).includes(key))?.[1].label ?? PREDICTOR_META[key]?.label ?? key;
}

/** Registries accept case-insensitive labels; saved presets retain that compatibility. */
export function canonicalComparisonLabels(labels: string[], options: Record<string, unknown>): string[] {
  return [...new Set(labels.map(label => Object.keys(options).find(key => key.toUpperCase() === label.toUpperCase()) ?? label))];
}
