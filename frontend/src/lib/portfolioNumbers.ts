export type PortfolioNumberKey = "initial_cash" | "transaction_cost_rate" | "risk_free_rate";
export type PortfolioNumbers = Partial<Record<PortfolioNumberKey, number>>;

/** Match the existing form bounds without converting an empty field to zero. */
export function portfolioNumberErrors(values: PortfolioNumbers, allowZeroCash = false): Partial<Record<PortfolioNumberKey, string>> {
  const errors: Partial<Record<PortfolioNumberKey, string>> = {};
  for (const [key, label, min, max] of [
    ["initial_cash", "Initial cash", allowZeroCash ? 0 : 1, Infinity],
    ["transaction_cost_rate", "Transaction cost", 0, 0.1],
    ["risk_free_rate", "Risk-free rate", 0, 1],
  ] as const) {
    const value = values[key];
    if (value === undefined || !Number.isFinite(value)) {
      errors[key] = `${label} requires a finite number.`;
    } else if (value < min || value > max) {
      errors[key] = max === Infinity ? `${label} must be at least ${min}.` : `${label} must be between ${min} and ${max}.`;
    }
  }
  return errors;
}
