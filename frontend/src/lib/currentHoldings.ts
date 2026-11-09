import type { HoldingInput } from "@/types/api";

export function currentHoldingsError(holdings: HoldingInput[], symbols: string[]): string | undefined {
  if (holdings.some(h => !symbols.includes(h.symbol))) return "Choose each holding from the available universe and dataset.";
  if (new Set(holdings.map(h => h.symbol)).size !== holdings.length) return "Holding symbols must be unique.";
  if (holdings.some(h => !Number.isFinite(h.quantity) || h.quantity <= 0)) return "Holding quantities must be finite and positive.";
}
