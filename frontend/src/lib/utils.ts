/** Tiny helpers used across the UI. */

import type { RunStatus } from "@/types/api";

interface ApiErrorLike {
  response?: {
    data?:
      | string
      | {
          detail?: unknown;
          message?: unknown;
          error?: unknown;
        };
  };
}

/** CSS class string for a run-status badge. */
export function statusColor(status: RunStatus): string {
  switch (status) {
    case "queued":
      return "bg-yellow-100 text-yellow-800";
    case "running":
      return "bg-blue-100 text-blue-800";
    case "completed":
      return "bg-green-100 text-green-800";
    case "failed":
      return "bg-red-100 text-red-800";
    case "cancelled":
      return "bg-gray-100 text-gray-600";
  }
}

/** Human-readable run-type label. */
export function runTypeLabel(rt: string): string {
  switch (rt) {
    case "execution":
      return "Execution";
    case "strategy_comparison":
      return "Strategy Comparison";
    case "solver_comparison":
      return "Solver Comparison";
    case "prediction_comparison":
      return "Prediction Comparison";
    default:
      return rt;
  }
}

/** Format an ISO datetime string to a locale string. */
export function fmtDate(iso: string | null): string {
  if (!iso) return "\u2014";
  return new Date(iso).toLocaleString();
}

/** Pull the most useful error message out of an API/client error. */
export function getErrorMessage(error: unknown, fallback: string): string {
  if (typeof error === "object" && error !== null) {
    const data = (error as ApiErrorLike).response?.data;
    if (typeof data === "string" && data.trim()) {
      return data;
    }
    if (data && typeof data === "object") {
      if (typeof data.detail === "string" && data.detail.trim()) {
        return data.detail;
      }
      if (Array.isArray(data.detail)) {
        const messages = data.detail.flatMap((item: unknown) => {
          if (!item || typeof item !== "object" || !("msg" in item) || typeof item.msg !== "string" || !item.msg.trim()) return [];
          const message = item.msg.trim().replace(/^Value error,\s*/i, "");
          if (!message) return [];
          const loc = "loc" in item && Array.isArray(item.loc) ? item.loc : [];
          const parts = loc.flatMap((part: unknown) => {
            if (typeof part === "number" && Number.isInteger(part) && part >= 0) return [String(part + 1)];
            if (typeof part !== "string" || ["body", "query", "path", "parameters"].includes(part)) return [];
            const label = part === "strategy_variants" ? "configuration" : part.replace(/_/g, " ");
            return label ? [label] : [];
          });
          const field = parts.join(" / ");
          return [field ? `${field[0].toUpperCase()}${field.slice(1)}: ${message}` : message];
        });
        if (messages.length) return [...new Set(messages)].join("; ");
      }
      if (typeof data.message === "string" && data.message.trim()) {
        return data.message;
      }
      if (typeof data.error === "string" && data.error.trim()) {
        return data.error;
      }
    }
  }

  if (error instanceof Error && error.message.trim()) {
    return error.message;
  }

  return fallback;
}
