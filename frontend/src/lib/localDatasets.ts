import type { LocalDataFileSummary } from "@/types/api";

export function datasetTitle(file: string): string {
  let title = file.replace(/\.pkl$/, "").replace(/^input_(?:data_)?(?:daily|weekly|monthly)_/, "")
    .replace(/_/g, " ").replace(/^./, letter => letter.toUpperCase());
  const validation = file.match(/^input_(?:daily|weekly|monthly)_validation_(\d{4})_filtered_news_(\d{4})_(\d{4})\.pkl$/);
  if (validation) title = `Validation ${validation[1]} · filtered news · history ${validation[2]}–${validation[3]}`;
  const opportunity = file.match(/^input_(?:daily|weekly|monthly)_(broad|balanced)_opportunity_(\d{4})_(?:alpaca_)?(?:research_)?(\d{4})_(\d{4})_prices\.pkl$/);
  if (opportunity) title = `${opportunity[1] === "broad" ? "Broad" : "Balanced"} opportunity ${opportunity[2]} · prices ${opportunity[3]}–${opportunity[4]}`;
  const snapshot = file.match(/^input_data_(daily|weekly|monthly)_(\d{4})-\d{2}-\d{2}_to_(\d{4})-\d{2}-\d{2}_\d+stocks_(\d{4}-\d{2}-\d{2})(_execution_enriched)?\.pkl$/);
  if (snapshot) title = `${snapshot[1][0].toUpperCase()}${snapshot[1].slice(1)} snapshot ${snapshot[2]}–${snapshot[3]} · saved ${snapshot[4]}${snapshot[5] ? " · execution prices" : ""}`;
  return title;
}

export function datasetLabel(summary: LocalDataFileSummary): string {
  const file = summary.file;
  if (summary.title) return `${summary.title} · ${summary.stock_count ?? "?"} stocks${summary.crypto_count ? ` · ${summary.crypto_count} crypto` : ""}${summary.features?.execution_quotes ? " · execution quotes" : ""}${summary.features?.news ? " · news" : ""}`;
  const title = datasetTitle(file);
  const standard = /^input_data_(daily|weekly|monthly)_/.test(file);
  const parts: string[] = standard ? [] : [title];
  const frequency = summary.frequency ?? file.match(/^input_(?:data_)?(daily|weekly|monthly)_/)?.[1];
  if (frequency) parts.push(frequency);
  if (summary.stock_count !== null) parts.push(`${summary.stock_count} stocks`);
  if (summary.start_date || summary.end_date) parts.push(`${summary.start_date ?? "?"} → ${summary.end_date ?? "?"}`);
  const generated = file.match(/_(\d{4}-\d{2}-\d{2})(?:_execution_enriched)?\.pkl$/)?.[1];
  if (generated) parts.push(`saved ${generated}`);
  if (file.endsWith("_execution_enriched.pkl")) parts.push("with execution prices");
  return parts.length ? parts.join(" · ") : title;
}


export function datasetFeature(value: boolean | null | undefined): string {
  return value === true ? "Present" : value === false ? "Absent" : "Unknown";
}
