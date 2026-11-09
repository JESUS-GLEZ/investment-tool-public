import { expect, it } from "vitest";
import { datasetLabel } from "@/lib/localDatasets";
import { effectiveExperimentData, initialExperimentData } from "@/lib/experimentData";

const summary = {frequency: null, start_date: null, end_date: null, stock_count: null};
it("makes research dataset names readable without changing file identities", () => {
  expect(datasetLabel({...summary, file: "input_monthly_validation_2018_filtered_news_2016_2018.pkl"})).toBe("Validation 2018 · filtered news · history 2016–2018 · monthly");
  expect(datasetLabel({...summary, file: "input_monthly_broad_opportunity_2018_alpaca_research_2016_2018_prices.pkl"})).toBe("Broad opportunity 2018 · prices 2016–2018 · monthly");
  expect(datasetLabel({...summary, file: "my_saved_prices.pkl"})).toBe("My saved prices");
});

it("distinguishes price-only and execution-enriched copies", () => {
  const original = {...summary, file: "input_data_monthly_2017-01-01_to_2018-09-30_6stocks_2018-10-02.pkl", frequency: "monthly", stock_count: 6, start_date: "2017-01-01", end_date: "2018-09-30"};
  const enriched = {...original, file: original.file.replace(".pkl", "_execution_enriched.pkl")};
  expect(datasetLabel(original)).toContain("6 stocks");
  expect(datasetLabel(original)).not.toEqual(datasetLabel(enriched));
  expect(datasetLabel(enriched)).toContain("with execution prices");
});

it("keeps dataset-recorded inputs authoritative after loading a preset for the same file", () => {
  const local = {...initialExperimentData, data_source: "local" as const, input_file: "prices.pkl"};
  const info = {file: "prices.pkl", frequency: "weekly" as const, asset_config_file: "recorded.json", start_date: null, end_date: null, stocks: [], crypto: [], stock_count: 0, crypto_count: 0};
  expect(effectiveExperimentData(local, info)).toMatchObject({frequency: "weekly", asset_config_file: "recorded.json"});
  expect(effectiveExperimentData({...local, input_file: "other.pkl"}, info)).toMatchObject({frequency: "monthly", asset_config_file: initialExperimentData.asset_config_file});
  expect(effectiveExperimentData({...local, data_source: "alpaca"}, info)).toMatchObject({input_file: null, asset_config_file: initialExperimentData.asset_config_file});
});
