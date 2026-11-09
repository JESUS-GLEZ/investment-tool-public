/**
 * Charts for prediction_comparison runs.
 *
 * Renders directly from the `results` list returned by the prediction
 * comparison coordinator using the locked local Plotly runtime.
 */
import { useEffect, useId, useRef, useState } from "react";
import Plotly, { type Data } from "plotly.js-dist-min";

// ── Types mirroring the sanitised coordinator output ────────────────────────

interface PredictorResult {
  config: { label: string };
  metrics: Record<string, number | null>;
  evaluation_dates?: string[];
  predictions_df: Record<string, number[]>;
  actuals_df: Record<string, number[]>;
  per_asset_metrics: Record<string, number[]>; // metric → [value per asset]
}

// ── Colour palette (consistent across charts) ───────────────────────────────
const COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"];

const PLOTLY_CONFIG = { responsive: true, displayModeBar: false, displaylogo: false, showSendToCloud: false };
const BASE_LAYOUT = {
  paper_bgcolor: "white",
  plot_bgcolor: "white",
  font: { family: "Inter, system-ui, sans-serif", size: 12 },
  margin: { l: 55, r: 20, t: 30, b: 60 },
};

// ── Tiny chart wrapper ───────────────────────────────────────────────────────

function Chart({
  data,
  layout,
  height = 320,
  title,
}: {
  data: unknown[];
  layout: Record<string, unknown>;
  height?: number;
  title?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    Plotly.newPlot(element, data as Data[], { ...BASE_LAYOUT, ...layout }, PLOTLY_CONFIG);
    return () => {
      Plotly.purge(element);
    };
  }, [data, layout]);

  return (
    <div className="overflow-hidden rounded-lg border border-gray-200 bg-white">
      {title && (
        <div className="border-b border-gray-100 px-4 py-2">
          <p className="text-sm font-medium text-gray-700">{title}</p>
        </div>
      )}
      <div ref={ref} style={{ height }} className="w-full" />
    </div>
  );
}

// ── Individual chart builders ────────────────────────────────────────────────

/** Bar chart: one group per predictor, one bar per summary metric */
function MetricsBarsChart({ results }: { results: PredictorResult[] }) {
  const METRICS = ["mae", "rmse", "directional_accuracy", "information_coefficient", "hit_rate_top_k"];
  const LABELS: Record<string, string> = {
    mae: "MAE",
    rmse: "RMSE",
    directional_accuracy: "Dir. Accuracy",
    information_coefficient: "IC",
    hit_rate_top_k: "Hit Rate (top-k)",
  };

  const data = results.map((r, i) => ({
    type: "bar",
    name: r.config.label,
    x: METRICS.filter(m => r.metrics[m] != null).map(m => LABELS[m] ?? m),
    y: METRICS.filter(m => r.metrics[m] != null).map(m => +r.metrics[m]!.toFixed(4)),
    marker: { color: COLOURS[i % COLOURS.length] },
  }));

  return (
    <Chart
      title="Prediction Accuracy Metrics by Predictor"
      data={data}
      layout={{ barmode: "group", xaxis: { title: { text: "" } }, yaxis: { title: { text: "Value" } } }}
      height={320}
    />
  );
}

/** Scatter: predicted vs actual returns (sampled across assets & periods) */
function PredVsActualChart({ results }: { results: PredictorResult[] }) {
  const data = results.map((r, i) => {
    const predicted: number[] = [];
    const actual: number[] = [];
    const assets = Object.keys(r.predictions_df);
    for (const asset of assets) {
      const preds = r.predictions_df[asset] ?? [];
      const acts = r.actuals_df[asset] ?? [];
      for (let t = 0; t < preds.length; t++) {
        if (preds[t] != null && acts[t] != null) {
          predicted.push(preds[t]);
          actual.push(acts[t]);
        }
      }
    }
    return {
      type: "scatter",
      mode: "markers",
      name: r.config.label,
      x: predicted,
      y: actual,
      marker: { color: COLOURS[i % COLOURS.length], size: 4, opacity: 0.5 },
    };
  });

  // 45° reference line
  const allVals = (data as Array<{x: number[], y: number[]}>).flatMap(d => [...d.x, ...d.y]);
  const min = allVals.length ? Math.min(...allVals) : -0.01;
  const max = allVals.length ? Math.max(...allVals) : 0.01;
  const refLine = {
    type: "scatter", mode: "lines", name: "Perfect", showlegend: false,
    x: [min, max], y: [min, max],
    line: { color: "#aaa", dash: "dash", width: 1 },
  };

  return (
    <Chart
      title="Predicted vs Actual Returns"
      data={[...data, refLine]}
      layout={{
        xaxis: { title: { text: "Predicted return" }, zeroline: true },
        yaxis: { title: { text: "Actual return" }, zeroline: true },
      }}
      height={360}
    />
  );
}

/** Heatmap: per-asset MAE for each predictor */
function PerAssetMAEChart({ results }: { results: PredictorResult[] }) {
  const assets = Object.keys(results[0]?.predictions_df ?? {});
  // Sort assets by average MAE descending so worst assets appear at top
  const avgMAE = assets.map((_, i) =>
    results.reduce((sum, r) => sum + (r.per_asset_metrics.mae?.[i] ?? 0), 0) / results.length
  );
  const order = avgMAE.map((_, i) => i).sort((a, b) => avgMAE[b] - avgMAE[a]);
  const sortedAssets = order.map(i => assets[i]);

  const z = results.map(r =>
    order.map(i => r.per_asset_metrics.mae?.[i] == null ? null : +r.per_asset_metrics.mae[i].toFixed(4))
  );
  const labels = results.map(r => r.config.label);

  return (
    <Chart
      title="Per-Asset MAE by Predictor"
      data={[{
        type: "heatmap",
        z,
        x: sortedAssets,
        y: labels,
        colorscale: "RdYlGn_r",
        colorbar: { title: { text: "MAE" } },
      }]}
      layout={{
        xaxis: { title: { text: "Asset" }, tickangle: -45 },
        yaxis: { title: { text: "" } },
        margin: { l: 80, r: 20, t: 20, b: 100 },
      }}
      height={Math.max(240, 80 + labels.length * 40)}
    />
  );
}

/** Heatmap: per-asset directional accuracy for each predictor */
function PerAssetDirAccChart({ results }: { results: PredictorResult[] }) {
  const assets = Object.keys(results[0]?.predictions_df ?? {});
  const avgDA = assets.map((_, i) =>
    results.reduce((sum, r) => sum + (r.per_asset_metrics.directional_accuracy?.[i] ?? 0), 0) / results.length
  );
  const order = avgDA.map((_, i) => i).sort((a, b) => avgDA[b] - avgDA[a]);
  const sortedAssets = order.map(i => assets[i]);

  const z = results.map(r =>
    order.map(i => r.per_asset_metrics.directional_accuracy?.[i] == null ? null : +r.per_asset_metrics.directional_accuracy[i].toFixed(4))
  );
  const labels = results.map(r => r.config.label);

  return (
    <Chart
      title="Per-Asset Directional Accuracy by Predictor"
      data={[{
        type: "heatmap",
        z,
        x: sortedAssets,
        y: labels,
        colorscale: "RdYlGn",
        zmid: 0.5,
        colorbar: { title: { text: "Dir. Acc." } },
      }]}
      layout={{
        xaxis: { title: { text: "Asset" }, tickangle: -45 },
        yaxis: { title: { text: "" } },
        margin: { l: 80, r: 20, t: 20, b: 100 },
      }}
      height={Math.max(240, 80 + labels.length * 40)}
    />
  );
}

/** Radar / bar chart comparing all metrics normalised */
function WallTimeChart({ results }: { results: PredictorResult[] }) {
  const data = [{
    type: "bar",
    x: results.map(r => r.config.label),
    y: results.map(r => +(r.metrics.wall_time_s ?? r.metrics.wall_time ?? 0).toFixed(3)),
    marker: { color: results.map((_, i) => COLOURS[i % COLOURS.length]) },
    text: results.map(r => `${(r.metrics.wall_time_s ?? r.metrics.wall_time ?? 0).toFixed(3)}s`),
    textposition: "auto",
  }];

  return (
    <Chart
      title="Computation Time per Predictor (seconds)"
      data={data}
      layout={{ xaxis: { title: { text: "Predictor" } }, yaxis: { title: { text: "Seconds" } } }}
      height={260}
    />
  );
}

// ── Main export ──────────────────────────────────────────────────────────────

export default function PredictionComparisonCharts({
  results,
}: {
  results: unknown[];
}) {
  const id = useId();
  const [chart, setChart] = useState("");
  const typed = results as PredictorResult[];
  if (!typed.length) return null;

  return (
    <div className="space-y-4">
      {typed.some(r => r.metrics.scored_pairs != null) && (
        <div className="rounded-lg border border-gray-200 bg-white p-4 text-sm">
          <h2 className="font-semibold text-gray-700">Evaluation coverage</h2>
          {typed[0].evaluation_dates?.length ? (
            <p>Forecast target periods: {typed[0].evaluation_dates[0].slice(0, 10)} to {typed[0].evaluation_dates.at(-1)!.slice(0, 10)}. Outcomes are close-to-close price returns.</p>
          ) : null}
          <p>Accuracy metrics use the same asset-period pairs for every predictor. Missing forecasts or outcomes reduce coverage; inspect forecast_pairs.csv before interpreting rankings.</p>
          {typed.map(r => (
            <p key={r.config.label}>
              {r.config.label}: {r.metrics.scored_pairs ?? 0}/{r.metrics.eligible_pairs ?? 0} pairs scored;
              {" "}{r.metrics.scored_periods ?? 0}/{r.metrics.evaluation_periods ?? 0} periods;
              {" "}{r.metrics.failed_periods ?? 0} failed periods.
            </p>
          ))}
        </div>
      )}
      <div>
        <label htmlFor={id} className="block text-sm font-medium text-gray-700">Prediction diagnostic</label>
        <select id={id} value={chart} onChange={event => setChart(event.target.value)} className="mt-1 block w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-sm">
          <option value="">Choose one diagnostic chart</option>
          <option value="metrics">Prediction accuracy metrics</option>
          <option value="scatter">Predicted vs actual returns</option>
          <option value="mae">Per-asset mean absolute error</option>
          <option value="direction">Per-asset directional accuracy</option>
          <option value="time">Computation time</option>
        </select>
      </div>
      {chart === "metrics" && <MetricsBarsChart results={typed} />}
      {chart === "scatter" && <PredVsActualChart results={typed} />}
      {chart === "mae" && <PerAssetMAEChart results={typed} />}
      {chart === "direction" && <PerAssetDirAccChart results={typed} />}
      {chart === "time" && <WallTimeChart results={typed} />}
    </div>
  );
}
