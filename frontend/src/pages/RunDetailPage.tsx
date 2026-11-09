import { RunCampaignAssignment } from "@/components/ui/Campaigns";
/**
 * Run Detail page — status, progress, grouped metrics, Plotly charts,
 * and (for comparison runs) a side-by-side strategy metrics table.
 */
import { useParams, useNavigate } from "react-router-dom";
import { useState, useEffect, lazy, Suspense, type ReactNode } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { getRun, listRunFiles, downloadRunFile, downloadAllRunFiles, cancelRun, deleteRun } from "@/api/endpoints";
import type { RunOutputFile } from "@/api/endpoints";
import type { RunDetail, RunSummary } from "@/types/api";
import { statusColor, runTypeLabel, fmtDate, getErrorMessage } from "@/lib/utils";
import { formatMetricValue, metricLabel, comparisonMetricLabel, metricDescription } from "@/lib/metrics";
import { useProgress } from "@/hooks";
import WorkerHealth from "@/components/ui/WorkerHealth";
import RunProgress from "@/components/ui/RunProgress";
import MetricsCards from "@/components/ui/MetricsCards";
import RunComparisonResults from "@/components/ui/RunComparisonResults";
import SavedChartBrowser from "@/components/ui/SavedChartBrowser";
import { savedCandidates } from "@/lib/savedComparison";
import SelectedPortfolio from "@/components/ui/SelectedPortfolio";
import ExecutionStrategySummary from "@/components/ui/ExecutionStrategySummary";
import { libraryRunType } from "@/lib/runLibrary";

const PredictionComparisonCharts = lazy(() => import("@/components/ui/PredictionComparisonCharts"));

// ── Comparison metrics table ────────────────────────────────────────────────

/**
 * For comparison runs the backend returns a dict keyed by strategy/solver/predictor
 * name, each value being a metrics dict.
 */
function ComparisonMetricsTable({
  data,
  definitions,
}: {
  data: Record<string, Record<string, unknown>>;
  definitions?: Record<string, Record<string, unknown>>;
}) {
  const names = Object.keys(data);
  if (names.length === 0) return null;

  // Collect all metric keys that appear across all entries (scalar only)
  const allKeys = Array.from(
    new Set(
      names.flatMap((n) =>
        Object.entries(data[n])
          .filter(([, v]) => typeof v !== "object" || v === null)
          .map(([k]) => k),
      ),
    ),
  );

  function label(key: string): string {
    return comparisonMetricLabel(key, names.map((n) => definitions?.[n]));
  }

  return (
    <div className="overflow-x-auto rounded-lg border border-gray-200 bg-white">
      <table className="min-w-full text-sm">
        <thead>
          <tr className="bg-gray-50 border-b border-gray-200">
            <th className="px-4 py-2 text-left text-xs font-semibold uppercase tracking-wide text-gray-500 sticky left-0 bg-gray-50">
              Metric
            </th>
            {names.map((n) => (
              <th
                key={n}
                className="px-4 py-2 text-right text-xs font-semibold uppercase tracking-wide text-gray-700 whitespace-nowrap"
              >
                {n.replace(/_/g, " ")}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100">
          {allKeys.map((key) => (
            <tr key={key} className="hover:bg-gray-50">
              <td className="px-4 py-2 text-xs font-medium text-gray-500 sticky left-0 bg-white whitespace-nowrap" title={metricDescription(key)}>
                {label(key)}
              </td>
              {names.map((n) => (
                <td
                  key={n}
                  className="px-4 py-2 text-right font-mono text-xs text-gray-900 whitespace-nowrap"
                  title={key === "annualized_return" ? metricLabel(key, definitions?.[n]) : undefined}
                >
                  {formatMetricValue(key, data[n][key])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ExpandableSection({
  title,
  summary,
  defaultOpen = false,
  renderContent,
}: {
  title: string;
  summary?: string;
  defaultOpen?: boolean;
  renderContent: () => ReactNode;
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <section className="overflow-hidden rounded-md border border-gray-200 bg-white">
      <div className="flex items-center justify-between gap-4 px-4 py-3">
        <div>
          <h2 className="text-sm font-medium text-gray-700">{title}</h2>
          {summary && <p className="mt-0.5 text-xs text-gray-500">{summary}</p>}
        </div>
        <button
          type="button"
          onClick={() => setIsOpen((open) => !open)}
          aria-expanded={isOpen}
          className="rounded border border-gray-300 bg-white px-2.5 py-1 text-xs font-medium text-gray-600 hover:bg-gray-50 hover:text-gray-900"
        >
          {isOpen ? "Hide" : "Show"}
        </button>
      </div>
      {isOpen && <div className="border-t border-gray-200 px-4 py-4">{renderContent()}</div>}
    </section>
  );
}

// ── Output files panel ──────────────────────────────────────────────────────

const FILE_ICONS: Record<string, string> = {
  "text/csv": "📄",
  "application/json": "{}",
  "text/html": "🌐",
  "image/png": "🖼",
};

function fileIcon(mime: string): string {
  return FILE_ICONS[mime] ?? "📁";
}

function fmtBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function RunFilesPanel({ runId, showTitle = true }: { runId: string; showTitle?: boolean }) {
  const { data: files, isLoading, error, isFetching, refetch } = useQuery({
    queryKey: ["runFiles", runId],
    queryFn: () => listRunFiles(runId),
  });
  const [downloadingAll, setDownloadingAll] = useState(false);
  const [downloadingFile, setDownloadingFile] = useState<string | null>(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const downloadPending = downloadingAll || downloadingFile !== null;

  if (isLoading) return <p role="status" className="text-sm text-gray-500">Loading output files…</p>;

  const fetchError = error ? (
    <div role="alert" className="mb-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">
      <p>{getErrorMessage(error, "Could not load output files.")}</p>
      <button type="button" disabled={isFetching} onClick={() => void refetch()} className="mt-2 underline disabled:opacity-50">
        {isFetching ? "Retrying files…" : "Retry file list"}
      </button>
    </div>
  ) : null;

  if (error && !files) return fetchError;

  if (!files || files.length === 0) {
    return (
      <div>
        {showTitle && <h2 className="mb-2 text-sm font-semibold text-gray-700">Output Files</h2>}
        {fetchError}
        <p className="text-xs text-gray-400">No output files found for this run.</p>
      </div>
    );
  }

  // Group by subdirectory
  const groups: Record<string, RunOutputFile[]> = {};
  for (const f of files) {
    const parts = f.path.split("/");
    const group = parts.length > 1 ? parts[0] : "root";
    (groups[group] ??= []).push(f);
  }

  async function handleDownloadAll() {
    if (downloadPending) return;
    setDownloadError(null);
    setDownloadingAll(true);
    try {
      await downloadAllRunFiles(runId);
    } catch (error) {
      setDownloadError(getErrorMessage(error, "Could not download the output archive."));
    } finally {
      setDownloadingAll(false);
    }
  }

  async function handleDownloadFile(filePath: string) {
    if (downloadPending) return;
    setDownloadError(null);
    setDownloadingFile(filePath);
    try {
      await downloadRunFile(runId, filePath);
    } catch (error) {
      setDownloadError(getErrorMessage(error, "Could not download the output file."));
    } finally {
      setDownloadingFile(null);
    }
  }

  return (
    <div>
      {fetchError}
      {downloadError && (
        <p role="alert" className="mb-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">
          {downloadError} Try the download again.
        </p>
      )}
      <div className={`mb-3 flex items-center justify-between ${showTitle ? "" : "mt-0"}`}>
        {showTitle ? <h2 className="text-sm font-semibold text-gray-700">Output Files</h2> : <div />}
        <button
          onClick={handleDownloadAll}
          disabled={downloadPending}
          className="rounded border border-gray-300 bg-white px-2.5 py-1 text-xs font-medium text-gray-600 hover:bg-gray-50 hover:text-gray-900 disabled:opacity-50"
        >
          {downloadingAll ? "Preparing…" : "Download All (.zip)"}
        </button>
      </div>
      <div className="space-y-3">
        {Object.entries(groups).map(([group, groupFiles]) => (
          <div key={group} className="rounded-md border border-gray-200 overflow-hidden">
            {group !== "root" && (
              <div className="bg-gray-50 px-3 py-1.5 text-xs font-medium text-gray-500 uppercase tracking-wide">
                {group}/
              </div>
            )}
            <ul className="divide-y divide-gray-100">
              {groupFiles.map((f) => (
                <li key={f.path} className="flex items-center justify-between px-3 py-2">
                  <span className="flex items-center gap-2 text-sm text-gray-700 min-w-0">
                    <span className="text-base">{fileIcon(f.type)}</span>
                    <span className="truncate">{f.path.split("/").pop()}</span>
                    <span className="text-xs text-gray-400 shrink-0">{fmtBytes(f.size)}</span>
                  </span>
                  <button
                    onClick={() => handleDownloadFile(f.path)}
                    disabled={downloadPending}
                    aria-label={`Download ${f.path}`}
                    className="ml-3 shrink-0 text-xs text-blue-600 hover:text-blue-800 hover:underline disabled:opacity-50"
                  >
                    {downloadingFile === f.path ? "…" : "Download"}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Main page ───────────────────────────────────────────────────────────────

const COMPARISON_TYPES = new Set([
  "strategy_comparison",
  "solver_comparison",
  "prediction_comparison",
]);

export default function RunDetailPage() {
  const { runId } = useParams<{ runId: string }>();
  const navigate = useNavigate();
  const [confirmDelete, setConfirmDelete] = useState(false);

  const {
    data: run,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["run", runId],
    queryFn: () => getRun(runId!),
    enabled: !!runId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === "completed" || status === "failed" || status === "cancelled") return false;
      return 3000;
    },
  });

  const isActive = run?.status === "queued" || run?.status === "running";
  const progress = useProgress(isActive ? runId : null);

  const queryClient = useQueryClient();
  const deleteMutation = useMutation({
    mutationFn: (id: string) => deleteRun(id),
    onSuccess: () => {
      queryClient.removeQueries({ queryKey: ["run", runId] });
      queryClient.removeQueries({ queryKey: ["runFiles", runId] });
      queryClient.invalidateQueries({ queryKey: ["runs"] });
      queryClient.invalidateQueries({ queryKey: ["campaigns"] });
      navigate("/");
    },
  });

  // When the run reaches a terminal state, invalidate the runs list so the
  // Dashboard reflects the updated status immediately on next visit.
  useEffect(() => {
    if (run?.status === "completed" || run?.status === "failed" || run?.status === "cancelled") {
      queryClient.invalidateQueries({ queryKey: ["runs"] });
    }
  }, [run?.status, queryClient]);

  const cancelMutation = useMutation({
    mutationFn: (id: string) => cancelRun(id),
    onMutate: async () => {
      await queryClient.cancelQueries({ queryKey: ["run", runId] });
    },
    onSuccess: (summary: RunSummary) => {
      queryClient.setQueryData<RunDetail | undefined>(["run", runId], (current) => {
        if (!current) return current;
        return {
          ...current,
          status: summary.status,
          completed_at: summary.completed_at,
          error_message: summary.error_message,
        };
      });
      queryClient.invalidateQueries({ queryKey: ["run", runId] });
    },
    onError: () => {
      queryClient.invalidateQueries({ queryKey: ["run", runId] });
    },
  });

  if (isLoading) {
    return <p className="mt-8 text-center text-gray-500">Loading run...</p>;
  }
  if (error || !run) {
    return <p className="mt-8 text-center text-red-600">Run not found.</p>;
  }

  const candidates = savedCandidates(run);
  const result = run.result as Record<string, unknown> | undefined | null;
  const isComparison = COMPARISON_TYPES.has(run.run_type);
  const showCancellationPending =
    (cancelMutation.isPending || cancelMutation.isSuccess) &&
    cancelMutation.variables === run.id &&
    (run.status === "queued" || run.status === "running");

  // Single-run fields
  const metrics =
    !isComparison && result && typeof result === "object"
      ? (result.metrics as Record<string, unknown> | undefined)
      : undefined;

  // Comparison field: { strategy_name: { metrics... }, ... }
  const comparisonMetrics: Record<string, Record<string, unknown>> | undefined =
    isComparison && result && typeof result === "object"
      ? (result.comparison_metrics as Record<string, Record<string, unknown>> | undefined) ??
        // Fallback: if the backend puts each strategy at top-level
        (result.results as Record<string, Record<string, unknown>> | undefined)
      : undefined;

  const figures =
    result && typeof result === "object"
      ? (result.figures as Record<string, unknown>[] | undefined)
      : undefined;
  const partialStrategies =
    result?.partial && result.individual_results && typeof result.individual_results === "object"
      ? (result.individual_results as Record<string, { results?: Array<{ date?: string; total_value?: number }> }>)
      : undefined;

  // Prediction comparison: results is a list of PredictorResult objects
  const predictionResults =
    run.run_type === "prediction_comparison" && result
      ? (result.results as unknown[] | undefined)
      : undefined;
  const hasGeneratedCharts = !!figures?.length;
  const hasPredictionCharts =
    !!predictionResults?.length && (!figures || figures.length === 0);
  const chartCount = figures?.length ?? predictionResults?.length ?? 0;

  return (
    <div className="space-y-6">
      {/* ── Header ── */}
      <div>
        <h1 className="text-xl font-semibold text-gray-900">
          {run.run_type === "execution" ? libraryRunType(run) : runTypeLabel(run.run_type)}
        </h1>
        <div className="mt-2 flex flex-wrap items-center gap-3 text-sm text-gray-500">
          <span
            className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-medium ${statusColor(run.status)}`}
          >
            {run.status}
          </span>
          <span>Created: {fmtDate(run.created_at)}</span>
          {run.completed_at && (
            <span>Completed: {fmtDate(run.completed_at)}</span>
          )}
          {isActive && (
            <button
              onClick={() => cancelMutation.mutate(run.id)}
              disabled={cancelMutation.isPending || showCancellationPending}
              className="ml-2 rounded border border-red-300 bg-white px-2.5 py-0.5 text-xs font-medium text-red-600 hover:bg-red-50 disabled:opacity-50"
            >
              {cancelMutation.isPending || showCancellationPending ? "Cancelling…" : "Cancel run"}
            </button>
          )}
          {!isActive && <button type="button" onClick={() => setConfirmDelete(true)}
            className="rounded border border-red-300 px-2.5 py-0.5 text-xs font-medium text-red-600 hover:bg-red-50">
            Delete run
          </button>}
        </div>
      </div>

      {run.run_type === "execution" && <ExecutionStrategySummary run={run} />}
      {!(run.run_type === "execution" && run.request_payload?.execution_mode === "investing") && <RunCampaignAssignment run={run} />}

      {confirmDelete && !isActive && <section role="alertdialog" aria-labelledby="delete-run-title" className="rounded-md border border-red-300 bg-red-50 p-4 space-y-3">
        <h2 id="delete-run-title" className="font-medium text-red-900">Permanently delete this run?</h2>
        <p className="text-sm text-red-800">This removes run {run.id}, its results, checkpoints, plots and output files. It cannot be undone. Shared input datasets, caches and saved presets are kept.</p>
        <div className="flex gap-3">
          <button type="button" disabled={deleteMutation.isPending} onClick={() => deleteMutation.mutate(run.id)} className="rounded bg-red-700 px-3 py-2 text-sm text-white disabled:opacity-50">{deleteMutation.isPending ? "Deleting…" : "Permanently delete run"}</button>
          <button type="button" disabled={deleteMutation.isPending} onClick={() => setConfirmDelete(false)} className="rounded border border-gray-300 bg-white px-3 py-2 text-sm">Keep run</button>
        </div>
        {deleteMutation.isError && <p role="alert" className="text-sm text-red-700">{getErrorMessage(deleteMutation.error, "Deletion failed. Please retry.")}</p>}
      </section>}

      {/* ── Live progress ── */}
      {isActive && (
        <div className="space-y-1">
          <RunProgress run={run} live={progress} />
          <WorkerHealth run={run} />
          {cancelMutation.isError && <p role="alert" className="text-sm text-red-700">{getErrorMessage(cancelMutation.error, "Cancellation could not be confirmed. Please retry.")}</p>}
          {showCancellationPending && (
            <p className="text-xs text-amber-600 pl-1">
              Cancellation requested. Waiting for the worker to stop this run.
            </p>
          )}
          {progress.message && (
            <p className="text-xs text-gray-500 pl-1">{progress.message}</p>
          )}
        </div>
      )}

      {/* ── Error ── */}
      {run.status === "failed" && run.error_message && (
        <div className="rounded-md bg-red-50 px-4 py-3 text-sm text-red-700">
          <strong>Error:</strong> {run.error_message}
        </div>
      )}

      {/* ── Cancelled notice ── */}
      {run.status === "cancelled" && (
        <div className="rounded-md bg-gray-50 px-4 py-3 text-sm text-gray-600">
          This run was cancelled. Previously saved output is retained; work still in memory when the worker stopped may not be available.
        </div>
      )}

      {partialStrategies && (
        <div className="rounded-md bg-amber-50 px-4 py-3 text-sm text-amber-900">
          <strong>Saved progress</strong>
          <p>These strategy periods finished before the run stopped. The comparison is incomplete.</p>
          <ul className="mt-2 list-disc pl-5">
            {Object.entries(partialStrategies).map(([name, output]) => {
              const periods = output.results ?? [];
              const last = periods[periods.length - 1];
              return <li key={name}>{name}: {periods.length} periods{last?.date ? `, through ${last.date}` : ""}</li>;
            })}
          </ul>
        </div>
      )}

      {run.run_type === "execution" && result && <SelectedPortfolio key={run.id} run={run} />}

      {/* ── Single-run metrics (grouped) ── */}
      {result && (metrics || comparisonMetrics) && (
        <p className="text-xs text-gray-500">Portfolio values are labelled by period end; execution times are reported separately. Runs saved before this update retain their original date labels.</p>
      )}
      {metrics && <MetricsCards metrics={metrics} />}

      {/* ── Comparison metrics table ── */}
      {candidates.length > 0 && isComparison ? <RunComparisonResults key={run.id} entries={candidates} /> : comparisonMetrics && Object.keys(comparisonMetrics).length > 0 && (
        <div>
          <h2 className="mb-3 text-sm font-semibold text-gray-700">
            Comparison Results
          </h2>
          <ComparisonMetricsTable data={comparisonMetrics} definitions={result?.metric_definitions as Record<string, Record<string, unknown>> | undefined} />
        </div>
      )}

      {/* ── Prediction comparison charts (fallback when no backend figures) ── */}
      {(hasGeneratedCharts || hasPredictionCharts) && (
        <ExpandableSection
          title="Charts"
          summary={`${chartCount} interactive chart${chartCount === 1 ? "" : "s"}. Load on demand to keep the page responsive.`}
          renderContent={() => (
            <Suspense fallback={<p role="status">Loading charts…</p>}>
              <div className="space-y-4">
              {hasPredictionCharts && predictionResults && (
                <PredictionComparisonCharts results={predictionResults} />
              )}
              {hasGeneratedCharts && figures && <SavedChartBrowser figures={figures} configurations={candidates.map(c => c.name)} />}
              </div>
            </Suspense>
          )}
        />
      )}

      {/* ── Output files ── */}
      {(run.status === "completed" || run.status === "failed" || run.status === "cancelled") && runId && (
        <ExpandableSection
          title="Output Files"
          summary="Load generated artifacts only when you need them."
          renderContent={() => <RunFilesPanel runId={runId} showTitle={false} />}
        />
      )}

      {/* ── Request payload ── */}
      <ExpandableSection
        title="Request Payload"
        summary="Inspect the exact submitted parameters."
        renderContent={() => (
          <pre className="overflow-x-auto text-xs text-gray-600">
            {JSON.stringify(run.request_payload, null, 2)}
          </pre>
        )}
      />

      {/* ── Raw result (collapsible) ── */}
      {result && (
        <ExpandableSection
          title="Raw Result JSON"
          summary="Keep the full result collapsed unless you need to inspect it."
          renderContent={() => (
            <pre className="max-h-96 overflow-auto text-xs text-gray-600">
              {JSON.stringify(result, null, 2)}
            </pre>
          )}
        />
      )}
    </div>
  );
}
