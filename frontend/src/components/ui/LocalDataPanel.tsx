/**
 * LocalDataPanel — file selector + metadata display for local .pkl datasets.
 *
 * Shows a dropdown of all available files (from GET /data/files) and detailed
 * metadata for the selected one (from GET /data/info?file=...).
 * The parent form passes `selectedFile` / `onFileChange` to keep the chosen
 * filename in the request payload.
 */
import { useEffect, useEffectEvent, useId } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { getLocalDataInfo, listLocalDataFiles } from "@/api/endpoints";
import { getErrorMessage } from "@/lib/utils";
import type { Frequency, LocalDataInfo } from "@/types/api";

import { datasetLabel, datasetTitle } from "@/lib/localDatasets";

interface Props {
  selectedFile: string | null;
  onFileChange: (file: string | null) => void;
  onFrequencyChange?: (frequency: Frequency) => void;
  onInfoChange?: (info: LocalDataInfo | null) => void;
}

export default function LocalDataPanel({
  selectedFile,
  onFileChange,
  onFrequencyChange,
  onInfoChange,
}: Props) {
  const datasetId = useId();
  const {
    data: files,
    isLoading: filesLoading,
    error: filesError,
    isFetching: filesFetching,
    refetch: retryFiles,
  } = useQuery({
    queryKey: ["localDataFiles"],
    queryFn: listLocalDataFiles,
    retry: false,
    staleTime: 0,
  });

  const notifyFile = useEffectEvent((file: string | null) => onFileChange(file));
  const notifyFrequency = useEffectEvent((frequency: Frequency) => onFrequencyChange?.(frequency));
  const notifyInfo = useEffectEvent((info: LocalDataInfo | null) => onInfoChange?.(info));

  // Keep the selection aligned with the current file list.
  useEffect(() => {
    if (!files) return;
    if (files.length === 0) {
      if (selectedFile !== null) {
        notifyFile(null);
      }
      notifyInfo(null);
      return;
    }

    const selectedEntry = selectedFile
      ? files.find((file) => file.file === selectedFile)
      : null;

    if (selectedEntry) {
      if (selectedEntry.frequency) notifyFrequency(selectedEntry.frequency as Frequency);
      return;
    }

    if (selectedFile) return;
    notifyFile(files[0].file);
    if (files[0].frequency) notifyFrequency(files[0].frequency as Frequency);
  }, [files, selectedFile]);

  const hasSelectedFile =
    selectedFile !== null &&
    !!files?.some((file) => file.file === selectedFile);

  const {
    data: info,
    error: infoError,
    isFetching: infoFetching,
    refetch: retryInfo,
  } = useQuery({
    queryKey: ["localDataInfo", selectedFile],
    queryFn: () => getLocalDataInfo(selectedFile ?? undefined),
    enabled: hasSelectedFile,
    retry: false,
    staleTime: 0,
  });

  useEffect(() => {
    if (!selectedFile || !hasSelectedFile || filesFetching || filesError || infoFetching || infoError || !info || info.file !== selectedFile) {
      notifyInfo(null);
      return;
    }
    if (info.frequency) notifyFrequency(info.frequency);
    notifyInfo(info);
  }, [info, infoFetching, infoError, filesFetching, filesError, hasSelectedFile, selectedFile]);

  if (filesLoading) {
    return (
      <div role="status" className="rounded-md border border-gray-200 bg-gray-50 px-4 py-3 text-sm text-gray-500">
        Loading available datasets…
      </div>
    );
  }

  if (filesError) {
    return (
      <div role="alert" className="rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
        {getErrorMessage(filesError, "Failed to load local datasets.")}
        <button type="button" disabled={filesFetching} onClick={() => { void retryFiles(); }} className="ml-2 underline disabled:opacity-50">Retry datasets</button>
      </div>
    );
  }

  if (!files || files.length === 0) {
    return (
      <div className="rounded-md border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-700">
        No local datasets found at <code>/app/data/input</code>. Mount a{" "}
        <code>.pkl</code> file or switch to <strong>alpaca</strong>.
      </div>
    );
  }

  const selectedSummary =
    files.find((file) => file.file === selectedFile);

  return (
    <div className="rounded-md border border-blue-200 bg-blue-50 px-4 py-3 space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold uppercase tracking-wide text-blue-600">
          Local Dataset
        </span>
        <span className="text-xs text-blue-400">{files.length} file{files.length !== 1 ? "s" : ""} available</span>
      </div>

      {/* File selector */}
      <div>
        <label htmlFor={datasetId} className="block text-xs font-medium text-gray-600 mb-1">Select dataset</label>
        <select
          id={datasetId}
          value={selectedFile ?? ""}
          onChange={(e) => {
            onInfoChange?.(null);
            onFileChange(e.target.value || null);
          }}
          className="block w-full rounded-md border border-blue-300 bg-white px-3 py-2 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none"
        >
          {selectedFile && !hasSelectedFile && <option value={selectedFile}>{selectedFile} (unavailable)</option>}
          {files.map((f) => (
            <option key={f.file} value={f.file}>
              {datasetLabel(f)}
            </option>
          ))}
        </select>
      </div>

      {selectedFile && !hasSelectedFile && <p role="alert" className="text-sm text-red-700">The saved dataset is unavailable. Select another dataset before launching.</p>}
      {selectedSummary && <div className="rounded-md border border-blue-200 bg-white px-3 py-2">
        <p className="text-sm font-medium text-gray-800">{info?.title ?? selectedSummary.title ?? datasetTitle(selectedSummary.file)}</p>
        <Link to={`/datasets?file=${encodeURIComponent(selectedSummary.file)}`} className="mt-1 inline-block text-xs text-blue-700 underline">Inspect in Data library</Link>
        <details className="mt-2 text-xs text-gray-500"><summary className="cursor-pointer">Technical filename</summary><p className="mt-1 break-all font-mono">{selectedSummary.file}</p></details>
      </div>}

      {/* Metadata for selected file */}
      {infoFetching && !infoError && (
        <p role="status" className="text-xs text-gray-500">Loading dataset details…</p>
      )}
      {infoError && (
        <div role="alert" className="rounded-md border border-red-200 bg-white px-3 py-2 text-xs text-red-700">
          {getErrorMessage(infoError, "Failed to load dataset details.")}
          <button type="button" disabled={infoFetching} onClick={() => { void retryInfo(); }} className="ml-2 underline disabled:opacity-50">Retry dataset details</button>
        </div>
      )}
      {info && !infoError && !infoFetching && (
        <>
          {info.time_contract_version === null && (
            <p className="text-xs text-amber-800">This legacy dataset needs an explicit timestamp upgrade before it can be used. Fetch a new dataset or follow the legacy-data upgrade instructions.</p>
          )}
          <div className="grid grid-cols-1 gap-3 text-sm sm:grid-cols-2">
            <div>
              <p className="text-xs text-gray-500 mb-0.5">Overall date span</p>
              <p className="font-medium text-gray-800">
                {info.start_date ?? "—"} → {info.end_date ?? "—"}
              </p>
              <p className="text-xs text-gray-400 mt-0.5">
                Individual assets may have shorter coverage. Use the historical data window below to restrict this range.
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-500 mb-0.5">Assets</p>
              <p className="font-medium text-gray-800">
                {info.stock_count} stocks · {info.crypto_count} crypto
              </p>
            </div>
          </div>

          {(info.stocks.length > 0 || info.crypto.length > 0) && (
            <div>
              <p className="text-xs text-gray-500 mb-1">Available tickers</p>
              <div className="flex flex-wrap gap-1 max-h-28 overflow-y-auto">
                {info.stocks.map((s) => (
                  <span
                    key={s}
                    className="rounded bg-white border border-blue-200 px-1.5 py-0.5 text-xs font-mono text-gray-700"
                  >
                    {s}
                  </span>
                ))}
                {info.crypto.map((c) => (
                  <span
                    key={c}
                    className="rounded bg-white border border-purple-200 px-1.5 py-0.5 text-xs font-mono text-purple-700"
                  >
                    {c}
                  </span>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
