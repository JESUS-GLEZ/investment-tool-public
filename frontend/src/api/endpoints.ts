/**
 * Typed API endpoint wrappers.
 *
 * Every function returns the Axios response's `.data` directly so callers
 * don't need to unwrap it.
 */
import client from "./client";
import { isAxiosError } from "axios";
import type {
  Campaign,
  CampaignFields,
  RunSummary,
  RunDetail,
  ExecutionRequest,
  StrategyComparisonRequest,
  SolverComparisonRequest,
  PredictionComparisonRequest,
  TokenResponse,
  SymbolsResponse,
  DefaultsResponse,
  LocalDataInfo,
  LocalDataFileSummary,
  AssetConfigSummary,
} from "@/types/api";

/* ---------- Auth ---------- */

export async function register(
  email: string,
  password: string,
): Promise<TokenResponse> {
  const { data } = await client.post<TokenResponse>("/auth/register", {
    email,
    password,
  });
  return data;
}

export async function login(
  email: string,
  password: string,
): Promise<TokenResponse> {
  const { data } = await client.post<TokenResponse>("/auth/login", {
    email,
    password,
  });
  return data;
}

/* ---------- Runs ---------- */

export async function createRun(body: ExecutionRequest): Promise<RunSummary> {
  const { data } = await client.post<RunSummary>("/runs", body);
  return data;
}

export async function listRuns(): Promise<RunSummary[]> {
  const { data } = await client.get<RunSummary[]>("/runs");
  return data;
}

export async function getRun(runId: string): Promise<RunDetail> {
  const { data } = await client.get<RunDetail>(`/runs/${runId}`);
  return data;
}

export async function deleteRun(runId: string): Promise<void> {
  await client.delete(`/runs/${runId}`);
}

export interface RunOutputFile {
  path: string;
  size: number;
  type: string;
}

export async function listRunFiles(runId: string): Promise<RunOutputFile[]> {
  const { data } = await client.get<RunOutputFile[]>(`/runs/${runId}/files`);
  return data;
}

/** Read the existing small manifest, when retained; never deserialize datasets. */
export async function getRunManifest(runId: string): Promise<unknown | null> {
  try {
    return (await client.get(`/runs/${runId}/files/run_manifest.json`, {responseType:"json"})).data;
  } catch (error) {
    if (isAxiosError(error) && error.response?.status === 404) return null;
    throw error;
  }
}

/** Returns the URL to download a file — used directly as an <a href>. */
export function runFileDownloadUrl(runId: string, filePath: string): string {
  return `/api/runs/${runId}/files/${filePath}`;
}

/** Returns the URL to download all output files as a ZIP archive. */
export function runFilesDownloadAllUrl(runId: string): string {
  return `/api/runs/${runId}/files/download-all`;
}

/**
 * Download a single run output file via the authenticated Axios client,
 * then trigger a browser save-as dialog using a temporary blob URL.
 */
export async function downloadRunFile(
  runId: string,
  filePath: string,
): Promise<void> {
  const { data } = await client.get(`/runs/${runId}/files/${filePath}`, {
    responseType: "blob",
  });
  const url = URL.createObjectURL(data as Blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filePath.split("/").pop() ?? filePath;
  a.click();
  URL.revokeObjectURL(url);
}

/**
 * Download all run output files as a ZIP via the authenticated Axios client,
 * then trigger a browser save-as dialog using a temporary blob URL.
 */
export async function downloadAllRunFiles(runId: string): Promise<void> {
  const { data } = await client.get(`/runs/${runId}/files/download-all`, {
    responseType: "blob",
  });
  const url = URL.createObjectURL(data as Blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `run_${runId}_files.zip`;
  a.click();
  URL.revokeObjectURL(url);
}

/* ---------- Comparisons ---------- */

export async function createStrategyComparison(
  body: StrategyComparisonRequest,
): Promise<RunSummary> {
  const { data } = await client.post<RunSummary>(
    "/comparisons/strategy",
    body,
  );
  return data;
}

export async function createSolverComparison(
  body: SolverComparisonRequest,
): Promise<RunSummary> {
  const { data } = await client.post<RunSummary>("/comparisons/solver", body);
  return data;
}

export async function createPredictionComparison(
  body: PredictionComparisonRequest,
): Promise<RunSummary> {
  const { data } = await client.post<RunSummary>(
    "/comparisons/prediction",
    body,
  );
  return data;
}

/* ---------- Reference data ---------- */

export async function getDefaults(): Promise<DefaultsResponse> {
  const { data } = await client.get<DefaultsResponse>("/defaults");
  return data;
}

export async function getSymbols(file?: string): Promise<SymbolsResponse> {
  const { data } = file
    ? await client.get<SymbolsResponse>("/symbols", {params: {file}})
    : await client.get<SymbolsResponse>("/symbols");
  return data;
}

export async function listAssetConfigs(): Promise<AssetConfigSummary[]> {
  const { data } = await client.get<AssetConfigSummary[]>("/asset-configs");
  return data;
}

export async function getLocalDataInfo(file?: string): Promise<LocalDataInfo> {
  const params = file ? { file } : {};
  const { data } = await client.get<LocalDataInfo>("/data/info", { params });
  return data;
}

export async function listLocalDataFiles(): Promise<LocalDataFileSummary[]> {
  const { data } = await client.get<LocalDataFileSummary[]>("/data/files");
  return data;
}

export async function cancelRun(runId: string): Promise<RunSummary> {
  const { data } = await client.post<RunSummary>(`/runs/${runId}/cancel`);
  return data;
}


/* ---------- Saved presets ---------- */
export interface RunPreset<T = object> {
  id: number;
  name: string;
  run_type: import("@/types/api").RunType;
  parameters: T;
}

export async function listPresets<T>(runType: RunPreset["run_type"]): Promise<RunPreset<T>[]> {
  return (await client.get("/presets", { params: { run_type: runType } })).data;
}

export async function savePreset<T>(body: Omit<RunPreset<T>, "id">, id?: number): Promise<RunPreset<T>> {
  return (id === undefined
    ? await client.post("/presets", body)
    : await client.put(`/presets/${id}`, body)).data;
}

export async function deletePreset(id: number): Promise<void> {
  await client.delete(`/presets/${id}`);
}

/* ---------- Research campaigns ---------- */
export async function listCampaigns(): Promise<Campaign[]> {
  return (await client.get<Campaign[]>("/campaigns")).data;
}
export async function saveCampaign(body: CampaignFields, id?: string): Promise<Campaign> {
  return (id
    ? await client.put<Campaign>(`/campaigns/${id}`, body)
    : await client.post<Campaign>("/campaigns", body)).data;
}
export async function assignRunCampaign(runId: string, campaignId: string | null): Promise<RunSummary> {
  return (await client.put<RunSummary>(`/runs/${runId}/campaign`, {campaign_id: campaignId})).data;
}
