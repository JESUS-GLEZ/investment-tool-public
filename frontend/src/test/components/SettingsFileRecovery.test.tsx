import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import * as api from "@/api/endpoints";
import type { RunDetail } from "@/types/api";
import SettingsPage from "@/pages/SettingsPage";
import RunDetailPage from "@/pages/RunDetailPage";

vi.mock("@/api/endpoints", () => ({
  getDefaults: vi.fn(), getSymbols: vi.fn(), listAssetConfigs:vi.fn(), getRun: vi.fn(), listRunFiles: vi.fn(),
  downloadRunFile: vi.fn(), downloadAllRunFiles: vi.fn(), cancelRun: vi.fn(), deleteRun: vi.fn(),
}));
vi.mock("@/hooks", () => ({ useProgress: () => ({ message: null, progress: null }) }));
vi.mock("@/components/ui/PlotlyChart", () => ({ default: () => null }));
vi.mock("@/components/ui/PredictionComparisonCharts", () => ({ default: () => null }));

const clients: QueryClient[] = [];
const files = [{ path: "tables/result.csv", size: 12, type: "text/csv" }];

function mount(settings = false) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  clients.push(client);
  render(<QueryClientProvider client={client}>
    <MemoryRouter initialEntries={["/runs/owned"]}>
      {settings ? <SettingsPage /> : <Routes><Route path="/runs/:runId" element={<RunDetailPage />} /></Routes>}
    </MemoryRouter>
  </QueryClientProvider>);
  return client;
}

async function openFiles() {
  const heading = await screen.findByRole("heading", { name: "Output Files" });
  fireEvent.click(within(heading.closest("section")!).getByRole("button", { name: "Show" }));
}

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.getDefaults).mockResolvedValue({ optimizer: { pop_size: 300 } });
  vi.mocked(api.listAssetConfigs).mockResolvedValue([{file:"example_asset_universe.json",name:"Example",stock_count:1,crypto_count:0}]);
  vi.mocked(api.getSymbols).mockResolvedValue({ stocks: ["SYNTH"], crypto: [] });
  vi.mocked(api.getRun).mockResolvedValue({ id: "owned", status: "completed", run_type: "execution", created_at: "2026-10-01T12:00:00Z", result: null } as RunDetail);
  vi.mocked(api.listRunFiles).mockResolvedValue(files);
  vi.mocked(api.downloadRunFile).mockResolvedValue(undefined);
  vi.mocked(api.downloadAllRunFiles).mockResolvedValue(undefined);
});

afterEach(() => {
  cleanup();
  clients.splice(0).forEach((client) => client.clear());
});

it("identifies both settings requests while they are loading", async () => {
  vi.mocked(api.getDefaults).mockReturnValue(new Promise(() => {}));
  vi.mocked(api.getSymbols).mockReturnValue(new Promise(() => {}));
  mount(true);
  await screen.findByText("Loading available symbols…");
  expect(screen.getAllByRole("status").map((item) => item.textContent)).toEqual([
    "Loading default parameters…", "Loading available symbols…",
  ]);
});

it.each(["defaults", "symbols"] as const)("retries only failed settings %s without hiding the other result", async (kind) => {
  const failed = kind === "defaults" ? api.getDefaults : api.getSymbols;
  vi.mocked(failed).mockRejectedValueOnce(new Error(`Temporary ${kind} failure`));
  mount(true);
  expect(await screen.findByRole("alert")).toHaveTextContent(`Temporary ${kind} failure`);
  if(kind === "symbols") fireEvent.click(screen.getByText("Advanced engine reference"));
  expect(await screen.findByText(kind === "defaults" ? "SYNTH" : "300")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: `Retry ${kind}` }));
  await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
  expect(api.getDefaults).toHaveBeenCalledTimes(kind === "defaults" ? 2 : 1);
  expect(api.getSymbols).toHaveBeenCalledTimes(kind === "symbols" ? 2 : 1);
  expect(api.getRun).not.toHaveBeenCalled();
});

it("shows file loading, then an error rather than a false empty list, and retries safely", async () => {
  let reject!: (error: Error) => void;
  vi.mocked(api.listRunFiles).mockReturnValueOnce(new Promise((_, fail) => { reject = fail; }));
  mount();
  await openFiles();
  expect(screen.getByRole("status")).toHaveTextContent("Loading output files");
  reject(new Error("File list unavailable"));
  expect(await screen.findByRole("alert")).toHaveTextContent("File list unavailable");
  expect(screen.queryByText("No output files found for this run.")).not.toBeInTheDocument();
  vi.mocked(api.listRunFiles).mockResolvedValueOnce([]);
  fireEvent.click(screen.getByRole("button", { name: "Retry file list" }));
  expect(await screen.findByText("No output files found for this run.")).toBeInTheDocument();
  expect(api.listRunFiles).toHaveBeenCalledTimes(2);
  expect(api.downloadRunFile).not.toHaveBeenCalled();
  expect(api.downloadAllRunFiles).not.toHaveBeenCalled();
});

it("retains cached files when a refresh fails", async () => {
  const client = mount();
  await openFiles();
  expect(await screen.findByText("result.csv")).toBeInTheDocument();
  vi.mocked(api.listRunFiles).mockRejectedValueOnce(new Error("Refresh unavailable"));
  await client.refetchQueries({ queryKey: ["runFiles", "owned"] });
  expect(await screen.findByRole("alert")).toHaveTextContent("Refresh unavailable");
  expect(screen.getByText("result.csv")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Download tables/result.csv" })).toBeEnabled();
});

it.each(["file", "archive"] as const)("catches %s download failure and allows an explicit retry without duplicate downloads", async (kind) => {
  const download = kind === "file" ? api.downloadRunFile : api.downloadAllRunFiles;
  let reject!: (error: Error) => void;
  vi.mocked(download).mockReturnValueOnce(new Promise((_, fail) => { reject = fail; }));
  mount();
  await openFiles();
  const single = await screen.findByRole("button", { name: "Download tables/result.csv" });
  const archive = screen.getByRole("button", { name: "Download All (.zip)" });
  const selected = kind === "file" ? single : archive;
  fireEvent.click(selected);
  expect(single).toBeDisabled();
  expect(archive).toBeDisabled();
  fireEvent.click(single);
  fireEvent.click(archive);
  expect(download).toHaveBeenCalledTimes(1);
  reject(new Error("Download unavailable"));
  expect(await screen.findByRole("alert")).toHaveTextContent("Download unavailable Try the download again.");
  expect(single).toBeEnabled();
  expect(archive).toBeEnabled();
  fireEvent.click(selected);
  await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
  expect(download).toHaveBeenCalledTimes(2);
  if (kind === "file") expect(api.downloadRunFile).toHaveBeenLastCalledWith("owned", "tables/result.csv");
  else expect(api.downloadAllRunFiles).toHaveBeenLastCalledWith("owned");
  expect(kind === "file" ? api.downloadAllRunFiles : api.downloadRunFile).not.toHaveBeenCalled();
  expect(api.cancelRun).not.toHaveBeenCalled();
  expect(api.deleteRun).not.toHaveBeenCalled();
});

vi.mock("@/components/ui/Campaigns", () => ({CampaignPanel: () => null, CampaignSelect: () => null, RunCampaignAssignment: () => null}));
