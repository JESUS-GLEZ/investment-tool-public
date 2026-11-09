import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Routes, Route } from "react-router-dom";
import { beforeEach, expect, it, vi } from "vitest";
import * as api from "@/api/endpoints";
import type { RunDetail } from "@/types/api";
import RunDetailPage from "@/pages/RunDetailPage";

vi.mock("@/api/endpoints", () => ({getRun: vi.fn(), listRunFiles: vi.fn(), deleteRun: vi.fn(), cancelRun: vi.fn(), downloadRunFile: vi.fn(), downloadAllRunFiles: vi.fn()}));
vi.mock("@/hooks", () => ({useProgress: () => ({message: null, progress: null})}));
vi.mock("@/components/ui/PlotlyChart", () => ({default: () => null}));
vi.mock("@/components/ui/PredictionComparisonCharts", () => ({default: () => null}));

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.getRun).mockResolvedValue({id: "aaaaaaaaaaaa", status: "completed", run_type: "strategy_comparison", created_at: "2026-10-01T12:00:00Z", result: null} as RunDetail);
  vi.mocked(api.listRunFiles).mockResolvedValue([]);
  vi.mocked(api.deleteRun).mockResolvedValue(undefined);
});

function mount() {
  render(<QueryClientProvider client={new QueryClient({defaultOptions:{queries:{retry:false}}})}>
    <MemoryRouter initialEntries={["/runs/aaaaaaaaaaaa"]}><Routes>
      <Route path="/runs/:runId" element={<RunDetailPage />} />
      <Route path="/" element={<p>Runs dashboard</p>} />
    </Routes></MemoryRouter>
  </QueryClientProvider>);
}

it("requires explicit confirmation and returns to the dashboard after deletion", async () => {
  mount();
  fireEvent.click(await screen.findByRole("button", {name:"Delete run"}));
  expect(api.deleteRun).not.toHaveBeenCalled();
  expect(screen.getByRole("alertdialog")).toHaveTextContent("Shared input datasets, caches and saved presets are kept");
  fireEvent.click(screen.getByRole("button", {name:"Permanently delete run"}));
  expect(await screen.findByText("Runs dashboard")).toBeInTheDocument();
  expect(api.deleteRun).toHaveBeenCalledWith("aaaaaaaaaaaa");
});

it("keeps the run when confirmation is dismissed", async () => {
  mount();
  fireEvent.click(await screen.findByRole("button", {name:"Delete run"}));
  fireEvent.click(screen.getByRole("button", {name:"Keep run"}));
  expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
  expect(api.deleteRun).not.toHaveBeenCalled();
});

it("does not expose deletion for a running run", async () => {
  vi.mocked(api.getRun).mockResolvedValue({id:"aaaaaaaaaaaa", status:"running", run_type:"execution", created_at:"2026-10-01T12:00:00Z", result:null} as RunDetail);
  mount();
  await screen.findByRole("button", {name:"Cancel run"});
  expect(screen.queryByRole("button", {name:"Delete run"})).not.toBeInTheDocument();
});

it("retains the confirmation and displays deletion failures", async () => {
  vi.mocked(api.deleteRun).mockRejectedValue(new Error("Worker is still stopping"));
  mount();
  fireEvent.click(await screen.findByRole("button", {name:"Delete run"}));
  fireEvent.click(screen.getByRole("button", {name:"Permanently delete run"}));
  await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("Worker is still stopping"));
  expect(screen.getByRole("alertdialog")).toBeInTheDocument();
});

vi.mock("@/components/ui/Campaigns", () => ({CampaignPanel: () => null, CampaignSelect: () => null, RunCampaignAssignment: () => <select aria-label="Campaign assignment"><option>Ungrouped</option></select>}));

it.each(["investing", "backtesting"])("only hides the campaign assignment for investment recommendations (%s)", async mode => {
  vi.mocked(api.getRun).mockResolvedValue({id: "aaaaaaaaaaaa", status: "completed", run_type: "execution", created_at: "2026-11-01", completed_at: null, error_message: null, request_payload: {execution_mode: mode}, result: null});
  mount();
  await screen.findByRole("heading", {name: mode === "investing" ? "Investing recommendation" : "Backtest"});
  if (mode === "investing") expect(screen.queryByLabelText("Campaign assignment")).not.toBeInTheDocument();
  else expect(screen.getByLabelText("Campaign assignment")).toBeInTheDocument();
});
