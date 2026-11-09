import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { expect, it, vi } from "vitest";
import { getRun, listRunFiles } from "@/api/endpoints";
import type { RunDetail } from "@/types/api";
import RunDetailPage from "@/pages/RunDetailPage";

vi.mock("@/api/endpoints", () => ({
  getRun: vi.fn(),
  listRunFiles: vi.fn(),
  deleteRun: vi.fn(),
  cancelRun: vi.fn(),
  downloadRunFile: vi.fn(),
  downloadAllRunFiles: vi.fn(),
}));
vi.mock("@/hooks", () => ({ useProgress: () => ({ message: null, progress: null }) }));
vi.mock("@/components/ui/PlotlyChart", () => ({ default: () => null }));
vi.mock("@/components/ui/PredictionComparisonCharts", () => ({ default: () => null }));

it.each(["execution", "strategy_comparison"] as const)(
  "preserves ratios, percentages, cents and counts in %s results",
  async (runType) => {
    vi.mocked(listRunFiles).mockResolvedValue([]);
    const metrics = {
      initial_value: 1000.55,
      total_periods: 12,
      profit_factor: 1.5,
      final_hhi: 0.25,
      total_return: 0.12,
    };
    vi.mocked(getRun).mockResolvedValue({
      id: "aaaaaaaaaaaa",
      status: "completed",
      run_type: runType,
      created_at: "2026-10-01T12:00:00Z",
      completed_at: "2026-10-01T12:01:00Z",
      error_message: null,
      request_payload: {},
      result: runType === "execution"
        ? { metrics }
        : { comparison_metrics: { baseline: metrics } },
    } satisfies RunDetail);

    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <MemoryRouter initialEntries={["/runs/aaaaaaaaaaaa"]}>
          <Routes>
            <Route path="/runs/:runId" element={<RunDetailPage />} />
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>,
    );

    expect(await screen.findByText("1.50")).toBeInTheDocument();
    expect(screen.getByText("0.25")).toBeInTheDocument();
    expect(screen.getByText("12.00%")).toBeInTheDocument();
    expect(screen.getByText("12")).toBeInTheDocument();
    const currency = `$${metrics.initial_value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    expect(screen.getByText(currency)).toBeInTheDocument();
    expect(screen.queryByText("150.00%")).not.toBeInTheDocument();
    expect(screen.queryByText("25.00%")).not.toBeInTheDocument();
    expect(screen.getByText("Final Capital HHI (0–1)")).toBeInTheDocument();
  },
);

it("identifies mixed return definitions and keeps each comparison value intact", async () => {
  vi.mocked(listRunFiles).mockResolvedValue([]);
  vi.mocked(getRun).mockResolvedValue({
    id: "bbbbbbbbbbbb", status: "completed", run_type: "strategy_comparison",
    created_at: "2026-10-01T12:00:00Z", completed_at: null, error_message: null,
    request_payload: {}, result: {
      comparison_metrics: { calendar: { annualized_return: 0.12 }, legacy: { annualized_return: 0.08 } },
      metric_definitions: { calendar: { version: 2, annualized_return: "cagr" } },
    },
  } satisfies RunDetail);
  render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
    <MemoryRouter initialEntries={["/runs/bbbbbbbbbbbb"]}><Routes>
      <Route path="/runs/:runId" element={<RunDetailPage />} />
    </Routes></MemoryRouter>
  </QueryClientProvider>);
  expect(await screen.findByText("Annualized Return (mixed definitions)")).toBeInTheDocument();
  expect(screen.getByText("12.00%")).toHaveAttribute("title", "CAGR");
  expect(screen.getByText("8.00%")).toHaveAttribute("title", "Annualized Return (definition unavailable)");
});

vi.mock("@/components/ui/Campaigns", () => ({CampaignPanel: () => null, CampaignSelect: () => null, RunCampaignAssignment: () => null}));
