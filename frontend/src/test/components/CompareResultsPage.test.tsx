import { cleanup, fireEvent, render, screen, within, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { getRun, getRunManifest, listRuns } from "@/api/endpoints";
import CompareResultsPage from "@/pages/CompareResultsPage";
import AppShell from "@/components/layout/AppShell";
import { backtest } from "../fixtures/savedRuns";
vi.mock("@/hooks", () => ({useAuth: () => ({logout: vi.fn()})}));
vi.mock("@/api/endpoints", () => ({getRun: vi.fn(), getRunManifest: vi.fn(), listRuns: vi.fn()}));
vi.mock("@/components/ui/PlotlyChart", () => ({default: () => <div>Saved curve</div>}));
const a = backtest("a"), b = {...backtest("b"), created_at: "2026-10-02T15:30:00Z"};
function editSelection() { const button = screen.queryByRole("button", {name: "Change selection"}); if (button) fireEvent.click(button); }
function Detail() { return <output>{JSON.stringify(useLocation().state)}</output>; }
function show(url = "/results/compare") {
  render(<QueryClientProvider client={new QueryClient({defaultOptions: {queries: {retry: false}}})}><MemoryRouter initialEntries={[url]}><Routes><Route element={<AppShell />}><Route path="/results/compare" element={<CompareResultsPage />} /><Route path="/runs/:id" element={<Detail />} /></Route></Routes></MemoryRouter></QueryClientProvider>);
}
beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(listRuns).mockResolvedValue([a, b, {...backtest("failed"), status: "failed"}, {...backtest("investing"), run_type: "execution", request_payload: {execution_mode: "investing"}}]);
  vi.mocked(getRun).mockImplementation(async id => id === "a" ? a : b);
  vi.mocked(getRunManifest).mockResolvedValue(null);
});
afterEach(cleanup);
it("identifies runs by execution and context, loading results only after explicit inspection", async () => {
  show();
  await screen.findByRole("button", {name: "Inspect a"});
  expect(getRun).not.toHaveBeenCalled();
  expect(screen.getByRole("region", {name: "Available runs"})).toHaveTextContent("2026-10-02 15:30 UTC");
  expect(screen.queryByRole("button", {name: "Inspect failed"})).not.toBeInTheDocument();
  expect(screen.queryByRole("button", {name: "Inspect investing"})).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", {name: "Inspect a"}));
  fireEvent.click(await screen.findByRole("button", {name: "Add result"}));
  expect(screen.getByRole("button", {name: "Add result"})).toBeDisabled();
  fireEvent.change(screen.getByLabelText("Configuration"), {target: {value: "a:Target"}});
  fireEvent.click(screen.getByRole("button", {name: "Add result"}));
  const results = await screen.findByRole("region", {name: "Backtest results"});
  expect(results).toHaveTextContent("20.00%");
  fireEvent.change(screen.getByLabelText("Display"), {target: {value: "differences"}});
  expect(results).toHaveTextContent("+10.00 pp");
  expect(screen.queryByLabelText("Performance curve")).not.toBeInTheDocument();
  expect(getRun).toHaveBeenCalledTimes(1);
  expect(getRunManifest).not.toHaveBeenCalled();
});
it("keeps legacy URL selections and reference through source navigation", async () => {
  show("/results/compare?mode=same-period&item=a%3ASharpe&item=a%3ATarget&reference=a%3ATarget");
  await screen.findByRole("region", {name: "Backtest results"});
  expect(screen.getByLabelText("Reference configuration")).toHaveTextContent("C2 · Target");
  expect(screen.queryByLabelText("Comparison mode")).not.toBeInTheDocument();
  fireEvent.click(screen.getAllByRole("link", {name: "Open full results"})[0]);
  expect(screen.getByRole("status")).toHaveTextContent("reference=a%3ATarget");
  expect(screen.getAllByRole("link", {name: "Compare Runs"})[0].getAttribute("href")).toContain("reference=a%3ATarget");
});
it("does not fetch another account's URL identities or display a partly resolved pool", async () => {
  show("/results/compare?item=foreign%3AX&item=a%3ASharpe");
  await screen.findByText("Configuration unavailable in the current run library.");
  expect(getRun).toHaveBeenCalledWith("a");
  expect(getRun).not.toHaveBeenCalledWith("foreign");
  expect(screen.queryByRole("region", {name: "Backtest results"})).not.toBeInTheDocument();
});
it("clears result and bulk selections on a family change", async () => {
  show("/results/compare?item=a%3ASharpe&item=a%3ATarget");
  await screen.findByRole("region", {name: "Backtest results"});
  editSelection();
  fireEvent.click(screen.getByLabelText("Select run b"));
  fireEvent.change(screen.getByLabelText("Result family"), {target: {value: "predictions"}});
  expect(screen.queryByRole("button", {name: "Remove Sharpe"})).not.toBeInTheDocument();
  expect(screen.queryByRole("region", {name: "Bulk configuration selection"})).not.toBeInTheDocument();
});
it("accepts different periods automatically without a mode switch or cross-year differences", async () => {
  const other = structuredClone(b);
  other.result!.individual_results = {Sharpe: {metrics: {total_return: 1}, results: [{date_basis: "period_end_label", period_start: "2025-01-01", period_end: "2025-12-31", total_value: 2000}]}};
  other.result!.comparison_metrics = {};
  vi.mocked(getRun).mockImplementation(async id => id === "a" ? a : other);
  show("/results/compare?item=a%3ASharpe&item=a%3ATarget");
  await screen.findByRole("region", {name: "Backtest results"});
  editSelection(); fireEvent.click(screen.getByRole("button", {name: "Inspect b"}));
  fireEvent.click(await screen.findByRole("button", {name: "Add result"}));
  expect(screen.getByRole("region", {name: "Backtest results"})).toHaveTextContent("2025-01-01 → 2025-12-31");
  expect(screen.queryByText("Saved curve")).not.toBeInTheDocument();
  expect(within(screen.getByRole("region", {name: "Results matrix"})).getAllByRole("row")).toHaveLength(3);
});
it("separates unverified cross-run context and permits retry without hiding reported results", async () => {
  vi.mocked(getRunManifest).mockRejectedValueOnce(new Error("network"));
  show("/results/compare?item=a%3ASharpe&item=b%3ATarget");
  await screen.findByRole("region", {name: "Backtest results"});
  await screen.findByRole("button", {name: "Retry manifests"});
  fireEvent.change(screen.getByLabelText("Display"), {target: {value: "differences"}});
  expect(screen.getByRole("region", {name: "Results matrix"})).toHaveTextContent("No reference in this period and context");
  fireEvent.click(screen.getByRole("button", {name: "Retry manifests"}));
});
it("recovers a selected-result load failure without dropping the selection", async () => {
  vi.mocked(getRun).mockRejectedValueOnce(new Error("network"));
  show("/results/compare?item=a%3ASharpe&item=a%3ATarget");
  const retries = await screen.findAllByRole("button", {name: "Retry a"});
  fireEvent.click(retries[0]);
  await screen.findByRole("region", {name: "Backtest results"});
});
it("loads explicitly checked runs, groups configuration choices and bulk-adds every matching execution", async () => {
  show();
  await screen.findByLabelText("Select run a");
  fireEvent.click(screen.getByLabelText("Select run a"));
  fireEvent.click(screen.getByLabelText("Select run b"));
  expect(getRun).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", {name: "Load configurations from 2 selected runs"}));
  const bulk = screen.getByRole("region", {name: "Bulk configuration selection"});
  const option = await within(bulk).findByRole("checkbox", {name: /C1 · Sharpe/});
  expect(bulk).toHaveTextContent("2 saved executions in 2 runs");
  fireEvent.click(option);
  fireEvent.click(within(bulk).getByRole("checkbox", {name: /C2 · Target/}));
  fireEvent.click(within(bulk).getByRole("button", {name: "Add selected configurations (4 results)"}));
  const matrix = await screen.findByRole("region", {name: "Results matrix"});
  expect(within(matrix).getAllByRole("columnheader")).toHaveLength(4); // period, benchmark, 2 configs
  expect(within(matrix).getAllByRole("row")).toHaveLength(3); // unknown cross-run provenance kept separate
  expect(getRun).toHaveBeenCalledTimes(2);
});
it("blocks a partial bulk load until failed runs are retried", async () => {
  vi.mocked(getRun).mockImplementation(async id => {if (id === "b") throw new Error("network"); return a;});
  show();
  await screen.findByLabelText("Select run a");
  fireEvent.click(screen.getByLabelText("Select run a")); fireEvent.click(screen.getByLabelText("Select run b"));
  fireEvent.click(screen.getByRole("button", {name: "Load configurations from 2 selected runs"}));
  await screen.findByRole("button", {name: "Retry selected runs"});
  fireEvent.click(screen.getByRole("checkbox", {name: /C1 · Sharpe/}));
  expect(screen.getByRole("button", {name: "Add selected configurations (1 results)"})).toBeDisabled();
  vi.mocked(getRun).mockResolvedValue(b);
  fireEvent.click(screen.getByRole("button", {name: "Retry selected runs"}));
  await waitFor(() => expect(screen.queryByRole("button", {name: "Retry selected runs"})).not.toBeInTheDocument());
});
it("retains selections beyond the old 24-result limit in a multi-year matrix", async () => {
  const annual = Array.from({length: 14}, (_, index) => {
    const run = structuredClone(backtest(String(index)));
    const year = 2010 + index;
    run.request_payload.historical_start_date = `${year - 2}-01-01`;
    const output = run.result!.individual_results as Record<string, {results: Record<string, unknown>[]}>;
    for (const item of Object.values(output)) item.results = [{period_start: `${year}-01-01`, period_end: `${year}-12-31`, date_basis: "period_end_label", total_value: 1100}];
    return run;
  });
  vi.mocked(listRuns).mockResolvedValue(annual);
  vi.mocked(getRun).mockImplementation(async id => annual.find(run => run.id === id)!);
  show(`/results/compare?${annual.flatMap(run => ["Sharpe", "Target"].map(name => `item=${run.id}%3A${name}`)).join("&")}`);
  const matrix = await screen.findByRole("region", {name: "Results matrix"});
  expect(within(matrix).getAllByRole("row")).toHaveLength(15);
  expect(within(matrix).getAllByRole("columnheader")).toHaveLength(4);
  expect(matrix).toHaveTextContent("2023-01-01 → 2023-12-31");
});

vi.mock("@/components/ui/Campaigns", () => ({CampaignPanel: () => null, CampaignSelect: () => null, RunCampaignAssignment: () => null}));
