import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { getRun, listCampaigns, listRuns } from "@/api/endpoints";
import type { RunSummary } from "@/types/api";
import DashboardPage from "@/pages/DashboardPage";
import AppShell from "@/components/layout/AppShell";
vi.mock("@/hooks", () => ({ useAuth: () => ({ logout: vi.fn() }) }));
vi.mock("@/api/endpoints", () => ({ getRun: vi.fn(), listRuns: vi.fn(), listCampaigns: vi.fn() }));
const fixture = (id: string, index = 1): RunSummary => ({ id, run_type: "execution", status: "completed", created_at: `2026-10-${String(index).padStart(2, "0")}T12:00:00Z`, completed_at: null, error_message: null, request_payload: { execution_mode: "investing", historical_start_date: "2016-01-01", historical_end_date: "2026-09-30", data_source: "local", input_file: `${id}.pkl`, strategy_configuration: { name: `Risk ${id}`, strategy_type: "optimization", selection: "return_target", return_target_fraction: 0.75, population_size: 200 } } });
const fixtures = [fixture("a"), { ...fixture("b", 2), run_type: "solver_comparison" as const, status: "failed" as const }];
function Location() { return <output aria-label="Current URL">{useLocation().search}</output>; }
function Detail() { const navigate = useNavigate(); return <button onClick={() => navigate(-1)}>Back to dashboard</button>; }
function show(url = "/") {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}>
    <MemoryRouter initialEntries={[url]}>
      <Location />
      <Routes>
        <Route element={<AppShell />}>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/runs/:id" element={<Detail />} />
        </Route>
      </Routes>
    </MemoryRouter>
  </QueryClientProvider>);
  return client;
}
beforeEach(() => { vi.resetAllMocks(); vi.mocked(listCampaigns).mockResolvedValue([]); vi.mocked(listRuns).mockResolvedValue(fixtures); });
afterEach(cleanup);
it("keeps rows compact and expands saved settings without fetching results", async () => {
  show();
  const open = await screen.findByRole("link", { name: "Open details for a" });
  expect(open).toHaveAttribute("href", "/runs/a");
  expect(screen.getByText("Investing recommendation")).toBeInTheDocument();
  expect(screen.getByText("Requested data cutoff")).toBeInTheDocument();
  expect(within(screen.getByRole("article", { name: "Investing recommendation a" })).getByText("Risk a · Optimization")).toBeInTheDocument();
  expect(screen.queryByText("Risk a")).not.toBeInTheDocument();
  expect(screen.queryByRole("checkbox")).not.toBeInTheDocument();
  expect(screen.queryByText("Compare saved settings")).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Show summary for a" }));
  expect(screen.getByText("Risk a")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Hide summary for a" })).toHaveAttribute("aria-expanded", "true");
  expect(within(screen.getByRole("article", { name: "Investing recommendation a" })).getByText("2016-01-01 → 2026-09-30")).toBeInTheDocument();
  expect(getRun).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Hide summary for a" }));
  expect(screen.queryByText("Risk a")).not.toBeInTheDocument();
});
it("combines search, type and status, can clear filters, and sorts deterministically", async () => {
  show();
  await screen.findByRole("link", { name: "Open details for a" });
  fireEvent.change(screen.getByLabelText("Search runs"), { target: { value: "RISK a" } });
  expect(screen.queryByRole("link", { name: "Open details for b" })).not.toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("Status"), { target: { value: "failed" } });
  expect(screen.getByText("No runs match these filters.")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Clear filters" }));
  const list = screen.getByRole("region", { name: "Runs" });
  expect(within(list).getAllByRole("article")[0]).toHaveAccessibleName("Solver benchmark b");
  fireEvent.change(screen.getByLabelText("Sort"), { target: { value: "oldest" } });
  expect(within(list).getAllByRole("article")[0]).toHaveAccessibleName("Investing recommendation a");
  fireEvent.change(screen.getByLabelText("Type"), { target: { value: "solver_comparison" } });
  expect(screen.queryByRole("link", { name: "Open details for a" })).not.toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Open details for b" })).toBeInTheDocument();
});
it("preserves filters, page and summary when returning from full details", async () => {
  const many = Array.from({ length: 14 }, (_, i) => fixture(`run${i + 1}`, i + 1));
  vi.mocked(listRuns).mockResolvedValue(many);
  show("/?status=completed&page=2&expanded=run4");
  fireEvent.click(await screen.findByRole("link", { name: "Open details for run4" }));
  fireEvent.click(screen.getByRole("button", { name: "Back to dashboard" }));
  expect(await screen.findByRole("link", { name: "Open details for run4" })).toBeInTheDocument();
  expect(screen.getByLabelText("Status")).toHaveValue("completed");
  expect(screen.getByText("Page 2 of 2")).toBeInTheDocument();
  expect(screen.getByText("Risk run4")).toBeInTheDocument();
  expect(screen.getByLabelText("Current URL")).toHaveTextContent("page=2");
  fireEvent.change(screen.getByLabelText("Search runs"), { target: { value: "run14" } });
  expect(screen.queryByLabelText("Run pages")).not.toBeInTheDocument();
  expect(screen.getByLabelText("Current URL")).not.toHaveTextContent("page=2");
});
it("clamps stale pagination when runs disappear and keeps the current filter on refresh", async () => {
  vi.mocked(listRuns).mockResolvedValue(Array.from({ length: 11 }, (_, i) => fixture(`run${i + 1}`, i + 1)));
  const client = show("/?status=completed&page=99");
  expect(await screen.findByText("Page 2 of 2")).toBeInTheDocument();
  await act(async () => { client.setQueryData(["runs"], [fixture("remaining")]); });
  expect(await screen.findByRole("link", { name: "Open details for remaining" })).toBeInTheDocument();
  expect(screen.getByLabelText("Status")).toHaveValue("completed");
});
it("handles retry, empty libraries and stale-data errors", async () => {
  vi.mocked(listRuns).mockRejectedValueOnce(new Error("offline")).mockResolvedValue(fixtures);
  const client = show();
  fireEvent.click(await screen.findByRole("button", { name: "Retry" }));
  expect(await screen.findByRole("link", { name: "Open details for a" })).toBeInTheDocument();
  vi.mocked(listRuns).mockRejectedValue(new Error("offline"));
  await act(async () => { await client.refetchQueries({ queryKey: ["runs"] }); });
  expect(await screen.findByRole("alert")).toHaveTextContent("Showing the last loaded list");
  expect(screen.getByRole("link", { name: "Open details for a" })).toBeInTheDocument();
  await act(async () => { client.setQueryData(["runs"], []); });
  expect(await screen.findByText(/No runs yet/)).toBeInTheDocument();
});
it("falls back safely from malformed URL filters and pagination", async () => {
  show("/?type=unknown&status=wrong&sort=other&page=NaN");
  expect(await screen.findByRole("link", { name: "Open details for a" })).toBeInTheDocument();
  expect(screen.getByLabelText("Type")).toHaveValue("all");
  expect(screen.getByLabelText("Status")).toHaveValue("all");
  expect(screen.getByLabelText("Sort")).toHaveValue("newest");
});

it.each([0, 1])("preserves library state through application Dashboard link %i", async (index) => {
  vi.mocked(listRuns).mockResolvedValue(Array.from({ length: 14 }, (_, i) => fixture(`run${i + 1}`, i + 1)));
  show("/?status=completed&page=2&expanded=run4");
  fireEvent.click(await screen.findByRole("link", { name: "Open details for run4" }));
  fireEvent.click(screen.getAllByRole("link", { name: "Dashboard" })[index]);
  expect(await screen.findByRole("link", { name: "Open details for run4" })).toBeInTheDocument();
  expect(screen.getByLabelText("Status")).toHaveValue("completed");
  expect(screen.getByText("Page 2 of 2")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Hide summary for run4" })).toHaveAttribute("aria-expanded", "true");
});

vi.mock("@/components/ui/Campaigns", () => ({CampaignPanel: () => null, CampaignSelect: () => null, RunCampaignAssignment: () => null}));

it("shows the campaign in each row and filters by clicking its name without loading results",async()=>{
 const name="A deliberately descriptive campaign name for multi-year validation";
 vi.mocked(listCampaigns).mockResolvedValue([{id:"aaaaaaaaaaaa",name,hypothesis:"",notes:"",created_at:"2026-10-01",updated_at:"2026-10-01",run_count:1}]);
 vi.mocked(listRuns).mockResolvedValue([{...fixtures[0],campaign_id:"aaaaaaaaaaaa"},fixtures[1]]);show();
 const row=await screen.findByRole("article",{name:"Investing recommendation a"});
 const campaign=await within(row).findByRole("button",{name:`Show campaign ${name}`});
 expect(campaign).toHaveAttribute("title",name);expect(within(screen.getByRole("article",{name:"Solver benchmark b"})).getByText("Ungrouped")).toBeInTheDocument();
 fireEvent.click(campaign);expect(screen.getByLabelText("Current URL")).toHaveTextContent("campaign=aaaaaaaaaaaa");
 expect(screen.queryByRole("link",{name:"Open details for b"})).not.toBeInTheDocument();expect(getRun).not.toHaveBeenCalled();
});
it("does not mislabel an unavailable campaign as ungrouped",async()=>{
 vi.mocked(listCampaigns).mockRejectedValue(new Error("Catalog unavailable"));
 vi.mocked(listRuns).mockResolvedValue([{...fixtures[0],campaign_id:"aaaaaaaaaaaa"}]);show();
 expect(await screen.findByText("Unavailable campaign")).toBeInTheDocument();expect(screen.queryByText("Ungrouped")).not.toBeInTheDocument();
});
