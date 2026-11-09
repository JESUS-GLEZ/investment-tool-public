import { cleanup, fireEvent, render, screen, within, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import * as api from "@/api/endpoints";
import DataLibraryPage from "@/pages/DataLibraryPage";
vi.mock("@/api/endpoints", () => ({ listLocalDataFiles: vi.fn(), getLocalDataInfo: vi.fn(), listRuns: vi.fn() }));
function mount(path = "/datasets") {
  render(<QueryClientProvider client={new QueryClient({defaultOptions: {queries: {retry: false}}})}><MemoryRouter initialEntries={[path]}><DataLibraryPage /></MemoryRouter></QueryClientProvider>);
}
beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.listLocalDataFiles).mockResolvedValue([{file: "one.pkl", title: "Research snapshot", frequency: "monthly", stock_count: 2, start_date: "2020-01-01", end_date: "2021-12-31"}, {file: "two.pkl", frequency: null, stock_count: null, start_date: null, end_date: null}]);
  vi.mocked(api.getLocalDataInfo).mockResolvedValue({file: "one.pkl", title: "Research snapshot", stocks: ["AAA", "BBB"], crypto: [], stock_count: 2, crypto_count: 0, start_date: "2020-01-01", end_date: "2021-12-31", features: {prices: true, execution_quotes: true, news: false}});
  vi.mocked(api.listRuns).mockResolvedValue([]);
});
afterEach(cleanup);
it("loads metadata only for the selected dataset, shows unknown provenance honestly and keeps technical filenames", async () => {
  mount(); await screen.findByText("Research snapshot");
  expect(api.getLocalDataInfo).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", {name: /Research snapshot/}));
  const details = screen.getByRole("region", {name: "Dataset details"});
  await within(details).findByText("Execution quotes");
  expect(api.getLocalDataInfo).toHaveBeenCalledWith("one.pkl");
  expect(within(details).getByText("Cached sentiment").parentElement).toHaveTextContent("Unknown");
  expect(within(details).getByText("News records").parentElement).toHaveTextContent("Absent");
  expect(within(details).getByText("one.pkl")).toBeInTheDocument();
  await waitFor(() => expect(api.listLocalDataFiles).toHaveBeenCalledTimes(2));
  expect(api.getLocalDataInfo).toHaveBeenCalledTimes(1);
});
it("does not inspect an unavailable URL filename and filters the dataset list", async () => {
  mount("/datasets?file=missing.pkl");
  await screen.findByRole("alert");
  expect(api.getLocalDataInfo).not.toHaveBeenCalled();
  fireEvent.change(screen.getByLabelText("Search datasets"), {target: {value: "Research"}});
  expect(screen.queryByRole("button", {name: /two.pkl/})).not.toBeInTheDocument();
});
