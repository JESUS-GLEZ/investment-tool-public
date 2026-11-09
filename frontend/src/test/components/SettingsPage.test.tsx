import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import * as api from "@/api/endpoints";
import SettingsPage from "@/pages/SettingsPage";
vi.mock("@/api/endpoints", () => ({ getDefaults: vi.fn(), getSymbols: vi.fn(), listAssetConfigs: vi.fn() }));
function mount() {
render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
    <MemoryRouter>
      <SettingsPage />
    </MemoryRouter>
  </QueryClientProvider>);
}
beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.listAssetConfigs).mockResolvedValue([{ file: "example_asset_universe.json", name: "Example", stock_count: 2, crypto_count: 1 }, { file: "expanded.json", name: "Expanded", stock_count: 300, crypto_count: 1, membership_mode: "dated", stock_limit: 300 }]);
  vi.mocked(api.getDefaults).mockResolvedValue({ portfolio: { transaction_cost_rate: 0.001 }, strategy_optimizer: { pop_size: 80, n_generations: 20000, n_restarts: 3, random_seed: 0, search_seconds: 300, seed_max_subsets: 1000, seed_beam_width: 5, seed_time_fraction: 0.4 }, optimizer: { pop_size: 300 } });
  vi.mocked(api.getSymbols).mockImplementation(async file => file === "expanded.json" ? { stocks: ["NVDA", "AMD"], crypto: ["BTC/USD"] } : { stocks: ["AAPL", "MSFT"], crypto: ["BTC/USD"] });
});
afterEach(cleanup);
it("separates form starting values, strategy search settings and engine references without promising unavailable edits", async () => {
  mount();
  expect(await within(screen.getByRole("region", { name: "Strategy search defaults" })).findByText("80")).toBeInTheDocument();
  expect(screen.queryByText(/coming soon/i)).not.toBeInTheDocument();
  expect(screen.getByRole("link", { name: "New Run" })).toHaveAttribute("href", "/runs/new");
  expect(screen.getByRole("region", { name: "Run starting values" })).toHaveTextContent("Trading cost rate0%");
  fireEvent.click(screen.getByText("Advanced engine reference"));
  fireEvent.click(screen.getByText("Portfolio and backtest engine"));
  expect(screen.getByRole("region", { name: "Engine reference defaults" })).toHaveTextContent("0.1%");
  expect(screen.getByText(/not account preferences/)).toBeInTheDocument();
});
it("scopes symbol results to the inspected universe and prevents stale symbols during a switch", async () => {
  mount(); await screen.findByText("AAPL");
  expect(api.getSymbols).toHaveBeenCalledWith("example_asset_universe.json");
  let resolve!: (data: { stocks: string[]; crypto: string[] }) => void;
  vi.mocked(api.getSymbols).mockReturnValueOnce(new Promise(done => { resolve = done; }));
  fireEvent.change(screen.getByLabelText("Universe to inspect"), { target: { value: "expanded.json" } });
  expect(screen.queryByText("AAPL")).not.toBeInTheDocument();
  expect(screen.getByRole("status")).toHaveTextContent("Loading available symbols");
  resolve({ stocks: ["NVDA", "AMD"], crypto: ["BTC/USD"] });
  await screen.findByText("NVDA");
  expect(api.getSymbols).toHaveBeenCalledWith("expanded.json");
  fireEvent.change(screen.getByLabelText("Search symbols"), { target: { value: "btc" } });
  expect(screen.getByRole("list", { name: "Universe symbols" })).toHaveTextContent("BTC/USD");
  expect(screen.queryByText("NVDA")).not.toBeInTheDocument();
});
it("does not fetch implicit default symbols when the universe catalog fails, and can retry", async () => {
  vi.mocked(api.listAssetConfigs).mockRejectedValueOnce(new Error("Catalog unavailable")); mount();
  expect(await screen.findByRole("alert")).toHaveTextContent("Catalog unavailable");
  expect(api.getSymbols).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Retry asset universes" }));
  await screen.findByText("AAPL");
});
it("keeps the full symbol list in a keyboard-focusable scroll panel and narrows it through search", async () => {
  vi.mocked(api.getSymbols).mockResolvedValue({ stocks: Array.from({ length: 120 }, (_, i) => `STOCK${i}`), crypto: ["BTC/USD"] }); mount();
  await screen.findByText("STOCK119");
  expect(screen.queryByText(/first 60 shown/)).not.toBeInTheDocument();
  expect(screen.getByRole("list", { name: "Universe symbols" })).toHaveAttribute("tabindex", "0");
  expect(within(screen.getByRole("list", { name: "Universe symbols" })).getAllByRole("listitem")).toHaveLength(121);
  fireEvent.change(screen.getByLabelText("Search symbols"), { target: { value: "stock119" } });
  await waitFor(() => expect(within(screen.getByRole("list", { name: "Universe symbols" })).getAllByRole("listitem")).toHaveLength(1));
});
