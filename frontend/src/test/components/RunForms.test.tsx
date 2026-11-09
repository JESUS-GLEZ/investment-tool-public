import { act, fireEvent, render, screen, waitFor, cleanup, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { periodInputValue } from "@/lib/periodDates";
import type { RunSummary } from "@/types/api";
import * as api from "@/api/endpoints";
import NewRunPage from "@/pages/NewRunPage";
import StrategyExperimentForm from "@/components/ui/StrategyExperimentForm";
import LocalDataPanel from "@/components/ui/LocalDataPanel";
import AssetUniverseSelect from "@/components/ui/AssetUniverseSelect";
import PresetControls from "@/components/ui/PresetControls";

vi.mock("@/api/endpoints", () => ({
  createRun: vi.fn(), createStrategyComparison: vi.fn(), createSolverComparison: vi.fn(), createPredictionComparison: vi.fn(),
  listPresets: vi.fn(), savePreset: vi.fn(), deletePreset: vi.fn(), listLocalDataFiles: vi.fn(), getLocalDataInfo: vi.fn(),
  listAssetConfigs: vi.fn(), getSymbols: vi.fn(),
  getDefaults: vi.fn(), listCampaigns: vi.fn(),
}));

function selectTask(kind: string) {fireEvent.change(screen.getByRole("combobox", {name: "What would you like to run?"}), {target: {value: kind === "solver" ? "solvers" : kind === "prediction" ? "predictors" : "strategy-comparison"}});}

it("keeps campaign editing out of recommendations and available for research tasks", async () => {
  await mount(<NewRunPage />);
  expect(screen.queryByLabelText("Campaign (optional)")).not.toBeInTheDocument();
  selectTask("strategy");
  expect(await screen.findByLabelText("Campaign (optional)")).toBeInTheDocument();
  fireEvent.change(screen.getByRole("combobox", {name: "What would you like to run?"}), {target: {value: "investment"}});
  expect(screen.queryByLabelText("Campaign (optional)")).not.toBeInTheDocument();
});

it("reviews fractional holdings with zero USD cash before submitting units", async () => {
  vi.mocked(api.createRun).mockResolvedValue({id:"holding-run",run_type:"execution",status:"queued",created_at:"2026-10-01",completed_at:null,error_message:null});
  await mount(<NewRunPage />);
  fireEvent.change(screen.getByLabelText("Available cash (USD)"), {target:{value:"0"}});
  expect(screen.getByRole("button",{name:"Review run"})).toBeDisabled();
  await waitFor(() => expect(screen.getByRole("button",{name:"Add holding"})).toBeEnabled());
  fireEvent.click(screen.getByRole("button",{name:"Add holding"}));
  fireEvent.change(screen.getByLabelText("Asset 1"), {target:{value:"BTC/USD"}});
  fireEvent.keyDown(screen.getByLabelText("Asset 1"),{key:"Enter"});
  fireEvent.change(screen.getByLabelText("Shares / units 1"), {target:{value:"0.003"}});
  fireEvent.click(screen.getByRole("button",{name:"Review run"}));
  expect(screen.getByText("BTC/USD: 0.003 shares/units")).toBeInTheDocument();
  expect(api.createRun).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button",{name:"Start run"}));
  await waitFor(() => expect(api.createRun).toHaveBeenCalledOnce());
  expect(vi.mocked(api.createRun).mock.calls[0][0]).toMatchObject({initial_cash:0,execution_mode:"investing",current_holdings:[{symbol:"BTC/USD",quantity:0.003}]});
});

it("preserves holdings for review when their universe changes", async () => {
  vi.mocked(api.getSymbols).mockImplementation(async file => ({stocks:file === "custom_example.json" ? ["MSFT"] : ["AAPL"],crypto:[]}));
  await mount(<NewRunPage />);
  await waitFor(() => expect(screen.getByRole("button",{name:"Add holding"})).toBeEnabled());
  fireEvent.click(screen.getByRole("button",{name:"Add holding"}));
  fireEvent.change(screen.getByLabelText("Asset 1"),{target:{value:"AAPL"}});
  fireEvent.keyDown(screen.getByLabelText("Asset 1"),{key:"Enter"});
  fireEvent.change(screen.getByLabelText("Shares / units 1"),{target:{value:"2.5"}});
  fireEvent.change(screen.getByLabelText("Asset universe"),{target:{value:"custom_example.json"}});
  await screen.findByText("Choose an eligible symbol from this universe.");
  expect(screen.getByLabelText("Asset 1")).toHaveValue("AAPL");
  expect(screen.getByLabelText("Shares / units 1")).toHaveValue(2.5);
  expect(screen.getByRole("button",{name:"Review run"})).toBeDisabled();
  expect(api.createRun).not.toHaveBeenCalled();
});

it("restores holdings from a preset without leaking them into backtests", async () => {
  vi.mocked(api.listPresets).mockResolvedValue([{id:199,name:"My actual holdings",run_type:"execution",parameters:{execution_mode:"investing",initial_cash:0,current_holdings:[{symbol:"AAPL",quantity:2.5}]}}]);
  await mount(<NewRunPage />);
  await screen.findByRole("option",{name:"My actual holdings"});
  fireEvent.change(screen.getByLabelText("Saved preset"),{target:{value:"199"}});
  expect(screen.getByLabelText("Shares / units 1")).toHaveValue(2.5);
  expect(screen.getByLabelText("Available cash (USD)")).toHaveValue(0);
  fireEvent.change(screen.getByRole("combobox",{name:"What would you like to run?"}),{target:{value:"strategy-comparison"}});
  expect(screen.queryByLabelText("Shares / units 1")).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button",{name:"Review run"}));
  expect(JSON.parse(screen.getByText(/"strategy_variants":/, {selector:"pre"}).textContent!)).not.toHaveProperty("current_holdings");
});

async function mount(element: React.ReactNode, url = "/runs/new") {
  const result = render(<QueryClientProvider client={new QueryClient({defaultOptions: {queries: {retry: false}}})}><MemoryRouter initialEntries={[url]}>{element}</MemoryRouter></QueryClientProvider>);
  await waitFor(() => expect(screen.queryAllByText("Loading strategy defaults…")).toHaveLength(0));
  return result;
}
function setDate(label: string, date: string) {
  const field = screen.getByLabelText(label) as HTMLInputElement;
  const frequency = field.type === "week" ? "weekly" : field.type === "month" ? "monthly" : "daily";
  fireEvent.change(field, {target:{value:periodInputValue(date, frequency)}});
}
function submitAttempt() {
  const review = screen.queryByRole("button", {name: "Review run"});
  if (review) fireEvent.click(review);
  const launch = screen.queryByRole("button", {name: "Start run"});
  if (launch) fireEvent.click(launch);
}

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.listCampaigns).mockResolvedValue([]);
  vi.mocked(api.getSymbols).mockResolvedValue({stocks: ["AAPL", "MSFT", "GLD"], crypto: ["BTC/USD"]});
  vi.mocked(api.getDefaults).mockResolvedValue({strategy_optimizer: {pop_size:300,n_generations:200000,n_restarts:3,random_seed:0,search_seconds:300,seed_max_subsets:10000,seed_beam_width:5,seed_time_fraction:0.4,refinement_supports:80}});
  vi.mocked(api.listPresets).mockResolvedValue([]);
  vi.mocked(api.listAssetConfigs).mockResolvedValue([
    {file: "example_asset_universe.json", name: "Example", stock_count: 6, crypto_count: 1},
    {file: "custom_example.json", name: "Expanded", stock_count: 2, crypto_count: 1},
  ]);
  vi.mocked(api.listLocalDataFiles).mockResolvedValue([{file: "prices.pkl", frequency: "monthly", start_date: "2020-01-01", end_date: "2026-01-31", stock_count: 5}]);
  vi.mocked(api.getLocalDataInfo).mockResolvedValue({file: "prices.pkl", frequency: "monthly", start_date: "2020-01-01", end_date: "2026-01-31", stock_count: 5, crypto_count: 0, stocks: ["ABC"], crypto: []});
});
afterEach(cleanup);

it("retains the configured strategy after a preset rejection and retries without launching", async () => {
  vi.mocked(api.savePreset).mockRejectedValueOnce(new Error("Temporary preset failure"));
  vi.mocked(api.savePreset).mockResolvedValueOnce({id: 41, name: "Bounds", run_type: "execution", parameters: {}});
  await mount(<StrategyExperimentForm mode="execution" />);
  fireEvent.click(screen.getByRole("button", {name: "Edit Combined 1"}));
  fireEvent.change(screen.getByLabelText("Min Assets"), {target: {value: "3"}});
  fireEvent.click(screen.getByRole("button", {name: "Save configuration"}));
  fireEvent.change(screen.getByLabelText("Preset name"), {target: {value: "Bounds"}});
  fireEvent.click(screen.getByRole("button", {name: "Save as new"}));
  expect(await screen.findByRole("alert")).toHaveTextContent("Temporary preset failure");
  expect(screen.getByRole("listitem", {name: "Combined 1"})).toHaveTextContent("3–5 assets");
  expect(screen.getByRole("spinbutton", {name: /^Initial Cash/})).toHaveValue(1000);
  fireEvent.click(screen.getByRole("button", {name: "Save as new"}));
  expect(await screen.findByRole("status")).toHaveTextContent("Preset saved. No run has been launched.");
  expect(api.savePreset).toHaveBeenCalledTimes(2);
  expect(vi.mocked(api.savePreset).mock.calls[1][0].parameters).toMatchObject({strategy_configuration: {min_assets: 3, max_assets: 5}, initial_cash: 1000});
  expect(api.createRun).not.toHaveBeenCalled();
});

describe("visible form labels", () => {
  const commonLabels = ["Data Source", "Frequency", "Validation Level", "Start Date", "End Date"];
  const cases = [
    {kind: "execution", labels: [...commonLabels, "Backtest start date (optional)"]},
    {kind: "strategy", labels: [...commonLabels, "Backtest start date (optional)"]},
    {kind: "solver", labels: [...commonLabels, "Population size", "Evaluations per restart", "Restarts", "Min Assets", "Max Assets"]},
    {kind: "prediction", labels: [...commonLabels, "Eval Periods", "Min History", "Top K"]},
  ];

  it.each(cases)("$kind: visible labels focus their controls and preserve local frequency locking", async ({kind, labels}) => {
    const user = userEvent.setup();
    await mount(kind === "execution" ? <StrategyExperimentForm mode="execution" /> : <NewRunPage defaultTask="strategy-comparison" />);
    if (kind === "solver" || kind === "prediction") selectTask(kind);
    for (const label of labels) {
      const control = screen.getByLabelText(label);
      const visibleLabel = screen.getByText(label, {selector: "label"});
      expect((visibleLabel as HTMLLabelElement).control).toBe(control);
      await user.click(visibleLabel);
      expect(control).toHaveFocus();
    }
    const source = screen.getByLabelText("Data Source");
    await user.click(source);
    await user.tab();
    expect(screen.getByRole("button", {name: "Explain Frequency"})).toHaveFocus();
    await user.tab();
    expect(screen.getByLabelText("Frequency")).toHaveFocus();
    await user.selectOptions(source, "local");
    expect(screen.getByLabelText("Frequency")).toBeDisabled();
    await user.selectOptions(source, "alpaca");
    expect(screen.getByLabelText("Frequency")).toBeEnabled();
  });

  it("keeps labels scoped to their form when multiple instances are mounted", async () => {
    const user = userEvent.setup();
    const {container} = await mount(<><StrategyExperimentForm mode="execution" /><StrategyExperimentForm mode="execution" /><NewRunPage defaultTask="strategy-comparison" /><NewRunPage defaultTask="strategy-comparison" /></>);
    const changedLabels = new Set(cases.flatMap(({labels}) => labels));
    const associatedLabels = Array.from(container.querySelectorAll<HTMLLabelElement>("label[for]")).filter(label => changedLabels.has(label.textContent ?? ""));
    const targets = associatedLabels.map(label => label.htmlFor);
    expect(new Set(targets).size).toBe(targets.length);
    for (const text of ["Data Source", "Start Date", "Backtest start date (optional)"]) {
      for (const label of screen.getAllByText(text, {selector: "label"})) {
        const control = (label as HTMLLabelElement).control;
        expect(control).not.toBeNull();
        await user.click(label);
        expect(control).toHaveFocus();
      }
    }
  });
});



describe("shared catalog recovery", () => {
  it.each(["execution", "strategy", "solver", "prediction"] as const)("%s: retries each failed catalog independently without losing values or writing", async kind => {
    const user = userEvent.setup();
    vi.mocked(api.listPresets).mockRejectedValue(new Error("Saved presets offline"));
    vi.mocked(api.listAssetConfigs).mockRejectedValue(new Error("Universes offline"));
    await mount(kind === "execution" ? <StrategyExperimentForm mode="execution" /> : <NewRunPage defaultTask="strategy-comparison" />);
    if (kind === "solver" || kind === "prediction") selectTask(kind);
    await screen.findByRole("button", {name: "Retry asset universes"});
    await screen.findByRole("button", {name: "Retry saved presets"});
    await waitFor(() => expect(screen.getByRole("button", {name: "Retry asset universes"})).toBeEnabled());
    const catalogReads = vi.mocked(api.listAssetConfigs).mock.calls.length;
    vi.mocked(api.listPresets).mockResolvedValue([]);
    fireEvent.change(screen.getByLabelText("Preset name"), {target: {value: "Keep entered name"}});
    fireEvent.change(screen.getByRole("spinbutton", {name: /^Initial Cash/}), {target: {value: "1700"}});
    await user.click(screen.getByRole("button", {name: "Retry saved presets"}));
    await waitFor(() => expect(screen.queryByRole("button", {name: "Retry saved presets"})).not.toBeInTheDocument());
    expect(screen.getByRole("button", {name: "Retry asset universes"})).toBeInTheDocument();
    expect(api.listAssetConfigs).toHaveBeenCalledTimes(catalogReads);
    vi.mocked(api.listAssetConfigs).mockResolvedValue([
      {file: "example_asset_universe.json", name: "Example", stock_count: 6, crypto_count: 1},
      {file: "custom_example.json", name: "Expanded", stock_count: 2, crypto_count: 1},
    ]);
    await user.click(screen.getByRole("button", {name: "Retry asset universes"}));
    await screen.findByRole("option", {name: /Expanded — 2 stocks/});
    expect(screen.getByLabelText("Preset name")).toHaveValue("Keep entered name");
    expect(screen.getByRole("spinbutton", {name: /^Initial Cash/})).toHaveValue(1700);
    expect(screen.getByLabelText("Asset universe")).toHaveValue("example_asset_universe.json");
    expect(api.savePreset).not.toHaveBeenCalled();
    expect(api.deletePreset).not.toHaveBeenCalled();
    expect(api.createRun).not.toHaveBeenCalled();
    expect(api.createStrategyComparison).not.toHaveBeenCalled();
    expect(api.createSolverComparison).not.toHaveBeenCalled();
    expect(api.createPredictionComparison).not.toHaveBeenCalled();
  });

  it("shows loading announcements for both catalogs", async () => {
    vi.mocked(api.listPresets).mockReturnValue(new Promise(() => {}));
    vi.mocked(api.listAssetConfigs).mockReturnValue(new Promise(() => {}));
    await mount(<StrategyExperimentForm mode="execution" />);
    expect(screen.getByText("Loading saved presets…")).toHaveAttribute("role", "status");
    expect(screen.getByText("Loading asset universes…")).toHaveAttribute("role", "status");
  });

  it("retains cached presets and typed names after refresh failure, guarding duplicate retries", async () => {
    const user = userEvent.setup();
    const client = new QueryClient({defaultOptions: {queries: {retry: false}}});
    const saved = [{id: 81, name: "Cached preset", run_type: "execution" as const, parameters: {initial_cash: 1000}}];
    vi.mocked(api.listPresets).mockResolvedValue(saved);
    const load = vi.fn();
    render(<QueryClientProvider client={client}><PresetControls runType="execution" value={{initial_cash: 1700}} onLoad={load} /></QueryClientProvider>);
    await screen.findByRole("option", {name: "Cached preset"});
    await user.selectOptions(screen.getByLabelText("Saved preset"), "81");
    fireEvent.change(screen.getByLabelText("Preset name"), {target: {value: "Edited name"}});
    vi.mocked(api.listPresets).mockRejectedValueOnce(new Error("Refresh offline"));
    await act(async () => { await client.invalidateQueries({queryKey: ["presets", "execution"]}); });
    expect(await screen.findByRole("alert")).toHaveTextContent("Refresh offline");
    expect(screen.getByLabelText("Saved preset")).toHaveValue("81");
    expect(screen.getByRole("option", {name: "Cached preset"})).toBeInTheDocument();
    let finish!: (items: typeof saved) => void;
    const response = new Promise<typeof saved>(resolve => { finish = resolve; });
    vi.mocked(api.listPresets).mockReturnValueOnce(response);
    await user.click(screen.getByRole("button", {name: "Retry saved presets"}));
    const retry = screen.getByRole("button", {name: "Retry saved presets"});
    expect(retry).toBeDisabled();
    await user.click(retry);
    expect(api.listPresets).toHaveBeenCalledTimes(3);
    await act(async () => { finish(saved); await response; });
    await waitFor(() => expect(screen.queryByRole("button", {name: "Retry saved presets"})).not.toBeInTheDocument());
    expect(screen.getByLabelText("Saved preset")).toHaveValue("81");
    expect(screen.getByLabelText("Preset name")).toHaveValue("Edited name");
    expect(load).toHaveBeenCalledTimes(1);
    expect(api.savePreset).not.toHaveBeenCalled();
    expect(api.deletePreset).not.toHaveBeenCalled();
  });

  it("retains cached universes and the selected file while retrying a refresh", async () => {
    const user = userEvent.setup();
    const client = new QueryClient({defaultOptions: {queries: {retry: false}}});
    const changed = vi.fn();
    render(<QueryClientProvider client={client}><AssetUniverseSelect value="custom_example.json" onChange={changed} /></QueryClientProvider>);
    await screen.findByRole("option", {name: /Expanded — 2 stocks/});
    vi.mocked(api.listAssetConfigs).mockRejectedValueOnce(new Error("Catalog offline"));
    await act(async () => { await client.invalidateQueries({queryKey: ["asset-configs"]}); });
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not load asset universes.");
    expect(screen.getByLabelText("Asset universe")).toHaveValue("custom_example.json");
    expect(screen.getByRole("option", {name: /Expanded — 2 stocks/})).toBeInTheDocument();
    const refreshed = [{file: "custom_example.json", name: "Expanded", stock_count: 3, crypto_count: 1}];
    let finish!: (items: typeof refreshed) => void;
    const response = new Promise<typeof refreshed>(resolve => { finish = resolve; });
    vi.mocked(api.listAssetConfigs).mockReturnValueOnce(response);
    await user.click(screen.getByRole("button", {name: "Retry asset universes"}));
    const retry = screen.getByRole("button", {name: "Retry asset universes"});
    expect(retry).toBeDisabled();
    await user.click(retry);
    expect(api.listAssetConfigs).toHaveBeenCalledTimes(3);
    await act(async () => { finish(refreshed); await response; });
    await screen.findByRole("option", {name: /Expanded — 3 stocks/});
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Asset universe")).toHaveValue("custom_example.json");
    expect(changed).not.toHaveBeenCalled();
  });

  it("does not offer a catalog retry for a preset write failure", async () => {
    vi.mocked(api.savePreset).mockRejectedValueOnce(new Error("Preset write failed"));
    await mount(<PresetControls runType="execution" value={{initial_cash: 1000}} onLoad={vi.fn()} />);
    fireEvent.change(screen.getByLabelText("Preset name"), {target: {value: "Write"}});
    fireEvent.click(screen.getByRole("button", {name: "Save as new"}));
    expect(await screen.findByRole("alert")).toHaveTextContent("Preset write failed");
    expect(screen.queryByRole("button", {name: "Retry saved presets"})).not.toBeInTheDocument();
  });

  it("keeps repeated asset-universe labels connected to their own selector", async () => {
    const user = userEvent.setup();
    await mount(<><AssetUniverseSelect value="example_asset_universe.json" onChange={vi.fn()} /><AssetUniverseSelect value="custom_example.json" onChange={vi.fn()} /></>);
    const controls = screen.getAllByLabelText("Asset universe");
    expect(controls[0].id).not.toBe(controls[1].id);
    for (const label of screen.getAllByText("Asset universe", {selector: "label"})) {
      await user.click(label);
      expect((label as HTMLLabelElement).control).toHaveFocus();
    }
  });
});

describe("native preset form validation", () => {
  for (const kind of ["execution", "strategy", "solver", "prediction"] as const) {
    const numericLabel = kind === "prediction" ? "Eval Periods" : "Min Assets";
    const cases = [
      ["Start Date", "", "2020-01-01"],
      ["End Date", "", "2026-01-31"],
      ["Start Date", "2030-01", "2020-01-01"],
      ["End Date", "2019-12", "2026-01-31"],
      ...(kind === "solver" || kind === "prediction" ? [[numericLabel, "0", "2"], [numericLabel, "1.5", "2"]] : []),
    ];
    it.each(cases)(`${kind}: rejects %s=%s before preset creation and saves the correction`, async (label, invalid, valid) => {
      vi.mocked(api.savePreset).mockResolvedValue({id: 71, name: "Native", run_type: kind === "execution" ? "execution" : `${kind}_comparison`, parameters: {}});
      await mount(kind === "execution" ? <StrategyExperimentForm mode="execution" /> : <NewRunPage defaultTask="strategy-comparison" />);
      if (kind === "solver" || kind === "prediction") selectTask(kind);
      await waitFor(() => expect(screen.getByRole("button", {name: "Review run"})).toBeEnabled());
      fireEvent.change(screen.getByLabelText("Preset name"), {target: {value: "Native"}});
        const field = screen.getByLabelText(label);
      fireEvent.change(field, {target: {value: invalid}});
      expect(field).toBeInvalid();
      fireEvent.click(screen.getByRole("button", {name: "Save as new"}));
      if (label.includes("Date")) expect(screen.getByRole("button", {name:"Save as new"})).toBeDisabled();
      else expect(screen.getByRole("alert")).toHaveTextContent("Check the highlighted form fields before saving this preset.");
      expect(api.savePreset).not.toHaveBeenCalled();
      expect(api.createRun).not.toHaveBeenCalled();
      expect(api.createStrategyComparison).not.toHaveBeenCalled();
      expect(api.createSolverComparison).not.toHaveBeenCalled();
      expect(api.createPredictionComparison).not.toHaveBeenCalled();
      expect(screen.getByLabelText("Preset name")).toHaveValue("Native");
      if ((label === "Start Date" || label === "End Date") && valid.includes("-")) setDate(label, valid);
      else fireEvent.change(field, {target: {value: valid}});
      expect(field).toBeValid();
      fireEvent.click(screen.getByRole("button", {name: "Save as new"}));
      await waitFor(() => expect(api.savePreset).toHaveBeenCalledTimes(1));
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
      expect(api.createRun).not.toHaveBeenCalled();
      expect(api.createStrategyComparison).not.toHaveBeenCalled();
      expect(api.createSolverComparison).not.toHaveBeenCalled();
      expect(api.createPredictionComparison).not.toHaveBeenCalled();
    });
  }

  it("validates updates while allowing load/delete and an optional blank field", async () => {
    vi.mocked(api.listPresets).mockResolvedValue([{id: 79, name: "Existing", run_type: "execution", parameters: {initial_cash: 1000}}]);
    vi.mocked(api.savePreset).mockResolvedValue({id: 79, name: "Existing", run_type: "execution", parameters: {initial_cash: 1000}});
    vi.mocked(api.deletePreset).mockResolvedValue(undefined);
    const load = vi.fn();
    await mount(<form><PresetControls runType="execution" value={{initial_cash: 1000}} onLoad={load} /><label>Required date<input aria-label="Required date" type="date" required defaultValue="2020-01-01" /></label><input aria-label="Optional date" type="date" /></form>);
    await screen.findByRole("option", {name: "Existing"});
    fireEvent.change(screen.getByLabelText("Saved preset"), {target: {value: "79"}});
    const field = screen.getByLabelText("Required date");
    fireEvent.change(field, {target: {value: ""}});
    fireEvent.click(screen.getByRole("button", {name: "Update selected"}));
    expect(api.savePreset).not.toHaveBeenCalled();
    expect(screen.getByRole("alert")).toHaveTextContent("Check the highlighted");
    fireEvent.change(screen.getByLabelText("Saved preset"), {target: {value: ""}});
    fireEvent.change(screen.getByLabelText("Saved preset"), {target: {value: "79"}});
    expect(load).toHaveBeenCalledTimes(2);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    fireEvent.change(field, {target: {value: "2020-01-01"}});
    fireEvent.click(screen.getByRole("button", {name: "Update selected"}));
    await waitFor(() => expect(api.savePreset).toHaveBeenCalledWith({name: "Existing", run_type: "execution", parameters: {initial_cash: 1000}}, 79));
    await waitFor(() => expect(screen.getByRole("button", {name: "Delete selected"})).toBeEnabled());
    fireEvent.change(field, {target: {value: ""}});
    fireEvent.click(screen.getByRole("button", {name: "Delete selected"}));
    await waitFor(() => expect(api.deletePreset).toHaveBeenCalledWith(79));
    expect(api.savePreset).toHaveBeenCalledTimes(1);
  });
});

describe("portfolio numbers", () => {
  it("keeps preset loading and deletion available while invalid fields block an update", async () => {
    vi.mocked(api.listPresets).mockResolvedValue([{id: 21, name: "Zero rates", run_type: "execution", parameters: {
      data_source: "alpaca", execution_mode: "backtesting", frequency: "monthly", strategy_type: "combined", validation_level: "warn", min_assets: 2, max_assets: 5,
      historical_start_date: "2020-01-01", historical_end_date: "2026-01-31", asset_config_file: "example_asset_universe.json", initial_cash: 1500, transaction_cost_rate: 0, risk_free_rate: 0,
    }}]);
    await mount(<StrategyExperimentForm mode="execution" />);
    await screen.findByRole("option", {name: "Zero rates"});
    const presets = screen.getByLabelText("Saved preset");
    fireEvent.change(presets, {target: {value: "21"}});
    const cost = screen.getByRole("spinbutton", {name: /^Transaction Cost/});
    expect(cost).toHaveValue(0);
    fireEvent.change(cost, {target: {value: ""}});
    expect(screen.getByRole("button", {name: "Update selected"})).toBeDisabled();
    expect(screen.getByRole("button", {name: "Delete selected"})).toBeEnabled();
    expect(presets).toBeEnabled();
    fireEvent.click(screen.getByRole("button", {name: "Update selected"}));
    expect(api.savePreset).not.toHaveBeenCalled();
    fireEvent.change(presets, {target: {value: "21"}});
    expect(cost).toHaveValue(0);
    expect(screen.getByRole("button", {name: "Update selected"})).toBeEnabled();
    vi.mocked(api.createRun).mockResolvedValue({id: "test"} as RunSummary);
    submitAttempt();
    await waitFor(() => expect(api.createRun).toHaveBeenCalledTimes(1));
    expect(vi.mocked(api.createRun).mock.calls[0][0]).toMatchObject({initial_cash: 1500, transaction_cost_rate: 0, risk_free_rate: 0});
  });

  const kinds = ["execution", "strategy", "solver", "prediction"] as const;
  const invalidFields = [
    ["Initial Cash", "", "1000"],
    ["Transaction Cost", "", "0"],
    ["Risk-free Rate", "", "0"],
    ["Initial Cash", "0", "1000"],
    ["Transaction Cost", "-0.001", "0"],
    ["Transaction Cost", "0.2", "0"],
    ["Risk-free Rate", "-0.01", "0"],
    ["Risk-free Rate", "1.1", "0"],
  ];

  for (const kind of kinds) {
    it.each(invalidFields)(`${kind}: blocks invalid %s=%s in both runs and presets, then recovers`, async (label, invalid, valid) => {
      await mount(kind === "execution" ? <StrategyExperimentForm mode="execution" /> : <NewRunPage defaultTask="strategy-comparison" />);
      if (kind === "solver" || kind === "prediction") selectTask(kind);
      const launch = screen.getByRole("button", {name: "Review run"});
      await waitFor(() => expect(launch).toBeEnabled());
      fireEvent.change(screen.getByLabelText("Preset name"), {target: {value: "finite"}});
      const save = screen.getByRole("button", {name: "Save as new"});
      expect(save).toBeEnabled();
      const field = screen.getByRole("spinbutton", {name: new RegExp(`^${label}`)});
      fireEvent.change(field, {target: {value: invalid}});
      expect(field).toHaveValue(invalid === "" ? null : Number(invalid));
      expect(field).toHaveAttribute("aria-invalid", "true");
      expect(field).toHaveAccessibleDescription(/requires a finite number|must be/);
      expect(save).toBeDisabled();
      expect(save).toHaveAccessibleDescription(/Resolve the configuration issue/);
      expect(launch).toBeDisabled();
      fireEvent.click(save);
      submitAttempt();
      const submit = {execution: api.createRun, strategy: api.createStrategyComparison, solver: api.createSolverComparison, prediction: api.createPredictionComparison}[kind];
      expect(api.savePreset).not.toHaveBeenCalled();
      expect(submit).not.toHaveBeenCalled();
      fireEvent.change(field, {target: {value: valid}});
      expect(field).toHaveAttribute("aria-invalid", "false");
      expect(save).toBeEnabled();
      expect(launch).toBeEnabled();
      vi.mocked(submit).mockResolvedValue({id: "test"} as RunSummary);
      submitAttempt();
      await waitFor(() => expect(submit).toHaveBeenCalledTimes(1));
      expect(vi.mocked(submit).mock.calls[0][0]).toMatchObject({initial_cash: 1000, transaction_cost_rate: 0, risk_free_rate: 0});
    });

    it(`${kind}: preserves explicit zero rates when saving a valid preset`, async () => {
      await mount(kind === "execution" ? <StrategyExperimentForm mode="execution" /> : <NewRunPage defaultTask="strategy-comparison" />);
      if (kind === "solver" || kind === "prediction") selectTask(kind);
      await waitFor(() => expect(screen.getByRole("button", {name: "Review run"})).toBeEnabled());
      fireEvent.change(screen.getByRole("spinbutton", {name: /^Initial Cash/}), {target: {value: "1500"}});
      fireEvent.change(screen.getByLabelText("Preset name"), {target: {value: "finite"}});
      // A save response does not launch a run; verify the outgoing effective values.
      vi.mocked(api.savePreset).mockResolvedValue({id: 99, name: "finite", run_type: kind === "execution" ? "execution" : `${kind}_comparison`, parameters: {initial_cash: 1500, transaction_cost_rate: 0, risk_free_rate: 0}});
      fireEvent.click(screen.getByRole("button", {name: "Save as new"}));
      await waitFor(() => expect(api.savePreset).toHaveBeenCalledTimes(1));
      expect(vi.mocked(api.savePreset).mock.calls[0][0].parameters).toMatchObject({initial_cash: 1500, transaction_cost_rate: 0, risk_free_rate: 0});
      expect(api.createRun).not.toHaveBeenCalled();
      expect(api.createStrategyComparison).not.toHaveBeenCalled();
      expect(api.createSolverComparison).not.toHaveBeenCalled();
      expect(api.createPredictionComparison).not.toHaveBeenCalled();
    });
  }
});

it.each([
  ["custom_example.json", "capital_weights", "monthly"],
  ["archived_example.json", "capital_weights", "monthly"],
  ["custom_example.json", "hybrid", "monthly"],
  ["custom_example.json", "hybrid", "weekly"],
  ["custom_example.json", "hybrid", "daily"],
])("restores named configurations and the saved universe %s with %s", async (assetConfig, balance, frequency) => {
  vi.mocked(api.listPresets).mockResolvedValue([{id: 5, name: "Paired configuration", run_type: "strategy_comparison", parameters: {
    strategy_types: ["optimization"], strategy_variants: [{name: "High return", strategy_type: "optimization", balance, news_shortlist: true, news_top_assets: 40, news_min_articles: 4, return_estimator: "ewm_12_mean_50", selection: "return_target", return_target_fraction: 0.6, volatility_increase: 0.25, n_restarts: 4, random_seed: 29, search_seconds: null, seed_max_subsets: 1000}],
    data_source: "alpaca", asset_config_file: assetConfig, frequency, validation_level: "warn", min_assets: 2, max_assets: 5, population_size: 300, n_generations: 200000,
    historical_start_date: "2020-01-01", historical_end_date: "2026-01-31", backtest_start_date: frequency === "weekly" ? "2023-10-02" : "2023-10-01", initial_cash: 1000, transaction_cost_rate: 0, risk_free_rate: 0, input_file: null,
  }}]);
  await mount(<NewRunPage defaultTask="strategy-comparison" />);
  await screen.findByRole("option", {name: "Paired configuration"});
  fireEvent.change(screen.getByLabelText("Saved preset"), {target: {value: "5"}});
  fireEvent.click(screen.getByRole("button", {name: "Edit High return"}));
  const variant = within(screen.getByRole("dialog"));
  expect(variant.getByLabelText("Position in feasible return range (%)")).toHaveValue(60);
  expect(variant.getByLabelText("Return estimator")).toHaveValue("ewm_12_mean_50");
  expect(variant.getByLabelText("Portfolio balance")).toHaveValue(balance);
  expect(variant.getByLabelText("News shortlist")).toBeChecked();
  expect(variant.getByLabelText("Top scored assets")).toHaveValue(40);
  expect(variant.getByLabelText("Minimum distinct articles")).toHaveValue(4);
  expect(screen.getByLabelText("Backtest start date (optional)")).toHaveValue(frequency === "monthly" ? "2023-10" : frequency === "weekly" ? "2023-W40" : "2023-10-01");
  expect(variant.getByLabelText("Independent restarts")).toHaveValue(4);
  expect(variant.getByLabelText("Random seed")).toHaveValue(29);
  expect(variant.getByLabelText("Fixed-work mode (no time cutoff)")).toBeChecked();
  expect(screen.getByLabelText("Asset universe")).toHaveValue(assetConfig);
  if (assetConfig.startsWith("broad_")) {
    expect(screen.getByRole("option", {name: `${assetConfig} (not in active catalog)`})).toBeInTheDocument();
  }
});

it("submits per-instance portfolio selection as a fraction", async () => {
  vi.mocked(api.createStrategyComparison).mockResolvedValue({id: "test"} as RunSummary);
  await mount(<NewRunPage defaultTask="strategy-comparison" />);
  fireEvent.click(screen.getByRole("button", {name: "Edit Optimization 1"}));
  fireEvent.change(screen.getByLabelText("Portfolio selection"), {target: {value: "return_with_risk_budget"}});
  fireEvent.change(screen.getByLabelText("Maximum increase in estimated volatility (%)"), {target: {value: "20"}});
  fireEvent.click(screen.getByRole("button", {name: "Save configuration"}));
  submitAttempt();
  await waitFor(() => expect(api.createStrategyComparison).toHaveBeenCalled());
  expect(vi.mocked(api.createStrategyComparison).mock.calls[0][0].strategy_variants![1]).toMatchObject({selection: "return_with_risk_budget", volatility_increase: 0.2});
});

it.each(["execution", "strategy", "solver", "prediction"] as const)(
  "submits the selected asset universe for %s",
  async (kind) => {
    await mount(kind === "execution" ? <StrategyExperimentForm mode="execution" /> : <NewRunPage defaultTask="strategy-comparison" />);
    if (kind === "solver" || kind === "prediction") {
      selectTask(kind);
    }
    await screen.findByRole("option", {name: /Expanded — 2 stocks/});
    fireEvent.change(screen.getByLabelText("Asset universe"), {target: {value: "custom_example.json"}});
    const submit = {execution: api.createRun, strategy: api.createStrategyComparison, solver: api.createSolverComparison, prediction: api.createPredictionComparison}[kind];
    vi.mocked(submit).mockResolvedValue({id: "test"} as RunSummary);
    submitAttempt();
    await waitFor(() => expect(submit).toHaveBeenCalled());
    expect(vi.mocked(submit).mock.calls[0][0]).toMatchObject({asset_config_file: "custom_example.json"});
  },
);

it("sends the selected universe with a local dataset", async () => {
  vi.mocked(api.createRun).mockResolvedValue({id: "test"} as RunSummary);
  await mount(<StrategyExperimentForm mode="execution" />);
  fireEvent.change(screen.getByRole("combobox", {name: "Data Source"}), {target: {value: "local"}});
  await screen.findByText("ABC");
  await screen.findByRole("option", {name: /Expanded — 2 stocks/});
  fireEvent.change(screen.getByLabelText("Asset universe"), {target: {value: "custom_example.json"}});
  fireEvent.click(screen.getByRole("button",{name:"Use full dataset date range"}));
  submitAttempt();
  await waitFor(() => expect(api.createRun).toHaveBeenCalled());
  expect(vi.mocked(api.createRun).mock.calls[0][0]).toMatchObject({
    input_file: "prices.pkl", asset_config_file: "custom_example.json",
  });
});


describe("local dataset recovery", () => {
  const kinds = ["execution", "strategy", "solver", "prediction"] as const;
  for (const kind of kinds) {
    for (const failure of ["catalog", "details"] as const) {
      it(`${kind}: retries ${failure} without losing values or launching work`, async () => {
        const user = userEvent.setup();
        if (failure === "catalog") vi.mocked(api.listLocalDataFiles).mockRejectedValueOnce(new Error("Dataset catalog offline"));
        else vi.mocked(api.getLocalDataInfo).mockRejectedValueOnce(new Error("Dataset details offline"));
        await mount(kind === "execution" ? <StrategyExperimentForm mode="execution" /> : <NewRunPage defaultTask="strategy-comparison" />);
        if (kind === "solver" || kind === "prediction") selectTask(kind);
        fireEvent.change(screen.getByLabelText("Preset name"), {target: {value: "Keep my setup"}});
        fireEvent.change(screen.getByRole("spinbutton", {name: /^Initial Cash/}), {target: {value: "1550"}});
        await user.selectOptions(screen.getByLabelText("Data Source"), "local");
        expect(await screen.findByRole("alert")).toHaveTextContent(failure === "catalog" ? "Dataset catalog offline" : "Dataset details offline");
        submitAttempt();
        expect(api.createRun).not.toHaveBeenCalled();
        expect(api.createStrategyComparison).not.toHaveBeenCalled();
        expect(api.createSolverComparison).not.toHaveBeenCalled();
        expect(api.createPredictionComparison).not.toHaveBeenCalled();
        await user.click(screen.getByRole("button", {name: failure === "catalog" ? "Retry datasets" : "Retry dataset details"}));
        await screen.findByText("ABC");
        expect(screen.getByLabelText("Preset name")).toHaveValue("Keep my setup");
        expect(screen.getByRole("spinbutton", {name: /^Initial Cash/})).toHaveValue(1550);
        const dataset = screen.getByLabelText("Select dataset");
        expect(dataset).toHaveValue("prices.pkl");
        await user.click(screen.getByText("Select dataset", {selector: "label"}));
        expect(dataset).toHaveFocus();
        expect(screen.getByLabelText("Frequency")).toBeDisabled();
        expect(failure === "catalog" ? api.listLocalDataFiles : api.getLocalDataInfo).toHaveBeenCalledTimes(2);
        expect(api.createRun).not.toHaveBeenCalled();
        expect(api.createStrategyComparison).not.toHaveBeenCalled();
        expect(api.createSolverComparison).not.toHaveBeenCalled();
        expect(api.createPredictionComparison).not.toHaveBeenCalled();
      });
    }
  }

  it("blocks stale metadata after a failed refresh and duplicate retries until recovery", async () => {
    const user = userEvent.setup();
    const client = new QueryClient({defaultOptions: {queries: {retry: false}}});
    render(<QueryClientProvider client={client}><MemoryRouter><StrategyExperimentForm mode="execution" /></MemoryRouter></QueryClientProvider>);
    await user.selectOptions(await screen.findByLabelText("Data Source"), "local");
    await screen.findByText("ABC");
    vi.mocked(api.getLocalDataInfo).mockRejectedValueOnce(new Error("Refresh failed"));
    await act(async () => { await client.invalidateQueries({queryKey: ["localDataInfo", "prices.pkl"]}); });
    expect(await screen.findByRole("alert")).toHaveTextContent("Refresh failed");
    expect(screen.queryByText("ABC")).not.toBeInTheDocument();
    submitAttempt();
    expect(api.createRun).not.toHaveBeenCalled();
    let finish!: (info: Awaited<ReturnType<typeof api.getLocalDataInfo>>) => void;
    const response = new Promise<Awaited<ReturnType<typeof api.getLocalDataInfo>>>(resolve => { finish = resolve; });
    vi.mocked(api.getLocalDataInfo).mockReturnValueOnce(response);
    await user.click(screen.getByRole("button", {name: "Retry dataset details"}));
    const retry = screen.getByRole("button", {name: "Retry dataset details"});
    expect(retry).toBeDisabled();
    await user.click(retry);
    expect(api.getLocalDataInfo).toHaveBeenCalledTimes(3);
    await act(async () => { finish({file: "prices.pkl", frequency: "monthly", start_date: "2020-01-01", end_date: "2026-01-31", stock_count: 5, crypto_count: 0, stocks: ["RECOVERED"], crypto: []}); await response; });
    await screen.findByText("RECOVERED");
    fireEvent.click(screen.getByRole("button",{name:"Use full dataset date range"}));
    vi.mocked(api.createRun).mockResolvedValue({id: "recovered"} as RunSummary);
    submitAttempt();
    await waitFor(() => expect(api.createRun).toHaveBeenCalledTimes(1));
    expect(vi.mocked(api.createRun).mock.calls[0][0]).toMatchObject({input_file: "prices.pkl", initial_cash: 1000});
  });

  it("keeps repeated dataset labels connected to their own selector", async () => {
    const user = userEvent.setup();
    await mount(<><LocalDataPanel selectedFile="prices.pkl" onFileChange={vi.fn()} /><LocalDataPanel selectedFile="prices.pkl" onFileChange={vi.fn()} /></>);
    const controls = await screen.findAllByLabelText("Select dataset");
    expect(controls).toHaveLength(2);
    expect(controls[0].id).not.toBe(controls[1].id);
    for (const label of screen.getAllByText("Select dataset", {selector: "label"})) {
      await user.click(label);
      expect((label as HTMLLabelElement).control).toHaveFocus();
    }
  });
});

describe("local historical windows", () => {
  it.each(["execution", "strategy", "solver", "prediction"] as const)("submits selected dates for %s", async (kind) => {
    await mount(kind === "execution" ? <StrategyExperimentForm mode="execution" /> : <NewRunPage defaultTask="strategy-comparison" />);
    if (kind === "solver" || kind === "prediction") selectTask(kind);
    fireEvent.change(screen.getByRole("combobox", {name: "Data Source"}), {target: {value: "local"}});
    await screen.findByText("ABC");
    expect(screen.getByText(/Choose the first and last complete month/)).toBeInTheDocument();
    setDate("Start Date", "2022-01-01");
    setDate("End Date", "2024-12-31");
    const submit = {execution: api.createRun, strategy: api.createStrategyComparison, solver: api.createSolverComparison, prediction: api.createPredictionComparison}[kind];
    vi.mocked(submit).mockResolvedValue({id: "test"} as RunSummary);
    submitAttempt();
    await waitFor(() => expect(submit).toHaveBeenCalled());
    expect(vi.mocked(submit).mock.calls[0][0]).toMatchObject({input_file: "prices.pkl", historical_start_date: "2022-01-01", historical_end_date: "2024-12-31"});
  });
});

it("explains that a legacy local dataset requires a timestamp upgrade", async () => {
  vi.mocked(api.getLocalDataInfo).mockResolvedValue({file: "prices.pkl", frequency: "monthly", start_date: "2020-01-01", end_date: "2026-01-31", stock_count: 5, crypto_count: 0, stocks: ["ABC"], crypto: [], time_contract_version: null});
  await mount(<StrategyExperimentForm mode="execution" />);
  fireEvent.change(screen.getByRole("combobox", {name: "Data Source"}), {target: {value: "local"}});
  expect(await screen.findByText(/legacy dataset needs an explicit timestamp upgrade/)).toBeInTheDocument();
});

it("loads, updates, saves and deletes presets without launching a run", async () => {
  const saved = {id: 7, name: "Quick", run_type: "execution" as const, parameters: {initial_cash: 1500}};
  vi.mocked(api.listPresets).mockResolvedValue([saved]);
  vi.mocked(api.savePreset).mockResolvedValue(saved);
  const load = vi.fn();
  await mount(<PresetControls runType="execution" value={{initial_cash: 2000}} onLoad={load} />);
  await screen.findByRole("option", {name: "Quick"});
  fireEvent.change(screen.getByLabelText("Saved preset"), {target: {value: "7"}});
  expect(load).toHaveBeenCalledWith({initial_cash: 1500});
  fireEvent.click(screen.getByRole("button", {name: "Update selected"}));
  await waitFor(() => expect(api.savePreset).toHaveBeenCalledWith({name: "Quick", run_type: "execution", parameters: {initial_cash: 2000}}, 7));
  await waitFor(() => expect(screen.getByRole("button", {name: "Save as new"})).toBeEnabled());
  fireEvent.click(screen.getByRole("button", {name: "Save as new"}));
  await waitFor(() => expect(api.savePreset).toHaveBeenLastCalledWith({name: "Quick", run_type: "execution", parameters: {initial_cash: 2000}}, undefined));
  await waitFor(() => expect(screen.getByRole("button", {name: "Delete selected"})).toBeEnabled());
  fireEvent.click(screen.getByRole("button", {name: "Delete selected"}));
  await waitFor(() => expect(api.deletePreset).toHaveBeenCalledWith(7));
  expect(api.createRun).not.toHaveBeenCalled();
});

it("reloading a preset for the current local file keeps the run launchable", async () => {
  const parameters = {data_source: "local", frequency: "monthly", execution_mode: "backtesting", strategy_type: "combined", validation_level: "warn", min_assets: 2, max_assets: 5, initial_cash: 1000, transaction_cost_rate: 0, risk_free_rate: 0, input_file: "prices.pkl", historical_start_date: "2022-01-01", historical_end_date: "2024-12-31"};
  vi.mocked(api.listPresets).mockResolvedValue([{id: 7, name: "Local", run_type: "execution", parameters}]);
  vi.mocked(api.createRun).mockResolvedValue({id: "test"} as RunSummary);
  await mount(<StrategyExperimentForm mode="execution" />);
  await screen.findByRole("option", {name: "Local"});
  fireEvent.change(screen.getByLabelText("Saved preset"), {target: {value: "7"}});
  await screen.findByText("ABC");
  fireEvent.change(screen.getByLabelText("Saved preset"), {target: {value: ""}});
  fireEvent.change(screen.getByLabelText("Saved preset"), {target: {value: "7"}});
  submitAttempt();
  await waitFor(() => expect(api.createRun).toHaveBeenCalled());
  expect(vi.mocked(api.createRun).mock.calls[0][0]).toMatchObject({data_source: parameters.data_source, input_file: parameters.input_file, historical_start_date: parameters.historical_start_date, historical_end_date: parameters.historical_end_date, strategy_configuration: {min_assets: 2, max_assets: 5}});
});

it("loads and submits Multi-Greedy and MV settings from a solver preset", async () => {
  const parameters = {data_source: "local", frequency: "monthly", validation_level: "warn", solvers: ["NSGA-II"], population_size: 40, n_generations: 200, n_restarts: 1, min_assets: 2, num_assets: 5, initial_cash: 1000, transaction_cost_rate: 0, risk_free_rate: 0, input_file: "prices.pkl", historical_start_date: "2022-01-01", historical_end_date: "2024-12-31", use_multi_greedy_seeds: true, seed_max_subsets: 123, seed_beam_width: 3, include_markowitz_reference: false};
  vi.mocked(api.listPresets).mockResolvedValue([{id: 8, name: "Seeds", run_type: "solver_comparison", parameters}]);
  vi.mocked(api.createSolverComparison).mockResolvedValue({id: "test"} as RunSummary);
  await mount(<NewRunPage defaultTask="strategy-comparison" />);
  selectTask("solver");
  await screen.findByRole("option", {name: "Seeds"});
  fireEvent.change(screen.getByLabelText("Saved preset"), {target: {value: "8"}});
  await screen.findByText("ABC");
  expect(screen.getByLabelText("Use Multi-Greedy risk-parity seeds")).toBeChecked();
  expect(screen.getByLabelText("Subset evaluation budget")).toHaveValue(123);
  expect(screen.getByLabelText("Candidates retained per risk preference")).toHaveValue(3);
  expect(screen.getByLabelText("Include convex mean–variance reference (without BPRC constraints)")).not.toBeChecked();
  fireEvent.click(screen.getByLabelText("Use Multi-Greedy risk-parity seeds"));
  submitAttempt();
  await waitFor(() => expect(api.createSolverComparison).toHaveBeenCalled());
  expect(vi.mocked(api.createSolverComparison).mock.calls[0][0]).toMatchObject({...parameters, use_multi_greedy_seeds: false});
});


it("copies backend search defaults into each independent instance", async () => {
  vi.mocked(api.getDefaults).mockResolvedValue({strategy_optimizer: {pop_size:80,n_generations:4321,n_restarts:7,random_seed:19,search_seconds:null,seed_max_subsets:2500,seed_beam_width:5,seed_time_fraction:0.4,refinement_supports:80}});
  vi.mocked(api.createStrategyComparison).mockResolvedValue({id: "test"} as RunSummary);
  await mount(<NewRunPage defaultTask="strategy-comparison" />);
  fireEvent.click(screen.getByRole("button", {name: "Edit Optimization 1"}));
  fireEvent.click(screen.getByText("Advanced search settings"));
  expect(screen.getByLabelText("Population size")).toHaveValue(80);
  expect(screen.getByLabelText("Evaluations per restart")).toHaveValue(4321);
  expect(screen.getByLabelText("Independent restarts")).toHaveValue(7);
  expect(screen.getByLabelText("Random seed")).toHaveValue(19);
  expect(screen.getByLabelText("Fixed-work mode (no time cutoff)")).toBeChecked();
  fireEvent.click(screen.getByRole("button", {name: "Cancel"}));
  submitAttempt();
  await waitFor(() => expect(api.createStrategyComparison).toHaveBeenCalled());
  expect(vi.mocked(api.createStrategyComparison).mock.calls[0][0].strategy_variants).toEqual([expect.objectContaining({population_size:80,n_generations:4321,n_restarts:7,random_seed:19,search_seconds:null}),expect.objectContaining({population_size:80,n_restarts:7})]);
});

it("blocks strategy setup until failed defaults recover", async () => {
  vi.mocked(api.getDefaults).mockRejectedValue(new Error("offline"));
  await mount(<NewRunPage defaultTask="strategy-comparison" />);
  expect(await screen.findByRole("alert")).toHaveTextContent("Strategy defaults could not be loaded");
  expect(screen.queryByRole("button", {name: "Review run"})).not.toBeInTheDocument();
  expect(api.createStrategyComparison).not.toHaveBeenCalled();
  vi.mocked(api.getDefaults).mockResolvedValue({strategy_optimizer: {pop_size:80,n_generations:4321,n_restarts:7,random_seed:19,search_seconds:null,seed_max_subsets:2500,seed_beam_width:5,seed_time_fraction:0.4}});
  fireEvent.click(screen.getByRole("button", {name: "Reload defaults"}));
  expect(await screen.findByRole("button", {name: "Review run"})).toBeEnabled();
});

it("explains dated universe coverage in the existing universe picker", async () => {
  vi.mocked(api.listAssetConfigs).mockResolvedValue([
    {file: "broad.json", name: "Broad dated", stock_count: 1200, crypto_count: 1,
      membership_mode: "dated", stock_limit: 1000},
  ]);
  await mount(<NewRunPage defaultTask="strategy-comparison" />);
  await screen.findByRole("option", {name: /Broad dated/});
  fireEvent.change(screen.getByLabelText("Asset universe"), {target: {value: "broad.json"}});
  expect(await screen.findByText(/up to 1000 equities per snapshot/)).toBeInTheDocument();
  expect(screen.getByText(/Historical snapshots are required/)).toBeInTheDocument();
});

it.each(["execution", "strategy", "solver", "prediction"] as const)("defaults to compact output and submits diagnostic output for %s", async (kind) => {
  await mount(kind === "execution" ? <StrategyExperimentForm mode="execution" /> : <NewRunPage defaultTask="strategy-comparison" />);
  if (kind === "solver" || kind === "prediction") selectTask(kind);
  expect(screen.getByLabelText("Saved output")).toHaveValue("standard");
  fireEvent.change(screen.getByLabelText("Saved output"), {target: {value: "diagnostic"}});
  const submit = {execution: api.createRun, strategy: api.createStrategyComparison, solver: api.createSolverComparison, prediction: api.createPredictionComparison}[kind];
  vi.mocked(submit).mockResolvedValue({id: "test"} as RunSummary);
  await waitFor(() => expect(screen.getByRole("button", {name: "Review run"})).toBeEnabled());
  submitAttempt();
  await waitFor(() => expect(submit).toHaveBeenCalled());
  expect(vi.mocked(submit).mock.calls[0][0]).toMatchObject({output_mode: "diagnostic"});
});

it("retains the output mode when saving and loading a comparison preset", async () => {
  vi.mocked(api.listPresets).mockResolvedValue([{id: 7, name: "Debug output", run_type: "strategy_comparison", parameters: {
    strategy_types: ["optimization"], data_source: "alpaca", frequency: "monthly", validation_level: "warn", output_mode: "diagnostic",
    historical_start_date: "2020-01-01", historical_end_date: "2024-12-31", initial_cash: 1000, transaction_cost_rate: 0, risk_free_rate: 0,
    population_size: 300, n_generations: 6000, min_assets: 2, max_assets: 5, include_benchmark: true, benchmark_asset: "SPY",
    optimization_selection: "sharpe", optimization_volatility_increase: 0.25, optimization_return_target_fraction: 0.75,
  }}]);
  vi.mocked(api.savePreset).mockResolvedValue({id: 7, name: "Debug output", run_type: "strategy_comparison", parameters: {output_mode: "standard"}});
  await mount(<NewRunPage defaultTask="strategy-comparison" />);
  await screen.findByRole("option", {name: "Debug output"});
  fireEvent.change(screen.getByLabelText("Saved preset"), {target: {value: "7"}});
  expect(screen.getByLabelText("Saved output")).toHaveValue("diagnostic");
  fireEvent.change(screen.getByLabelText("Saved output"), {target: {value: "standard"}});
  fireEvent.click(screen.getByRole("button", {name: "Update selected"}));
  await waitFor(() => expect(api.savePreset).toHaveBeenCalledWith(expect.objectContaining({parameters: expect.objectContaining({output_mode: "standard"})}), 7));
});



it.each(["capital_weights", "hybrid"])("duplicates Optimization independently with %s and reviews before launching", async balance => {
  const user = userEvent.setup();
  vi.mocked(api.createStrategyComparison).mockResolvedValue({id: "builder"} as RunSummary);
  await mount(<NewRunPage defaultTask="strategy-comparison" />);
  await user.click(screen.getByRole("button", {name: "Duplicate Optimization 1"}));
  const dialog = within(screen.getByRole("dialog", {name: "Configure Optimization"}));
  expect(dialog.getByLabelText("Name")).toHaveFocus();
  fireEvent.change(dialog.getByLabelText("Portfolio balance"), {target: {value: balance}});
  fireEvent.change(dialog.getByLabelText("Portfolio selection"), {target: {value: "return_target"}});
  fireEvent.change(dialog.getByLabelText("Position in feasible return range (%)"), {target: {value: "75"}});
  fireEvent.click(dialog.getByText("Advanced search settings"));
  fireEvent.change(dialog.getByLabelText("Population size"), {target: {value: "150"}});
  fireEvent.change(dialog.getByLabelText("Seed beam width"), {target: {value: "8"}});
  expect(dialog.getByLabelText("Maximum seed subsets")).toHaveAttribute("max", "360000");
  fireEvent.change(dialog.getByLabelText("Maximum seed subsets"), {target: {value: "360000"}});
  fireEvent.click(dialog.getByRole("button", {name: "Save configuration"}));
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(screen.getByRole("button", {name: "Duplicate Optimization 1"})).toHaveFocus();
  expect(screen.getByRole("listitem", {name: "Optimization 1"})).toHaveTextContent("Maximum Sharpe");
  expect(screen.getByRole("listitem", {name: "Optimization 1"})).toHaveTextContent("Population 300");
  expect(screen.getByRole("listitem", {name: "Optimization 2"})).toHaveTextContent("Return target 75%");
  fireEvent.click(screen.getByRole("button", {name: "Review run"}));
  expect(await screen.findByRole("heading", {name: "3. Review run"})).toBeInTheDocument();
  expect(api.createStrategyComparison).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", {name: "Start run"}));
  await waitFor(() => expect(api.createStrategyComparison).toHaveBeenCalledTimes(1));
  const payload = vi.mocked(api.createStrategyComparison).mock.calls[0][0];
  expect(payload.strategy_types).toEqual([]);
  expect(payload.strategy_variants).toEqual([
    expect.objectContaining({name: "Combined 1", population_size: 300}),
    expect.objectContaining({name: "Optimization 1", selection: "sharpe", population_size: 300, balance: "risk_contributions"}),
    expect.objectContaining({name: "Optimization 2", selection: "return_target", return_target_fraction: .75, balance, population_size: 150, seed_beam_width: 8, seed_max_subsets: 360000}),
  ]);
});

it("validates duplicate names and bounds inside the drawer and discards cancelled edits", async () => {
  await mount(<NewRunPage defaultTask="strategy-comparison" />);
  fireEvent.click(screen.getByRole("button", {name: "Duplicate Optimization 1"}));
  fireEvent.change(screen.getByLabelText("Name"), {target: {value: "optimization 1"}});
  fireEvent.click(screen.getByRole("button", {name: "Save configuration"}));
  expect(screen.getByRole("alert")).toHaveTextContent("unique name");
  fireEvent.change(screen.getByLabelText("Name"), {target: {value: "High return"}});
  fireEvent.change(screen.getByLabelText("Min Assets"), {target: {value: "6"}});
  fireEvent.click(screen.getByRole("button", {name: "Save configuration"}));
  expect(screen.getByRole("alert")).toHaveTextContent("Min assets cannot be greater");
  fireEvent.click(screen.getByRole("button", {name: "Cancel"}));
  expect(screen.getAllByRole("listitem")).toHaveLength(2);
  fireEvent.click(screen.getByRole("button", {name: "Edit Optimization 1"}));
  expect(screen.getByLabelText("Min Assets")).toHaveValue(2);
  fireEvent.change(screen.getByLabelText("Min Assets"), {target: {value: ""}});
  expect(screen.getByLabelText("Min Assets")).toBeInvalid();
  fireEvent.click(screen.getByRole("button", {name: "Save configuration"}));
  expect(screen.getByRole("dialog")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", {name: "Cancel"}));
  expect(api.createStrategyComparison).not.toHaveBeenCalled();
});

it("uses the same single-run editor and supports searchable replacement without stale overrides", async () => {
  vi.mocked(api.createRun).mockResolvedValue({id: "single"} as RunSummary);
  await mount(<StrategyExperimentForm mode="execution" />);
  fireEvent.click(screen.getByRole("button", {name: "Replace strategy"}));
  fireEvent.change(screen.getByLabelText("Search strategies"), {target: {value: "optimization"}});
  expect(screen.queryByRole("button", {name: "Add Buy & Hold"})).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", {name: "Add Optimization"}));
  fireEvent.change(screen.getByLabelText("Portfolio balance"), {target: {value: "hybrid"}});
  fireEvent.change(screen.getByLabelText("Portfolio selection"), {target: {value: "return_target"}});
  fireEvent.change(screen.getByLabelText("Min Assets"), {target: {value: "3"}});
  fireEvent.click(screen.getByRole("button", {name: "Save configuration"}));
  setDate("Backtest start date (optional)", "2021-01-01");
  fireEvent.click(screen.getByRole("button", {name: "Review run"}));
  expect(api.createRun).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", {name: "Start run"}));
  await waitFor(() => expect(api.createRun).toHaveBeenCalledTimes(1));
  expect(vi.mocked(api.createRun).mock.calls[0][0]).toMatchObject({strategy_type: "optimization", backtest_start_date: "2021-01-01", strategy_configuration: {strategy_type: "optimization", balance: "hybrid", selection: "return_target", min_assets: 3, max_assets: 5, population_size: 300}});
});

it("can add two independently configured baseline strategies and remove a row", async () => {
  vi.mocked(api.createStrategyComparison).mockResolvedValue({id: "baselines"} as RunSummary);
  await mount(<NewRunPage defaultTask="strategy-comparison" />);
  fireEvent.click(screen.getByRole("button", {name: "Remove Combined 1"}));
  fireEvent.click(screen.getByRole("button", {name: "Add strategy"}));
  fireEvent.click(screen.getByRole("button", {name: "Add Buy & Hold"}));
  expect(screen.queryByLabelText("Portfolio selection")).not.toBeInTheDocument();
  expect(screen.queryByLabelText("Min Assets")).not.toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("Asset symbol"), {target: {value: "QQQ"}});
  fireEvent.click(screen.getByRole("button", {name: "Save configuration"}));
  fireEvent.click(screen.getByRole("button", {name: "Duplicate Buy & Hold 1"}));
  fireEvent.change(screen.getByLabelText("Asset symbol"), {target: {value: "GLD"}});
  fireEvent.click(screen.getByRole("button", {name: "Save configuration"}));
  fireEvent.click(screen.getByRole("button", {name: "Add strategy"}));
  fireEvent.click(screen.getByRole("button", {name: "Add Momentum Risk Parity"}));
  fireEvent.change(screen.getByLabelText("Maximum selected assets"), {target: {value: "5"}});
  fireEvent.change(screen.getByLabelText("Momentum lookback (periods)"), {target: {value: "12"}});
  fireEvent.click(screen.getByRole("button", {name: "Save configuration"}));
  submitAttempt();
  await waitFor(() => expect(api.createStrategyComparison).toHaveBeenCalledTimes(1));
  expect(vi.mocked(api.createStrategyComparison).mock.calls[0][0].strategy_variants).toEqual([
    expect.objectContaining({strategy_type: "optimization"}),
    expect.objectContaining({strategy_type: "buy_and_hold", asset_symbol: "QQQ"}),
    expect.objectContaining({strategy_type: "buy_and_hold", asset_symbol: "GLD"}),
    expect.objectContaining({strategy_type: "momentum_risk_parity", top_n: 5, lookback: 12}),
  ]);
});

it("keeps strategy settings when visiting the other comparison tabs and invalidates review on edits", async () => {
  await mount(<NewRunPage defaultTask="strategy-comparison" />);
  fireEvent.change(screen.getByRole("spinbutton", {name: /^Initial Cash/}), {target: {value: "1700"}});
  fireEvent.click(screen.getByRole("button", {name: "Edit Optimization 1"}));
  fireEvent.change(screen.getByLabelText("Max Assets"), {target: {value: "4"}});
  fireEvent.click(screen.getByRole("button", {name: "Save configuration"}));
  selectTask("solver");
  expect(screen.getAllByLabelText("Data Source")).toHaveLength(1);
  selectTask("strategy");
  expect(screen.getByRole("spinbutton", {name: /^Initial Cash/})).toHaveValue(1700);
  expect(screen.getByRole("listitem", {name: "Optimization 1"})).toHaveTextContent("2–4 assets");
  fireEvent.click(screen.getByRole("button", {name: "Review run"}));
    setDate("End Date", "2025-12-31");
  expect(screen.queryByRole("button", {name: "Start run"})).not.toBeInTheDocument();
  expect(screen.getByRole("button", {name: "Review run"})).toBeEnabled();
  expect(api.createStrategyComparison).not.toHaveBeenCalled();
});

it("does not launch twice and preserves configuration after a launch error", async () => {
  let fail!: (reason: Error) => void;
  vi.mocked(api.createRun).mockImplementationOnce(() => new Promise((_, reject) => {fail = reject;}));
  await mount(<StrategyExperimentForm mode="execution" />);
  fireEvent.click(screen.getByRole("button", {name: "Review run"}));
  const launch = screen.getByRole("button", {name: "Start run"});
  fireEvent.click(launch);
  fireEvent.click(launch);
  await waitFor(() => expect(api.createRun).toHaveBeenCalledTimes(1));
  expect(screen.getByRole("button", {name: "Submitting…"})).toBeDisabled();
  await act(async () => fail(new Error("Queue unavailable")));
  expect(await screen.findByRole("alert")).toHaveTextContent("Queue unavailable");
  expect(screen.getByRole("listitem", {name: "Combined 1"})).toBeInTheDocument();
  expect(screen.getByRole("button", {name: "Start run"})).toBeEnabled();
});

it.each([false, true])("preserves legacy QQQ Buy & Hold and its benchmark role with auto-add=%s", async includeBenchmark => {
  vi.mocked(api.listPresets).mockResolvedValue([{id: 88, name: "Legacy QQQ", run_type: "strategy_comparison", parameters: {
    strategy_types: ["combined", "buy_and_hold"], benchmark_asset: "QQQ", include_benchmark: includeBenchmark,
  }}]);
  vi.mocked(api.createStrategyComparison).mockResolvedValue({id: "legacy-qqq"} as RunSummary);
  await mount(<NewRunPage defaultTask="strategy-comparison" />);
  await screen.findByRole("option", {name: "Legacy QQQ"});
  fireEvent.change(screen.getByLabelText("Saved preset"), {target: {value: "88"}});
  expect(screen.getByRole("listitem", {name: "Buy & Hold 1"})).toHaveTextContent("100% QQQ");
  expect(screen.getByLabelText("Include Buy & Hold benchmark")).toBeChecked();
  expect(screen.getByLabelText("Benchmark asset")).toHaveValue("QQQ");
  fireEvent.click(screen.getByRole("button", {name: "Review run"}));
  const review = screen.getByRole("region", {name: "3. Review run"});
  expect(review).toHaveTextContent("QQQ Buy & Hold (uses Buy & Hold 1)");
  expect(review).not.toHaveTextContent("plus one benchmark");
  fireEvent.click(screen.getByRole("button", {name: "Start run"}));
  await waitFor(() => expect(api.createStrategyComparison).toHaveBeenCalledTimes(1));
  expect(vi.mocked(api.createStrategyComparison).mock.calls[0][0]).toMatchObject({
    strategy_types: [], include_benchmark: true, benchmark_asset: "QQQ",
    strategy_variants: [expect.objectContaining({strategy_type:"combined"}), expect.objectContaining({strategy_type:"buy_and_hold", asset_symbol:"QQQ"})],
  });
});

it("does not promote an overridden named Buy & Hold configuration to a legacy benchmark", async () => {
  vi.mocked(api.listPresets).mockResolvedValue([{id: 89, name: "Named baseline", run_type: "strategy_comparison", parameters: {
    strategy_types: ["buy_and_hold"], benchmark_asset: "QQQ", include_benchmark: false,
    strategy_variants: [{name:"Gold baseline", strategy_type:"buy_and_hold", asset_symbol:"GLD"}],
  }}]);
  await mount(<NewRunPage defaultTask="strategy-comparison" />);
  await screen.findByRole("option", {name: "Named baseline"});
  fireEvent.change(screen.getByLabelText("Saved preset"), {target: {value:"89"}});
  expect(screen.getByLabelText("Include Buy & Hold benchmark")).not.toBeChecked();
  expect(screen.getAllByRole("listitem")).toHaveLength(1);
  expect(screen.getByRole("listitem", {name:"Gold baseline"})).toHaveTextContent("100% GLD");
});

describe("consistent comparison workflow", () => {
  it.each(["solver", "prediction"] as const)("%s: requires review and clears it when the setup changes", async kind => {
    const user = userEvent.setup();
    const submit = kind === "solver" ? api.createSolverComparison : api.createPredictionComparison;
    vi.mocked(submit).mockResolvedValue({id: "test"} as RunSummary);
    await mount(<NewRunPage defaultTask="strategy-comparison" />);
    selectTask(kind);
    const headings = screen.getAllByRole("heading").map(heading => heading.textContent);
    expect(headings.indexOf("1. Shared setup")).toBeLessThan(headings.indexOf(kind === "solver" ? "2. Solver selection" : "2. Predictor selection"));
    await user.click(screen.getByRole("button", {name: "Review run"}));
    expect(screen.getByRole("heading", {name: "3. Review run"})).toBeInTheDocument();
    expect(submit).not.toHaveBeenCalled();
    setDate("Start Date", "2021-01-01");
    expect(screen.queryByRole("heading", {name: "3. Review run"})).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", {name: "Review run"}));
    await user.click(screen.getByRole("button", {name: "Start run"}));
    await waitFor(() => expect(submit).toHaveBeenCalledTimes(1));
    expect(vi.mocked(submit).mock.calls[0][0]).toMatchObject({historical_start_date: "2021-01-01"});
  });

  it("preserves each comparison’s draft and selection across tab switches", async () => {
    const user = userEvent.setup();
    await mount(<NewRunPage defaultTask="strategy-comparison" />);
    selectTask("solver");
    fireEvent.change(screen.getByLabelText("Population size"), {target: {value: "80"}});
    await user.click(screen.getByRole("button", {name: "Remove SPEA2"}));
    selectTask("prediction");
    fireEvent.change(screen.getByLabelText("Top K"), {target: {value: "8"}});
    selectTask("solver");
    expect(screen.getByLabelText("Population size")).toHaveValue(80);
    expect(screen.queryByRole("listitem", {name: "SPEA2"})).not.toBeInTheDocument();
    selectTask("prediction");
    expect(screen.getByLabelText("Top K")).toHaveValue(8);
  });

  it("configures predictors before adding them, and validates monthly-only choices", async () => {
    const user = userEvent.setup();
    await mount(<NewRunPage defaultTask="strategy-comparison" />);
    selectTask("prediction");
    expect(screen.queryByRole("button", {name: "Add MomentumRidge"})).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", {name: "Add predictor"}));
    await user.type(screen.getByLabelText("Search predictors"), "MomentumRidge");
    expect(screen.queryByRole("button", {name: "Add HistoricalMean"})).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", {name: "Add MomentumRidge"}));
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.queryByRole("listitem", {name: "MomentumRidge"})).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", {name: "Save configuration"}));
    expect(screen.getAllByRole("listitem", {name: "MomentumRidge"})).toHaveLength(1);
    await user.selectOptions(screen.getByLabelText("Frequency"), "weekly");
    expect(screen.getByRole("button", {name: "Review run"})).toBeDisabled();
    expect(screen.getAllByText(/require monthly data/).length).toBeGreaterThan(0);
    await user.click(screen.getByRole("button", {name: "Remove MomentumRidge"}));
    expect(screen.getByRole("button", {name: "Review run"})).toBeEnabled();
  });

  it("preserves the existing sentiment option from prediction presets", async () => {
    vi.mocked(api.listPresets).mockImplementation(async runType => runType === "prediction_comparison" ? [{id: 18, name: "No sentiment", run_type: runType, parameters: {compute_sentiment: false, predictors: ["EWM-6"]}}] : []);
    vi.mocked(api.createPredictionComparison).mockResolvedValue({id: "test"} as RunSummary);
    await mount(<NewRunPage defaultTask="strategy-comparison" />);
    selectTask("prediction");
    await screen.findByRole("option", {name: "No sentiment"});
    fireEvent.change(screen.getByLabelText("Saved preset"), {target: {value: "18"}});
    expect(screen.getByLabelText("Compute sentiment for predictors that use news")).not.toBeChecked();
    submitAttempt();
    await waitFor(() => expect(api.createPredictionComparison).toHaveBeenCalledTimes(1));
    expect(vi.mocked(api.createPredictionComparison).mock.calls[0][0]).toMatchObject({compute_sentiment: false, predictors: ["EWM-6"]});
  });
});

describe("recorded local dataset configuration", () => {
  it.each(["execution", "strategy", "solver", "prediction"] as const)("%s: uses the recorded universe and frequency instead of the Alpaca selection", async kind => {
    vi.mocked(api.getLocalDataInfo).mockResolvedValue({file: "prices.pkl", frequency: "weekly", asset_config_file: "archived-research.json", asset_config_name: "Recorded research universe", asset_config_available: true, stocks: ["ABC"], crypto: [], stock_count: 1, crypto_count: 0, start_date: "2020-01-01", end_date: "2026-01-31"});
    const submit = {execution: api.createRun, strategy: api.createStrategyComparison, solver: api.createSolverComparison, prediction: api.createPredictionComparison}[kind];
    vi.mocked(submit).mockResolvedValue({id: "test"} as RunSummary);
    await mount(kind === "execution" ? <StrategyExperimentForm mode="execution" /> : <NewRunPage defaultTask="strategy-comparison" />);
    if (kind === "solver" || kind === "prediction") selectTask(kind);
    fireEvent.change(screen.getByLabelText("Data Source"), {target: {value: "local"}});
    await screen.findByText("Recorded research universe");
    expect(screen.queryByLabelText("Asset universe")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Frequency")).toHaveValue("weekly");
    expect(screen.getByLabelText("Frequency")).toBeDisabled();
    setDate("Start Date", "2020-01-06"); setDate("End Date", "2026-01-25");
    submitAttempt();
    await waitFor(() => expect(submit).toHaveBeenCalledTimes(1));
    expect(vi.mocked(submit).mock.calls[0][0]).toMatchObject({asset_config_file: "archived-research.json", frequency: "weekly", input_file: "prices.pkl"});
  });

  it("blocks a dataset with an unavailable recorded universe without silently substituting the default", async () => {
    vi.mocked(api.getLocalDataInfo).mockResolvedValue({file: "prices.pkl", frequency: "monthly", asset_config_file: "missing.json", asset_config_available: false, stocks: ["ABC"], crypto: [], stock_count: 1, crypto_count: 0, start_date: "2020-01-01", end_date: "2026-01-31"});
    await mount(<StrategyExperimentForm mode="execution" />);
    fireEvent.change(screen.getByLabelText("Data Source"), {target: {value: "local"}});
    await screen.findByText("Dataset universe");
    submitAttempt();
    expect(screen.getAllByText(/recorded universe is unavailable/).length).toBeGreaterThan(0);
    expect(api.createRun).not.toHaveBeenCalled();
    expect(screen.queryByLabelText("Asset universe")).not.toBeInTheDocument();
  });
});

it.each([
  ["solver", "EPSNSGAII", "EpsNSGAII"],
  ["prediction", "ewm-6", "EWM-6"],
] as const)("%s retains case-insensitive preset labels", async (kind, savedLabel, canonicalLabel) => {
  const runType = kind === "solver" ? "solver_comparison" : "prediction_comparison";
  vi.mocked(api.listPresets).mockResolvedValue([{id: 29, name: "Registry label", run_type: runType, parameters: kind === "solver" ? {solvers: [savedLabel, canonicalLabel]} : {predictors: [savedLabel, canonicalLabel]}}]);
  const submit = kind === "solver" ? api.createSolverComparison : api.createPredictionComparison;
  vi.mocked(submit).mockResolvedValue({id: "test"} as RunSummary);
  await mount(<NewRunPage defaultTask="strategy-comparison" />);
  selectTask(kind);
  await screen.findByRole("option", {name: "Registry label"});
  fireEvent.change(screen.getByLabelText("Saved preset"), {target: {value: "29"}});
  expect(screen.getByRole("button", {name: "Review run"})).toBeEnabled();
  submitAttempt();
  await waitFor(() => expect(submit).toHaveBeenCalledTimes(1));
  expect(vi.mocked(submit).mock.calls[0][0]).toMatchObject(kind === "solver" ? {solvers: [canonicalLabel]} : {predictors: [canonicalLabel]});
});

it.each(["solver", "prediction"] as const)("%s clears the inapplicable hidden backtest start from old API presets", async kind => {
  const runType = kind === "solver" ? "solver_comparison" : "prediction_comparison";
  vi.mocked(api.listPresets).mockResolvedValue([{id: 31, name: "Historical API preset", run_type: runType, parameters: {backtest_start_date: "2021-01-01"}}]);
  const submit = kind === "solver" ? api.createSolverComparison : api.createPredictionComparison;
  vi.mocked(submit).mockResolvedValue({id: "test"} as RunSummary);
  await mount(<NewRunPage defaultTask="strategy-comparison" />);
  selectTask(kind);
  await screen.findByRole("option", {name: "Historical API preset"});
  fireEvent.change(screen.getByLabelText("Saved preset"), {target: {value: "31"}});
    setDate("Start Date", "2022-01-01");
  submitAttempt();
  await waitFor(() => expect(submit).toHaveBeenCalledTimes(1));
  expect(vi.mocked(submit).mock.calls[0][0]).toMatchObject({historical_start_date: "2022-01-01", backtest_start_date: null});
});

it("uses dataset coverage only on request and restores the Alpaca universe after switching back", async () => {
  vi.mocked(api.getLocalDataInfo).mockResolvedValue({file: "prices.pkl", frequency: "monthly", asset_config_file: "recorded.json", asset_config_available: true, stocks: ["ABC"], crypto: [], stock_count: 1, crypto_count: 0, start_date: "2021-01-01", end_date: "2025-12-31"});
  const user = userEvent.setup();
  await mount(<StrategyExperimentForm mode="execution" />);
  await screen.findByRole("option", {name: /Expanded — 2 stocks/});
  await user.selectOptions(screen.getByLabelText("Asset universe"), "custom_example.json");
  await user.selectOptions(screen.getByLabelText("Data Source"), "local");
  await screen.findByText("Dataset universe");
  expect(screen.getByLabelText("Start Date")).toHaveValue("2020-01");
  await user.click(screen.getByRole("button", {name: "Use full dataset date range"}));
  expect(screen.getByLabelText("Start Date")).toHaveValue("2021-01");
  expect(screen.getByLabelText("End Date")).toHaveValue("2025-12");
  await user.selectOptions(screen.getByLabelText("Data Source"), "alpaca");
  expect(screen.getByLabelText("Asset universe")).toHaveValue("custom_example.json");
});

describe("independent solver and predictor configurations", () => {
  it("offers configurable predictor families rather than parameter-specific aliases", async () => {
    const user = userEvent.setup();
    await mount(<NewRunPage defaultTask="strategy-comparison" />);
    selectTask("prediction");
    expect(screen.getByRole("listitem", {name: "EWM"})).toHaveTextContent("periods): 6");
    await user.click(screen.getByRole("button", {name: "Add predictor"}));
    expect(screen.getByRole("button", {name: "Add EWM"})).toBeEnabled();
    for (const label of ["EWM-3", "EWM-6", "EWM-12", "Sentiment EWM-6", "EWM12-Mean-25", "RidgeVARX (lag 5)", "FuzzyThesis-fixed", "FuzzyRidge-fixed"]) {
      expect(screen.queryByRole("button", {name: `Add ${label}`})).not.toBeInTheDocument();
    }
    await user.click(screen.getByRole("button", {name: "Add EWM"}));
    expect(screen.getByRole("dialog", {name: "Configure EWM"})).toBeInTheDocument();
    expect(screen.getByLabelText("Name")).toHaveValue("EWM 2");
    fireEvent.change(screen.getByLabelText("EWM span (periods)"), {target: {value: "3"}});
    await user.click(screen.getByRole("button", {name: "Save configuration"}));
    expect(screen.getByRole("listitem", {name: "EWM 2"})).toHaveTextContent("periods): 3");
  });

  it("loads legacy EWM span presets without collapsing their distinct configurations", async () => {
    const user = userEvent.setup();
    vi.mocked(api.listPresets).mockResolvedValue([{id: 46, name: "Legacy spans", run_type: "prediction_comparison", parameters: {predictors: ["EWM-3", "EWM-6", "EWM-12"]}}]);
    await mount(<NewRunPage defaultTask="strategy-comparison" />);
    selectTask("prediction");
    await screen.findByRole("option", {name: "Legacy spans"});
    await user.selectOptions(screen.getByLabelText("Saved preset"), "46");
    for (const span of [3, 6, 12]) expect(screen.getByRole("listitem", {name: `EWM-${span}`})).toHaveTextContent(`periods): ${span}`);
    fireEvent.change(screen.getByLabelText("Preset name"), {target: {value: "Saved legacy spans"}});
    await user.click(screen.getByRole("button", {name: "Save as new"}));
    await waitFor(() => expect(api.savePreset).toHaveBeenCalledTimes(1));
    expect(vi.mocked(api.savePreset).mock.calls[0][0].parameters).toMatchObject({predictor_variants: [
      {name: "EWM-3", predictor: "EWM-3", span: 3},
      {name: "EWM-6", predictor: "EWM-6", span: 6},
      {name: "EWM-12", predictor: "EWM-12", span: 12},
    ]});
  });

  it("duplicates a solver, overrides only its budget and reviews the effective total", async () => {
    const user=userEvent.setup();
    vi.mocked(api.createSolverComparison).mockResolvedValue({id:"named"} as RunSummary);
    await mount(<NewRunPage defaultTask="strategy-comparison" />);
    selectTask("solver");
    await user.click(screen.getByRole("button",{name:"Duplicate NSGA-II"}));
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    await user.clear(screen.getByLabelText("Name"));
    await user.type(screen.getByLabelText("Name"),"NSGA larger");
    await user.click(screen.getByLabelText("Use shared solver budgets"));
    const dialog=within(screen.getByRole("dialog"));
    fireEvent.change(dialog.getByLabelText("Population size"),{target:{value:"80"}});
    fireEvent.change(dialog.getByLabelText("Evaluations per restart"),{target:{value:"2000"}});
    fireEvent.change(dialog.getByLabelText("Random seed"),{target:{value:"17"}});
    await user.click(screen.getByRole("button",{name:"Save configuration"}));
    expect(screen.getByRole("listitem",{name:"NSGA larger"})).toHaveTextContent("80 · 2000 evaluations × 3");
    expect(screen.getByRole("listitem",{name:"NSGA-II"})).toHaveTextContent("100 · 500 evaluations × 3");
    await user.click(screen.getByRole("button",{name:"Review run"}));
    expect(screen.getByText(/Search budgets differ/)).toBeInTheDocument();
    expect(screen.getByText(/10,500 requested evaluations/)).toBeInTheDocument();
    expect(api.createSolverComparison).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button",{name:"Start run"}));
    await waitFor(()=>expect(api.createSolverComparison).toHaveBeenCalledTimes(1));
    const payload=vi.mocked(api.createSolverComparison).mock.calls[0][0];
    expect(payload.solver_variants).toEqual(expect.arrayContaining([
      expect.objectContaining({name:"NSGA-II",solver:"NSGA-II"}),
      expect.objectContaining({name:"NSGA larger",solver:"NSGA-II",population_size:80,n_generations:2000,random_seed:17}),
    ]));
  });

  it("edits two EWM instances independently and rejects duplicate configuration names", async()=>{
    const user=userEvent.setup();
    vi.mocked(api.createPredictionComparison).mockResolvedValue({id:"named"} as RunSummary);
    await mount(<NewRunPage defaultTask="strategy-comparison" />);
    selectTask("prediction");
    await user.click(screen.getByRole("button",{name:"Duplicate EWM"}));
    fireEvent.change(screen.getByLabelText("Name"),{target:{value:"EWM"}});
    await user.click(screen.getByRole("button",{name:"Save configuration"}));
    expect(screen.getByRole("alert")).toHaveTextContent("names must be unique");
    fireEvent.change(screen.getByLabelText("Name"),{target:{value:"Slow EWM"}});
    fireEvent.change(screen.getByLabelText("EWM span (periods)"),{target:{value:"18"}});
    await user.click(screen.getByRole("button",{name:"Save configuration"}));
    expect(screen.getByRole("listitem",{name:"EWM"})).toHaveTextContent("periods): 6");
    expect(screen.getByRole("listitem",{name:"Slow EWM"})).toHaveTextContent("periods): 18");
    submitAttempt();
    await waitFor(()=>expect(api.createPredictionComparison).toHaveBeenCalledTimes(1));
    expect(vi.mocked(api.createPredictionComparison).mock.calls[0][0].predictor_variants).toEqual(expect.arrayContaining([
      expect.objectContaining({name:"EWM",predictor:"EWM-6",span:6}),
      expect.objectContaining({name:"Slow EWM",predictor:"EWM-6",span:18}),
    ]));
  });

  it("resolves omitted predictor settings in named presets and saves without forcing manual defaults", async()=>{
    const user=userEvent.setup();
    vi.mocked(api.listPresets).mockResolvedValue([{id:44,name:"Inherited predictor",run_type:"prediction_comparison",parameters:{predictor_variants:[{name:"Smooth",predictor:"ewm-6",span:null}]}}]);
    await mount(<NewRunPage defaultTask="strategy-comparison" />);
    selectTask("prediction");
    await screen.findByRole("option",{name:"Inherited predictor"});
    await user.selectOptions(screen.getByLabelText("Saved preset"),"44");
    expect(screen.getByRole("listitem",{name:"Smooth"})).toHaveTextContent("periods): 6");
    await user.click(screen.getByRole("button",{name:"Edit Smooth"}));
    expect(screen.getByLabelText("EWM span (periods)")).toHaveValue(6);
    expect(screen.getByLabelText("EWM span (periods)")).toBeValid();
    await user.click(screen.getByRole("button",{name:"Save configuration"}));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("retains named solver settings through preset save and restores focus on Cancel", async()=>{
    const user=userEvent.setup();
    vi.mocked(api.savePreset).mockResolvedValue({id:45,name:"Named",run_type:"solver_comparison",parameters:{}});
    await mount(<NewRunPage defaultTask="strategy-comparison" />);
    selectTask("solver");
    await user.click(screen.getByRole("button",{name:"Edit NSGA-II"}));
    await user.click(screen.getByLabelText("Override mutation probability"));
    fireEvent.change(screen.getByLabelText("Mutation probability"),{target:{value:"0.35"}});
    await user.click(screen.getByRole("button",{name:"Save configuration"}));
    await user.click(screen.getByRole("button",{name:"Edit NSGA-II"}));
    expect(screen.getByLabelText("Mutation probability")).toHaveValue(0.35);
    await user.click(screen.getByRole("button",{name:"Cancel"}));
    expect(screen.getByRole("button",{name:"Edit NSGA-II"})).toHaveFocus();
    fireEvent.change(screen.getByLabelText("Preset name"),{target:{value:"Named"}});
    await user.click(screen.getByRole("button",{name:"Save as new"}));
    await waitFor(()=>expect(api.savePreset).toHaveBeenCalledTimes(1));
    expect(vi.mocked(api.savePreset).mock.calls[0][0].parameters).toMatchObject({solver_variants:expect.arrayContaining([expect.objectContaining({name:"NSGA-II",mutation_prob:0.35})])});
    expect(api.createSolverComparison).not.toHaveBeenCalled();
  });
});

it("offers four tasks in priority order and defaults to investment portfolios",async()=>{
 await mount(<NewRunPage/>);
 const purpose=screen.getByRole("combobox",{name:"What would you like to run?"});
 expect(within(purpose).getAllByRole("option").map(option=>option.textContent)).toEqual(["Generate an investment portfolio","Backtest investment strategies","Compare return predictors","Benchmark portfolio optimization solvers"]);
 expect(purpose).toHaveValue("investment");
 fireEvent.change(screen.getByRole("spinbutton",{name:"Available cash (USD)"}),{target:{value:"1750"}});
 for(const task of ["predictors","solvers","strategy-comparison","investment"]) fireEvent.change(purpose,{target:{value:task}});
 expect(screen.getByRole("spinbutton",{name:"Available cash (USD)"})).toHaveValue(1750);
 expect(screen.queryByLabelText("Backtest start date (optional)")).not.toBeInTheDocument();
 expect(api.createRun).not.toHaveBeenCalled();expect(api.createStrategyComparison).not.toHaveBeenCalled();
});
it("backtests a single configuration through the shared strategies backtest without an additional benchmark",async()=>{
 await mount(<NewRunPage defaultTask="strategy-comparison"/>);
 fireEvent.click(screen.getByRole("button",{name:"Remove Combined 1"}));
 fireEvent.click(screen.getByLabelText("Include Buy & Hold benchmark"));
 vi.mocked(api.createStrategyComparison).mockResolvedValue({id:"one"} as RunSummary);
 submitAttempt();submitAttempt();
 await waitFor(()=>expect(api.createStrategyComparison).toHaveBeenCalledTimes(1));
 expect(vi.mocked(api.createStrategyComparison).mock.calls[0][0]).toMatchObject({strategy_types:[],strategy_variants:[expect.objectContaining({name:"Optimization 1"})],include_benchmark:false});
 expect(api.createRun).not.toHaveBeenCalled();
});
it("makes a loaded execution preset's task explicit instead of silently changing its purpose",async()=>{
  vi.mocked(api.listPresets).mockResolvedValue([{id:91,name:"My allocation",run_type:"execution",parameters:{execution_mode:"investing",initial_cash:2000,strategy_type:"optimization"}}]);
  await mount(<NewRunPage/>);
  await screen.findByRole("option",{name:"My allocation"});
  fireEvent.change(screen.getByLabelText("Saved preset"),{target:{value:"91"}});
  expect(screen.getByRole("combobox",{name:"What would you like to run?"})).toHaveValue("investment");
  expect(screen.getByRole("spinbutton",{name:"Available cash (USD)"})).toHaveValue(2000);
  expect(screen.queryByLabelText("Backtest start date (optional)")).not.toBeInTheDocument();
  expect(api.createRun).not.toHaveBeenCalled();
});

it("restores the requested run purpose from its URL",async()=>{
 await mount(<NewRunPage/>,"/runs/new?task=investment");
 expect(screen.getByRole("combobox",{name:"What would you like to run?"})).toHaveValue("investment");
 expect(screen.queryByLabelText("Backtest start date (optional)")).not.toBeInTheDocument();
 expect(api.createRun).not.toHaveBeenCalled();
});

it("maps older single-backtest URLs into the shared strategies backtest",async()=>{
 await mount(<NewRunPage/>,"/runs/new?task=backtest");
 expect(screen.getByRole("combobox",{name:"What would you like to run?"})).toHaveValue("strategy-comparison");
 expect(screen.getByRole("button",{name:"Add strategy"})).toBeInTheDocument();
});
it("imports an older backtest preset as one configuration without modifying the original",async()=>{
 vi.mocked(api.listPresets).mockImplementation(async type=> type === "execution" ? [{id:92,name:"Old backtest",run_type:"execution",parameters:{execution_mode:"backtesting",strategy_type:"optimization",initial_cash:2500,population_size:80,n_generations:1234}}] : []);
 await mount(<NewRunPage defaultTask="strategy-comparison"/>);
 fireEvent.click(screen.getByRole("button",{name:"Import an older single-strategy backtest preset"}));
 await screen.findByRole("option",{name:"Old backtest"});
 fireEvent.change(screen.getByLabelText("Older backtest preset"),{target:{value:"92"}});
 expect(screen.getByRole("spinbutton",{name:/^Initial Cash/})).toHaveValue(2500);
 expect(screen.getAllByRole("listitem")).toHaveLength(1);
 expect(screen.getByLabelText("Include Buy & Hold benchmark")).not.toBeChecked();
 expect(screen.getByRole("button",{name:"Update selected"})).toBeDisabled();
 expect(api.savePreset).not.toHaveBeenCalled();expect(api.deletePreset).not.toHaveBeenCalled();expect(api.createRun).not.toHaveBeenCalled();
});

it("does not offer historical backtest presets as investment-portfolio presets",async()=>{
 vi.mocked(api.listPresets).mockResolvedValue([
  {id:93,name:"Historical run",run_type:"execution",parameters:{execution_mode:"backtesting"}},
  {id:94,name:"Portfolio allocation",run_type:"execution",parameters:{execution_mode:"investing"}}
 ]);
 await mount(<NewRunPage/>);
 await screen.findByRole("option",{name:"Portfolio allocation"});
 expect(screen.queryByRole("option",{name:"Historical run"})).not.toBeInTheDocument();
 expect(screen.getByRole("combobox",{name:"What would you like to run?"})).toHaveValue("investment");
});
it("retries an older-preset import failure independently without writing or launching",async()=>{
 let failed=false;
 vi.mocked(api.listPresets).mockImplementation(async type=>{
  if(type !== "execution") return [];
  if(!failed){failed=true;throw new Error("Older presets unavailable");}
  return [{id:95,name:"Recovered backtest",run_type:"execution",parameters:{execution_mode:"backtesting"}}];
 });
 await mount(<NewRunPage defaultTask="strategy-comparison"/>);
 fireEvent.click(screen.getByRole("button",{name:"Import an older single-strategy backtest preset"}));
 await screen.findByText(/Older presets unavailable/);
 fireEvent.click(screen.getByRole("button",{name:"Retry older backtest presets"}));
 await screen.findByRole("option",{name:"Recovered backtest"});
 expect(api.createRun).not.toHaveBeenCalled();expect(api.createStrategyComparison).not.toHaveBeenCalled();expect(api.savePreset).not.toHaveBeenCalled();
});

it.each(["investment","strategy-comparison","solvers","predictors"])("passes campaign membership separately for %s and keeps it out of presets",async task=>{
 const id="aaaaaaaaaaaa";
 vi.mocked(api.listCampaigns).mockResolvedValue([{id,name:"Validation",hypothesis:"",notes:"",created_at:"2026-10-01",updated_at:"2026-10-01",run_count:0}]);
 const submit=task==="investment"?api.createRun:task==="strategy-comparison"?api.createStrategyComparison:task==="solvers"?api.createSolverComparison:api.createPredictionComparison;
 vi.mocked(submit).mockResolvedValue({id:"launched"} as RunSummary);
 await mount(<NewRunPage/>,`/runs/new?task=${task}&campaign=${id}`);
 if(task==="investment")expect(screen.queryByRole("combobox",{name:"Campaign (optional)"})).not.toBeInTheDocument();
 else {expect(await screen.findByRole("combobox",{name:"Campaign (optional)"})).toHaveValue(id);await screen.findByRole("option",{name:"Validation"});}
 fireEvent.click(screen.getByRole("button",{name:"Review run"}));
 await waitFor(()=>expect(screen.getByText("Campaign:").parentElement).toHaveTextContent("Campaign: Validation"));
 fireEvent.click(screen.getByRole("button",{name:"Start run"}));
 await waitFor(()=>expect(submit).toHaveBeenCalledOnce());
 expect(vi.mocked(submit).mock.calls[0][0]).toMatchObject({campaign_id:id});
});

it("requires another review after changing campaign without clearing configuration",async()=>{
 vi.mocked(api.listCampaigns).mockResolvedValue([{id:"aaaaaaaaaaaa",name:"Validation",hypothesis:"",notes:"",created_at:"2026-10-01",updated_at:"2026-10-01",run_count:0}]);
 await mount(<NewRunPage/>,"/runs/new?task=strategy-comparison");
 fireEvent.change(screen.getByRole("spinbutton",{name:/^Initial Cash/}),{target:{value:"2500"}});
 fireEvent.click(screen.getByRole("button",{name:"Review run"}));
 expect(screen.getByRole("button",{name:"Start run"})).toBeInTheDocument();
 fireEvent.change(screen.getByRole("combobox",{name:"Campaign (optional)"}),{target:{value:"aaaaaaaaaaaa"}});
 expect(screen.getByRole("button",{name:"Review run"})).toBeInTheDocument();
 expect(screen.getByRole("spinbutton",{name:/^Initial Cash/})).toHaveValue(2500);expect(api.createStrategyComparison).not.toHaveBeenCalled();
});

it.each([
  ["monthly", "2024-01-15", "2024-04-01", "2024-02-01", "2024-03-31"],
  ["weekly", "2024-01-17", "2024-02-04", "2024-01-22", "2024-02-04"],
])("requires whole %s periods and constrains the backtest start", async (frequency, start, end, validStart, validEnd) => {
  vi.mocked(api.listPresets).mockResolvedValue([{id:81,name:"Incompatible dates",run_type:"strategy_comparison",parameters:{frequency,historical_start_date:start,historical_end_date:end}}]);
  await mount(<NewRunPage defaultTask="strategy-comparison" />);
  await screen.findByRole("option",{name:"Incompatible dates"});
  fireEvent.change(screen.getByLabelText("Saved preset"),{target:{value:"81"}});
  expect(screen.queryByLabelText("Date selection")).not.toBeInTheDocument();
  expect(screen.getByRole("button",{name:"Review run"})).toBeDisabled();
  setDate("Start Date", validStart); setDate("End Date", validEnd);
  setDate("Backtest start date (optional)", validStart);
  expect(screen.getByRole("button",{name:"Review run"})).toBeEnabled();
  expect(screen.getByLabelText("Backtest start date (optional)")).toHaveAttribute("min",periodInputValue(validStart, frequency as "monthly" | "weekly"));
  expect(api.createStrategyComparison).not.toHaveBeenCalled();
});

it.each(["execution","strategy","solver","prediction"] as const)("%s blocks an out-of-coverage local window in review and presets, then recovers", async kind => {
  vi.mocked(api.getLocalDataInfo).mockResolvedValue({file:"prices.pkl",frequency:"monthly",time_contract_version:1,start_date:"2022-01-01",end_date:"2024-12-31",stocks:["ABC"],crypto:[],stock_count:1,crypto_count:0});
  await mount(kind === "execution" ? <StrategyExperimentForm mode="execution" /> : <NewRunPage defaultTask="strategy-comparison" />);
  if (kind === "solver" || kind === "prediction") selectTask(kind);
  fireEvent.change(screen.getByLabelText("Data Source"),{target:{value:"local"}});
  await screen.findByText("ABC");
  fireEvent.change(screen.getByLabelText("Preset name"),{target:{value:"Coverage"}});
  expect(screen.getByLabelText("Start Date")).toHaveAttribute("min","2022-01");
  expect(screen.getByLabelText("End Date")).toHaveAttribute("max","2024-12");
  expect(screen.getByRole("button",{name:"Review run"})).toBeDisabled();
  expect(screen.getByRole("button",{name:"Save as new"})).toBeDisabled();
  fireEvent.click(screen.getByRole("button",{name:"Use full dataset date range"}));
  expect(screen.getByLabelText("Start Date")).toHaveValue("2022-01");
  expect(screen.getByLabelText("End Date")).toHaveValue("2024-12");
  expect(screen.getByRole("button",{name:"Review run"})).toBeEnabled();
  expect(screen.getByRole("button",{name:"Save as new"})).toBeEnabled();
  // A manually supplied value or loaded preset cannot bypass the native max attribute.
  setDate("End Date","2025-12-31");
  expect(screen.getByRole("button",{name:"Review run"})).toBeDisabled();
  expect(screen.getByRole("button",{name:"Save as new"})).toBeDisabled();
  expect(api.createRun).not.toHaveBeenCalled(); expect(api.createStrategyComparison).not.toHaveBeenCalled();
  expect(api.createSolverComparison).not.toHaveBeenCalled(); expect(api.createPredictionComparison).not.toHaveBeenCalled();
});
it("disables legacy dataset range shortcuts and review until timestamps are upgraded", async () => {
  vi.mocked(api.getLocalDataInfo).mockResolvedValue({file:"prices.pkl",frequency:"monthly",time_contract_version:null,start_date:"2022-12-31",end_date:"2026-05-01",stocks:["ABC"],crypto:[],stock_count:1,crypto_count:0});
  await mount(<NewRunPage />);
  fireEvent.change(screen.getByLabelText("Data Source"),{target:{value:"local"}});
  await screen.findByText("ABC");
  expect(screen.getByRole("button",{name:"Use full dataset date range"})).toBeDisabled();
  expect(screen.getByRole("button",{name:"Review run"})).toBeDisabled();
  expect(screen.getByRole("button",{name:"Use full dataset date range"})).toHaveAccessibleDescription("Upgrade the dataset timestamps before selecting its range.");
});
it("aligns the full-range shortcut inward to complete periods for a supported dataset", async () => {
  vi.mocked(api.getLocalDataInfo).mockResolvedValue({file:"prices.pkl",frequency:"monthly",time_contract_version:1,start_date:"2022-01-15",end_date:"2024-12-10",stocks:["ABC"],crypto:[],stock_count:1,crypto_count:0});
  await mount(<NewRunPage defaultTask="strategy-comparison" />);
  fireEvent.change(screen.getByLabelText("Data Source"),{target:{value:"local"}});
  await screen.findByText("ABC");
  fireEvent.click(screen.getByRole("button",{name:"Use full dataset date range"}));
  expect(screen.getByLabelText("Start Date")).toHaveValue("2022-02");
  expect(screen.getByLabelText("End Date")).toHaveValue("2024-11");
  expect(screen.getByRole("button",{name:"Review run"})).toBeEnabled();
});
it("blocks review after leaving an invalid asset edit instead of using the prior holding", async () => {
  const user = userEvent.setup();
  await mount(<NewRunPage />);
  await user.click(await screen.findByRole("button",{name:"Add holding"}));
  const asset=screen.getByLabelText("Asset 1");
  await user.click(asset); await user.type(asset,"AAPL"); await user.keyboard("{Enter}");
  await user.type(screen.getByLabelText("Shares / units 1"),"2.5");
  expect(screen.getByRole("button",{name:"Review run"})).toBeEnabled();
  await user.click(asset); await user.type(asset,"BOGUS");
  await user.click(screen.getByRole("button",{name:"Review run"}));
  expect(screen.queryByRole("button",{name:"Start run"})).not.toBeInTheDocument();
  expect(screen.getByRole("button",{name:"Review run"})).toBeDisabled();
  expect(asset).toHaveValue("BOGUS"); expect(api.createRun).not.toHaveBeenCalled();
});

it("invalidates a reviewed portfolio before an edited holding can be submitted", async () => {
  const user=userEvent.setup();
  await mount(<NewRunPage />);
  await user.click(await screen.findByRole("button",{name:"Add holding"}));
  const asset=screen.getByLabelText("Asset 1");
  await user.click(asset); await user.type(asset,"AAPL"); await user.keyboard("{Enter}");
  await user.type(screen.getByLabelText("Shares / units 1"),"2");
  await user.click(screen.getByRole("button",{name:"Review run"}));
  await user.click(asset); await user.type(asset,"MSFT");
  // Blur commits the draft and clears the prior review before a launch is allowed.
  await user.click(screen.getByRole("button",{name:"Start run"}));
  expect(api.createRun).not.toHaveBeenCalled();
  expect(screen.getByText("MSFT: 2 shares/units")).toBeInTheDocument();
  expect(screen.queryByText("AAPL: 2 shares/units")).not.toBeInTheDocument();
  expect(screen.getByRole("button",{name:"Start run"})).toBeEnabled();
});
