import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, expect, it, vi } from "vitest";
import BacktestRunComparison from "@/components/ui/BacktestRunComparison";
import { savedCandidates } from "@/lib/savedComparison";
import { backtest } from "../fixtures/savedRuns";
const {draw} = vi.hoisted(() => ({draw: vi.fn()}));
vi.mock("@/components/ui/PlotlyChart", () => ({default: ({figure}: {figure: unknown}) => {draw(figure); return <div data-testid="curve" />;}}));
afterEach(() => {cleanup(); vi.clearAllMocks();});
function entries() { return savedCandidates(structuredClone(backtest())).slice(0, 2); }
function mount(selected = entries()) {
  render(<MemoryRouter><BacktestRunComparison entries={selected} candidates={selected} manifests={{}} referenceId={selected[0].id} onReferenceChange={vi.fn()} comparisonSearch="?item=a%3ASharpe" /></MemoryRouter>);
}
it("shows configuration differences before a table-first matrix without a chart or mode selector", () => {
  mount();
  const matrix = screen.getByRole("region", {name: "Results matrix"});
  expect(matrix).toHaveTextContent("10.00%");
  expect(matrix).toHaveTextContent("20.00%");
  expect(screen.getByRole("region", {name: "Configuration differences"})).toHaveTextContent("Population");
  expect(screen.queryByLabelText("Comparison mode")).not.toBeInTheDocument();
  expect(screen.queryByTestId("curve")).not.toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("Display"), {target: {value: "differences"}});
  expect(matrix).toHaveTextContent("+10.00 pp");
  expect(matrix).toHaveTextContent("Reference");
});
it("shows multiple years as rows, missing configurations as blanks and single configuration as a series", () => {
  const selected = entries();
  selected[1].periods = ["2025-01-01 → 2025-12-31"];
  mount(selected);
  const rows = within(screen.getByRole("region", {name: "Results matrix"})).getAllByRole("row");
  expect(rows).toHaveLength(3);
  expect(rows[1]).toHaveTextContent("2025-01-01 → 2025-12-31");
  expect(rows[1]).toHaveTextContent("—");
  fireEvent.change(screen.getByLabelText("Display"), {target: {value: "differences"}});
  expect(rows[1]).toHaveTextContent("No reference in this period and context");
  expect(screen.getByRole("region", {name: "Consistency summary"})).toHaveTextContent("0 / 2");
});
it("keeps repeated executions visible and never averages or silently selects one", () => {
  const selected = entries(), duplicate = structuredClone(selected[0]);
  duplicate.id = "a:Duplicate"; duplicate.name = "Duplicate";
  duplicate.metrics.total_return = 0.3;
  mount([...selected, duplicate]);
  const matrix = screen.getByRole("region", {name: "Results matrix"});
  expect(matrix).toHaveTextContent("2 executions; shown separately");
  expect(matrix).toHaveTextContent("10.00%");
  expect(matrix).toHaveTextContent("30.00%");
  fireEvent.change(screen.getByLabelText("Display"), {target: {value: "differences"}});
  expect(matrix).toHaveTextContent("Multiple reference executions");
});
it("scopes optional curves and saved returns to an inspected period, retaining the initial drawdown baseline", async () => {
  const selected = entries();
  selected[0].points = [{date: "2026-01-31", value: 800}, {date: "2026-02-28", value: 900}];
  selected[0].observations = [{start: "2026-01-01", end: "2026-01-31", value: 0}];
  mount(selected);
  fireEvent.click(screen.getByRole("button", {name: "Inspect period details"}));
  expect(screen.getByRole("link", {name: "A · Sharpe"})).toHaveAttribute("href", "/runs/a");
  fireEvent.click(screen.getByText("Growth and drawdown curves"));
  fireEvent.change(screen.getByLabelText("Performance curve"), {target: {value: "drawdown"}});
  await screen.findByTestId("curve");
  const figure = draw.mock.calls.at(-1)![0] as {data: {name: string; y: number[]}[]};
  expect(figure.data[0].name).toBe("A · Sharpe");
  expect(figure.data[0].y[0]).toBeCloseTo(-20);
  fireEvent.click(screen.getByText("Saved period returns"));
  const row = await screen.findByRole("rowheader", {name: "2026-01-01 → 2026-01-31"});
  expect(row.closest("tr")).toHaveTextContent("0.00%—");
});
it("flags metric definitions without hiding recorded numeric values", () => {
  const selected = entries(); selected[1].definitions = {};
  mount(selected);
  fireEvent.change(screen.getByLabelText("Metric"), {target: {value: "sharpe_ratio"}});
  const matrix = screen.getByRole("region", {name: "Results matrix"});
  expect(matrix).toHaveTextContent("1.20");
  expect(matrix).toHaveTextContent("Reported only");
  fireEvent.change(screen.getByLabelText("Display"), {target: {value: "differences"}});
  expect(matrix).toHaveTextContent("Not comparable");
});
