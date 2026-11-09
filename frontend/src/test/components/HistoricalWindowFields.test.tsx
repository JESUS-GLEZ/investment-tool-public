import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { afterEach, expect, it, vi } from "vitest";
import HistoricalWindowFields from "@/components/ui/HistoricalWindowFields";
import PeriodDateField from "@/components/ui/PeriodDateField";
import type { Frequency } from "@/types/api";
function Example({frequency = "monthly", initialStart = "2024-01-01", initialEnd = "2024-12-31", bounds}: {frequency?: Frequency; initialStart?: string; initialEnd?: string; bounds?: {start:string;end:string}}) {
  const [start, setStart] = useState(initialStart), [end, setEnd] = useState(initialEnd);
  return <><HistoricalWindowFields bounds={bounds} frequency={frequency} start={start} end={end} onStart={setStart} onEnd={setEnd} /><output data-testid="window">{start}/{end}</output></>;
}
afterEach(() => {cleanup(); vi.restoreAllMocks();});
it("uses one native month input and sends inclusive leap-year boundaries", () => {
  render(<Example />);
  expect(screen.queryByLabelText("Date selection")).not.toBeInTheDocument();
  expect(screen.queryByLabelText("Start Date year")).not.toBeInTheDocument();
  expect(screen.getByLabelText("Start Date")).toHaveAttribute("type","month");
  fireEvent.change(screen.getByLabelText("End Date"), {target: {value: "2024-02"}});
  expect(screen.getByTestId("window")).toHaveTextContent("2024-01-01/2024-02-29");
});
it("uses native ISO weeks across calendar years", () => {
  render(<Example frequency="weekly" initialStart="2024-12-30" initialEnd="2025-01-12" />);
  expect(screen.getByLabelText("Start Date")).toHaveAttribute("type","week");
  expect(screen.getByLabelText("Start Date")).toHaveValue("2025-W01");
  fireEvent.change(screen.getByLabelText("End Date"), {target: {value: "2025-W01"}});
  expect(screen.getByTestId("window")).toHaveTextContent("2024-12-30/2025-01-05");
});
it("explains incompatible saved dates beside the field and hides a misleading requested window", () => {
  render(<Example initialStart="2024-01-15" initialEnd="2024-04-01" />);
  expect(screen.getByLabelText("Start Date")).toHaveValue("");
  expect(screen.getByLabelText("Start Date")).toHaveAccessibleDescription("Choose a complete month for this frequency.");
  expect(screen.queryByText(/Requested window:/)).not.toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("Start Date"),{target:{value:"2024-02"}});
  fireEvent.change(screen.getByLabelText("End Date"),{target:{value:"2024-03"}});
  expect(screen.getByTestId("window")).toHaveTextContent("2024-02-01/2024-03-31");
});
it("keeps valid ISO week 53 and rejects an invalid week with field-level guidance", () => {
  render(<Example frequency="weekly" initialStart="2020-12-28" initialEnd="2021-12-26" />);
  expect(screen.getByLabelText("Start Date")).toHaveValue("2020-W53");
  fireEvent.change(screen.getByLabelText("Start Date"),{target:{value:"2021-W53"}});
  expect(screen.getByLabelText("Start Date")).toBeInvalid();
  expect(screen.getByLabelText("Start Date")).toHaveAccessibleDescription("Choose a valid ISO week.");
  fireEvent.change(screen.getByLabelText("Start Date"),{target:{value:"2021-W01"}});
  expect(screen.getByTestId("window")).toHaveTextContent("2021-01-04/2021-12-26");
});
it("uses native daily dates without assuming exchange sessions", () => {
  render(<Example frequency="daily" />);
  expect(screen.getByLabelText("Start Date")).toHaveAttribute("type", "date");
  fireEvent.change(screen.getByLabelText("Start Date"),{target:{value:"2024-02-03"}});
  expect(screen.getByTestId("window")).toHaveTextContent("2024-02-03/2024-12-31");
});
it("bounds native inputs and flags dates outside dataset coverage", () => {
  render(<Example initialEnd="2025-12-31" bounds={{start:"2024-01-01",end:"2024-12-31"}} />);
  expect(screen.getByLabelText("Start Date")).toHaveAttribute("min","2024-01");
  expect(screen.getByLabelText("End Date")).toHaveAttribute("max","2024-12");
  expect(screen.getByLabelText("End Date")).toHaveAccessibleDescription("Choose a month on or before 2024-12-31.");
  expect(screen.getByText(/Choose dates within dataset coverage/)).toBeInTheDocument();
});
it("keeps an empty optional backtest date valid when historical bounds change", () => {
  const {rerender} = render(<PeriodDateField id="backtest" label="Backtest start" value="" frequency="monthly" optional min="2024-01-01" max="2024-12-31" onChange={() => {}} />);
  rerender(<PeriodDateField id="backtest" label="Backtest start" value="" frequency="monthly" optional min="2020-01-01" max="2020-12-31" onChange={() => {}} />);
  expect(screen.getByLabelText("Backtest start")).toHaveValue("");
  expect(screen.getByLabelText("Backtest start")).toBeValid();
});

it.each(["monthly","weekly"] as const)("supports incremental %s text fallback entry and blocks an incomplete optional date", frequency => {
  function Fallback() {
    const [value,setValue]=useState("");
    return <><PeriodDateField id="fallback" label="Backtest start" value={value} frequency={frequency} optional onChange={setValue}/><output data-testid="date">{value}</output></>;
  }
  // Emulate the platform retaining text type when a month/week type is requested.
  const setter=Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,"type")!.set!;
  vi.spyOn(HTMLInputElement.prototype,"type","set").mockImplementation(function(this:HTMLInputElement,type:string) {
    setter.call(this,type === "month" || type === "week" ? "text" : type);
  });
  render(<Fallback />);
  const input=screen.getByLabelText("Backtest start") as HTMLInputElement;
  expect(input.type).toBe("text");
  fireEvent.change(input,{target:{value:"2"}});
  expect(input).toHaveValue("2"); expect(input.checkValidity()).toBe(false);
  expect(screen.getByTestId("date")).toHaveTextContent("");
  fireEvent.change(input,{target:{value:"2024-"}});
  expect(input).toHaveValue("2024-"); expect(input.checkValidity()).toBe(false);
  fireEvent.change(input,{target:{value:frequency === "weekly" ? "2024-W01" : "2024-01"}});
  expect(input.checkValidity()).toBe(true);
  expect(screen.getByTestId("date")).toHaveTextContent("2024-01-01");
  fireEvent.change(input,{target:{value:"x"}});
  fireEvent.click(screen.getByRole("button",{name:"Use automatic warm-up"}));
  expect(input).toHaveValue(""); expect(input.checkValidity()).toBe(true);
});
