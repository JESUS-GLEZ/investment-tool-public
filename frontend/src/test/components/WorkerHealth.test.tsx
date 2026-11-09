import { render, screen, cleanup } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import WorkerHealth from "@/components/ui/WorkerHealth";
import type { RunSummary } from "@/types/api";

afterEach(cleanup);
const run: RunSummary = {id: "test", run_type: "solver_comparison", status: "running", created_at: "2026-01-01", completed_at: null, error_message: null};
describe("worker health", () => {
  it("distinguishes a live worker from stagnant progress", () => {
    render(<WorkerHealth run={{...run, worker_health: "alive", progress_fraction: 0}} />);
    expect(screen.getByRole("status")).toHaveTextContent("Worker alive");
  });
  it("warns when heartbeat has stopped", () => {
    render(<WorkerHealth run={{...run, worker_health: "unresponsive"}} />);
    expect(screen.getByRole("alert")).toHaveTextContent("no heartbeat for over 2 minutes");
  });
  it("does not equate connection with worker health", () => {
    render(<WorkerHealth run={run} />);
    expect(screen.getByRole("status")).toHaveTextContent("Worker activity is unverified");
  });
  it("hides liveness for terminal runs", () => {
    const {container} = render(<WorkerHealth run={{...run, status: "cancelled", worker_health: "unresponsive"}} />);
    expect(container).toBeEmptyDOMElement();
  });
});

import RunProgress from "@/components/ui/RunProgress";
it("restores persisted progress when the socket has no update after reload", () => {
  render(<RunProgress run={{...run, progress_fraction: 0.6, progress_message: "SPEA2: restart 2/3"}} live={{fraction: 0, message: "", isConnected: false}} />);
  expect(screen.getByText(/60%/)).toBeInTheDocument();
  expect(screen.getByText("SPEA2: restart 2/3")).toBeInTheDocument();
});
it("uses a newer solver update before the next database heartbeat", () => {
  render(<RunProgress run={{...run, progress_fraction: 0.4, progress_message: "old"}} live={{fraction: 0.6, message: "current", isConnected: true}} />);
  expect(screen.getByText(/60%/)).toBeInTheDocument();
  expect(screen.getByText("current")).toBeInTheDocument();
});
