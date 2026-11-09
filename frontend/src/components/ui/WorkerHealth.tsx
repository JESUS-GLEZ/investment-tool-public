import type { RunSummary } from "@/types/api";

export default function WorkerHealth({ run }: { run: RunSummary }) {
  if (run.status !== "running") return null;
  const stale = run.worker_health === "unresponsive";
  return <p role={stale ? "alert" : "status"} className={`text-xs ${stale ? "text-amber-700" : "text-gray-500"}`}>
    {stale ? "Worker unresponsive: no heartbeat for over 2 minutes. The run may have stopped; you can cancel it."
      : run.worker_health === "alive" ? "Worker alive. Progress reflects reported work, not elapsed time."
      : "Worker activity is unverified; a live connection alone does not mean the solver is running."}
    {run.heartbeat_at && ` Last heartbeat: ${new Date(run.heartbeat_at).toLocaleString()}.`}
  </p>;
}
