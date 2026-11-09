import type { RunSummary } from "@/types/api";
import ProgressBar from "./ProgressBar";

/** Keep the last persisted progress visible after reload or a dropped socket. */
export default function RunProgress({ run, live }: {
  run: RunSummary;
  live: { fraction: number; message: string; isConnected: boolean };
}) {
  const persisted = run.progress_fraction ?? 0;
  return <ProgressBar
    fraction={Math.max(live.fraction, persisted)}
    message={persisted > live.fraction || !live.message ? run.progress_message || "Waiting for worker progress" : live.message}
    isConnected={live.isConnected}
  />;
}
