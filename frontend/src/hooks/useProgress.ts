/**
 * Hook that subscribes to real-time progress for a run via WebSocket.
 *
 * Returns `{ fraction, message, isConnected }`.
 * Automatically closes when the server sends a sentinel (fraction >= 1.0).
 */
import { useEffect, useRef, useState } from "react";
import { getAccessToken } from "@/lib/tokens";
import type { ProgressMessage } from "@/types/api";

interface UseProgressResult {
  fraction: number;
  message: string;
  isConnected: boolean;
}

interface ProgressState extends UseProgressResult {
  runId: string | null;
}

const DEFAULT_PROGRESS_STATE: ProgressState = {
  runId: null,
  fraction: 0,
  message: "",
  isConnected: false,
};

export function useProgress(
  runId: string | null | undefined,
): UseProgressResult {
  const [state, setState] = useState<ProgressState>(DEFAULT_PROGRESS_STATE);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!runId) return;

    const token = getAccessToken();
    if (!token) return;

    // Build WS URL — uses Vite proxy in dev, same-origin in prod
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${protocol}//${window.location.host}/ws/progress/${runId}?token=${token}`;

    const ws = new WebSocket(url);
    const currentRunId = runId;
    wsRef.current = ws;

    ws.onopen = () => {
      setState({
        runId: currentRunId,
        fraction: 0,
        message: "",
        isConnected: true,
      });
    };

    ws.onmessage = (event) => {
      try {
        const data: ProgressMessage = JSON.parse(event.data);
        const isTerminal =
          data.fraction >= 1 &&
          (data.message.startsWith("completed") ||
            data.message.startsWith("failed") ||
            data.message.startsWith("cancelled"));

        setState({
          runId: currentRunId,
          fraction: data.fraction,
          message: data.message,
          isConnected: !isTerminal,
        });

        if (isTerminal) {
          ws.close();
        }
      } catch {
        // non-JSON message, ignore
      }
    };

    ws.onclose = () => {
      setState((prev) =>
        prev.runId === currentRunId ? { ...prev, isConnected: false } : prev,
      );
    };
    ws.onerror = () => {
      setState((prev) =>
        prev.runId === currentRunId ? { ...prev, isConnected: false } : prev,
      );
    };

    return () => {
      ws.close();
      if (wsRef.current === ws) {
        wsRef.current = null;
      }
    };
  }, [runId]);

  const visibleState = state.runId === runId ? state : DEFAULT_PROGRESS_STATE;
  return {
    fraction: visibleState.fraction,
    message: visibleState.message,
    isConnected: visibleState.isConnected,
  };
}
