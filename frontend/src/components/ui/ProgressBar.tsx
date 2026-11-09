/**
 * Progress bar component — shows fraction + message for an active run.
 */

interface Props {
  fraction: number;
  message: string;
  isConnected: boolean;
}

export default function ProgressBar({ fraction, message, isConnected }: Props) {
  const pct = Math.max(0, Math.min(100, Math.round(fraction * 100)));

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs text-gray-500">
        <span>{message || "Waiting for progress..."}</span>
        <span>
          {pct}%{" "}
          {isConnected ? (
            <span className="text-green-600" title="Connection to progress updates; worker health is reported separately">connected</span>
          ) : (
            <span className="text-gray-400">disconnected</span>
          )}
        </span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-gray-200">
        <div
          className="h-full rounded-full bg-blue-600 transition-all duration-300"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
