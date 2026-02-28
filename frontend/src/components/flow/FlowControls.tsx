import { Play, Pause, Flag, RotateCcw, Loader2 } from "lucide-react";
import type { FlowExecStatus } from "@/types";

interface FlowControlsProps {
  flowStatus: FlowExecStatus;
  finishing: boolean;
  onStart: () => void;
  onPause: () => void;
  onResume: () => void;
  onFinish: () => void;
  onReset: () => void;
}

export function FlowControls({
  flowStatus,
  finishing,
  onStart,
  onPause,
  onResume,
  onFinish,
  onReset,
}: FlowControlsProps) {
  return (
    <div className="flex items-center gap-2">
      {flowStatus === "idle" && (
        <button
          onClick={onStart}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-primary text-white forgis-text-label font-normal font-forgis-digit cursor-pointer border-none hover:bg-[var(--accent-hover)]"
        >
          <Play size={12} /> Run
        </button>
      )}

      {flowStatus === "running" && (
        <>
          <button
            onClick={onPause}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-muted text-foreground forgis-text-label font-normal font-forgis-digit cursor-pointer border-none hover:bg-muted/70"
          >
            <Pause size={12} /> Pause
          </button>
          {finishing ? (
            <span className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-muted text-muted-foreground forgis-text-label font-normal font-forgis-digit">
              <Loader2 size={12} className="animate-spin" /> Finishing…
            </span>
          ) : (
            <button
              onClick={onFinish}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-muted text-foreground forgis-text-label font-normal font-forgis-digit cursor-pointer border-none hover:bg-muted/70"
            >
              <Flag size={12} /> Finish
            </button>
          )}
        </>
      )}

      {flowStatus === "paused" && (
        <>
          <button
            onClick={onResume}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-primary text-white forgis-text-label font-normal font-forgis-digit cursor-pointer border-none hover:bg-[var(--accent-hover)]"
          >
            <Play size={12} /> Resume
          </button>
          {finishing ? (
            <span className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-muted text-muted-foreground forgis-text-label font-normal font-forgis-digit">
              <Loader2 size={12} className="animate-spin" /> Finishing…
            </span>
          ) : (
            <button
              onClick={onFinish}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-muted text-foreground forgis-text-label font-normal font-forgis-digit cursor-pointer border-none hover:bg-muted/70"
            >
              <Flag size={12} /> Finish
            </button>
          )}
        </>
      )}

      {flowStatus === "completed" && (
        <>
          <button
            onClick={onReset}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-muted text-foreground forgis-text-label font-normal font-forgis-digit cursor-pointer border-none hover:bg-muted/70"
          >
            <RotateCcw size={12} /> Reset
          </button>
          <button
            onClick={onStart}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-primary text-white forgis-text-label font-normal font-forgis-digit cursor-pointer border-none hover:bg-[var(--accent-hover)] whitespace-nowrap"
          >
            <Play size={12} /> Re-run
          </button>
        </>
      )}

      {flowStatus !== "idle" && (
        <span className="forgis-text-detail text-muted-foreground uppercase tracking-wider ml-2 font-forgis-digit">
          {flowStatus}
        </span>
      )}
    </div>
  );
}
