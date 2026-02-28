import { Handle, Position, type NodeProps } from "@xyflow/react";
import { Cpu } from "lucide-react";
import { EXECUTOR_ICONS } from "@/constants/executorConfig";
import {
  getNodeBorderColor,
  getNodeOpacity,
  type ExecNode,
} from "@/services/flowTransformService";
import type { NodeExecStatus } from "@/types";

function ExecutionNode({ data }: NodeProps<ExecNode>) {
  const { label, steps, nodeType, execState, flowStatus, style, message } = data;
  const status = (execState?.status ?? "idle") as NodeExecStatus;
  const borderColor = getNodeBorderColor(status, flowStatus);
  const opacity = getNodeOpacity(status, flowStatus);
  const isRunning = status === "running";

  // Count completed steps for progress
  const totalSteps = steps?.length ?? 0;
  const doneSteps = steps?.filter(
    (s) => execState?.stepStates?.[s.id]?.status === "success",
  ).length ?? 0;

  // Start/End nodes - simple pill shape
  if (nodeType === "start" || nodeType === "end") {
    return (
      <>
        {nodeType === "end" && (
          <Handle
            type="target"
            position={Position.Top}
            id="top"
            className="!w-2 !h-2 !border !border-solid"
            style={{ background: "var(--handle-bg)", borderColor: "var(--handle-border)" }}
          />
        )}
        <div
          className="rounded-full px-6 py-2 text-center"
          style={{
            background: "var(--gunmetal-50)",
            opacity: 0.9,
          }}
        >
          <div className="forgis-text-reading font-normal text-white font-forgis-digit">{label}</div>
        </div>
        {nodeType === "start" && (
          <Handle
            type="source"
            position={Position.Bottom}
            id="bottom"
            className="!w-2 !h-2 !border !border-solid"
            style={{ background: "var(--handle-bg)", borderColor: "var(--handle-border)" }}
          />
        )}
      </>
    );
  }

  // State nodes - container with steps inside
  return (
    <>
      <Handle
        type="target"
        position={Position.Top}
        id="top"
        className="!w-2 !h-2 !border !border-solid"
            style={{ background: "var(--handle-bg)", borderColor: "var(--handle-border)" }}
      />
      {/* Left-side handles for loop edges */}
      <Handle
        type="source"
        position={Position.Left}
        id="loop-source"
        className="!w-2 !h-2 !border !border-solid"
            style={{ background: "var(--handle-bg)", borderColor: "var(--handle-border)" }}
      />
      <Handle
        type="target"
        position={Position.Left}
        id="loop-target"
        className="!w-2 !h-2 !border !border-solid"
            style={{ background: "var(--handle-bg)", borderColor: "var(--handle-border)" }}
      />
      <div
        className="rounded-lg overflow-hidden"
        style={{
          width: style?.width ?? 300,
          minHeight: 80,
          background: "var(--card)",
          border: `2px solid ${borderColor}`,
          opacity,
          boxShadow: isRunning ? "0 0 16px 4px var(--node-glow)" : undefined,
          transition: "border-color 0.3s, opacity 0.3s, box-shadow 0.3s",
        }}
      >
        {/* Header */}
        <div
          className="px-3 py-2 flex items-center justify-between"
          style={{ background: "var(--node-header)" }}
        >
          <span className="forgis-text-reading font-normal text-[var(--gunmetal)] dark:text-foreground font-forgis-digit uppercase tracking-wide">
            {label}
          </span>
          {totalSteps > 0 && status !== "idle" && (
            <span className="forgis-text-caption text-muted-foreground font-forgis-digit tabular-nums">
              {doneSteps}/{totalSteps}
            </span>
          )}
        </div>

        {/* Progress bar (thin, below header) */}
        {totalSteps > 0 && status === "running" && (
          <div className="h-[2px] bg-muted/30">
            <div
              className="h-full transition-all duration-300"
              style={{
                width: `${(doneSteps / totalSteps) * 100}%`,
                background: "var(--status-healthy)",
              }}
            />
          </div>
        )}

        {/* Steps list / empty message */}
        <div className="p-2">
          {!steps?.length && message && (
            <p className="px-2 py-3 forgis-text-label text-muted-foreground text-center font-forgis-body">{message}</p>
          )}
          {steps?.map((step, idx) => {
            const stepState = execState?.stepStates?.[step.id];
            const stepStatus = stepState?.status ?? "idle";
            const ExecIcon = EXECUTOR_ICONS[step.executor] ?? Cpu;

            // Status-based styling
            let rowBg = "transparent";
            let ringClass = "";
            if (stepStatus === "running") {
              rowBg = "color-mix(in srgb, var(--primary) 8%, transparent)";
              ringClass = "ring-1 ring-primary/40";
            } else if (stepStatus === "success") {
              rowBg = "color-mix(in srgb, var(--status-healthy) 6%, transparent)";
            } else if (stepStatus === "failure") {
              rowBg = "color-mix(in srgb, var(--status-critical) 8%, transparent)";
            }

            return (
              <div key={step.id}>
                {/* Dotted separator between steps */}
                {idx > 0 && (
                  <div className="flex items-center px-2 py-0.5">
                    <div className="w-5" />
                    <div className="flex-1 border-t border-dotted border-border/40" />
                  </div>
                )}
                <div
                  className={`flex items-stretch rounded forgis-text-label cursor-pointer transition-all ${ringClass}`}
                  style={{ background: rowBg }}
                  onClick={() => {
                    data.onSelectStep?.(data.nodeId, step);
                  }}
                >
                  {/* Accent strip */}
                  <div
                    className="w-1 shrink-0 rounded-l bg-border/60"
                  />

                  {/* Content */}
                  <div className="flex-1 px-2 py-1.5 min-w-0">
                    <div className="flex items-center gap-1.5">
                      {/* Status dot */}
                      {stepStatus === "running" && (
                        <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse shrink-0" />
                      )}
                      {stepStatus === "success" && (
                        <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: "var(--status-healthy)" }} />
                      )}
                      {stepStatus === "failure" && (
                        <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: "var(--status-critical)" }} />
                      )}

                      {/* Skill name */}
                      <span className="font-normal text-[var(--gunmetal)] dark:text-foreground truncate font-forgis-digit">{step.skill}</span>

                    </div>

                    {/* Executor + step id */}
                    <div className="flex items-center gap-1 mt-0.5 pl-5">
                      <ExecIcon size={9} className="shrink-0 text-muted-foreground" />
                      <span className="forgis-text-caption text-muted-foreground font-forgis-body">{step.executor}</span>
                      <span className="forgis-text-caption text-muted-foreground/40 mx-0.5">|</span>
                      <span className="forgis-text-caption text-muted-foreground/60 font-forgis-body truncate">{step.id}</span>
                    </div>

                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
      <Handle
        type="source"
        position={Position.Bottom}
        id="bottom"
        className="!w-2 !h-2 !border !border-solid"
            style={{ background: "var(--handle-bg)", borderColor: "var(--handle-border)" }}
      />
    </>
  );
}

// Defined outside component to avoid re-renders
export const nodeTypes = { execution: ExecutionNode };
