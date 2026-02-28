import { MarkerType, type Node, type Edge } from "@xyflow/react";
import type {
  Flow,
  FlowStep,
  FlowExecStatus,
  NodeExecState,
  NodeExecStatus,
} from "@/types";

// ── Shared type: data contract between this service and ExecutionNode ──

export type ExecNodeData = {
  label: string;
  steps?: FlowStep[];
  nodeType: string;
  nodeId: string;
  execState: NodeExecState | undefined;
  flowStatus: FlowExecStatus;
  style?: { width: number; height: number };
  message?: string;
  onSelectStep?: (nodeId: string, step: FlowStep) => void;
  [key: string]: unknown;
};

export type ExecNode = Node<ExecNodeData, "execution">;

// ── Node styling helpers ─────────────────────────────────────

export function getNodeBorderColor(
  execStatus: NodeExecStatus | undefined,
  flowStatus: FlowExecStatus,
): string {
  if (flowStatus === "idle") return "var(--node-header)";
  switch (execStatus) {
    case "running":
      return "var(--gunmetal-50)";
    case "success":
      return "var(--status-healthy)";
    case "failure":
      return "var(--orange)";
    default:
      return "var(--node-header)";
  }
}

export function getNodeOpacity(
  execStatus: NodeExecStatus | undefined,
  flowStatus: FlowExecStatus,
): number {
  if (flowStatus === "idle") return 0.9;
  switch (execStatus) {
    case "running":
      return 1;
    case "success":
    case "failure":
      return 0.9;
    default:
      return 0.3;
  }
}

// ── Flow → ReactFlow nodes ──────────────────────────────────

export function toReactFlowNodes(
  flow: Flow,
  nodeStates: Record<string, NodeExecState>,
  flowStatus: FlowExecStatus,
  onSelectStep?: (nodeId: string, step: FlowStep) => void,
): ExecNode[] {
  return flow.nodes.map((n) => ({
    id: n.id,
    type: "execution" as const,
    position: { ...n.position },
    data: {
      label: n.label,
      steps: n.steps,
      nodeType: n.type,
      nodeId: n.id,
      execState: nodeStates[n.id],
      flowStatus,
      style: n.style,
      onSelectStep,
    },
  }));
}

// ── Flow → ReactFlow edges (initial static styling) ─────────

export function toReactFlowEdges(flow: Flow): Edge[] {
  return flow.edges.map((e) => {
    const isLoop = e.data?.isLoop === true;
    const isConditional = e.data?.transitionType === "conditional";
    const condition = e.data?.condition as string | undefined;

    const label = isConditional && condition ? condition : undefined;

    return {
      id: e.id,
      source: e.source,
      target: e.target,
      type: "smoothstep",
      sourceHandle: isLoop ? "loop-source" : "bottom",
      targetHandle: isLoop ? "loop-target" : "top",
      label,
      labelStyle: label
        ? { fontSize: "var(--text-size-caption)", fill: "var(--muted-foreground)", fontFamily: "inherit" }
        : undefined,
      labelBgStyle: label
        ? { fill: "var(--muted)", stroke: "var(--border)", strokeWidth: 1, rx: 4, ry: 4 }
        : undefined,
      labelBgPadding: label ? ([8, 4] as [number, number]) : undefined,
      style: isLoop
        ? { stroke: "var(--accent-hover)", strokeWidth: 2, strokeDasharray: "6 3" }
        : isConditional
          ? { stroke: "var(--vertical-quality)", strokeWidth: 2 }
          : { stroke: "var(--steel)", strokeWidth: 2 },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: isLoop ? "var(--accent-hover)" : isConditional ? "var(--vertical-quality)" : "var(--steel)",
      },
    };
  });
}

// ── Execution-time edge styling ─────────────────────────────

export function updateEdgeStyles(
  edges: Edge[],
  nodeStates: Record<string, NodeExecState>,
  flowStatus: FlowExecStatus,
): Edge[] {
  return edges.map((e) => {
    const isLoop = e.sourceHandle === "loop-source";
    const isConditional = e.label !== undefined;
    const isStartEdge = e.source === "start";
    const srcState = nodeStates[e.source];
    const tgtState = nodeStates[e.target];
    const srcDone = isStartEdge
      ? flowStatus !== "idle"
      : srcState?.status === "success";
    const isEndEdge = e.target === "end";
    const tgtRunning = tgtState?.status === "running";
    const tgtDone = isEndEdge
      ? flowStatus === "completed"
      : tgtState?.status === "success" || tgtState?.status === "failure";

    let stroke = isLoop ? "var(--accent-hover)" : isConditional ? "var(--vertical-quality)" : "var(--steel)";
    let edgeOpacity = flowStatus === "idle" ? 1 : 0.3;
    let animated = false;

    if (srcDone && tgtDone) {
      stroke = "var(--status-healthy)";
      edgeOpacity = 1;
    } else if (srcDone && tgtRunning) {
      stroke = "var(--status-healthy)";
      edgeOpacity = 1;
      animated = true;
    }

    return {
      ...e,
      animated,
      style: {
        stroke,
        opacity: edgeOpacity,
        strokeWidth: 2,
        ...(isLoop ? { strokeDasharray: "6 3" } : {}),
      },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: stroke,
      },
    };
  });
}

// ── Utility ─────────────────────────────────────────────────

/** Finds which state node contains a given step ID. */
export function findStateForStep(flow: Flow, stepId: string): string | null {
  for (const node of flow.nodes) {
    if (node.steps?.some((s) => s.id === stepId)) {
      return node.id;
    }
  }
  return null;
}
