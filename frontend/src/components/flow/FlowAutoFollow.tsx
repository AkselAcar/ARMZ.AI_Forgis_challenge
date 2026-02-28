import { useEffect } from "react";
import { useReactFlow } from "@xyflow/react";
import type { NodeExecState, FlowExecStatus } from "@/types";

export function FlowAutoFollow({
  nodeStates,
  flowStatus,
}: {
  nodeStates: Record<string, NodeExecState>;
  flowStatus: FlowExecStatus;
}) {
  const { setCenter, getNode } = useReactFlow();

  useEffect(() => {
    if (flowStatus !== "running" && flowStatus !== "paused") return;

    const runningId = Object.keys(nodeStates).find(
      (id) => nodeStates[id].status === "running",
    );
    if (!runningId) return;

    const node = getNode(runningId);
    if (!node) return;

    const x = node.position.x + (node.measured?.width ?? 280) / 2;
    const y = node.position.y + (node.measured?.height ?? 120) / 2;
    setCenter(x, y, { zoom: 1.1, duration: 600 });
  }, [nodeStates, flowStatus, setCenter, getNode]);

  return null;
}
