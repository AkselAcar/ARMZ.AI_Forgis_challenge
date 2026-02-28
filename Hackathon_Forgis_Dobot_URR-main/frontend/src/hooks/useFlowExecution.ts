import { useCallback, useEffect, useRef, useState } from "react";
import {
  startFlow as apiStartFlow,
  abortFlow as apiAbortFlow,
  pauseFlow as apiPauseFlow,
  resumeFlow as apiResumeFlow,
  finishFlow as apiFinishFlow,
} from "@/api/flowApi";
import { createFlowSocket } from "@/api/flowSocket";
import { startCameraStream, stopCameraStream } from "@/api/cameraApi";
import { findStateForStep } from "@/services/flowTransformService";
import type { CameraCallbacks } from "./useCamera";
import type {
  Flow,
  FlowExecStatus,
  NodeExecState,
  ServerMessage,
  StepExecState,
} from "@/types";

export function useFlowExecution(
  flow: Flow | null,
  camera: CameraCallbacks,
) {
  const [flowStatus, setFlowStatus] = useState<FlowExecStatus>("idle");
  const [nodeStates, setNodeStates] = useState<Record<string, NodeExecState>>({});
  const [finishing, setFinishing] = useState(false);

  const socketRef = useRef<{ close: () => void } | null>(null);
  const handleMessageRef = useRef<(msg: ServerMessage) => void>(() => {});

  // ── WebSocket Event Handling ────────────────────────────────

  const handleMessage = useCallback(
    (msg: ServerMessage) => {
      switch (msg.type) {
        case "connected":
          break;

        case "flow_started":
          setFlowStatus("running");
          break;

        case "flow_completed":
          setFlowStatus("completed");
          setFinishing(false);
          stopCameraStream();
          camera.onStreamStop();
          break;

        case "flow_aborted":
          setFlowStatus("idle");
          setFinishing(false);
          stopCameraStream();
          camera.onStreamStop();
          break;

        case "flow_paused":
          setFlowStatus("paused");
          break;

        case "flow_resumed":
          setFlowStatus("running");
          break;

        case "flow_error":
          setFlowStatus("error");
          stopCameraStream();
          camera.onStreamStop();
          console.error("[ws] Flow error:", msg.error);
          break;

        case "loop_restart":
          if (flow) {
            const reset: Record<string, NodeExecState> = {};
            for (const node of flow.nodes) {
              reset[node.id] = { status: "idle", stepStates: {} };
            }
            setNodeStates(reset);
          }
          break;

        case "state_entered":
          setNodeStates((prev) => ({
            ...prev,
            [msg.state]: {
              ...prev[msg.state],
              status: "running",
            },
          }));
          break;

        case "state_completed":
          setNodeStates((prev) => ({
            ...prev,
            [msg.state]: {
              ...prev[msg.state],
              status: "success",
            },
          }));
          break;

        case "waiting_condition":
          break;

        case "step_started": {
          if (!flow) break;
          const stateId = findStateForStep(flow, msg.step_id);
          if (stateId) {
            setNodeStates((prev) => {
              const nodeState = prev[stateId] || { status: "running", stepStates: {} };
              const stepStates: Record<string, StepExecState> = {
                ...(nodeState.stepStates || {}),
                [msg.step_id]: { status: "running" },
              };
              return {
                ...prev,
                [stateId]: {
                  ...nodeState,
                  status: "running",
                  currentStep: msg.step_id,
                  stepStates,
                },
              };
            });
          }
          break;
        }

        case "step_completed": {
          // Capture label from get_label skill results
          if (msg.result?.label && typeof msg.result.label === "string") {
            camera.onLabel(msg.result.label);
          }

          if (!flow) break;
          const stateId = findStateForStep(flow, msg.step_id);
          if (stateId) {
            setNodeStates((prev) => {
              const nodeState = prev[stateId] || { status: "running", stepStates: {} };
              const stepStates: Record<string, StepExecState> = {
                ...(nodeState.stepStates || {}),
                [msg.step_id]: { status: "success", retries: msg.retries, result: msg.result },
              };
              return {
                ...prev,
                [stateId]: {
                  ...nodeState,
                  stepStates,
                },
              };
            });
          }
          break;
        }

        case "step_error": {
          if (!flow) break;
          const stateId = findStateForStep(flow, msg.step_id);
          if (stateId) {
            setNodeStates((prev) => {
              const nodeState = prev[stateId] || { status: "running", stepStates: {} };
              const stepStates: Record<string, StepExecState> = {
                ...(nodeState.stepStates || {}),
                [msg.step_id]: { status: "failure", error: msg.error },
              };
              return {
                ...prev,
                [stateId]: {
                  ...nodeState,
                  status: "failure",
                  stepStates,
                },
              };
            });
          }
          break;
        }

        case "step_retry":
          break;

        case "step_skipped":
          break;

        case "camera_frame":
          camera.onFrame(msg.frame, msg.timestamp);
          break;

        case "bounding_box":
          camera.onBbox(msg.bbox, msg.frame_width, msg.frame_height, msg.display_duration_ms);
          break;
      }
    },
    [flow, camera]
  );

  // Keep ref updated with latest handleMessage to avoid stale closures in WebSocket
  useEffect(() => {
    handleMessageRef.current = handleMessage;
  }, [handleMessage]);

  // ── WebSocket connection ────────────────────────────────────

  const connectWebSocket = useCallback(async () => {
    if (socketRef.current) return;
    socketRef.current = await createFlowSocket(
      (msg) => handleMessageRef.current(msg),
      () => { socketRef.current = null; }
    );
  }, []);

  // ── Reset execution state when a new flow is generated ──────

  useEffect(() => {
    socketRef.current?.close();
    socketRef.current = null;

    if (flow) {
      const initial: Record<string, NodeExecState> = {};
      for (const node of flow.nodes) {
        initial[node.id] = { status: "idle", stepStates: {} };
      }
      setNodeStates(initial);
    } else {
      setNodeStates({});
    }
    setFlowStatus("idle");
  }, [flow]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      socketRef.current?.close();
    };
  }, []);

  // ── Flow control functions ──────────────────────────────────

  const startFlow = useCallback(async () => {
    if (!flow) return;

    const initial: Record<string, NodeExecState> = {};
    for (const node of flow.nodes) {
      initial[node.id] = { status: "idle", stepStates: {} };
    }
    setNodeStates(initial);
    setFinishing(false);

    await connectWebSocket();
    try {
      await startCameraStream(15);
    } catch (err) {
      console.warn("Camera stream unavailable, starting flow without camera:", err);
    }
    await apiStartFlow(flow.id);
  }, [flow, connectWebSocket]);

  const pauseFlow = useCallback(async () => {
    try {
      await apiPauseFlow();
    } catch (err) {
      console.error("Failed to pause flow:", err);
    }
  }, []);

  const resumeFlow = useCallback(async () => {
    try {
      await apiResumeFlow();
    } catch (err) {
      console.error("Failed to resume flow:", err);
    }
  }, []);

  const abortFlow = useCallback(async () => {
    try {
      await apiAbortFlow();
    } catch (err) {
      console.error("Failed to abort flow:", err);
    }
    setFlowStatus("idle");
  }, []);

  const finishFlow = useCallback(async () => {
    setFinishing(true);
    try {
      await apiFinishFlow();
    } catch (err) {
      console.error("Failed to finish flow:", err);
    }
  }, []);

  const resetFlow = useCallback(async () => {
    setFinishing(false);
    socketRef.current?.close();
    socketRef.current = null;

    await stopCameraStream();
    camera.onReset(); // clears frame, labels, QC, bbox

    if (flow) {
      const initial: Record<string, NodeExecState> = {};
      for (const node of flow.nodes) {
        initial[node.id] = { status: "idle", stepStates: {} };
      }
      setNodeStates(initial);
    }
    setFlowStatus("idle");
  }, [flow, camera]);

  return {
    flowStatus,
    nodeStates,
    finishing,
    startFlow,
    pauseFlow,
    resumeFlow,
    abortFlow,
    finishFlow,
    resetFlow,
  };
}
