import { useEffect } from "react";
import { useReactFlow } from "@xyflow/react";

export function FlowAutoFit({ flowId }: { flowId: string | undefined }) {
  const { fitView } = useReactFlow();
  useEffect(() => {
    fitView({ padding: 0.3, maxZoom: 0.85, duration: 0 });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [flowId]);
  return null;
}
