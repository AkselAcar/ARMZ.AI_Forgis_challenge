import type { ServerMessage } from "@/types";

const WS_URL = `ws://${window.location.host}/ws`;

export function createFlowSocket(
  onMessage: (msg: ServerMessage) => void,
  onClose?: () => void,
): Promise<{ close: () => void }> {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      // Connection established — from here on, errors surface via onclose
      ws.onerror = null;
      resolve({ close: () => ws.close() });
    };

    ws.onerror = () => reject(new Error("WebSocket connection failed"));

    ws.onmessage = (event) => {
      try {
        onMessage(JSON.parse(event.data) as ServerMessage);
      } catch {
        // Ignore non-JSON messages (e.g. binary frames)
      }
    };

    ws.onclose = () => onClose?.();
  });
}
