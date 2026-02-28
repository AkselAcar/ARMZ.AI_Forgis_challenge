import { postJson } from "./httpClient";

export async function startCameraStream(fps = 15): Promise<void> {
  await postJson("/camera/stream/start", { fps });
}

export async function stopCameraStream(): Promise<void> {
  await postJson("/camera/stream/stop");
}
