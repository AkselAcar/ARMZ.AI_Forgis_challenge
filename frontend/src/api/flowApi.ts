import { postJson } from "./httpClient";
import type { GenerateResult } from "@/types";

export async function generateFlow(
  prompt: string,
  fileBase64?: string,
  fileMimeType?: string,
): Promise<GenerateResult> {
  return postJson<GenerateResult>("/flows/generate", {
    prompt,
    file_base64: fileBase64 ?? null,
    file_mime_type: fileMimeType ?? null,
  });
}

export async function startFlow(flowId: string): Promise<void> {
  await postJson(`/flows/${flowId}/start`);
}

export async function pauseFlow(): Promise<void> {
  await postJson("/flows/pause");
}

export async function resumeFlow(): Promise<void> {
  await postJson("/flows/resume");
}

export async function abortFlow(): Promise<void> {
  await postJson("/flows/abort");
}

export async function finishFlow(): Promise<void> {
  await postJson("/flows/finish");
}
