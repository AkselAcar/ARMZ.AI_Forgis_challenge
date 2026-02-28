const API_BASE = "/api";

export function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

export async function postJson<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(apiUrl(path), {
    method: "POST",
    headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `Request failed: ${response.statusText}`);
  }

  const text = await response.text();
  return text ? (JSON.parse(text) as T) : (undefined as T);
}

export async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(apiUrl(path), { 
    method: "GET" });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `Request failed: ${response.statusText}`);
  }

  return response.json() as Promise<T>;
}
