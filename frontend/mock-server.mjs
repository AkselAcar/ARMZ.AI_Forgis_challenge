/**
 * mock-server.mjs
 *
 * Standalone mock backend for frontend development without a real robot/backend.
 * Listens on :8000 — Vite's proxy already forwards /api and /ws there.
 *
 * - Flow generation calls Gemini Vision when GEMINI_API_KEY is set (reads ../.env)
 * - Falls back to a hardcoded mock flow otherwise
 *
 * Usage:
 *   npm run dev:mock
 */

import http from "http";
import path from "path";
import { fileURLToPath } from "url";
import { WebSocketServer } from "ws";
import { createRequire } from "module";
import { readFileSync } from "fs";

// ── Load ../.env ─────────────────────────────────────────────────────────────

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const envPath = path.resolve(__dirname, "../.env");

try {
  const { config } = await import("dotenv");
  config({ path: envPath });
  console.log("[mock] Loaded .env from", envPath);
} catch {
  // dotenv not available / .env missing — continue without it
}

const GEMINI_API_KEY = process.env.GEMINI_API_KEY ?? "";

// ── Gemini setup ─────────────────────────────────────────────────────────────

let genAI = null;
if (GEMINI_API_KEY) {
  try {
    const { GoogleGenerativeAI } = await import("@google/generative-ai");
    genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
    console.log("[mock] Gemini Vision ready (gemini-2.5-flash)");
  } catch (e) {
    console.warn("[mock] @google/generative-ai not available:", e.message);
  }
} else {
  console.warn("[mock] GEMINI_API_KEY not set — will use hardcoded mock flow");
}

const GEMINI_SYSTEM_PROMPT = `You are an expert robotics automation engineer assistant.
You help users design and understand robot execution flows.

Return ONLY a raw valid JSON object (no markdown, no explanation, no code fences).

RESPONSE FORMAT
===============
{
  "message": "A helpful natural language response to the user.",
  "flow": <flow object or null>
}

Rules:
- Set "flow" to null for conversational messages (greetings, questions, clarifications, when the request cannot be fulfilled).
- Set "flow" to a full flow object when the user asks to generate, create, modify, or build a flow.
- If you cannot fulfil the request with the available skills, set flow to null and explain why in the message, then give an example prompt that would work.
- All position values must be {"x": 0, "y": 0} — the frontend computes layout.
- A flow always starts with a "start" node and ends with an "end" node.
- Each state node must have at least one step.
- Edge ids must be unique.

FLOW OBJECT SCHEMA
==================
{
  "id": "snake_case_id",
  "name": "Human Readable Title",
  "loop": false,
  "nodes": [
    {"id": "start", "type": "start", "label": "Start", "position": {"x": 0, "y": 0}},
    {
      "id": "state_id",
      "type": "state",
      "label": "State Label",
      "steps": [
        {"id": "step_id", "skill": "skill_name", "executor": "executor_name", "params": {}}
      ],
      "position": {"x": 0, "y": 0}
    },
    {"id": "end", "type": "end", "label": "End", "position": {"x": 0, "y": 0}}
  ],
  "edges": [
    {"id": "e-start-state1", "source": "start", "target": "state1"},
    {"id": "e-state1-end", "source": "state1", "target": "end"}
  ]
}

COMPLETE SKILL CATALOG
======================
Below is the exhaustive list of every skill the robot can execute.
You MUST only use skills from this list. Do NOT invent skills.

1. move_joint (executor: "robot")
   Move robot joints to target positions.
   Params:
   - target_joints_deg: list[float] — 6 joint angles in degrees (REQUIRED)
   - acceleration: float (0.1–2.0, default 2.0) — rad/s²
   - velocity: float (0.1–2.0, default 2.0) — rad/s
   - tolerance_deg: float (0.1–10.0, default 1.0)

2. move_linear (executor: "robot")
   Move robot TCP in a straight line to a Cartesian pose.
   Params:
   - target_pose: list[float] — [x, y, z, rx, ry, rz] in metres and radians (REQUIRED)
   - z_offset: float (default 0.0) — vertical offset in metres (useful for approach/retract)
   - acceleration: float (0.01–3.0, default 1.2) — m/s²
   - velocity: float (0.01–1.0, default 0.8) — m/s

3. palletize (executor: "robot")
   Pop next position from a palletizing list variable and move there.
   Params:
   - positions_var: str — name of a flow variable holding a positions map (REQUIRED)
   - key: str — zone key to look up (REQUIRED, can be a {{variable}} reference)
   - default_key: str (default "default") — fallback key
   - z_offset: float (default 0.0)
   - acceleration: float (default 1.2)
   - velocity: float (default 0.8)

4. set_tool_output (executor: "robot")
   Control DOBOT tool digital output (pneumatic gripper).
   Params:
   - index: int ≥ 1 — output index (1 = close gripper, 2 = open gripper) (REQUIRED)
   - status: int (0 or 1, default 1)

5. get_bounding_box (executor: "camera")
   Run YOLOv8 object detection, return bounding box for a class.
   Params:
   - object_class: str (REQUIRED)
   - confidence_threshold: float (0.0–1.0, default 0.5)

6. get_label (executor: "camera")
   Read labels / OCR via GPT-4V Vision.
   Params:
   - prompt: str — instruction for the vision model (REQUIRED)
   - use_bbox: bool (default true) — crop to last detection
   - crop_margin: float (0.0–0.5, default 0.1)
   Use store_result to save the output (e.g. "label_data"), then reference {{label_data.label}} in later steps.

7. start_streaming (executor: "camera")
   Start camera frame streaming over WebSocket.
   Params:
   - fps: int (1–30, default 15)

8. stop_streaming (executor: "camera")
   Stop camera streaming. No params.

9. set_grip (executor: "hand")
   Activate a predefined COVVI hand grip.
   Params:
   - grip: one of "POWER", "TRIPOD", "TRIPOD_OPEN", "PREC_OPEN", "PREC_CLOSED",
           "TRIGGER", "KEY", "FINGER", "CYLINDER", "COLUMN", "RELAXED", "GLOVE",
           "TAP", "GRAB" (REQUIRED)

10. set_finger_positions (executor: "hand")
    Set individual finger positions on the COVVI hand.
    Params:
    - speed: int (15–100, default 50)
    - thumb, index, middle, ring, little, rotate: Optional[int] (0=open, 100=closed)

11. grip_until_contact (executor: "hand")
    Close fingers slowly until stall-based contact detected.
    Params:
    - speed: int (15–100, default 25)
    - fingers: list of "thumb"|"index"|"middle"|"little" (REQUIRED)
    - min_contacts: int (1–4, default 2)
    - timeout_s: float (0.1–30.0, default 5.0)

12. io_set_digital_output (executor: "io_robot" or "io_machine")
    Set a digital output pin HIGH or LOW.
    Params:
    - pin: int (0–7) (REQUIRED)
    - value: bool (REQUIRED)

13. wait_digital_input (executor: "io_robot" or "io_machine")
    Block until a digital input matches expected value.
    Params:
    - pin: int (0–7) (REQUIRED)
    - expected_value: bool (REQUIRED)
    - poll_interval_ms: int (10–1000, default 100)

VARIABLE INTERPOLATION
======================
Step params support these patterns:
- {{var_name}} — substitute a flow variable (supports dot-paths: {{label_data.label}})
- {{lookup:map_name:key_path:default_key}} — look up a value in a map variable
- {{pop:map_name:key_path:default_key}} — pop first element from a list in a map
You can use store_result on a step to save its output into a variable for later use.

EXAMPLE: PICK AND PLACE WITH VISION + CONDITIONAL ROUTING
=========================================================
{
  "message": "I created a pick-and-place flow that reads a product label with the camera, picks the object, then places it in the correct zone based on the label.",
  "flow": {
    "id": "pick_and_place_v2",
    "name": "Pick and Place with Conditional Routing",
    "loop": true,
    "nodes": [
      {"id": "start", "type": "start", "label": "Start", "position": {"x": 0, "y": 0}},
      {
        "id": "go_home",
        "type": "state",
        "label": "Go Home",
        "steps": [
          {"id": "move_home", "skill": "move_joint", "executor": "robot", "params": {"target_joints_deg": [-122.69, -109.26, -44.24, -112.69, 89.94, 118.72]}}
        ],
        "position": {"x": 0, "y": 0}
      },
      {
        "id": "read_label",
        "type": "state",
        "label": "Read Label",
        "steps": [
          {"id": "get_product_label", "skill": "get_label", "executor": "camera", "params": {"prompt": "Read the product label. Return only the product code.", "use_bbox": false}, "store_result": "label_data"}
        ],
        "position": {"x": 0, "y": 0}
      },
      {
        "id": "pick_object",
        "type": "state",
        "label": "Pick Object",
        "steps": [
          {"id": "move_to_pick", "skill": "move_joint", "executor": "robot", "params": {"target_joints_deg": [-141.27, -129.96, -80.55, -59.00, 88.25, 118.72]}},
          {"id": "vacuum_on", "skill": "io_set_digital_output", "executor": "io_robot", "params": {"pin": 0, "value": true}}
        ],
        "position": {"x": 0, "y": 0}
      },
      {
        "id": "place_object",
        "type": "state",
        "label": "Place Object",
        "steps": [
          {"id": "move_to_place", "skill": "move_joint", "executor": "robot", "params": {"target_joints_deg": [-58.50, -103.66, -75.31, -90.16, 89.94, 118.72]}},
          {"id": "vacuum_off", "skill": "io_set_digital_output", "executor": "io_robot", "params": {"pin": 0, "value": false}}
        ],
        "position": {"x": 0, "y": 0}
      },
      {"id": "end", "type": "end", "label": "End", "position": {"x": 0, "y": 0}}
    ],
    "edges": [
      {"id": "e-start-go_home", "source": "start", "target": "go_home"},
      {"id": "e-go_home-read_label", "source": "go_home", "target": "read_label"},
      {"id": "e-read_label-pick_object", "source": "read_label", "target": "pick_object"},
      {"id": "e-pick_object-place_object", "source": "pick_object", "target": "place_object"},
      {"id": "e-place_object-end", "source": "place_object", "target": "end"}
    ]
  }
}`;


async function generateWithGemini(prompt, fileBase64, mimeType) {
  if (!genAI) return null;
  try {
    const model = genAI.getGenerativeModel({
      model: "gemini-2.5-flash",
      systemInstruction: GEMINI_SYSTEM_PROMPT,
    });

    const parts = [{ text: prompt }];
    if (fileBase64 && mimeType) {
      parts.push({ inlineData: { mimeType, data: fileBase64 } });
    }

    const result = await model.generateContent({
      contents: [{ role: "user", parts }],
      generationConfig: { responseMimeType: "application/json" },
    });

    const raw = result.response.text().trim();
    console.log("[mock] Gemini response preview:", raw.slice(0, 200));
    return JSON.parse(raw);
  } catch (err) {
    console.error("[mock] Gemini call failed:", err.message);
    return null;
  }
}

const PORT = 8000;

// ── Mock flow (frontend Flow schema) ────────────────────────────────────────

const MOCK_FLOW = {
  id: "mock_pick_and_place",
  name: "Pick and Place",
  loop: false,
  nodes: [
    {
      id: "start",
      type: "start",
      label: "Start",
      position: { x: 0, y: 0 },
    },
    {
      id: "go_home",
      type: "state",
      label: "Go Home",
      steps: [
        {
          id: "move_home",
          skill: "move_joint",
          executor: "robot",
          params: { target_joints_deg: [0, -90, 0, -90, 0, 0] },
        },
      ],
      position: { x: 0, y: 100 },
    },
    {
      id: "pick",
      type: "state",
      label: "Pick",
      steps: [
        {
          id: "open_gripper",
          skill: "grip_until_contact",
          executor: "hand",
          params: {},
        },
        {
          id: "move_pick",
          skill: "move_joint",
          executor: "robot",
          params: { target_joints_deg: [-90, -110, -80, -60, 90, 0] },
        },
      ],
      position: { x: 0, y: 270 },
    },
    {
      id: "place",
      type: "state",
      label: "Place",
      steps: [
        {
          id: "move_place",
          skill: "move_joint",
          executor: "robot",
          params: { target_joints_deg: [-45, -100, -70, -85, 90, 0] },
        },
        {
          id: "release",
          skill: "grip_until_contact",
          executor: "hand",
          params: {},
        },
      ],
      position: { x: 0, y: 440 },
    },
    {
      id: "end",
      type: "end",
      label: "End",
      position: { x: 0, y: 610 },
    },
  ],
  edges: [
    { id: "e-start-home", source: "start", target: "go_home" },
    { id: "e-home-pick", source: "go_home", target: "pick" },
    { id: "e-pick-place", source: "pick", target: "place" },
    { id: "e-place-end", source: "place", target: "end" },
  ],
};

// ── WebSocket clients registry ───────────────────────────────────────────────

const clients = new Set();

function broadcast(msg) {
  const data = JSON.stringify(msg);
  for (const ws of clients) {
    if (ws.readyState === 1 /* OPEN */) ws.send(data);
  }
}

// ── Flow simulation ──────────────────────────────────────────────────────────

const delay = (ms) => new Promise((r) => setTimeout(r, ms));

let simulationRunning = false;
let aborted = false;

async function simulateFlow(flow) {
  simulationRunning = true;
  aborted = false;

  broadcast({ type: "flow_started", flow_id: flow.id, name: flow.name, timestamp: Date.now() });

  const stateNodes = flow.nodes.filter((n) => n.type === "state");

  outer: for (const node of stateNodes) {
    if (aborted) break;

    await delay(700);
    broadcast({ type: "state_entered", state: node.id, timestamp: Date.now() });

    for (const step of node.steps ?? []) {
      if (aborted) break outer;

      await delay(500);
      broadcast({ type: "step_started", step_id: step.id, state: node.id, timestamp: Date.now() });

      await delay(1000);
      if (aborted) break outer;

      broadcast({
        type: "step_completed",
        step_id: step.id,
        state: node.id,
        retries: 0,
        result: {},
        timestamp: Date.now(),
      });
    }

    await delay(400);
    broadcast({ type: "state_completed", state: node.id, timestamp: Date.now() });
  }

  if (!aborted) {
    await delay(400);
    broadcast({ type: "flow_completed", flow_id: flow.id, timestamp: Date.now() });
  }

  simulationRunning = false;
}

// ── HTTP server ──────────────────────────────────────────────────────────────

let currentFlow = MOCK_FLOW;

const server = http.createServer((req, res) => {
  // CORS — needed when Vite proxy is not used (e.g. direct fetch during tests)
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") {
    res.writeHead(204);
    res.end();
    return;
  }

  const url = req.url ?? "";

  // Collect body
  let body = "";
  req.on("data", (chunk) => (body += chunk));
  req.on("end", async () => {
    const json = (data, status = 200) => {
      res.writeHead(status, { "Content-Type": "application/json" });
      res.end(JSON.stringify(data));
    };
    const ok = () => {
      res.writeHead(200);
      res.end();
    };

    // ── Routes ──────────────────────────────────────────────

    // Flow generation — call Gemini Vision if available, else return mock flow
    if (req.method === "POST" && url === "/api/flows/generate") {
      let parsed = {};
      try { parsed = JSON.parse(body); } catch { /* ignore */ }
      const hasFile = !!(parsed.file_base64);
      const mimeType = parsed.file_mime_type ?? "";
      const prompt = parsed.prompt || "Generate a robot flow.";
      console.log(`[mock] POST /api/flows/generate  prompt="${prompt}"  mime=${mimeType || "none"}`);

      if (genAI) {
        const result = await generateWithGemini(prompt, parsed.file_base64 ?? null, mimeType);
        if (result) {
          // result is { message, flow } — flow may be null for conversational replies
          if (result.flow) currentFlow = result.flow;
          json({ message: result.message, flow: result.flow ?? null });
          return;
        }
        console.warn("[mock] Gemini returned null — falling back to mock flow");
      }

      // Fallback
      setTimeout(() => {
        const label = mimeType === "application/pdf" ? "PDF-Generated Flow (mock)"
          : hasFile ? "Vision-Generated Flow (mock)" : MOCK_FLOW.name;
        const flow = { ...MOCK_FLOW, name: label };
        currentFlow = flow;
        json({ message: "Here is the default mock flow.", flow });
      }, hasFile ? 1400 : 900);
      return;
    }

    // Start flow
    if (req.method === "POST" && /^\/api\/flows\/[^/]+\/start$/.test(url)) {
      console.log("[mock] POST", url);
      if (!simulationRunning) simulateFlow(currentFlow);
      ok();
      return;
    }

    // Pause
    if (req.method === "POST" && url === "/api/flows/pause") {
      console.log("[mock] POST /api/flows/pause");
      broadcast({ type: "flow_paused", timestamp: Date.now() });
      ok();
      return;
    }

    // Resume
    if (req.method === "POST" && url === "/api/flows/resume") {
      console.log("[mock] POST /api/flows/resume");
      broadcast({ type: "flow_resumed", timestamp: Date.now() });
      ok();
      return;
    }

    // Abort
    if (req.method === "POST" && url === "/api/flows/abort") {
      console.log("[mock] POST /api/flows/abort");
      aborted = true;
      simulationRunning = false;
      broadcast({ type: "flow_aborted", flow_id: currentFlow.id, timestamp: Date.now() });
      ok();
      return;
    }

    // Finish (graceful stop)
    if (req.method === "POST" && url === "/api/flows/finish") {
      console.log("[mock] POST /api/flows/finish");
      aborted = true;
      simulationRunning = false;
      broadcast({ type: "flow_completed", flow_id: currentFlow.id, timestamp: Date.now() });
      ok();
      return;
    }

    // Camera (no-op)
    if (req.method === "POST" && url === "/api/camera/stream/start") {
      ok();
      return;
    }
    if (req.method === "POST" && url === "/api/camera/stream/stop") {
      ok();
      return;
    }

    // 404
    console.warn("[mock] 404:", req.method, url);
    json({ detail: `Not found: ${url}` }, 404);
  });
});

// ── WebSocket server ─────────────────────────────────────────────────────────

const wss = new WebSocketServer({ server, path: "/ws" });

wss.on("connection", (ws) => {
  clients.add(ws);
  console.log(`[mock] WS client connected (total: ${clients.size})`);
  ws.send(
    JSON.stringify({ type: "connected", message: "Mock server connected", timestamp: Date.now() })
  );
  ws.on("close", () => {
    clients.delete(ws);
    console.log(`[mock] WS client disconnected (total: ${clients.size})`);
  });
});

// ── Start ────────────────────────────────────────────────────────────────────

server.listen(PORT, () => {
  console.log(`\n🟢  Mock backend running on http://localhost:${PORT}`);
  console.log(`    WebSocket : ws://localhost:${PORT}/ws`);
  console.log(`    Vite proxy: http://localhost:3000  →  :${PORT}\n`);
  console.log("    Available mock endpoints:");
  console.log("      POST /api/flows/generate");
  console.log("      POST /api/flows/:id/start");
  console.log("      POST /api/flows/pause | resume | abort | finish");
  console.log("      POST /api/camera/stream/start | stop\n");
});
