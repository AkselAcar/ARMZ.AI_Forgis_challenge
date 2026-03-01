"""REST endpoints for flow management."""

import base64
import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from flow import FlowSchema, FlowStatusResponse

# ── Gemini SDK (optional) ────────────────────────────────────────────────────
_GEMINI_AVAILABLE = False
try:
    import google.generativeai as genai
    _GEMINI_AVAILABLE = True
except ImportError:
    genai = None  # type: ignore
    logger.info("google-generativeai not installed — Gemini features disabled")

router = APIRouter(prefix="/api/flows", tags=["flows"])

# FlowManager will be injected via app state
_flow_manager = None


def set_flow_manager(manager) -> None:
    """Set the flow manager instance (called during app initialization)."""
    global _flow_manager
    _flow_manager = manager


def get_manager():
    """Get the flow manager, raising if not initialized."""
    if _flow_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Flow manager not initialized",
        )
    return _flow_manager


# --- Request/Response Models ---


class FlowListResponse(BaseModel):
    """Response for listing flows."""

    flows: list[str]


class FlowCreateResponse(BaseModel):
    """Response for creating/updating a flow."""

    success: bool
    message: str
    flow_id: Optional[str] = None


class FlowStartResponse(BaseModel):
    """Response for starting a flow."""

    success: bool
    message: str


class FlowAbortResponse(BaseModel):
    """Response for aborting a flow."""

    success: bool
    message: str


class FlowGenerateRequest(BaseModel):
    """Request for generating a flow from prompt."""

    prompt: str
    file_base64: Optional[str] = None      # base64-encoded file (image or PDF)
    file_mime_type: Optional[str] = None   # e.g. "image/jpeg", "application/pdf"


class FlowStep(BaseModel):
    """Step within a state (aligned with backend naming)."""

    id: str
    skill: str
    executor: str
    params: Optional[dict[str, Any]] = None


class FlowNode(BaseModel):
    """Node in the frontend flow format (aligned with backend naming)."""

    id: str
    type: str  # "state", "start", "end"
    label: str
    steps: Optional[list[FlowStep]] = None  # For state nodes
    position: dict[str, float]
    style: Optional[dict[str, Any]] = None  # For sizing


class FlowEdge(BaseModel):
    """Edge in the frontend flow format."""

    id: str
    source: str
    target: str
    type: str = "transitionEdge"
    data: Optional[dict[str, Any]] = None


class FlowGenerateResponse(BaseModel):
    """Response with frontend-compatible flow format."""

    id: str
    name: str
    loop: bool = False
    nodes: list[FlowNode]
    edges: list[FlowEdge]


class GenerateResult(BaseModel):
    """Wrapper returned by /flows/generate."""

    message: str
    flow: Optional[FlowGenerateResponse] = None  # None for conversational replies


def convert_backend_to_frontend(flow: FlowSchema) -> FlowGenerateResponse:
    """
    Convert backend flow format to frontend node/edge format.

    Backend: states with steps, transitions between states
    Frontend: start node, state nodes (containing steps), end node, edges

    Uses actual transitions from the flow definition.
    """
    nodes: list[FlowNode] = []
    edges: list[FlowEdge] = []

    # Positions are set to (0,0) — the frontend's layoutFlow() computes real positions.

    # Add start node
    start_node_id = "start"
    nodes.append(FlowNode(
        id=start_node_id,
        type="start",
        label="Start",
        position={"x": 0, "y": 0},
    ))

    # Convert each state to a node with steps inside
    for state in flow.states:
        node_id = state.name

        steps = [
            FlowStep(
                id=step.id,
                skill=step.skill,
                executor=step.executor,
                params=step.params,
            )
            for step in state.steps
        ]

        nodes.append(FlowNode(
            id=node_id,
            type="state",
            label=state.name,
            steps=steps,
            position={"x": 0, "y": 0},
        ))

    # Add end node
    end_node_id = "end"
    nodes.append(FlowNode(
        id=end_node_id,
        type="end",
        label="End",
        position={"x": 0, "y": 0},
    ))

    # Edge from start to initial state
    edges.append(FlowEdge(
        id=f"e_{start_node_id}_{flow.initial_state}",
        source=start_node_id,
        target=flow.initial_state,
    ))

    # Convert actual transitions to edges
    for i, t in enumerate(flow.transitions):
        edge_data: dict[str, Any] = {"transitionType": t.type}
        if t.condition:
            edge_data["condition"] = t.condition

        edges.append(FlowEdge(
            id=f"e_{t.from_state}_{t.to_state}_{i}",
            source=t.from_state,
            target=t.to_state,
            data=edge_data,
        ))

    # Find terminal states (no outgoing transitions)
    states_with_outgoing = {t.from_state for t in flow.transitions}
    terminal_states = [s.name for s in flow.states if s.name not in states_with_outgoing]

    # Add loop-back and/or end edges for terminal states
    for state_name in terminal_states:
        if flow.loop:
            edges.append(FlowEdge(
                id=f"e_loop_{state_name}_{flow.initial_state}",
                source=state_name,
                target=flow.initial_state,
                data={"isLoop": True},
            ))
        edges.append(FlowEdge(
            id=f"e_{state_name}_{end_node_id}",
            source=state_name,
            target=end_node_id,
        ))

    return FlowGenerateResponse(
        id=flow.id,
        name=flow.name,
        loop=flow.loop,
        nodes=nodes,
        edges=edges,
    )


# --- Endpoints ---


@router.get("", response_model=FlowListResponse)
async def list_flows():
    """List all available flows."""
    manager = get_manager()
    return FlowListResponse(flows=manager.list_flows())


@router.get("/status", response_model=FlowStatusResponse)
async def get_status():
    """Get current execution status."""
    manager = get_manager()
    return manager.get_status()


@router.get("/{flow_id}", response_model=FlowSchema)
async def get_flow(flow_id: str):
    """Get a flow definition by ID."""
    manager = get_manager()
    flow = manager.get_flow(flow_id)
    if flow is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Flow '{flow_id}' not found",
        )
    return flow


@router.post("", response_model=FlowCreateResponse)
async def create_flow(flow: FlowSchema):
    """Create or update a flow."""
    manager = get_manager()
    success, error = manager.save_flow(flow)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error or "Failed to save flow",
        )
    return FlowCreateResponse(
        success=True,
        message=f"Flow '{flow.id}' saved",
        flow_id=flow.id,
    )


@router.delete("/{flow_id}")
async def delete_flow(flow_id: str):
    """Delete a flow."""
    manager = get_manager()
    if not manager.delete_flow(flow_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Flow '{flow_id}' not found",
        )
    return {"success": True, "message": f"Flow '{flow_id}' deleted"}


@router.post("/{flow_id}/start", response_model=FlowStartResponse)
async def start_flow(flow_id: str):
    """Start executing a flow."""
    manager = get_manager()
    success, message = await manager.start_flow(flow_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT
            if "already running" in message.lower()
            else status.HTTP_400_BAD_REQUEST,
            detail=message,
        )
    return FlowStartResponse(success=True, message=message)


@router.post("/abort", response_model=FlowAbortResponse)
async def abort_flow():
    """Abort the currently running flow."""
    manager = get_manager()
    success, message = await manager.abort_flow()
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message,
        )
    return FlowAbortResponse(success=True, message=message)


class FlowFinishResponse(BaseModel):
    """Response for finishing a flow."""

    success: bool
    message: str


class FlowPauseResponse(BaseModel):
    """Response for pausing a flow."""

    success: bool
    message: str


class FlowResumeResponse(BaseModel):
    """Response for resuming a flow."""

    success: bool
    message: str


@router.post("/finish", response_model=FlowFinishResponse)
async def finish_flow():
    """Request graceful finish — complete current loop cycle then stop."""
    manager = get_manager()
    success, message = await manager.finish_flow()
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message,
        )
    return FlowFinishResponse(success=True, message=message)


@router.post("/pause", response_model=FlowPauseResponse)
async def pause_flow():
    """Pause the currently running flow."""
    manager = get_manager()
    success, message = await manager.pause_flow()
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message,
        )
    return FlowPauseResponse(success=True, message=message)


@router.post("/resume", response_model=FlowResumeResponse)
async def resume_flow():
    """Resume a paused flow."""
    manager = get_manager()
    success, message = await manager.resume_flow()
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message,
        )
    return FlowResumeResponse(success=True, message=message)


# Default flow loaded when no AI generation is implemented.
_DEFAULT_FLOW_ID = "test_vision_speed"


# ── Gemini system prompt ─────────────────────────────────────────────────────
_GEMINI_SYSTEM_PROMPT = """\
You are an expert robot-programming assistant for the ARMZ.AI platform.
Your job is to help users build automation flows for a UR3 robotic arm
equipped with a vacuum gripper, IO modules, and a camera.

## RESPONSE ENVELOPE
Always reply with a JSON object:

{ "message": "<human-readable text>", "flow": <FlowObject or null> }

* If the user is just chatting (greeting, question, clarification, thanks, etc.)
  set `flow` to `null` and put your answer in `message`.
* If the user asks you to build, modify, or demonstrate a robotic flow, create
  the flow object AND put a short description in `message`.
* If the user asks for something impossible with this robot, politely decline in
  `message` and set `flow` to `null`.

## FLOW SCHEMA
```
{
  "id": "<kebab-id>",
  "name": "<Human Name>",
  "loop": false,
  "states": [ ... ],
  "transitions": [ ... ],
  "initial_state": "<first-state-name>"
}
```

### State
```
{ "name": "<state-name>", "steps": [ <Step>, ... ] }
```

### Step
```
{ "id": "<unique-step-id>", "skill": "<skill_name>", "executor": "<executor>", "params": { ... } }
```

### Transition
```
{
  "from_state": "<state-name>",
  "to_state": "<state-name>",
  "type": "sequential" | "conditional" | "immediate",
  "condition": null | "<python-expression referencing variables.*>"
}
```

## AVAILABLE SKILLS

### Executor: "robot"
1. **move_joint** – Joint-space move (degrees)
   params: { "target_joints_deg": [j0, j1, j2, j3, j4, j5],  // REQUIRED, 6 floats in DEGREES
              "velocity"?: 0.1-2.0 (default 2.0, rad/s),
              "acceleration"?: 0.1-2.0 (default 2.0, rad/s²),
              "tolerance_deg"?: 0.1-10.0 (default 1.0) }

2. **move_linear** – Cartesian straight-line move
   params: { "target_pose": [x, y, z, rx, ry, rz],  // REQUIRED, 6 floats — metres & radians
              "velocity"?: 0.01-1.0 (default 0.8, m/s),
              "acceleration"?: 0.01-3.0 (default 1.2, m/s²) }

3. **palletize** – Pop next position from a list variable
   params: { "positions_var": string,   // REQUIRED — name of flow variable holding positions map
              "key": string,             // REQUIRED — zone key e.g. "Zone_A"
              "default_key"?: string (default "default"),
              "velocity"?: 0.01-1.0 (default 0.8),
              "acceleration"?: 0.01-3.0 (default 1.2) }

4. **set_tool_output** – Control tool digital output (vacuum gripper)
   params: { "index": int (≥1),   // REQUIRED — 1=close gripper, 2=open on dual-solenoid
              "status"?: 0|1 (default 1) }

5. **pick_and_place** – Complete pick-and-place cycle with Gemini vision zone routing (compound skill)
   params: { "pick_pose": [x,y,z,rx,ry,rz],       // REQUIRED, metres & radians
              "positions_var"?: string (default "place_positions"),
                // Name of the flow variable holding a zone→poses dict, e.g.:
                // { "Zone_A": [[x,y,z,rx,ry,rz], ...], "Zone_B": [...], "default": [...] }
              "default_zone"?: string (default "default"),
                // Zone key used when Gemini label is unrecognised
              "waypoint_c_joints"?: [j0,j1,j2,j3,j4,j5] or null (default null),
                // Optional joint-space detour taken only for waypoint_c_zone
              "waypoint_c_zone"?: string (default "Zone_C"),
              "label_prompt"?: string (default: MXP-30/Speed/Torque zone prompt),
              "vacuum_pin"?: 0-7 (default 0),
              "vacuum_settle_s"?: 0-2 (default 0.3),
              "pick_velocity"?: 0.01-2.0 (default 1.0, m/s),
              "place_velocity"?: 0.01-2.0 (default 1.0, m/s),
              "acceleration"?: 0.01-5.0 (default 2.0, m/s²) }
   NOTE: place_pose is NO LONGER a parameter. Placement position is chosen
   automatically by reading the product label with Gemini and looking it up
   in the positions_var dict. Zone positions wrap around (round-robin).
   The flow MUST have a flow-level variable named "place_positions" (or whatever
   positions_var is set to) with the zone dict.
   This skill uses "robot", "io_robot" AND "camera" executors internally.
   Set executor to "robot" in the step.

### Executor: "camera"
5. **get_bounding_box** – Local YOLOv8 detection (fast, ~50-200 ms)
   params: { "object_class": string,  // REQUIRED — class name to detect e.g. "bottle", "box"
              "confidence_threshold"?: 0.0-1.0 (default 0.5) }
   sets: variables.bbox = {x_center, y_center, width, height, confidence, label}

6. **get_label** – Vision-LLM query via Gemini (2-5 s)
   params: { "prompt"?: string (default "Read any text visible in the image…"),
              "use_bbox"?: bool (default true — crop to last bbox),
              "crop_margin"?: 0.0-0.5 (default 0.1) }
   sets: variables.label = "<text>"

7. **start_streaming** – Start camera MJPEG stream
   params: { "fps"?: 1-30 (default 15) }

8. **stop_streaming** – Stop camera MJPEG stream
   params: {}   // no parameters

### Executor: "hand"
9. **set_grip** – Activate a predefined grip posture
   params: { "grip": string }  // REQUIRED — one of: "POWER", "TRIPOD", "TRIPOD_OPEN",
            // "PREC_OPEN", "PREC_CLOSED", "TRIGGER", "KEY", "FINGER",
            // "CYLINDER", "COLUMN", "RELAXED", "GLOVE", "TAP", "GRAB"

10. **set_finger_positions** – Per-finger control (0=open, 100=closed)
    params: { "speed"?: 15-100 (default 50),
              "thumb"?: 0-100, "index"?: 0-100, "middle"?: 0-100,
              "ring"?: 0-100, "little"?: 0-100, "rotate"?: 0-100 }

11. **grip_until_contact** – Close fingers until stall detected
    params: { "speed"?: 15-100 (default 25),
              "fingers"?: ["thumb","index","middle","little"] (default all four),
              "min_contacts"?: 1-4 (default 2),
              "timeout_s"?: 0.1-30.0 (default 5.0) }

### Executor: "io_robot"
12. **io_set_digital_output** – Set a digital output pin
    params: { "pin": 0-7,   // REQUIRED
              "value": bool  // REQUIRED — true=HIGH, false=LOW }

13. **wait_digital_input** – Wait for a digital input pin value
    params: { "pin": 0-7,            // REQUIRED
              "expected_value": bool,  // REQUIRED — true=HIGH, false=LOW
              "poll_interval_ms"?: 10-1000 (default 100) }

## IMPORTANT NOTES
- For the UR3 + vacuum gripper setup: the vacuum is typically controlled via
  `io_set_digital_output` with executor "io_robot" (pin 0, value true=on / false=off).
- Joint values for `move_joint` are in DEGREES (not radians).
- Pose values for `move_linear` are in METRES and RADIANS (not mm or degrees).
- If the user doesn't provide exact coordinates, tell them you need the values
  rather than making them up. You may use placeholder values like [0,0,0,0,0,0]
  with a note to the user.

## VARIABLE INTERPOLATION
Step params may reference runtime variables with `{{variable_name}}`:
  - `{{hover_joints}}` — references flow-level variable
  - `{{vision_data.label}}` — nested access
Transition conditions can also reference variables, e.g. `{{vision_data.label}} == 'BOX_FOUND'`.
Flow-level variables are defined in the top-level `"variables"` object.

## EXAMPLE FLOW (detect box → pick → place with vacuum)
```json
{
  "id": "simple-pick-and-place",
  "name": "Simple Pick & Place",
  "loop": false,
  "initial_state": "detect",
  "states": [
    {
      "name": "detect",
      "steps": [
        { "id": "s1", "skill": "get_bounding_box", "executor": "camera", "params": { "object_class": "box" } }
      ]
    },
    {
      "name": "pick",
      "steps": [
        { "id": "s2", "skill": "move_joint", "executor": "robot", "params": { "target_joints_deg": [0, -90, -90, -90, 90, 0], "velocity": 1.5 } },
        { "id": "s3", "skill": "io_set_digital_output", "executor": "io_robot", "params": { "pin": 0, "value": true } }
      ]
    },
    {
      "name": "place",
      "steps": [
        { "id": "s4", "skill": "move_linear", "executor": "robot", "params": { "target_pose": [-0.3, -0.2, 0.15, 0.0, 3.14, 0.0], "velocity": 0.5 } },
        { "id": "s5", "skill": "io_set_digital_output", "executor": "io_robot", "params": { "pin": 0, "value": false } }
      ]
    }
  ],
  "transitions": [
    { "from_state": "detect", "to_state": "pick", "type": "sequential", "condition": null },
    { "from_state": "pick", "to_state": "place", "type": "sequential", "condition": null }
  ]
}
```

## ROBOT CONFIGURATION (use these values — do NOT ask the user for them)

These are the pre-calibrated values for this specific robot cell.
Always use them when building pick-and-place flows unless the user explicitly
provides different ones.

### Standard pick pose (conveyor pick position) 
"pick_pose": [-0.449, 0.05, 0.103, 0.0, 3.14, 0.0]

### Standard midair/home joint position (safe transit pose)
"midair_joints": [-172.0, -136.01, -77.93, -58.64, 89.0, -55.0]

### Standard waypoint for Zone_C (joint detour to avoid collision)
"waypoint_c_joints": [-26.66, -76.09, -89.12, -99.54, 88.68, 154.01]

### Standard place_positions (zone → list of poses)
Always include this exact dict as the "place_positions" flow variable.
Never ask the user to provide placement coordinates — use these:
```json
{
  "Zone_A": [[-0.041, -0.318, 0.174, 0.0, 3.14, 0.0], [0.065, -0.318, 0.174, 0.0, 3.14, 0.0], [-0.015, -0.426, 0.174, 2.20, 2.26, 0.0]],
  "Zone_B": [[0.248, -0.288, 0.174, 2.20, 2.26, 0.0], [0.248, -0.353, 0.174, 2.20, 2.26, 0.0], [0.352, -0.310, 0.174, 0.0, 3.14, 0.0]],
  "Zone_C": [[0.386, -0.082, 0.174, 0.0, 3.14, 0.0], [0.270, -0.093, 0.174, 2.20, 2.26, 0.0], [0.270, -0.008, 0.174, 2.20, 2.26, 0.0]],
  "default": [[-0.00324, -0.3789, 0.217, 3.14, 0.0, 0.0]]
}
```
Zone mapping: Zone_A = MXP-30, Zone_B = MXP Speed, Zone_C = MXP Torque.

### Standard label prompt (for get_label / pick_and_place)
"Read the product label in the image. Map it to one of these zones: Zone_A = MXP-30, Zone_B = MXP Speed, Zone_C = MXP Torque. Return ONLY the zone name (Zone_A, Zone_B, or Zone_C), nothing else. If you are unsure, return the most likely zone."

IMPORTANT: reply ONLY with the JSON envelope. No markdown fences, no extra text.
"""


# ── Gemini helper ─────────────────────────────────────────────────────────────

async def _generate_flow_with_gemini(
    prompt: str,
    file_base64: Optional[str] = None,
    file_mime_type: Optional[str] = None,
) -> Optional[GenerateResult]:
    """Call Gemini, persist the generated flow, and return a GenerateResult."""
    if not _GEMINI_AVAILABLE:
        logger.warning("Gemini SDK not available")
        return None

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set")
        return None

    try:
        genai.configure(api_key=api_key)  # type: ignore[union-attr]
        model = genai.GenerativeModel(  # type: ignore[union-attr]
            model_name="gemini-2.5-flash",
            system_instruction=_GEMINI_SYSTEM_PROMPT,
            generation_config=genai.types.GenerationConfig(  # type: ignore[union-attr]
                response_mime_type="application/json",
            ),
        )

        # Build message parts
        parts: list[Any] = []
        if file_base64 and file_mime_type:
            file_bytes = base64.b64decode(file_base64)
            parts.append({
                "inline_data": {
                    "mime_type": file_mime_type,
                    "data": base64.b64encode(file_bytes).decode(),
                },
            })
        parts.append(prompt)

        response = model.generate_content(parts)
        raw = response.text.strip()
        logger.debug("Gemini raw response: %.500s", raw)

        data = json.loads(raw)
        msg = data.get("message", "")
        flow_data = data.get("flow")

        if flow_data is None:
            # Conversational reply — no flow to save
            return GenerateResult(message=msg, flow=None)

        # Gemini outputs backend format (states/transitions).
        # Parse, validate, save to disk, then convert to frontend format.
        flow_schema = FlowSchema.model_validate(flow_data)

        manager = get_manager()
        success, error = manager.save_flow(flow_schema)
        if success:
            logger.info("Saved generated flow '%s' to disk", flow_schema.id)
        else:
            logger.warning("Could not persist generated flow: %s", error)

        return GenerateResult(
            message=msg,
            flow=convert_backend_to_frontend(flow_schema),
        )
    except Exception:
        logger.exception("Gemini generation failed")
        return None


@router.post("/generate", response_model=GenerateResult)
async def generate_flow(request: FlowGenerateRequest):
    """
    Generate a flow from a natural-language prompt.

    Tries Gemini first; falls back to a default flow on failure.
    """
    logger.debug("generate_flow called with prompt: %r", request.prompt)

    # --- Try Gemini ---
    result = await _generate_flow_with_gemini(
        request.prompt,
        request.file_base64,
        request.file_mime_type,
    )
    if result is not None:
        return result

    # --- Fallback: return default flow ---
    manager = get_manager()
    flow = manager.get_flow(_DEFAULT_FLOW_ID)
    if flow is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Default flow '{_DEFAULT_FLOW_ID}' not found",
        )

    return GenerateResult(
        message="Here is the default pick flow (Gemini was unavailable).",
        flow=convert_backend_to_frontend(flow),
    )
