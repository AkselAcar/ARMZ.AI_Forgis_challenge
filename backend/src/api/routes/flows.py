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
_DEFAULT_FLOW_ID = "dobot_test_pick"


# ── Gemini system prompt ─────────────────────────────────────────────────────
_GEMINI_SYSTEM_PROMPT = """\
You are an expert robot-programming assistant for the ARMZ.AI platform.
Your job is to help users build automation flows for a Dobot Nova 5 robotic arm
equipped with a Covvi prosthetic hand, IO modules, and a camera.

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
  "type": "immediate" | "conditional",
  "condition": null | "<python-expression referencing variables.*>"
}
```

## AVAILABLE SKILLS (13 total)

### Executor: "robot"
1. **move_joint** – Joint-space move
   params: { joints: number[6], speed?: 0-100 (default 30), acceleration?: 0-100 (default 30) }
2. **move_linear** – Cartesian linear move
   params: { x, y, z: mm; rx?, ry?, rz?: degrees; speed?: 0-100 (default 30); acceleration?: 0-100 (default 30) }
3. **palletize** – Palletize helper
   params: { corner_a, corner_b, corner_c: {x,y,z,rx,ry,rz}; rows: int; cols: int; approach_height?: mm; current_index?: int }

### Executor: "camera"
4. **get_bounding_box** – Local YOLOv8 detection (fast, ~50-200 ms)
   params: { label: string; camera_id?: string (default "bridge") }
   sets: variables.bbox = {x_center, y_center, width, height, confidence, label}
5. **get_label** – Vision-LLM label via GPT-4o (slower, 2-5 s)
   params: { prompt: string; camera_id?: string (default "bridge") }
   sets: variables.label = "<text>"
6. **start_streaming** – Start MJPEG stream
   params: { camera_id?: string }
7. **stop_streaming** – Stop MJPEG stream
   params: { camera_id?: string }

### Executor: "hand"
8. **set_grip** – Open/close whole hand
   params: { grip_percentage: 0-100 }
9. **set_finger_positions** – Per-finger control
   params: { thumb?: 0-100; index?: 0-100; middle?: 0-100; ring?: 0-100; little?: 0-100 }
10. **grip_until_contact** – Close until force threshold
    params: { force_threshold?: 0-100 (default 50); timeout?: seconds (default 5); speed?: 0-100 (default 50) }

### Executor: "io_robot" / "io_machine"
11. **io_set_digital_output** – Set digital output pin
    params: { index: int; value: 0 | 1 }
12. **wait_digital_input** – Wait for digital input
    params: { index: int; value: 0 | 1; timeout?: seconds (default 30) }

## VARIABLE INTERPOLATION
Step params may reference runtime variables with `{{variables.xxx}}`:
  - `{{variables.bbox.x_center}}`, `{{variables.bbox.y_center}}`
  - `{{variables.label}}`
  - `{{variables.palletize.current_index}}`
Transition conditions can also use `variables.*`, e.g. `variables.label == 'box'`.

## EXAMPLE FLOW (pick & place)
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
        { "id": "s1", "skill": "get_bounding_box", "executor": "camera", "params": { "label": "box" } }
      ]
    },
    {
      "name": "pick",
      "steps": [
        { "id": "s2", "skill": "move_joint", "executor": "robot", "params": { "joints": [0, 45, -90, 0, 45, 0], "speed": 50 } },
        { "id": "s3", "skill": "grip_until_contact", "executor": "hand", "params": { "force_threshold": 60 } }
      ]
    },
    {
      "name": "place",
      "steps": [
        { "id": "s4", "skill": "move_linear", "executor": "robot", "params": { "x": 300, "y": -200, "z": 150, "speed": 40 } },
        { "id": "s5", "skill": "set_grip", "executor": "hand", "params": { "grip_percentage": 0 } }
      ]
    }
  ],
  "transitions": [
    { "from_state": "detect", "to_state": "pick", "type": "immediate", "condition": null },
    { "from_state": "pick", "to_state": "place", "type": "immediate", "condition": null }
  ]
}
```

IMPORTANT: reply ONLY with the JSON envelope. No markdown fences, no extra text.
"""


# ── Gemini helper ─────────────────────────────────────────────────────────────

async def _generate_flow_with_gemini(
    prompt: str,
    file_base64: Optional[str] = None,
    file_mime_type: Optional[str] = None,
) -> Optional[GenerateResult]:
    """Call Gemini and return a GenerateResult, or None on failure."""
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
            return GenerateResult(message=msg, flow=None)

        return GenerateResult(
            message=msg,
            flow=FlowGenerateResponse(**flow_data),
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
