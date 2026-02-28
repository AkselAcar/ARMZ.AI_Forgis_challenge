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

try:
    import google.generativeai as genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False
    logger.warning("google-generativeai not installed — Gemini Vision unavailable")

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
    """Frontend-compatible flow format (nodes + edges)."""

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


# Default flow loaded when no AI generation is available.
_DEFAULT_FLOW_ID = "dobot_test_pick"

# ── Gemini Vision integration ─────────────────────────────────────────────────

_GEMINI_FLOW_SCHEMA = """{
  "message": "A helpful natural language response to the user.",
  "flow": {
    "id": "string (snake_case, no spaces)",
    "name": "string (human readable title)",
    "loop": false,
    "nodes": [
      {"id": "start", "type": "start", "label": "Start", "position": {"x": 0, "y": 0}},
      {
        "id": "state_id (snake_case)",
        "type": "state",
        "label": "Human Label",
        "steps": [
          {"id": "step_id", "skill": "skill_name", "executor": "executor_name", "params": {}}
        ],
        "position": {"x": 0, "y": 0}
      },
      {"id": "end", "type": "end", "label": "End", "position": {"x": 0, "y": 0}}
    ],
    "edges": [
      {"id": "e-start-first_state", "source": "start", "target": "first_state_id"},
      {"id": "e-last_state-end", "source": "last_state_id", "target": "end"}
    ]
  }
}"""

_GEMINI_SYSTEM_PROMPT = f"""You are an expert robotics automation engineer assistant.
You help users design and understand robot execution flows.

Return ONLY a raw valid JSON object (no markdown) with this structure:
{_GEMINI_FLOW_SCHEMA}

Rules for the "flow" field:
- Set "flow" to null for conversational messages (greetings, questions, clarifications).
- Set "flow" to a full flow object when the user asks you to generate, create, or build a flow.
- If the user provides an image or PDF, analyse it and use it to inform the flow.
- A flow always starts with a 'start' node and ends with an 'end' node.
- Each state node must have at least one step.
- Edge ids must be unique.
- Position values must always be {{"x": 0, "y": 0}} — the frontend handles layout.

Available executors and their skills:
- executor "robot":      move_joint, move_linear
- executor "camera":     get_label, get_bounding_box, start_streaming, stop_streaming
- executor "hand":       grip_until_contact
- executor "io_robot":   wait_digital_input, set_digital_output
- executor "io_machine": wait_digital_input, set_digital_output
"""


async def _generate_flow_with_gemini(
    prompt: str,
    file_base64: Optional[str],
    file_mime_type: Optional[str],
) -> Optional[GenerateResult]:
    """Call Gemini to generate a flow or answer conversationally. Returns None on failure."""
    if not _GEMINI_AVAILABLE:
        logger.warning("Gemini not available — falling back to default flow")
        return None

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set — falling back to default flow")
        return None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=_GEMINI_SYSTEM_PROMPT,
        )

        parts: list = [prompt]

        if file_base64 and file_mime_type:
            file_bytes = base64.b64decode(file_base64)
            parts.append({"mime_type": file_mime_type, "data": file_bytes})

        response = await model.generate_content_async(
            parts,
            generation_config={"response_mime_type": "application/json"},
        )

        raw = response.text.strip()
        logger.debug("Gemini raw response: %s", raw[:500])

        data = json.loads(raw)
        flow_data = data.get("flow")
        return GenerateResult(
            message=data.get("message", "Done."),
            flow=FlowGenerateResponse(**flow_data) if flow_data else None,
        )

    except Exception as exc:
        logger.error("Gemini call failed: %s", exc, exc_info=True)
        return None


@router.post("/generate", response_model=GenerateResult)
async def generate_flow(request: FlowGenerateRequest):
    """
    Generate a robot execution flow from a natural language prompt.
    Returns a { message, flow } envelope — flow is null for conversational replies.
    """
    logger.debug(
        "generate_flow called with prompt: %r, has_file: %s",
        request.prompt,
        bool(request.file_base64),
    )

    if request.file_base64 or os.environ.get("GEMINI_API_KEY"):
        result = await _generate_flow_with_gemini(request.prompt, request.file_base64, request.file_mime_type)
        if result is not None:
            return result
        logger.info("Gemini generation failed — falling back to default flow")

    # Fallback: load default flow from disk
    manager = get_manager()
    flow = manager.get_flow(_DEFAULT_FLOW_ID)
    if flow is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Default flow '{_DEFAULT_FLOW_ID}' not found",
        )
    return GenerateResult(
        message="Here is the default flow.",
        flow=convert_backend_to_frontend(flow),
    )
