"""File-based flow persistence."""

import json
import logging
from pathlib import Path
from typing import Optional

from .schemas import FlowSchema

logger = logging.getLogger(__name__)


class FlowLoader:
    """
    Load and save flows from/to JSON files.

    Flows are stored in a configured directory, one file per flow.
    File naming: {flow_id}.json
    """

    def __init__(self, flows_dir: str = "/app/flows"):
        self._flows_dir = Path(flows_dir)
        self._flows_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"FlowLoader initialized with directory: {self._flows_dir}")

    def _flow_path(self, flow_id: str) -> Path:
        """Get file path for a flow ID."""
        # Sanitize flow_id to prevent path traversal
        safe_id = "".join(c for c in flow_id if c.isalnum() or c in "-_")
        return self._flows_dir / f"{safe_id}.json"

    def list_flows(self) -> list[str]:
        """List all available flow IDs."""
        flow_ids = []
        for path in self._flows_dir.glob("*.json"):
            flow_ids.append(path.stem)
        return sorted(flow_ids)

    def load(self, flow_id: str) -> Optional[FlowSchema]:
        """
        Load a flow by ID.

        Returns:
            FlowSchema if found, None otherwise.
        """
        path = self._flow_path(flow_id)
        if not path.exists():
            logger.warning(f"Flow file not found: {path}")
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            flow = FlowSchema.model_validate(data)
            logger.info(f"Loaded flow: {flow_id}")
            return flow
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in flow file {path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading flow {flow_id}: {e}")
            return None

    def save(self, flow: FlowSchema) -> bool:
        """
        Save a flow to file.

        Returns:
            True if saved successfully, False otherwise.
        """
        path = self._flow_path(flow.id)

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(flow.model_dump(), f, indent=2)
            logger.info(f"Saved flow: {flow.id} to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving flow {flow.id}: {e}")
            return False

    def delete(self, flow_id: str) -> bool:
        """
        Delete a flow file.

        Returns:
            True if deleted, False if not found or error.
        """
        path = self._flow_path(flow_id)
        if not path.exists():
            return False

        try:
            path.unlink()
            logger.info(f"Deleted flow: {flow_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting flow {flow_id}: {e}")
            return False

    def load_all(self) -> list[FlowSchema]:
        """Load all available flows."""
        flows = []
        for flow_id in self.list_flows():
            flow = self.load(flow_id)
            if flow:
                flows.append(flow)
        return flows
