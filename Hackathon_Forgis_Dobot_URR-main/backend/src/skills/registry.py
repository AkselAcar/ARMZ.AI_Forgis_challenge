"""Skill registry with auto-discovery via @register_skill decorator."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Skill

_REGISTRY: dict[str, "Skill"] = {}


def register_skill(cls: type["Skill"]) -> type["Skill"]:
    """
    Class decorator to register a skill in the global registry.

    Usage:
        @register_skill
        class MoveJointSkill(Skill[MoveJointParams]):
            name = "move_joint"
            ...
    """
    instance = cls()
    if not instance.name:
        raise ValueError(f"Skill class {cls.__name__} must define a 'name' attribute")
    if instance.name in _REGISTRY:
        raise ValueError(f"Skill '{instance.name}' is already registered")
    _REGISTRY[instance.name] = instance
    return cls


def get_skill(name: str) -> "Skill":
    """
    Get a skill instance by name.

    Raises:
        KeyError: If skill is not registered.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Skill '{name}' not found. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]


def list_skills() -> list[dict]:
    """
    List all registered skills with their metadata.

    Returns:
        List of skill metadata dicts (name, executor_type, description, params_schema).
    """
    return [skill.get_metadata() for skill in _REGISTRY.values()]


def get_skills_by_executor(executor_type: str) -> list["Skill"]:
    """Get all skills for a specific executor type."""
    return [s for s in _REGISTRY.values() if s.executor_type == executor_type]


def clear_registry() -> None:
    """Clear the registry (useful for testing)."""
    _REGISTRY.clear()
