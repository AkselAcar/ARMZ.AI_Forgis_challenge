"""REST endpoints for skill discovery."""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from skills import list_skills

router = APIRouter(prefix="/api/skills", tags=["skills"])


class SkillInfo(BaseModel):
    """Skill metadata for API response."""

    name: str
    executor_type: str
    description: str
    params_schema: dict[str, Any]


class SkillListResponse(BaseModel):
    """Response for listing skills."""

    skills: list[SkillInfo]


@router.get("", response_model=SkillListResponse)
async def get_skills():
    """List all available skills with their metadata."""
    skills_metadata = list_skills()
    return SkillListResponse(
        skills=[SkillInfo(**skill) for skill in skills_metadata]
    )


@router.get("/{skill_name}", response_model=SkillInfo)
async def get_skill(skill_name: str):
    """Get metadata for a specific skill."""
    from fastapi import HTTPException, status
    from skills import get_skill as get_skill_instance

    try:
        skill = get_skill_instance(skill_name)
        metadata = skill.get_metadata()
        return SkillInfo(**metadata)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Skill '{skill_name}' not found",
        )
