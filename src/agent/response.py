"""Agent response models and schemas."""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    """Unified response model returned by the Aster & Row agent."""
    answer: str = Field(description="The customer-facing response text.")
    sources: List[str] = Field(
        default_factory=list,
        description="List of cited source filenames and headings, e.g. '01-returns-policy-current.md > Standard return window'"
    )
    handoff_recommended: bool = Field(
        default=False,
        description="Whether a human support specialist should review or handle this inquiry."
    )
    tool_called: Optional[str] = Field(
        default=None,
        description="Name of the tool called during processing, e.g. 'order_lookup'."
    )
    tool_arguments: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Arguments supplied to the tool."
    )
    tool_result_sanitized: Optional[str] = Field(
        default=None,
        description="Sanitized result output from the tool."
    )
    debug_trace: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured debug trace metadata for observability."
    )
