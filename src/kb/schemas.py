"""Schemas for Knowledge Base and Retrieval."""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata extracted from Markdown YAML front-matter."""
    document_id: str
    title: str
    status: str  # "active", "superseded", "draft"
    effective_date: Optional[str] = None
    last_reviewed: Optional[str] = None
    audience: str = "customer"  # "customer", "internal"
    policy_authority: str = "official"  # "official", "none"
    supersedes: Optional[str] = None
    superseded_by: Optional[str] = None
    customer_answering: bool = True
    filename: str = ""


class DocChunk(BaseModel):
    """A cohesive passage/section from a document."""
    chunk_id: str
    document_id: str
    filename: str
    title: str
    heading: str
    content: str
    full_text: str  # Includes metadata header + heading + content for indexing
    metadata: DocumentMetadata
    citation: str = Field(description="Formatted source citation e.g. '01-returns-policy-current.md > Standard return window'")


class RetrievalResult(BaseModel):
    """Result of retrieving relevant chunks for a query."""
    chunk: DocChunk
    score: float
    is_authoritative: bool
    status: str
    policy_authority: str


class SearchResponse(BaseModel):
    """Aggregated search result with metadata and conflict indicators."""
    query: str
    results: List[RetrievalResult]
    has_conflict: bool = False
    conflict_details: Optional[str] = None
    citations: List[str] = Field(default_factory=list)
