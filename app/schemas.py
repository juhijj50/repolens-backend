"""
Pydantic schemas for request/response bodies.
Keep DB models (in models.py) separate from API contracts.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    repo_url: str = Field(..., min_length=3, max_length=512)


class AnalyzeResponse(BaseModel):
    job_id: int
    status: str
    cached: bool = False  # True if we already had this repo+commit analyzed


class JobStatusResponse(BaseModel):
    job_id: int
    status: str
    stage_message: Optional[str] = None
    error_message: Optional[str] = None
    repo_id: Optional[int] = None


class RepoOverview(BaseModel):
    what_it_does: str
    tech_stack: list[str]
    architecture_pattern: Optional[str] = None
    entry_points: list[str] = []
    notable_dependencies: list[str] = []


class GraphNode(BaseModel):
    id: str
    label: str
    files: list[str] = []


class GraphEdge(BaseModel):
    from_: str = Field(..., alias="from")
    to: str
    label: Optional[str] = None

    model_config = {"populate_by_name": True}


class ModuleGraph(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class RepoDetail(BaseModel):
    id: int
    owner: str
    name: str
    commit_sha: str
    stars: Optional[int]
    primary_language: Optional[str]
    files_analyzed: Optional[int]
    total_files_in_repo: Optional[int]
    overview: Optional[RepoOverview]
    module_graph: Optional[ModuleGraph]
    analyzed_at: datetime


class RepoSummary(BaseModel):
    """Compact version of RepoDetail for list endpoints."""
    id: int
    owner: str
    name: str
    stars: Optional[int]
    primary_language: Optional[str]
    what_it_does: Optional[str]
    analyzed_at: datetime


class RepoLookupResponse(BaseModel):
    """
    Response for GET /repos/by-url — tells the frontend what state a repo
    is in so it can decide what to render.

    Possible shapes:
      - not_found:      no cached analysis; frontend should POST /analyze
      - up_to_date:     cached and current commit matches; render immediately
      - stale:          cached but repo has new commits; frontend can choose
                        to render cache + offer "re-analyze" button
      - in_progress:    an analysis job is currently running for this repo
    """
    status: str  # 'not_found' | 'up_to_date' | 'stale' | 'in_progress'
    repo_id: Optional[int] = None
    cached_commit_sha: Optional[str] = None
    current_commit_sha: Optional[str] = None
    in_progress_job_id: Optional[int] = None


class ChatMessage(BaseModel):
    role: str  # 'user' | 'assistant'
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    answer: str
    cited_files: list[str] = []
