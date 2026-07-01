from datetime import datetime
from pydantic import BaseModel, HttpUrl


# Workspace

class WorkspaceCreate(BaseModel):
    name: str
    description: str | None = None


class WorkspaceResponse(BaseModel):
    id: str
    name: str
    description: str | None
    document_count: int = 0
    created_at: datetime

    model_config = {"from_attributes": True}


# Document

class DocumentResponse(BaseModel):
    id: str
    title: str
    source_type: str
    source_url: str | None
    file_name: str | None
    status: str
    chunk_count: int
    token_count: int
    created_at: datetime

    model_config = {"from_attributes": True}


class IngestURLRequest(BaseModel):
    url: HttpUrl
    workspace_id: str


class IngestTextRequest(BaseModel):
    title: str
    content: str
    workspace_id: str


# Query

class QueryRequest(BaseModel):
    question: str
    workspace_id: str
    top_k: int = 5
    stream: bool = True


class SourceChunk(BaseModel):
    document_id: str
    document_title: str
    chunk_text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    prompt_tokens: int
    completion_tokens: int


# Health

class HealthResponse(BaseModel):
    status: str
    version: str = "0.1.0"
    environment: str