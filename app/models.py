"""
ORM models for RepoLens.

Schema:
  repos          — one row per (owner, name, commit_sha). Cached across users.
  chunks         — code/doc chunks with pgvector embeddings.
  rate_limits    — IP-based daily counters for analyze endpoint.
  analysis_jobs  — tracks in-progress analysis so /status endpoint can poll.
"""
from datetime import datetime
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class Repo(Base):
    __tablename__ = "repos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    commit_sha: Mapped[str] = mapped_column(String(40), nullable=False)

    # Populated after analysis completes
    overview: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    module_graph: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Stats for the UI
    total_files_in_repo: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    files_analyzed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    primary_language: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    stars: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    analyzed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="repo", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("owner", "name", "commit_sha", name="uq_repo_commit"),
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    repo_id: Mapped[int] = mapped_column(
        ForeignKey("repos.id", ondelete="CASCADE"), nullable=False
    )

    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    chunk_type: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # 'function' | 'class' | 'module' | 'readme' | 'file'
    content: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    # 384 dims = all-MiniLM-L6-v2 output size
    embedding = Column(Vector(384), nullable=True)

    repo: Mapped["Repo"] = relationship(back_populates="chunks")

    __table_args__ = (
        # ivfflat needs data to be tuned; we'll create it manually after inserts
        # in Phase 2. For now, a btree index on repo_id is enough for filtering.
        Index("ix_chunks_repo_id", "repo_id"),
        Index("ix_chunks_file_path", "file_path"),
    )


class RateLimit(Base):
    __tablename__ = "rate_limits"

    ip: Mapped[str] = mapped_column(String(64), primary_key=True)
    analyses_today: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    reset_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class AnalysisJob(Base):
    """
    Tracks the state of an in-progress analysis so the frontend can poll.
    Rows are short-lived — cleaned up periodically or on failure.
    """
    __tablename__ = "analysis_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    repo_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("repos.id", ondelete="SET NULL"), nullable=True
    )
    repo_url: Mapped[str] = mapped_column(String(512), nullable=False)

    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="pending"
    )  # 'pending' | 'fetching' | 'selecting' | 'chunking' | 'analyzing' | 'ready' | 'failed'

    stage_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
