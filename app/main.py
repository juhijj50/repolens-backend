"""
RepoLens API entrypoint.

Endpoints:
  POST   /analyze                     — kick off analysis for a repo URL
  GET    /jobs/{job_id}               — poll analysis status
  GET    /repos/{repo_id}             — fetch analyzed overview + module graph
  POST   /repos/{repo_id}/chat        — ask a question about the repo (Phase 3)
  GET    /health                      — liveness check
"""
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db, init_db
from app.github_client import GitHubClient, GitHubError, parse_repo_url
from app.models import AnalysisJob, Repo
from app.rate_limit import check_and_increment, get_client_ip
from app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ChatRequest,
    ChatResponse,
    JobStatusResponse,
    RepoDetail,
    RepoLookupResponse,
    RepoSummary,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Runs once at startup: enable pgvector, create tables, fail orphaned jobs."""
    init_db()
    _fail_orphaned_jobs()
    yield


def _fail_orphaned_jobs() -> None:
    """
    Background tasks live in process memory, so a deploy or Render's idle
    spin-down kills any analysis mid-flight and leaves its job row stuck in a
    non-terminal status forever. The frontend would poll it endlessly and the
    in-flight guard in /analyze would keep handing it out.

    Safe with a single worker (WEB_CONCURRENCY=1): any unfinished job at boot
    belonged to a process that is gone or about to be.
    """
    from app.db import db_session

    with db_session() as db:
        db.query(AnalysisJob).filter(
            AnalysisJob.status.notin_(["ready", "failed"]),
        ).update(
            {
                AnalysisJob.status: "failed",
                AnalysisJob.error_message: "Server restarted during analysis. Please try again.",
            },
            synchronize_session=False,
        )


app = FastAPI(
    title="RepoLens API",
    version="0.1.0",
    description="Understand any GitHub repo in 5 minutes.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ------------------------------------------------------------
# Health
# ------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


# ------------------------------------------------------------
# Analyze
# ------------------------------------------------------------

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    body: AnalyzeRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> AnalyzeResponse:
    """
    Validate the URL, check cache, rate-limit the caller, and either:
      - return the cached repo immediately (cached=True), or
      - create a job row and schedule background analysis (cached=False).
    """
    # Parse first — cheap and fails fast on bad input
    try:
        ref = parse_repo_url(body.repo_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Fetch metadata to get the current commit SHA (needed for cache lookup).
    # If GitHub redirected (renamed repo), use the resolved ref for caching.
    try:
        async with GitHubClient() as gh:
            ref, meta = await gh.get_repo_meta(ref)
    except GitHubError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Cache hit? Return the existing repo without touching rate limits —
    existing = (
        db.query(Repo)
        .filter_by(owner=ref.owner, name=ref.name, commit_sha=meta.commit_sha)
        .first()
    )
    if existing and existing.overview is not None:
        job = AnalysisJob(
            repo_id=existing.id,
            repo_url=body.repo_url,
            status="ready",
            stage_message="Loaded from cache",
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        return AnalyzeResponse(job_id=job.id, status="ready", cached=True)

    # Guard: is an analysis already running for this exact commit?
    # If so, return that job's id instead of starting a duplicate.
    in_flight = (
        db.query(AnalysisJob)
        .filter(
            AnalysisJob.repo_url.ilike(f"%{ref.owner}/{ref.name}%"),
            AnalysisJob.status.notin_(["ready", "failed"]),
        )
        .order_by(AnalysisJob.created_at.desc())
        .first()
    )
    if in_flight is not None:
        return AnalyzeResponse(
            job_id=in_flight.id, status=in_flight.status, cached=False
        )

    # Cache miss and no in-flight job — enforce rate limit before spending LLM tokens
    ip = get_client_ip(request)
    check_and_increment(db, ip)

    # Size guardrails
    if meta.size_kb > settings.max_repo_size_mb * 1024:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Repo is {meta.size_kb // 1024}MB; max is "
                f"{settings.max_repo_size_mb}MB for the free tier."
            ),
        )

    # Cleanup: delete analysis_jobs older than 24h so the table stays small.
    # Runs opportunistically on every /analyze call — cheap and self-maintaining.
    from datetime import datetime, timedelta, timezone
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    db.query(AnalysisJob).filter(AnalysisJob.created_at < cutoff).delete()

    # Create the job row and schedule the background task
    job = AnalysisJob(
        repo_url=body.repo_url,
        status="pending",
        stage_message="Queued",
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    background_tasks.add_task(
        _run_analysis,
        job_id=job.id,
        owner=ref.owner,
        name=ref.name,
        commit_sha=meta.commit_sha,
        stars=meta.stars,
        primary_language=meta.primary_language,
    )

    return AnalyzeResponse(job_id=job.id, status="pending", cached=False)


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: int, db: Session = Depends(get_db)) -> JobStatusResponse:
    job = db.get(AnalysisJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        stage_message=job.stage_message,
        error_message=job.error_message,
        repo_id=job.repo_id,
    )


# ------------------------------------------------------------
# Repo detail
# ------------------------------------------------------------

@app.get("/repos/by-url", response_model=RepoLookupResponse)
async def lookup_repo_by_url(
    url: str,
    db: Session = Depends(get_db),
) -> RepoLookupResponse:
    """
    Frontend calls this on page load. Returns whether we have a cached
    analysis, whether it's up to date with the current commit, or whether
    an analysis is currently running.

    This lets the frontend decide: render immediately, offer re-analyze,
    show progress, or kick off a fresh analysis.
    """
    try:
        ref = parse_repo_url(url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Get the current commit SHA from GitHub — cheap, one API call
    try:
        async with GitHubClient() as gh:
            ref, meta = await gh.get_repo_meta(ref)
    except GitHubError as e:
        raise HTTPException(status_code=404, detail=str(e))

    current_sha = meta.commit_sha

    # Check for an in-progress job for this repo (matched by owner/name in URL)
    # We look at jobs that are neither ready nor failed
    in_progress = (
        db.query(AnalysisJob)
        .filter(
            AnalysisJob.repo_url.ilike(f"%{ref.owner}/{ref.name}%"),
            AnalysisJob.status.notin_(["ready", "failed"]),
        )
        .order_by(AnalysisJob.created_at.desc())
        .first()
    )
    if in_progress is not None:
        return RepoLookupResponse(
            status="in_progress",
            in_progress_job_id=in_progress.id,
            current_commit_sha=current_sha,
        )

    # Look up the most recent completed analysis for this repo
    latest = (
        db.query(Repo)
        .filter_by(owner=ref.owner, name=ref.name)
        .filter(Repo.overview.isnot(None))
        .order_by(Repo.analyzed_at.desc())
        .first()
    )

    if latest is None:
        return RepoLookupResponse(
            status="not_found",
            current_commit_sha=current_sha,
        )

    if latest.commit_sha == current_sha:
        return RepoLookupResponse(
            status="up_to_date",
            repo_id=latest.id,
            cached_commit_sha=latest.commit_sha,
            current_commit_sha=current_sha,
        )

    return RepoLookupResponse(
        status="stale",
        repo_id=latest.id,
        cached_commit_sha=latest.commit_sha,
        current_commit_sha=current_sha,
    )


@app.get("/repos", response_model=list[RepoSummary])
def list_repos(
    limit: int = 20,
    db: Session = Depends(get_db),
) -> list[RepoSummary]:
    """
    List recently analyzed repos, most recent first. Used by the frontend
    to show a "recently analyzed" section on the landing page.
    """
    limit = max(1, min(limit, 50))  # clamp
    rows = (
        db.query(Repo)
        .filter(Repo.overview.isnot(None))
        .order_by(Repo.analyzed_at.desc())
        .limit(limit)
        .all()
    )
    return [
        RepoSummary(
            id=r.id,
            owner=r.owner,
            name=r.name,
            stars=r.stars,
            primary_language=r.primary_language,
            what_it_does=(r.overview or {}).get("what_it_does"),
            analyzed_at=r.analyzed_at,
        )
        for r in rows
    ]


@app.get("/repos/{repo_id}", response_model=RepoDetail)
def get_repo(repo_id: int, db: Session = Depends(get_db)) -> RepoDetail:
    repo = db.get(Repo, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="Repo not found")
    return RepoDetail(
        id=repo.id,
        owner=repo.owner,
        name=repo.name,
        commit_sha=repo.commit_sha,
        stars=repo.stars,
        primary_language=repo.primary_language,
        files_analyzed=repo.files_analyzed,
        total_files_in_repo=repo.total_files_in_repo,
        overview=repo.overview,
        module_graph=repo.module_graph,
        analyzed_at=repo.analyzed_at,
    )


# ------------------------------------------------------------
# Chat (Phase 3 — stubbed for now)
# ------------------------------------------------------------

@app.post("/repos/{repo_id}/chat", response_model=ChatResponse)
def chat(
    repo_id: int,
    body: ChatRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> ChatResponse:
    repo = db.get(Repo, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="Repo not found")
    if repo.overview is None:
        raise HTTPException(
            status_code=409, detail="Repo analysis not complete yet"
        )

    # Rate limit: 30 messages / IP / hour
    ip = get_client_ip(request)
    from app.rate_limit import check_and_increment_chat
    check_and_increment_chat(db, ip, hourly_limit=30)

    from app.chat import answer_question

    # Convert Pydantic history models to plain dicts for the prompt builder
    history = [{"role": m.role, "content": m.content} for m in body.history]

    answer, cited = answer_question(db, repo_id, body.message, history)

    return ChatResponse(answer=answer, cited_files=cited)


# ------------------------------------------------------------
# Background analysis (Phase 2 — stubbed with the right shape)
# ------------------------------------------------------------

async def _run_analysis(
    job_id: int,
    owner: str,
    name: str,
    commit_sha: str,
    stars: int,
    primary_language: str | None,
) -> None:
    """
    Runs in a background task so /analyze returns immediately.
    All error handling is inside run_analysis — this never raises.
    """
    from app.analyzer import run_analysis
    from app.github_client import RepoRef

    await run_analysis(
        job_id=job_id,
        ref=RepoRef(owner=owner, name=name),
        commit_sha=commit_sha,
        stars=stars,
        primary_language=primary_language,
    )
