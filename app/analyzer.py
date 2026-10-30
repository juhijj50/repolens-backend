"""
The analysis pipeline. Called from a background task in main.py.

Stages:
  1. fetching   — GitHub tree + selected file contents
  2. selecting  — LLM picks 15 important files
  3. chunking   — split files into meaningful pieces
  4. embedding  — sentence-transformers on all chunks
  5. analyzing  — LLM generates overview + module graph
  6. ready      — everything saved, repo_id set on the job

Each stage updates AnalysisJob.stage_message so the frontend polling
sees real progress.
"""
import asyncio
import logging
from typing import Any

import httpx
from sqlalchemy.orm import Session

from app.chunker import Chunk, chunk_file
from app.db import db_session
from app.embedder import embed_texts
from app.file_selector import filter_tree, llm_select_files
from app.github_client import GitHubClient, GitHubError, RepoRef
from app.llm import call_json
from app.models import AnalysisJob, Chunk as ChunkModel, Repo

logger = logging.getLogger(__name__)


async def run_analysis(
    job_id: int,
    ref: RepoRef,
    commit_sha: str,
    stars: int = 0,
    primary_language: str | None = None,
) -> None:
    """
    Top-level entrypoint. Catches any error and marks the job as failed
    with a user-facing message. Should never raise.
    """
    try:
        await _run_pipeline(job_id, ref, commit_sha, stars, primary_language)
    except GitHubError as e:
        _mark_failed(job_id, f"GitHub error: {e}")
    except ValueError as e:
        _mark_failed(job_id, f"Analysis error: {e}")
    except Exception as e:
        logger.exception("Unexpected error in analysis pipeline")
        _mark_failed(job_id, f"Unexpected error: {type(e).__name__}: {e}")


async def _run_pipeline(
    job_id: int, ref: RepoRef, commit_sha: str, stars: int, primary_language: str | None
) -> None:
    # ---- Stage 1: fetch the tree ----
    await asyncio.to_thread(_update_stage, job_id, "fetching", "Reading file structure...")

    async with GitHubClient() as gh:
        tree = await gh.get_tree(ref, commit_sha)
        total_files = sum(1 for e in tree if e.type == "blob")

        filtered = filter_tree(tree)
        candidate_paths = [e.path for e in filtered]
        logger.info("Repo %s: %d total files, %d after filtering",
                    ref.full_name, total_files, len(candidate_paths))

        if not candidate_paths:
            raise ValueError("No analyzable source files found in this repo")

        # ---- Stage 2: LLM picks important files ----
        await asyncio.to_thread(_update_stage, job_id, "selecting", "Picking the files that matter most...")
        selected_paths = await asyncio.to_thread(
            llm_select_files, candidate_paths, ref.full_name
        )
        logger.info("Selected %d files for analysis", len(selected_paths))

        # ---- Stage 1 (continued): fetch selected file contents in parallel ----
        await asyncio.to_thread(_update_stage, job_id, "fetching", f"Fetching {len(selected_paths)} files...")
        file_contents = await _fetch_files_parallel(gh, ref, commit_sha, selected_paths)

    if not file_contents:
        raise ValueError("Could not read any of the selected files")

    # ---- Stage 3: chunk ----
    await asyncio.to_thread(_update_stage, job_id, "chunking", "Splitting code into meaningful pieces...")
    all_chunks: list[Chunk] = []
    for path, content in file_contents.items():
        all_chunks.extend(chunk_file(path, content))
    logger.info("Generated %d chunks from %d files", len(all_chunks), len(file_contents))

    # ---- Stage 4: embed ----
    await asyncio.to_thread(_update_stage, job_id, "embedding", f"Understanding {len(all_chunks)} code pieces...")
    embeddings = await asyncio.to_thread(
        embed_texts, [c.content for c in all_chunks]
    )

    # ---- Stage 5: LLM overview + module graph ----
    await asyncio.to_thread(_update_stage, job_id, "analyzing", "Writing the overview...")
    overview = await asyncio.to_thread(_generate_overview, ref.full_name, file_contents)

    await asyncio.to_thread(_update_stage, job_id, "analyzing", "Building the module map...")
    module_graph = await asyncio.to_thread(
        _generate_module_graph, ref.full_name, list(file_contents.keys()), overview
    )

    # ---- Persist everything atomically ----
    await asyncio.to_thread(
        _persist_results,
        job_id=job_id,
        ref=ref,
        commit_sha=commit_sha,
        overview=overview,
        module_graph=module_graph,
        chunks=all_chunks,
        embeddings=embeddings,
        total_files=total_files,
        files_analyzed=len(file_contents),
        stars=stars,
        primary_language=primary_language,
    )

    await asyncio.to_thread(_update_stage, job_id, "ready", "Analysis complete")


# ------------------------------------------------------------
# File fetching (parallel with concurrency cap)
# ------------------------------------------------------------

async def _fetch_files_parallel(
    gh: GitHubClient,
    ref: RepoRef,
    commit_sha: str,
    paths: list[str],
) -> dict[str, str]:
    """Fetch files in parallel with a concurrency cap. Skips files that fail."""
    semaphore = asyncio.Semaphore(5)

    async def fetch_one(path: str) -> tuple[str, str] | None:
        async with semaphore:
            # One retry on transport errors: GitHub may have closed a pooled
            # keep-alive connection, and the retry gets a fresh one.
            for attempt in range(2):
                try:
                    content = await gh.get_file_content(ref, path, commit_sha)
                    return (path, content)
                except httpx.TransportError as e:
                    if attempt == 0:
                        continue
                    logger.warning("Skipping %s: %s", path, e)
                except (GitHubError, httpx.HTTPStatusError) as e:
                    logger.warning("Skipping %s: %s", path, e)
                    return None
            return None

    results = await asyncio.gather(*[fetch_one(p) for p in paths])
    return {path: content for r in results if r for path, content in [r]}


# ------------------------------------------------------------
# LLM prompts
# ------------------------------------------------------------

OVERVIEW_SYSTEM_PROMPT = """You are analyzing a code repository to help a developer decide whether to use or contribute to it.

Read the provided files carefully. Then output a structured analysis.

Rules:
- "what_it_does": 2-3 sentences in plain English. No marketing fluff, no "cutting-edge" or "revolutionary". Just what it does and who it's for.
- "tech_stack": the actual technologies used. ONLY include names you can verify from imports, requires, dependencies fields, or package manifests in the provided files. Do NOT guess or invent library names. If unsure, leave it out. Names only, no versions.
- "architecture_pattern": one short phrase describing the shape (e.g. "REST API with background worker", "CLI tool", "React SPA with REST backend"). If unclear, use "library".
- "entry_points": file paths a new contributor should read first, in order. Max 5. Only use paths that appear in the provided files.
- "notable_dependencies": libraries or frameworks that meaningfully shape the design. Max 6. ONLY include names you can verify from imports or package manifests. Skip generic ones like "requests" or "lodash". If you cannot verify a dependency exists, omit it.

Critical: it is better to return a shorter, accurate list than a longer list with guessed names. Never invent library names to fill space.

Respond in JSON matching this schema:
{
  "what_it_does": "...",
  "tech_stack": ["..."],
  "architecture_pattern": "...",
  "entry_points": ["path1", "path2"],
  "notable_dependencies": ["..."]
}"""


# Groq's on-demand tier caps openai/gpt-oss-120b at 8000 tokens/minute.
# Budget total file content (not per-file) so this holds regardless of
# how many files were selected or how large any one of them is.
OVERVIEW_TOTAL_CHAR_BUDGET = 20000


def _truncate_at_line(text: str, max_chars: int) -> str:
    """Truncate to max_chars, but back off to the nearest newline to
    avoid cutting mid-line. Falls back to hard truncation if no newline
    exists in the range."""
    if len(text) <= max_chars:
        return text
    cutoff = text.rfind("\n", 0, max_chars)
    if cutoff < max_chars // 2:  # if newline is way too early, hard cut
        cutoff = max_chars
    return text[:cutoff] + "\n[...truncated]"


def _generate_overview(repo_name: str, files: dict[str, str]) -> dict[str, Any]:
    """LLM call #2: structured overview of the project."""
    # Build a compact "here are the files" prompt, splitting the total
    # character budget evenly so no single large file crowds out the rest.
    per_file_limit = max(300, OVERVIEW_TOTAL_CHAR_BUDGET // max(1, len(files)))

    parts = [f"Repository: {repo_name}\n"]
    for path, content in files.items():
        truncated = _truncate_at_line(content, per_file_limit)
        parts.append(f"\n=== {path} ===\n{truncated}")

    user_prompt = "\n".join(parts)
    return call_json(OVERVIEW_SYSTEM_PROMPT, user_prompt, max_tokens=1024)


GRAPH_SYSTEM_PROMPT = """You are building an architecture diagram for a code repository.

Given a list of file paths and a project overview, identify 5-8 logical modules
in this codebase and how they relate. A module is a coherent group of files
serving one purpose (auth, API routes, database layer, background jobs, UI
components, etc.).

Rules:
- 5-8 nodes. Fewer is fine for small projects.
- Each node has: id (short lowercase, no spaces), label (2-3 word display name), files (list of paths from the input).
- Edges show real relationships: "calls", "reads", "renders", "extends", "configures", etc.
- Don't invent nodes for things that aren't in the file list.
- Every file in a node's "files" must be from the input list.

Respond in JSON matching this schema:
{
  "nodes": [
    {"id": "api", "label": "API Routes", "files": ["app/routes.py"]}
  ],
  "edges": [
    {"from": "api", "to": "db", "label": "queries"}
  ]
}"""


def _generate_module_graph(
    repo_name: str, file_paths: list[str], overview: dict[str, Any]
) -> dict[str, Any]:
    """LLM call #3: nodes + edges for the React Flow diagram."""
    user_prompt = (
        f"Repository: {repo_name}\n"
        f"Overview: {overview.get('what_it_does', '')}\n"
        f"Pattern: {overview.get('architecture_pattern', '')}\n\n"
        f"Files analyzed:\n" + "\n".join(file_paths)
    )
    graph = call_json(GRAPH_SYSTEM_PROMPT, user_prompt, max_tokens=1024)

    # Defensive: ensure shape is valid
    if not isinstance(graph.get("nodes"), list) or not isinstance(graph.get("edges"), list):
        raise ValueError("LLM returned malformed module graph")

    return graph


# ------------------------------------------------------------
# Persistence
# ------------------------------------------------------------

def _persist_results(
    job_id: int,
    ref: RepoRef,
    commit_sha: str,
    overview: dict[str, Any],
    module_graph: dict[str, Any],
    chunks: list[Chunk],
    embeddings: list[list[float]],
    total_files: int,
    files_analyzed: int,
    stars: int,
    primary_language: str | None,
) -> None:
    """One transaction: upsert repo, insert chunks, link job to repo."""
    with db_session() as db:
        # Upsert repo row
        repo = (
            db.query(Repo)
            .filter_by(owner=ref.owner, name=ref.name, commit_sha=commit_sha)
            .first()
        )
        if repo is None:
            repo = Repo(
                owner=ref.owner,
                name=ref.name,
                commit_sha=commit_sha,
            )
            db.add(repo)

        repo.overview = overview
        repo.module_graph = module_graph
        repo.total_files_in_repo = total_files
        repo.files_analyzed = files_analyzed
        repo.stars = stars
        repo.primary_language = primary_language
        db.flush()  # populates repo.id

        # Wipe any old chunks for this repo (idempotent re-runs)
        db.query(ChunkModel).filter_by(repo_id=repo.id).delete()

        # Insert new chunks
        for chunk, embedding in zip(chunks, embeddings):
            db.add(
                ChunkModel(
                    repo_id=repo.id,
                    file_path=chunk.file_path,
                    chunk_type=chunk.chunk_type,
                    content=chunk.content,
                    language=chunk.language,
                    embedding=embedding,
                )
            )

        # Link the job to the completed repo
        job = db.get(AnalysisJob, job_id)
        if job is not None:
            job.repo_id = repo.id


# ------------------------------------------------------------
# Job status helpers
# ------------------------------------------------------------

def _update_stage(job_id: int, status: str, message: str) -> None:
    """Update job status. Runs in its own short transaction."""
    with db_session() as db:
        job = db.get(AnalysisJob, job_id)
        if job is not None:
            job.status = status
            job.stage_message = message


def _mark_failed(job_id: int, message: str) -> None:
    with db_session() as db:
        job = db.get(AnalysisJob, job_id)
        if job is not None:
            job.status = "failed"
            job.error_message = message
