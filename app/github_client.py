"""
Thin wrapper around GitHub REST API.

Uses:
  - GET /repos/{owner}/{repo}                        → default branch, stars, language
  - GET /repos/{owner}/{repo}/git/trees/{sha}?recursive=1  → entire file tree in one call
  - GET /repos/{owner}/{repo}/contents/{path}        → file content (base64)

No cloning. This keeps the free-tier backend disk-free and fast.
"""
import base64
import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import httpx

from app.config import settings


GITHUB_API = "https://api.github.com"


@dataclass
class RepoRef:
    owner: str
    name: str

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"


@dataclass
class RepoMeta:
    default_branch: str
    commit_sha: str
    stars: int
    primary_language: Optional[str]
    size_kb: int  # total repo size reported by GitHub


@dataclass
class TreeEntry:
    path: str
    type: str  # 'blob' | 'tree'
    size: Optional[int]  # bytes; None for trees
    sha: str


class GitHubError(Exception):
    """Raised for any GitHub API error we can't recover from."""
    pass


def parse_repo_url(url: str) -> RepoRef:
    """
    Accepts any of:
      https://github.com/owner/repo
      https://github.com/owner/repo.git
      https://github.com/owner/repo/tree/main
      git@github.com:owner/repo.git
      owner/repo
    Returns RepoRef or raises ValueError.
    """
    url = url.strip()

    # Bare owner/repo shorthand
    if re.fullmatch(r"[\w.-]+/[\w.-]+", url):
        owner, name = url.split("/")
        return RepoRef(owner=owner, name=_strip_git_suffix(name))

    # SSH format
    ssh_match = re.match(r"git@github\.com:([\w.-]+)/([\w.-]+)", url)
    if ssh_match:
        return RepoRef(
            owner=ssh_match.group(1),
            name=_strip_git_suffix(ssh_match.group(2)),
        )

    # HTTPS URL
    parsed = urlparse(url)
    if parsed.netloc not in ("github.com", "www.github.com"):
        raise ValueError(f"Not a GitHub URL: {url}")

    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"URL missing owner/repo: {url}")

    return RepoRef(owner=parts[0], name=_strip_git_suffix(parts[1]))


def _strip_git_suffix(name: str) -> str:
    return name[:-4] if name.endswith(".git") else name


class GitHubClient:
    """Async client. Instantiate once per request/analysis job."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {settings.github_token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "RepoLens/0.1",
            },
            timeout=30.0,
            follow_redirects=True,  # GitHub returns 301 for renamed/transferred repos
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "GitHubClient":
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.close()

    async def get_repo_meta(self, ref: RepoRef) -> tuple[RepoRef, RepoMeta]:
        """
        Fetch repo metadata. If GitHub redirected (renamed/transferred repo),
        returns the resolved RepoRef so callers can update their cache key.
        """
        r = await self._client.get(f"{GITHUB_API}/repos/{ref.full_name}")
        if r.status_code == 404:
            raise GitHubError("Repo not found or not public")
        r.raise_for_status()
        data = r.json()

        # If GitHub resolved to a different owner/name (redirect), use that
        resolved = RepoRef(
            owner=data["owner"]["login"],
            name=data["name"],
        )

        # Second call to get the SHA of the default branch tip
        branch = data["default_branch"]
        br = await self._client.get(
            f"{GITHUB_API}/repos/{resolved.full_name}/branches/{branch}"
        )
        br.raise_for_status()
        commit_sha = br.json()["commit"]["sha"]

        meta = RepoMeta(
            default_branch=branch,
            commit_sha=commit_sha,
            stars=data.get("stargazers_count", 0),
            primary_language=data.get("language"),
            size_kb=data.get("size", 0),
        )
        return resolved, meta

    async def get_tree(self, ref: RepoRef, sha: str) -> list[TreeEntry]:
        """
        Returns the full recursive tree at the given commit SHA.
        GitHub caps at ~100k entries; if truncated, we still get what fits.
        """
        r = await self._client.get(
            f"{GITHUB_API}/repos/{ref.full_name}/git/trees/{sha}",
            params={"recursive": "1"},
        )
        r.raise_for_status()
        data = r.json()

        entries = [
            TreeEntry(
                path=e["path"],
                type=e["type"],
                size=e.get("size"),
                sha=e["sha"],
            )
            for e in data.get("tree", [])
        ]
        return entries

    async def get_file_content(self, ref: RepoRef, path: str, sha: str) -> str:
        """
        Fetches a single file's content at a specific commit.
        Returns decoded UTF-8 text. Raises GitHubError for binary/oversized files.
        """
        r = await self._client.get(
            f"{GITHUB_API}/repos/{ref.full_name}/contents/{path}",
            params={"ref": sha},
        )
        if r.status_code == 404:
            raise GitHubError(f"File not found: {path}")
        r.raise_for_status()
        data = r.json()

        if data.get("encoding") != "base64":
            raise GitHubError(f"Unexpected encoding for {path}")

        raw = base64.b64decode(data["content"])
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            raise GitHubError(f"Binary or non-UTF-8 file: {path}")
