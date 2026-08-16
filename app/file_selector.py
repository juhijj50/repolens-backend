"""
Two-stage file selection.

Stage 1 (deterministic): drop paths that are never useful — vendored code,
build artifacts, binaries, tests, generated code. This trims a 5000-file
tree to something the LLM can reason about in one prompt.

Stage 2 (LLM): pass remaining paths to Groq and ask it to pick the 15
files most important for understanding what this project does and how
it's built.
"""
import logging
from pathlib import PurePosixPath

from app.github_client import TreeEntry
from app.llm import call_json

logger = logging.getLogger(__name__)


# Directory names we skip entirely (any file underneath is dropped)
SKIP_DIRS = {
    "node_modules", "vendor", "third_party", "third-party",
    ".git", ".github", ".vscode", ".idea", ".cache",
    "dist", "build", "out", "target", "bin", "obj",
    ".next", ".nuxt", ".svelte-kit", ".turbo", ".parcel-cache",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox",
    "coverage", ".nyc_output", "htmlcov",
    "venv", "env", ".venv", ".env",
    "migrations", "alembic",  # usually noisy, not architecturally interesting
}

# File extensions we skip (binaries, media, lockfiles, generated)
SKIP_EXTENSIONS = {
    # Binaries / compiled
    ".pyc", ".pyo", ".so", ".dll", ".dylib", ".class", ".jar",
    ".exe", ".o", ".a", ".lib",
    # Media
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp", ".bmp",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".webm",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    # Archives
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    # Data blobs
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".db", ".sqlite", ".sqlite3",
    # Generated / minified
    ".min.js", ".min.css", ".map",
}

# Exact filenames we skip (lockfiles, IDE junk)
SKIP_FILENAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "Cargo.lock", "poetry.lock", "Pipfile.lock", "uv.lock",
    "Gemfile.lock", "composer.lock", "go.sum",
    ".DS_Store", "Thumbs.db",
    ".env", ".env.local", ".env.production", ".env.development",
    ".env.example", ".env.sample",
}


def filter_tree(entries: list[TreeEntry]) -> list[TreeEntry]:
    """
    Deterministic first pass. Keep only source-shaped files under a
    reasonable size, drop known-noisy paths.
    """
    kept: list[TreeEntry] = []
    for entry in entries:
        if entry.type != "blob":
            continue

        path = PurePosixPath(entry.path)

        # Skip if any parent directory is in SKIP_DIRS
        if any(part in SKIP_DIRS for part in path.parts[:-1]):
            continue

        filename = path.name
        if filename in SKIP_FILENAMES:
            continue

        # Handle multi-suffix like .min.js
        lower = filename.lower()
        if any(lower.endswith(ext) for ext in SKIP_EXTENSIONS):
            continue

        # Drop very large files (probably generated/data)
        if entry.size is not None and entry.size > 200_000:
            continue

        kept.append(entry)

    return kept


SELECTION_SYSTEM_PROMPT = """You are analyzing a code repository to help a developer understand it quickly.

Your job: from a list of file paths, pick the 15 files that best answer:
"What does this project do, and how is it built?"

Prioritize:
- README, main docs
- Entry points (main.py, index.ts, app.ts, cli.ts, __main__.py)
- Primary source modules (routes, models, core business logic)
- Key configuration (package.json, pyproject.toml, requirements.txt) — but only ONE, whichever tells you the most about the stack
- Type definitions or schemas if they reveal domain concepts

Deprioritize / skip:
- Tests (unless the project IS a testing library)
- Example / tutorial / demo files (paths containing "example", "demo", "tutorial", "docs_src", "samples")
- CI config
- Documentation for contributors
- Boilerplate __init__.py files with no content

Rule: if a path is under docs_src/, examples/, tutorial/, demo/, or samples/, skip it unless there is nothing else to pick from.

Return ONLY files that exist in the input list. Return at most 15 paths.
Fewer is fine for small repos.

Respond in JSON: {"files": ["path1", "path2", ...]}"""


def llm_select_files(paths: list[str], repo_name: str) -> list[str]:
    """
    Ask the LLM which files matter most. Falls back to a heuristic
    selection if the LLM fails or returns garbage.
    """
    # If the tree is already small, just take everything
    if len(paths) <= 15:
        return paths

    # Build a compact prompt. Sort so LLM sees the structure clearly.
    sorted_paths = sorted(paths)
    tree_str = "\n".join(sorted_paths)

    user_prompt = (
        f"Repository: {repo_name}\n"
        f"Total files: {len(paths)}\n\n"
        f"File tree:\n{tree_str}"
    )

    try:
        result = call_json(SELECTION_SYSTEM_PROMPT, user_prompt, max_tokens=1024)
        selected = result.get("files", [])
        if not isinstance(selected, list):
            raise ValueError("'files' field is not a list")

        # Trust but verify: filter to only paths that actually exist in the tree
        valid_paths = set(paths)
        selected = [p for p in selected if isinstance(p, str) and p in valid_paths]

        if not selected:
            logger.warning("LLM returned no valid files, falling back to heuristic")
            return _heuristic_select(paths)

        return selected[:15]

    except Exception as e:
        logger.warning("LLM file selection failed: %s — falling back", e)
        return _heuristic_select(paths)


# Ordered by importance — earlier patterns win
IMPORTANT_FILENAMES = [
    "readme.md", "readme.rst", "readme.txt", "readme",
    "package.json", "pyproject.toml", "cargo.toml", "go.mod",
    "main.py", "app.py", "cli.py", "__main__.py",
    "index.ts", "index.js", "main.ts", "main.js", "app.ts", "app.js",
    "server.py", "server.ts", "server.js",
]


def _heuristic_select(paths: list[str]) -> list[str]:
    """
    Fallback when the LLM call fails. Picks files by name importance and
    directory shallowness (root-level files usually matter more).
    """
    def score(path: str) -> tuple[int, int, str]:
        p = PurePosixPath(path)
        name_lower = p.name.lower()

        # Lower is better in Python's sort
        name_score = 999
        for i, pattern in enumerate(IMPORTANT_FILENAMES):
            if name_lower == pattern:
                name_score = i
                break

        depth = len(p.parts)
        return (name_score, depth, path)

    return sorted(paths, key=score)[:15]
