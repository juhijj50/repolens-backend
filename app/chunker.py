"""
Language-agnostic chunking.

Rather than parsing code with tree-sitter (which had install issues on
Python 3.13 Windows), we use a simpler approach: split code files at
lines that look like top-level definitions in common languages. Docs and
small files stay as single chunks.

Trade-off: won't perfectly capture nested classes or edge cases like
decorators-only-on-one-line. But embeddings are robust to imperfect
chunk boundaries, and this works for every language without a compiled
dependency. Good enough for retrieval quality; honest trade-off you can
name in the README.
"""
import re
from dataclasses import dataclass
from pathlib import PurePosixPath


# Extensions that get treated as documentation (chunked as one piece)
DOC_EXTENSIONS = {".md", ".rst", ".txt", ".mdx", ".adoc"}

# Extensions that get treated as config (one piece, small anyway)
CONFIG_EXTENSIONS = {
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env", ".xml",
}

# Language extension → tuple of regex patterns for top-level definitions
# Patterns match the START of a line (^ anchor), so they only fire at
# the top level (not inside nested blocks). Keep them broad to be safe.
LANGUAGE_PATTERNS: dict[str, tuple[str, ...]] = {
    ".py": (
        r"^def\s+\w+",
        r"^async\s+def\s+\w+",
        r"^class\s+\w+",
        r"^@\w+",  # decorators — capture with the function below
    ),
    ".js": (
        r"^function\s+\w+",
        r"^async\s+function\s+\w+",
        r"^class\s+\w+",
        r"^export\s+(default\s+)?(function|class|const|let|var|async)\s+",
        r"^const\s+\w+\s*=\s*(async\s*)?\(",
        r"^const\s+\w+\s*=\s*(async\s*)?function",
    ),
    ".ts": (
        r"^function\s+\w+",
        r"^async\s+function\s+\w+",
        r"^class\s+\w+",
        r"^interface\s+\w+",
        r"^type\s+\w+",
        r"^enum\s+\w+",
        r"^export\s+(default\s+)?(function|class|const|let|var|async|interface|type|enum)\s+",
        r"^const\s+\w+\s*=\s*(async\s*)?\(",
    ),
    ".go": (
        r"^func\s+",
        r"^type\s+\w+",
        r"^var\s+",
        r"^const\s+",
    ),
    ".rs": (
        r"^fn\s+\w+",
        r"^pub\s+fn\s+\w+",
        r"^struct\s+\w+",
        r"^enum\s+\w+",
        r"^impl\s+",
        r"^trait\s+\w+",
    ),
    ".java": (
        r"^\s*(public|private|protected)?\s*(static\s+)?(class|interface|enum)\s+\w+",
        r"^\s*(public|private|protected)\s+.*\s+\w+\s*\(",
    ),
    ".rb": (
        r"^def\s+\w+",
        r"^class\s+\w+",
        r"^module\s+\w+",
    ),
}

# Aliases — treat these extensions like the mapped one
EXTENSION_ALIASES = {
    ".jsx": ".js",
    ".mjs": ".js",
    ".cjs": ".js",
    ".tsx": ".ts",
    ".mts": ".ts",
    ".cts": ".ts",
}


@dataclass
class Chunk:
    file_path: str
    chunk_type: str  # 'function' | 'class' | 'module' | 'readme' | 'file' | 'config'
    content: str
    language: str | None


# Chunk size targets. Under min → merge with previous. Over max → split anyway.
MIN_CHUNK_CHARS = 200
MAX_CHUNK_CHARS = 4000


def chunk_file(file_path: str, content: str) -> list[Chunk]:
    """
    Split a file into semantically meaningful chunks.
    """
    ext = PurePosixPath(file_path).suffix.lower()
    lang = ext[1:] if ext else None

    # Docs and configs: single chunk
    if ext in DOC_EXTENSIONS:
        chunk_type = "readme" if "readme" in PurePosixPath(file_path).name.lower() else "module"
        return [Chunk(file_path, chunk_type, content, lang)]

    if ext in CONFIG_EXTENSIONS:
        return [Chunk(file_path, "config", content, lang)]

    # Small files: single chunk regardless of language
    if len(content) < MIN_CHUNK_CHARS * 2:
        return [Chunk(file_path, "file", content, lang)]

    # Get language patterns
    normalized_ext = EXTENSION_ALIASES.get(ext, ext)
    patterns = LANGUAGE_PATTERNS.get(normalized_ext)

    if patterns is None:
        # Unknown language — fall back to size-based splitting
        return _split_by_size(file_path, content, lang)

    return _split_by_definitions(file_path, content, lang, patterns)


def _split_by_definitions(
    file_path: str, content: str, lang: str | None, patterns: tuple[str, ...]
) -> list[Chunk]:
    """
    Split content at lines matching any of the given patterns.
    Merges chunks smaller than MIN_CHUNK_CHARS with the previous chunk.
    """
    combined_pattern = re.compile("|".join(f"({p})" for p in patterns), re.MULTILINE)

    # Find all split points (start-of-line positions of definitions)
    split_indices = [0]
    for match in combined_pattern.finditer(content):
        # Only split at the start of a line
        if match.start() == 0 or content[match.start() - 1] == "\n":
            split_indices.append(match.start())
    split_indices.append(len(content))
    split_indices = sorted(set(split_indices))

    raw_chunks: list[str] = []
    for i in range(len(split_indices) - 1):
        segment = content[split_indices[i]:split_indices[i + 1]].strip()
        if segment:
            raw_chunks.append(segment)

    if not raw_chunks:
        return [Chunk(file_path, "file", content, lang)]

    # Merge tiny chunks into the previous one
    merged: list[str] = []
    for c in raw_chunks:
        if merged and len(c) < MIN_CHUNK_CHARS:
            merged[-1] = merged[-1] + "\n\n" + c
        elif merged and len(merged[-1]) < MIN_CHUNK_CHARS:
            merged[-1] = merged[-1] + "\n\n" + c
        else:
            merged.append(c)

    # Split any chunk that's too big
    final: list[str] = []
    for c in merged:
        if len(c) <= MAX_CHUNK_CHARS:
            final.append(c)
        else:
            final.extend(_hard_split(c, MAX_CHUNK_CHARS))

    return [
        Chunk(
            file_path=file_path,
            chunk_type=_infer_chunk_type(content_segment),
            content=content_segment,
            language=lang,
        )
        for content_segment in final
    ]


def _split_by_size(file_path: str, content: str, lang: str | None) -> list[Chunk]:
    """Fallback for unknown languages: split at paragraph breaks around target size."""
    pieces = _hard_split(content, MAX_CHUNK_CHARS)
    return [Chunk(file_path, "file", p, lang) for p in pieces]


def _hard_split(text: str, max_size: int) -> list[str]:
    """Split a string at newlines to stay under max_size."""
    if len(text) <= max_size:
        return [text]

    lines = text.split("\n")
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0

    for line in lines:
        line_size = len(line) + 1  # +1 for the newline
        if current_size + line_size > max_size and current:
            chunks.append("\n".join(current))
            current = [line]
            current_size = line_size
        else:
            current.append(line)
            current_size += line_size

    if current:
        chunks.append("\n".join(current))

    return chunks


def _infer_chunk_type(content: str) -> str:
    """Look at the first non-empty line to guess whether this is a class or function."""
    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("class ", "struct ", "interface ", "enum ", "trait ")):
            return "class"
        if stripped.startswith(("def ", "async def ", "function ", "fn ", "func ")):
            return "function"
        if stripped.startswith(("export class", "export default class")):
            return "class"
        if stripped.startswith(("export function", "export default function")):
            return "function"
        break
    return "file"
