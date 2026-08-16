"""
Chat endpoint logic.

Given a user question about a specific analyzed repo, we:
  1. Embed the question using the same model that embedded the chunks
  2. Run TWO retrievals in parallel:
       a. Semantic — top 5 chunks by cosine similarity
       b. Keyword — top 3 chunks where file path or content contains
          significant words from the question
  3. Merge, dedupe by chunk id, cap at ~8 chunks total
  4. Build a context prompt with the retrieved chunks (labeled by file path)
  5. Ask Groq to answer, citing which files it used

Why hybrid retrieval?
  Pure semantic can miss the obvious. If the question is "where's the auth
  code" and there's an `auth.py` file, semantic search might rank other
  chunks higher (since "auth" as a concept lives in many files). Keyword
  search catches the direct filename match. Best of both.
"""
import logging
import re
from dataclasses import dataclass

from sqlalchemy import func, or_, text
from sqlalchemy.orm import Session

from app.embedder import embed_one
from app.llm import call_text
from app.models import Chunk

logger = logging.getLogger(__name__)


# How many chunks each retrieval strategy contributes
SEMANTIC_TOP_K = 5
KEYWORD_TOP_K = 3

# Total cap after merging — LLM context stays reasonable
MAX_CONTEXT_CHUNKS = 8

# Total character budget for context sent to LLM. Same load-shedding
# principle as _generate_overview in analyzer.py — bounded prompt size
# regardless of how much retrieval returned.
MAX_CONTEXT_CHARS = 12000


# Stop words we ignore when extracting keywords from questions.
# Kept small on purpose — real signal words like "route", "model", "auth"
# should always come through.
STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "can", "shall",
    "i", "you", "he", "she", "it", "we", "they", "them", "their", "this",
    "that", "these", "those", "what", "which", "who", "whom", "whose",
    "where", "when", "why", "how",
    "and", "or", "but", "if", "then", "else", "as", "of", "at", "by",
    "for", "with", "about", "in", "on", "to", "from", "into", "through",
    "me", "my", "please", "tell", "show", "explain", "help",
    "get", "make", "use", "used", "using", "there", "here",
    "code", "codebase", "repo", "repository", "project", "file", "files",
})


@dataclass
class RetrievedChunk:
    id: int
    file_path: str
    content: str
    language: str | None
    score: float  # higher = more relevant; scale differs per source
    source: str   # 'semantic' | 'keyword'


def extract_keywords(question: str) -> list[str]:
    """
    Pull salient keywords from a natural-language question.
    Split on non-word chars, lowercase, drop stop words and single chars.
    """
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", question.lower())
    keywords = [t for t in tokens if len(t) > 1 and t not in STOP_WORDS]
    # Dedupe while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for k in keywords:
        if k not in seen:
            seen.add(k)
            unique.append(k)
    return unique[:10]  # cap so keyword SQL doesn't blow up


def semantic_search(
    db: Session, repo_id: int, query_vector: list[float], k: int = SEMANTIC_TOP_K
) -> list[RetrievedChunk]:
    """
    Cosine-similarity search using pgvector's <=> operator.
    <=> returns *distance* (0 = identical, 2 = opposite), so we sort ASC
    and convert to a similarity score for ranking downstream.
    """
    rows = (
        db.query(
            Chunk.id,
            Chunk.file_path,
            Chunk.content,
            Chunk.language,
            Chunk.embedding.cosine_distance(query_vector).label("distance"),
        )
        .filter(Chunk.repo_id == repo_id)
        .order_by("distance")
        .limit(k)
        .all()
    )
    return [
        RetrievedChunk(
            id=r.id,
            file_path=r.file_path,
            content=r.content,
            language=r.language,
            score=1.0 - float(r.distance),  # convert distance → similarity
            source="semantic",
        )
        for r in rows
    ]


def keyword_search(
    db: Session, repo_id: int, keywords: list[str], k: int = KEYWORD_TOP_K
) -> list[RetrievedChunk]:
    """
    Simple keyword match: score = weighted count of matches across
    (file_path + content). File path matches score higher because a file
    literally named after the query term is usually the answer.

    Not TF-IDF; not BM25. Deliberately simple — it just needs to catch
    the obvious "there's a file literally named after what they asked".
    """
    if not keywords:
        return []

    from sqlalchemy import case

    # One CASE per keyword: 2 points for path match, 1 for content match.
    # Sum them into a single scoring expression.
    score_terms = [
        case(
            (Chunk.file_path.ilike(f"%{kw}%"), 2),
            (Chunk.content.ilike(f"%{kw}%"), 1),
            else_=0,
        )
        for kw in keywords
    ]
    score_sum = sum(score_terms)

    rows = (
        db.query(
            Chunk.id,
            Chunk.file_path,
            Chunk.content,
            Chunk.language,
            score_sum.label("score"),
        )
        .filter(Chunk.repo_id == repo_id)
        .filter(score_sum > 0)
        .order_by(score_sum.desc(), func.length(Chunk.file_path).asc())
        .limit(k)
        .all()
    )

    return [
        RetrievedChunk(
            id=r.id,
            file_path=r.file_path,
            content=r.content,
            language=r.language,
            score=float(r.score),
            source="keyword",
        )
        for r in rows
    ]


def merge_results(
    semantic: list[RetrievedChunk],
    keyword: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """
    Dedupe by chunk id. When a chunk appears in both, prefer the semantic
    version but tag the source as 'both' via score bump.
    Order: put keyword-only hits FIRST (they're likely exact filename
    matches — the developer probably wants to see them), then semantic.
    """
    seen: dict[int, RetrievedChunk] = {}

    # Keyword hits go in first — they're often the most obvious answer
    for c in keyword:
        seen[c.id] = c

    # Semantic hits fill in the rest, but don't overwrite keyword hits
    for c in semantic:
        if c.id not in seen:
            seen[c.id] = c

    merged = list(seen.values())
    return merged[:MAX_CONTEXT_CHUNKS]


CHAT_SYSTEM_PROMPT = """You are a code assistant helping a developer understand a specific repository.

You will be given several code and documentation snippets from the repo, each labeled with its file path. Answer the user's question based ONLY on those snippets.

Absolute rules:
- If the answer is not in the provided snippets, say exactly this and STOP: "I don't have enough context from the analyzed files to answer that. This might be in a file that wasn't among the 15 selected for analysis, or the feature might not exist in this repo."
- Never guess, invent, or reason about code you were not shown. Even if you know the library or pattern from other repos, do not describe it as if it's in THIS repo unless the snippets show it.
- Ground every claim in the provided snippets. Cite the file paths inline in your answer, e.g. "In `app/auth.py`, the `login` function...".
- Be concise. Two to five short paragraphs, or a short list.
- Prefer showing small code excerpts (5-10 lines) over describing them abstractly when a snippet directly answers the question.
- Don't apologize for missing context. Just state what you can answer from what you have, or use the exact "don't have enough context" phrasing above.
"""


def build_context_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    history: list[dict],
) -> tuple[str, str]:
    """
    Build (system_prompt, user_prompt) for the chat LLM call.
    Enforces a total character budget for the retrieved chunks so the
    prompt stays bounded regardless of chunk sizes.
    """
    # Split budget evenly across chunks — same principle as _generate_overview
    if chunks:
        per_chunk_budget = MAX_CONTEXT_CHARS // len(chunks)
    else:
        per_chunk_budget = 0

    context_parts: list[str] = []
    for c in chunks:
        content = c.content
        if len(content) > per_chunk_budget:
            # Try to cut at a newline to avoid breaking mid-statement
            cutoff = content.rfind("\n", 0, per_chunk_budget)
            if cutoff < per_chunk_budget // 2:
                cutoff = per_chunk_budget
            content = content[:cutoff] + "\n[...truncated]"
        context_parts.append(f"=== {c.file_path} ===\n{content}")

    context_block = "\n\n".join(context_parts) if context_parts else "(No relevant snippets found in the analyzed files.)"

    # Include recent history if any (last 3 exchanges, keeps prompt tight)
    history_block = ""
    if history:
        recent = history[-6:]  # 3 user + 3 assistant turns
        history_lines = [f"{h['role'].upper()}: {h['content']}" for h in recent]
        history_block = "Recent conversation:\n" + "\n".join(history_lines) + "\n\n"

    user_prompt = (
        f"{history_block}"
        f"Relevant code snippets from the repo:\n\n{context_block}\n\n"
        f"Question: {question}"
    )

    return CHAT_SYSTEM_PROMPT, user_prompt


def answer_question(
    db: Session,
    repo_id: int,
    question: str,
    history: list[dict],
) -> tuple[str, list[str]]:
    """
    Top-level entrypoint called by the /repos/{id}/chat endpoint.
    Returns (answer_text, list_of_cited_file_paths).
    """
    # Embed the question
    query_vector = embed_one(question)

    # Extract keywords for the keyword search
    keywords = extract_keywords(question)
    logger.info("Chat query — repo=%s keywords=%s", repo_id, keywords)

    # Run both retrievals
    semantic = semantic_search(db, repo_id, query_vector)
    keyword = keyword_search(db, repo_id, keywords)

    # Merge
    chunks = merge_results(semantic, keyword)
    logger.info("Retrieved %d semantic + %d keyword → %d after merge",
                len(semantic), len(keyword), len(chunks))

    # Build prompt and call LLM
    system_prompt, user_prompt = build_context_prompt(question, chunks, history)
    answer = call_text(system_prompt, user_prompt, max_tokens=1024, temperature=0.3)

    # Cited files = unique file paths from the chunks we sent
    cited = list(dict.fromkeys(c.file_path for c in chunks))

    return answer, cited
