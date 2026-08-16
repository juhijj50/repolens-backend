# RepoLens — Backend

> Understand any public GitHub repo in 5 minutes. Paste a URL, get a
> plain-English overview, an interactive module diagram, and a chat that
> has actually read the code — with citations back to the source.

**Live demo:** [repolens-frontend-three.vercel.app](https://repolens-frontend-three.vercel.app)
**Frontend repo:** [juhijj50/repolens-frontend](https://github.com/juhijj50/repolens-frontend)

---

## What this repo is

This is the backend service that powers RepoLens. It:

- Fetches a GitHub repo's file tree without cloning
- Uses an LLM to pick the ~15 most architecturally important files
- Chunks, embeds, and stores those files in Postgres with pgvector
- Generates a structured overview and module dependency graph
- Answers questions about the code via hybrid semantic + keyword retrieval

Every response cites the exact files it drew from, at the exact commit
that was analyzed.

---

## Stack

| Piece | Choice |
|-------|--------|
| Web framework | FastAPI |
| Database | Neon (Postgres) + pgvector |
| LLM | Groq (`openai/gpt-oss-120b`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, CPU) |
| GitHub access | REST API (tree + contents, no cloning) |
| Deployment | Render (free tier) |

---

## Interesting problems I solved

This section is the point of reading this README.

### Job queue instead of blocking requests

Analysis takes 30–90 seconds. HTTP requests have timeouts at every hop
(browser, CDN, proxy). If the analysis takes longer than the shortest
timeout in the chain, the connection dies and the user sees an error
even though the backend is still working.

Solution: `POST /analyze` returns a `job_id` in under a second, the
actual work runs in a background task, and the frontend polls
`GET /jobs/{id}` for progress. This also unlocks a live progress bar
instead of a blank spinner — much better UX.

### Caching by commit SHA, not by URL

The same repo URL at different commits should produce different
analyses. The same repo URL at the same commit should be an instant
cache hit for anyone who visits. The cache key is `(owner, name,
commit_sha)`, resolved from GitHub's API on every lookup. Popular
repos cost me zero LLM tokens after the first analysis.

Extended for the frontend: `GET /repos/by-url` compares the cached
commit SHA against the current HEAD and returns one of four states —
`up_to_date`, `stale`, `not_found`, `in_progress` — so the frontend
can decide whether to render immediately, offer a re-analyze button,
kick off a new job, or hook into an existing job's progress.

### No cloning — GitHub API only

Cloning a repo would:
- Eat disk (Render's free tier gives you ~512MB ephemeral disk)
- Download the entire git history when I only need ~15 files
- Take 10-30 seconds just for the network transfer

Instead I use `GET /repos/{owner}/{repo}/git/trees/{sha}?recursive=1`
to fetch the full file tree in one call, then parallel-fetch only the
selected files via the contents API. Sub-second, disk-free.

### LLM file selection with heuristic fallback

Given a tree of thousands of files, an LLM decides which ~15 matter.
This is genuinely non-trivial reasoning — it has to distinguish
entry points, primary source modules, and key config from tests,
examples, and vendored code.

**But** if the LLM call fails (rate limit, malformed JSON, API outage),
a deterministic fallback picks by filename importance (README, main.py,
package.json, etc.) so the pipeline never dies at the first LLM call.
Graceful degradation is a real production pattern.

### Fixed-total-budget prompt construction

An early bug: analyzing `psf/requests` blew past Groq's 8000 tokens/min
free tier limit. I was capping each file at 3000 chars — but 15 files
× 3000 chars = ~12000 tokens, over the limit.

Fix: set a total character budget (20000 chars) and divide across
however many files were selected. Prompt size becomes bounded regardless
of repo size. This is a load-shedding principle: predictable resource
use regardless of input size.

Used again in the chat endpoint for the same reason.

### Hybrid retrieval (semantic + keyword)

Pure vector search misses obvious things. Question: "where's the auth
code?" Repo: has a file literally named `auth.py`. Pure semantic search
might rank other chunks higher because "auth" as a concept lives in
many files. Keyword search catches the direct filename match.

Solution: run both in parallel, weight path matches 2x content matches
(a file named after the query term is usually the answer), merge and
dedupe. Best of both.

### Grounding via prompt constraints, not fine-tuning

The chat can't fine-tune away hallucinations. Instead, an aggressive
system prompt requires the model to either ground every claim in the
retrieved snippets or say the exact phrase "I don't have enough
context…" and stop. Combined with citation requirements, the model
either answers with references or admits it can't.

Tested: asking "how does the payment system work" on a repo with no
payment code returns the canned refusal, not a made-up answer.

### pgvector over a dedicated vector DB

I use Postgres (via Neon) for everything: repos, chunks, embeddings,
rate limits, job state. Using a separate vector DB like Pinecone would
create a two-system problem — atomic writes across systems are hard,
so you get orphaned records when one write succeeds and the other
fails. With pgvector, one transaction inserts chunks and embeddings
together, or nothing at all.

For 200k vectors (RepoLens' expected demo lifetime) pgvector is
faster than the network round-trip to a dedicated service anyway.

### Idle cost as a design constraint

Free-tier economics: Render's free web service sleeps after 15 min
of inactivity. Neon's compute scales to zero. Vercel's edge is
free at low volume. Total idle cost of running RepoLens: $0.

Cold-start trade-off: ~30s to wake Render, ~500ms to wake Neon.
Mitigated with a cron ping every 10 min during typical demo hours.

---

## Architecture

```
Browser
  │
  ▼
Vercel (Next.js frontend)
  │
  ▼
Render (FastAPI, this repo)
  ├─→ GitHub REST API      (tree + contents)
  ├─→ Groq                 (llama / gpt-oss for reasoning)
  ├─→ sentence-transformers (local CPU embeddings)
  └─→ Neon Postgres + pgvector
        ├─ repos           (cached overviews + module graphs)
        ├─ chunks          (code with 384-dim embeddings)
        ├─ analysis_jobs   (progress tracking, 24h TTL)
        └─ rate_limits     (per-IP daily / hourly caps)
```

### Analysis pipeline

```
POST /analyze
  → parse URL, look up cache, guard against duplicate in-flight
  → return job_id immediately

background task:
  1. fetching   — GitHub tree API (one call)
  2. selecting  — LLM picks ~15 files from the tree
  3. fetching   — parallel-fetch selected files (semaphore=5)
  4. chunking   — split by top-level definitions (regex per language)
  5. embedding  — sentence-transformers on all chunks
  6. analyzing  — LLM generates overview
  7. analyzing  — LLM generates module graph
  8. persist all in one transaction
  → job.status = "ready"

GET /jobs/{id} (polled every 1.5s by frontend)
GET /repos/{id} (once ready)
```

### Chat pipeline

```
POST /repos/{id}/chat
  → rate limit (30/hour/IP)
  → embed the question
  → semantic search (top 5 by cosine similarity)  ┐
  → keyword search  (top 3, path-weighted)        ┴─→ merge, dedupe
  → build prompt with cited snippets, bounded to 12k chars
  → Groq call with grounding constraints
  → return answer + list of cited file paths
```

---

## API endpoints

| Method | Path | What it does |
|--------|------|--------------|
| `GET` | `/health` | Liveness check (used by keep-warm cron) |
| `POST` | `/analyze` | Kick off analysis for a repo URL |
| `GET` | `/jobs/{id}` | Poll job status + stage message |
| `GET` | `/repos` | List recently analyzed repos |
| `GET` | `/repos/by-url?url=...` | Smart lookup: cached / stale / in-progress / not-found |
| `GET` | `/repos/{id}` | Get full overview + module graph |
| `POST` | `/repos/{id}/chat` | Ask a question about the repo |

Swagger docs at `/docs`.

---

## Run locally

Prerequisites: Python 3.13, a Neon database, a Groq API key, a GitHub
personal access token with `public_repo` scope.

```bash
# 1. Install deps
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# then fill in GITHUB_TOKEN, GROQ_API_KEY, DATABASE_URL

# 3. Run
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/docs for the interactive API.

The first analysis takes ~90 seconds because `sentence-transformers`
downloads its ~80MB embedding model. Subsequent analyses take 10–30
seconds; cached hits are instant.

---

## Environment variables

| Variable | Purpose |
|----------|---------|
| `GITHUB_TOKEN` | Personal access token, `public_repo` scope |
| `GROQ_API_KEY` | From console.groq.com |
| `DATABASE_URL` | Neon Postgres URL with `?sslmode=require` |
| `CORS_ORIGINS` | Comma-separated list of allowed frontend origins |
| `MAX_ANALYSES_PER_IP_PER_DAY` | Default 10 |
| `MAX_REPO_SIZE_MB` | Default 50 |
| `MAX_REPO_FILES` | Default 2000 |

---

## What I'd do next

- **Streaming chat responses** — Groq supports SSE; the whole stack
  is ready for it, I just haven't wired the frontend up.
- **Tree-sitter chunking** — the regex chunker works but has edge
  cases (nested classes, decorators on separate lines). Tree-sitter
  would be more precise. Kept regex for now because it works on every
  language with zero native compilation.
- **Public analyses feed** — the `GET /repos` endpoint already returns
  recently analyzed repos; a real frontend page for browsing them
  would be a nice touch.
- **Analysis re-runs on-demand** — currently cached forever; a "force
  re-analyze" query param would let people refresh their own analyses.

---

## Trade-offs worth knowing

- **Regex chunking over tree-sitter**: portable across languages, no
  native compilation on Windows/Python 3.13, but occasional imperfect
  boundaries. Embeddings are robust to this.
- **50MB repo size cap**: sized empirically. Repos above this take
  too long to select + embed on free-tier hardware.
- **15 file selection cap**: LLM context and prompt-size trade-off.
  Very large monorepos lose fidelity.
- **Chat has no streaming yet**: full response is generated then
  returned. Feels slower than it needs to.
- **No auth**: intentional — this is a public read-only tool. Rate
  limiting is per-IP.
- **Chat history is client-side only**: no persistence across sessions.
  A design choice for demo simplicity; adding server-side history
  would need thought about user identity.

---

## Built with

FastAPI · Neon · pgvector · Groq · sentence-transformers · Render

Related: [Frontend repo](https://github.com/juhijj50/repolens-frontend)