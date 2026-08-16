# RepoLens API

Backend for **RepoLens** — paste any public GitHub repo, get a plain-English
overview, an interactive module diagram, and a chat that has actually read
the code.

## Status

**Phase 1 complete.** This ships:
- FastAPI app skeleton with all endpoints
- Neon Postgres + pgvector database schema
- GitHub REST API client (tree fetching, file contents, no cloning)
- IP-based rate limiting (10 analyses / day / IP by default)
- Repo caching by `(owner, name, commit_sha)` — same commit = instant reuse
- Background job runner shape (pipeline stubs land in Phase 2)
- Full Pydantic request/response schemas

## Local setup

```bash
# 1. Create virtualenv
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate

# 2. Install deps
pip install -r requirements.txt

# 3. Copy env template and fill in keys
cp .env.example .env
# then edit .env with your GITHUB_TOKEN, GROQ_API_KEY, DATABASE_URL

# 4. Run
uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000/docs` for the auto-generated Swagger UI.
The `init_db()` call on startup creates all tables and enables the
`vector` extension on your Neon database.

## Test the skeleton

```bash
# Should return {"status":"ok"}
curl http://localhost:8000/health

# Kick off analysis (will currently fail cleanly since Phase 2 isn't built)
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/tiangolo/fastapi"}'

# Poll the job (use the id from the previous response)
curl http://localhost:8000/jobs/1
```

You should see the job move from `pending` to `failed` with the
"Phase 2 not implemented yet" message. That confirms the whole plumbing
works: URL parsing → GitHub API → cache lookup → rate limit → job row →
background task → status polling.

## Architecture

```
Client (Vercel)
      │
      ▼
FastAPI (Render)
      │
      ├─ GitHub REST API   (tree, contents)
      ├─ Groq              (Llama 3.1 70B for reasoning)  ← Phase 2
      ├─ sentence-transformers  (local embeddings)         ← Phase 2
      └─ Neon Postgres + pgvector
           ├─ repos          (cached overviews)
           ├─ chunks         (code + embeddings)
           ├─ rate_limits    (per-IP counters)
           └─ analysis_jobs  (progress polling)
```

## What's next

- **Phase 2:** the actual analysis pipeline — file selection prompt,
  tree-sitter chunking, embedding, LLM summary + module graph generation.
- **Phase 3:** chat endpoint with hybrid retrieval.
- **Phase 4:** Next.js frontend.
- **Phase 5:** Deploy to Render + Vercel.
