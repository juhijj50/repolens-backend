"""
Database connection and session management.
Uses SQLAlchemy 2.0 style with Neon Postgres + pgvector.
"""
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings


# pool_pre_ping catches Neon's idle-connection drops (Neon suspends idle computes)
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for request-scoped sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def db_session() -> Generator[Session, None, None]:
    """For use outside request context (e.g. background tasks)."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """
    Enable pgvector extension, create tables, and set up the vector index.
    Called once at app startup. Idempotent — safe to run every boot.
    """
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    # Import models so they're registered with Base.metadata
    from app import models  # noqa: F401
    Base.metadata.create_all(bind=engine)

    # ivfflat index for fast approximate nearest-neighbor search on
    # chunk embeddings. Requires pgvector's `vector_cosine_ops` operator
    # class since our similarity metric is cosine.
    #
    # lists=100 is a middle-of-the-road choice; the pgvector docs suggest
    # rows/1000 for small tables. We can rebuild with a larger `lists`
    # value once we have millions of chunks.
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_chunks_embedding_cosine
            ON chunks USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """))
        conn.commit()
