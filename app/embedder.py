"""
Embedding via sentence-transformers.

Model: all-MiniLM-L6-v2
  - 384-dim output (matches the vector column in models.py)
  - ~80MB on disk, runs on CPU in a few ms per chunk
  - Downloaded on first use, cached to disk after

Loaded lazily so the FastAPI app boots fast — we only pay the model
load cost the first time an analysis runs.
"""
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_model: "SentenceTransformer | None" = None


def _get_model() -> "SentenceTransformer":
    global _model
    if _model is None:
        # Import here so app startup doesn't pull in torch
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model %s (first-time download may take a minute)", MODEL_NAME)
        _model = SentenceTransformer(MODEL_NAME)
        logger.info("Embedding model loaded")
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts. Returns a list of 384-float vectors.
    Batch size handled internally by sentence-transformers.
    """
    if not texts:
        return []
    model = _get_model()
    # convert_to_numpy=True gives np arrays; we convert to plain lists for pgvector
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return [v.tolist() for v in vectors]


def embed_one(text: str) -> list[float]:
    """Embed a single text. Used by the chat endpoint (Phase 3)."""
    return embed_texts([text])[0]
