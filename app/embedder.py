"""
Embedding via fastembed (ONNX runtime, no PyTorch).

Model: all-MiniLM-L6-v2
  - 384-dim output (matches the vector column in models.py)
  - ~90MB on disk, runs on CPU in a few ms per chunk
  - Same weights as the sentence-transformers model, so vectors are
    identical to ones stored before the switch
  - PyTorch would need ~500MB RAM, more than Render's free tier allows
  - Cached in the project dir (not /tmp) so the build step can pre-download
    it — run `python -m app.embedder` at build time. Otherwise the first
    analysis after every boot pays a ~90MB download inside "Understanding..."

Loaded lazily so the FastAPI app boots fast — we only pay the model
load cost the first time an analysis runs.
"""
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
CACHE_DIR = Path(__file__).resolve().parent.parent / ".fastembed_cache"

_model: "TextEmbedding | None" = None


def _get_model() -> "TextEmbedding":
    global _model
    if _model is None:
        # Import here so app startup doesn't pull in onnxruntime
        from fastembed import TextEmbedding
        logger.info("Loading embedding model %s (first-time download may take a minute)", MODEL_NAME)
        # Tuned for Render's 512MB limit: onnxruntime's CPU memory arena
        # keeps a high-water-mark pool that never shrinks, and each extra
        # thread holds its own scratch buffers. Both pushed us past 512MB.
        _model = TextEmbedding(
            MODEL_NAME,
            cache_dir=str(CACHE_DIR),
            threads=1,
            enable_cpu_mem_arena=False,
        )
        logger.info("Embedding model loaded")
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts. Returns a list of 384-float vectors.
    Batched internally by fastembed. Small batches keep peak memory low;
    throughput barely matters at our chunk counts.
    """
    if not texts:
        return []
    model = _get_model()
    # embed() yields np arrays; we convert to plain lists for pgvector
    return [v.tolist() for v in model.embed(texts, batch_size=8)]


def embed_one(text: str) -> list[float]:
    """Embed a single text. Used by the chat endpoint (Phase 3)."""
    return embed_texts([text])[0]


if __name__ == "__main__":
    # Build-time warmup: downloads the model into CACHE_DIR
    logging.basicConfig(level=logging.INFO)
    embed_one("warmup")
