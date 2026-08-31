import random
import time

import logfire
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import settings


# ── Configuration ─────────────────────────────────────────────────────────────

BATCH_SIZE = 10
_GEMINI_DIM = 3072

_active_model = None


# ── Model initialisation ──────────────────────────────────────────────────────

def _init():
    """Initialise Gemini embedding model once per process."""
    global _active_model

    if _active_model is not None:
        return

    if not settings.GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. "
            "Gemini embeddings cannot be initialized."
        )

    _active_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2-preview",
        google_api_key=settings.GEMINI_API_KEY,
    )

    # Probe the API once so configuration/API failures happen immediately.
    try:
        _active_model.embed_query("probe")

        logfire.info(
            "Gemini embeddings ready "
            "(gemini-embedding-2-preview, 3072-dim)."
        )

    except Exception as e:
        _active_model = None

        logfire.error(
            f"Gemini embedding initialization failed: "
            f"{type(e).__name__}: {e}"
        )

        raise


# ── Public helpers ────────────────────────────────────────────────────────────

def get_embedding_dim() -> int:
    """Return the dimension of the active embedding model."""
    _init()
    return _GEMINI_DIM


# ── Batch embedding with retry ────────────────────────────────────────────────

def _embed_batch(batch: list[str]) -> list[list[float]]:
    """
    Embed one batch using Gemini.

    Only Gemini is supported.
    There is intentionally NO local fallback model.
    """

    _init()

    max_attempts = 6

    for attempt in range(max_attempts):
        try:
            embeddings = _active_model.embed_documents(batch)

            # Safety check: make sure Gemini returned the expected dimension.
            if embeddings:
                actual_dim = len(embeddings[0])

                if actual_dim != _GEMINI_DIM:
                    raise RuntimeError(
                        f"Gemini returned unexpected embedding dimension: "
                        f"{actual_dim}. Expected {_GEMINI_DIM}."
                    )

            return embeddings

        except Exception as e:
            err = str(e).lower()

            is_rate_limit = any(
                token in err
                for token in (
                    "429",
                    "rate",
                    "quota",
                    "resource_exhausted",
                )
            )

            if is_rate_limit and attempt < max_attempts - 1:
                # 5s → 10s → 20s → 40s → 60s
                wait = min(60, 5 * (2 ** attempt))

                # Small jitter prevents synchronized retries.
                wait += random.uniform(0, 2)

                logfire.warning(
                    f"Gemini rate limit hit — "
                    f"retrying in {wait:.1f}s "
                    f"(attempt {attempt + 1}/{max_attempts})."
                )

                time.sleep(wait)
                continue

            logfire.error(
                f"Gemini embedding failed after "
                f"{attempt + 1} attempt(s): {e}"
            )

            raise

    raise RuntimeError(
        "Gemini embedding failed after maximum retry attempts."
    )


# ── Public API ────────────────────────────────────────────────────────────────

def embed_query(query: str) -> list[float]:
    """Generate a single 3072-dimensional Gemini embedding."""
    _init()

    embedding = _active_model.embed_query(query)

    if len(embedding) != _GEMINI_DIM:
        raise RuntimeError(
            f"Gemini returned dimension {len(embedding)}, "
            f"expected {_GEMINI_DIM}."
        )

    return embedding


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate Gemini embeddings for multiple texts."""
    _init()

    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]

        with logfire.span(
            "Embed batch",
            model="gemini-embedding-2-preview",
            dimension=_GEMINI_DIM,
            start=i,
            size=len(batch),
        ):
            all_embeddings.extend(_embed_batch(batch))

    return all_embeddings