"""
Semantic search over the local catalog index (see catalog_dataset.py for
why this index exists at all — data.gov.my's real API has no search
endpoint).

Design: embed_texts() is the ONLY function here that touches the network
(OpenRouter's /embeddings endpoint). Every other function is pure/testable
without mocking anything beyond that one boundary — same pattern used
throughout ai_service.py and catalog_sync.py in this codebase.

No pgvector, no vector DB — at data.gov.my's catalog scale (hundreds of
datasets, not millions) a brute-force in-memory cosine similarity pass
over numpy arrays is effectively instant and avoids depending on a
Postgres extension that may not even be available on the hosting
provider.
"""
import json

import numpy as np
from sqlalchemy.orm import Session

from app.models.catalog_dataset import CatalogDataset
from app.services.ai_service import client

# Reuses the same OpenRouter client + API key already configured for chat
# calls in ai_service.py — OpenRouter's /embeddings endpoint is OpenAI-
# compatible, same as /chat/completions, so no new client or credential is
# needed.
EMBEDDING_MODEL = "openai/text-embedding-3-small"

# How many texts to send in a single embeddings API call. Batching
# matters here specifically because the catalog can be hundreds of rows —
# one API call per row would be needlessly slow and chatty.
DEFAULT_BATCH_SIZE = 20


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    The one network call in this module. Returns one embedding vector per
    input text, in the same order as `texts`.
    """
    if not texts:
        return []

    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def build_searchable_text(row: CatalogDataset) -> str:
    """
    What actually gets embedded for a catalog row. Combines the fields a
    user's free-text query is most likely to conceptually overlap with —
    title, category, subcategory, and source agency. Deliberately leaves
    out geography/demography/frequency: those are filters a user might
    apply, not things they'd phrase a search query around.
    """
    parts = [row.title_en, row.category_en, row.subcategory_en, row.source]
    return " — ".join(p for p in parts if p)


def generate_catalog_embeddings(db: Session, force: bool = False, batch_size: int = DEFAULT_BATCH_SIZE) -> dict:
    """
    Embeds every catalog row that doesn't have an embedding yet (or every
    row, if force=True — e.g. after switching embedding models). Batches
    API calls rather than one-per-row.

    Returns {"embedded": int, "skipped": int, "total": int}.
    """
    query = db.query(CatalogDataset)
    if not force:
        query = query.filter(CatalogDataset.embedding.is_(None))

    rows_to_embed = query.all()
    total_rows = db.query(CatalogDataset).count()
    skipped = total_rows - len(rows_to_embed)

    embedded_count = 0
    for i in range(0, len(rows_to_embed), batch_size):
        batch = rows_to_embed[i:i + batch_size]
        texts = [build_searchable_text(row) for row in batch]
        vectors = embed_texts(texts)

        for row, vector in zip(batch, vectors):
            row.embedding = json.dumps(vector)
            embedded_count += 1

        db.commit()

    return {
        "embedded": embedded_count,
        "skipped": skipped,
        "total": total_rows,
    }


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)

    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


def search_catalog(db: Session, query: str, top_k: int = 5) -> list[dict]:
    """
    Embeds the free-text query and ranks every embedded catalog row by
    cosine similarity. Rows with no embedding yet (embedding IS NULL —
    e.g. added by a sync but not yet processed by
    generate_catalog_embeddings) are excluded rather than erroring, since
    a partially-embedded catalog is an expected, normal state, not a
    failure.

    Returns a list of {"dataset": CatalogDataset, "score": float},
    sorted by score descending, longest list length top_k.
    """
    query_vector = embed_texts([query])[0]

    candidates = db.query(CatalogDataset).filter(CatalogDataset.embedding.isnot(None)).all()

    scored = []
    for row in candidates:
        row_vector = json.loads(row.embedding)
        score = _cosine_similarity(query_vector, row_vector)
        scored.append({"dataset": row, "score": score})

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]
