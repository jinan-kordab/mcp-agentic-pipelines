"""© JINAN KORDAB — 2026 AI HYBRID AGENTIC RETRIEVAL-AUGMENTED GENERATION RAG PIPELINE - PERSONAL PROJECT"""

import hashlib
from typing import Tuple


def make_query_hash(query_tokens: Tuple[str, ...]) -> str:
    """SHA-256 hash of pipe-joined query tokens. Deterministic cache key for MemoryAgent."""
    joined = "|".join(query_tokens)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def make_document_hash(source_doc: str, source_page: int, source_position: int) -> str:
    """Unique hash for a document location (file + page + position). Used for citation deduplication."""
    combined = f"{source_doc}|{source_page}|{source_position}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


def make_session_key(session_id: str) -> str:
    """Normalized session key. Strips whitespace, lowercases, hashes."""
    normalized = session_id.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
