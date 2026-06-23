"""© JINAN KORDAB — 2026 AI HYBRID AGENTIC RETRIEVAL-AUGMENTED GENERATION RAG PIPELINE - PERSONAL PROJECT

Adds embedding-based retrieval alongside the hash index for multi-source fusion.
Uses all-MiniLM-L6-v2 (80MB model, 384-dim vectors) — runs locally, no API calls.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class VectorIndex:
    """Semantic document retrieval using FAISS in-memory vector index.

    Complements the hash index by finding semantically similar content
    that doesn't share exact words with the query.

    Parameters
    ----------
    model_name : str
        HuggingFace sentence-transformers model name.
    encode_batch_size : int
        Batch size for model.encode().  Larger values (64–128) encode
        many chunks faster on CPU; reduce to 8–16 if GPU OOM occurs.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 encode_batch_size: int = 64) -> None:
        if not HAS_FAISS:
            raise ImportError("faiss-cpu and sentence-transformers required: pip install faiss-cpu sentence-transformers")
        
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()  # 384
        self.encode_batch_size = encode_batch_size
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine with normalized vectors)
        self.chunks: List[Dict[str, Any]] = []  # {"text": ..., "source": ..., "page": ...}
        self._chunk_vectors: Optional[np.ndarray] = None

    def index_text(self, text: str, source: str = "unknown",
                   chunk_size: int = 300, chunk_overlap: int = 50) -> int:
        """Split text into overlapping chunks and index as vectors. Returns chunk count."""
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append({"text": chunk, "source": source, "page": 1})
            start += chunk_size - chunk_overlap
        
        if not chunks:
            return 0
        
        # Encode in batches for throughput — sentence-transformers uses an
        # internal batch loop; a larger batch_size reduces Python→C round-trips.
        vectors = self.model.encode(
            [c["text"] for c in chunks],
            batch_size=self.encode_batch_size,
            show_progress_bar=False,
        )
        # Normalize for cosine similarity (inner product on unit vectors = cosine)
        faiss.normalize_L2(vectors)
        self.index.add(vectors.astype(np.float32))
        self.chunks.extend(chunks)
        return len(chunks)

    def search(self, query: str, top_k: int = 10,
               source_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Search for semantically similar chunks. Returns scored results.

        Parameters
        ----------
        source_filter : Optional[List[str]]
            If provided, only return chunks from these source documents
            (case-insensitive basename matching).
        """
        if self.index.ntotal == 0:
            return []
        
        query_vec = self.model.encode([query], show_progress_bar=False)
        faiss.normalize_L2(query_vec)
        # Fetch more candidates than needed so filtering doesn't starve results
        fetch_k = min(top_k * 3, self.index.ntotal) if source_filter else min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vec.astype(np.float32), fetch_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                chunk = self.chunks[idx]
                # ── Document-scope filter ────────────────────────
                if source_filter:
                    import os
                    filter_set = set()
                    for f in source_filter:
                        f = str(f).lower().strip()
                        f = os.path.basename(f)
                        if f:
                            filter_set.add(f)
                    if filter_set and os.path.basename(chunk.get("source", "").lower().strip()) not in filter_set:
                        continue
                results.append({
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "page": chunk.get("page", 1),
                    "score": round(float(score), 3),  # Cosine similarity, 0-1
                    "match_type": "semantic",
                })
                if len(results) >= top_k:
                    break
        return results

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_vectors": int(self.index.ntotal),
            "total_chunks": len(self.chunks),
            "dimension": self.dimension,
        }
