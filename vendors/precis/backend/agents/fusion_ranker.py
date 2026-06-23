"""© JINAN KORDAB — 2026 AI HYBRID AGENTIC RETRIEVAL-AUGMENTED GENERATION RAG PIPELINE - PERSONAL PROJECT

Implements RRF (Reciprocal Rank Fusion): score = Σ 1/(k + rank_i)
where k=60 is the standard constant, and rank_i is the result's rank in each source.

Produces a single ranked list from multiple retrieval backends.
"""

from typing import Any, Dict, List


class FusionRanker:
    """Combines results from multiple retrieval engines using RRF."""

    def __init__(self, k: int = 60) -> None:
        self.k = k  # RRF constant

    def fuse(self, sources: Dict[str, List[Dict[str, Any]]],
             top_k: int = 15) -> List[Dict[str, Any]]:
        """Fuse multiple ranked result lists into one.

        Args:
            sources: {"hash": [...], "vector": [...]} — each list pre-sorted by score
            top_k: Max results to return
        """
        # Assign RRF scores
        fused: Dict[str, Dict[str, Any]] = {}  # key = text[:100]
        
        for source_name, results in sources.items():
            for rank, item in enumerate(results):
                key = item.get("text", "")[:100]  # Dedup key
                rrf_score = 1.0 / (self.k + rank + 1)
                
                if key in fused:
                    fused[key]["rrf_score"] += rrf_score
                    fused[key]["sources"].add(source_name)
                    # Keep the higher original score
                    if item.get("score", 0) > fused[key].get("original_score", 0):
                        fused[key]["original_score"] = item.get("score", 0)
                        fused[key]["match_type"] = item.get("match_type", "")
                        fused[key]["source"] = item.get("source", "")
                        fused[key]["page"] = item.get("page", 1)
                else:
                    fused[key] = {
                        "text": item.get("text", ""),
                        "source": item.get("source", ""),
                        "page": item.get("page", 1),
                        "rrf_score": rrf_score,
                        "original_score": item.get("score", 0),
                        "match_type": item.get("match_type", ""),
                        "sources": {source_name},
                    }
        
        # Sort by RRF score, return top_k
        ranked = sorted(fused.values(), key=lambda x: -x["rrf_score"])[:top_k]
        
        # Normalize scores to 0-1
        if ranked:
            max_rrf = ranked[0]["rrf_score"]
            for item in ranked:
                item["score"] = round(item["rrf_score"] / max_rrf, 3) if max_rrf > 0 else 0
                item["match_type"] = "fusion:" + "+".join(sorted(item["sources"]))
        
        return ranked
