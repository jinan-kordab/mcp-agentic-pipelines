"""© JINAN KORDAB — 2026 AI HYBRID AGENTIC RETRIEVAL-AUGMENTED GENERATION RAG PIPELINE - PERSONAL PROJECT"""

import math
from typing import Dict, List, Tuple
from collections import defaultdict


class PMIScorer:
    """Computes PMI-based relevance scores using incremental corpus token statistics."""

    def __init__(self) -> None:
        self.token_count: Dict[str, int] = defaultdict(int)
        self.pair_count: Dict[Tuple[str, str], int] = defaultdict(int)
        self.total_tokens: int = 0

    def ingest_tokens(self, tokens: List[str]) -> None:
        """Update corpus statistics with a sequence of stemmed tokens from one multi-token."""
        for i, t in enumerate(tokens):
            self.token_count[t] += 1
            self.total_tokens += 1
            for j in range(i + 1, len(tokens)):
                pair = (t, tokens[j]) if t <= tokens[j] else (tokens[j], t)
                self.pair_count[pair] += 1

    def score(self, query_tokens: List[str], chunk_tokens: List[str]) -> float:
        """Average PMI across all (query, chunk) token pairs. Higher = more relevant."""
        if not query_tokens or not chunk_tokens:
            return 0.0
        pmi_values: List[float] = []
        for q in query_tokens:
            for c in chunk_tokens:
                pair = (q, c) if q <= c else (c, q)
                joint = self.pair_count.get(pair, 0)
                if joint == 0:
                    continue
                p_q = self.token_count.get(q, 0) / max(1, self.total_tokens)
                p_c = self.token_count.get(c, 0) / max(1, self.total_tokens)
                p_joint = joint / max(1, self.total_tokens)
                if p_q > 0 and p_c > 0:
                    pmi = math.log2(p_joint / (p_q * p_c))
                    pmi_values.append(pmi)
        return sum(pmi_values) / len(pmi_values) if pmi_values else 0.0

    def normalize_score(self, raw_score: float, max_observed: float = 10.0) -> float:
        """Clip and normalize to [0, 1]."""
        return min(max(raw_score, 0.0), max_observed) / max_observed


# Global singleton — shared across index, search, and upload
_pmi_scorer: PMIScorer = PMIScorer()


def get_pmi_scorer() -> PMIScorer:
    return _pmi_scorer
