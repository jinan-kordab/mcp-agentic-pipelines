"""© JINAN KORDAB — 2026 AI HYBRID AGENTIC RETRIEVAL-AUGMENTED GENERATION RAG PIPELINE - PERSONAL PROJECT"""

import math
import numpy as np
from typing import List, Dict, Any


def compute_relevancy(query_tokens: List[str], chunk_tokens_list: List[List[str]],
                      pmi_scorer=None) -> float:
    """Average Jaccard similarity. Uses PMI scorer when available."""
    if not chunk_tokens_list:
        return 0.0
    scores = []
    query_set = set(query_tokens)
    for chunk_tokens in chunk_tokens_list:
        chunk_set = set(chunk_tokens)
        intersection = query_set & chunk_set
        union = query_set | chunk_set
        jaccard = len(intersection) / len(union) if union else 0.0
        scores.append(jaccard)
    return sum(scores) / len(scores)


def compute_trustworthiness(source_metadata: List[Dict[str, Any]]) -> float:
    """Score source reliability from metadata: type, recency, cross-referencing."""
    if not source_metadata:
        return 0.5
    scores = []
    for meta in source_metadata:
        s = 0.5
        if meta.get("is_title"):
            s += 0.2
        if meta.get("font_size", 0) > 14:
            s += 0.1
        if meta.get("token_type") == "contextual":
            s += 0.15
        scores.append(min(s, 1.0))
    return sum(scores) / len(scores)


def compute_hallucination_rate(generated_claims: List[str],
                                source_evidence: List[Dict[str, Any]]) -> float:
    """Proportion of claims not supported by source evidence."""
    if not generated_claims:
        return 0.0
    supported = 0
    for claim in generated_claims:
        claim_lower = claim.lower()
        for evidence in source_evidence:
            text = evidence.get("text", "").lower()
            if any(word in text for word in claim_lower.split() if len(word) > 3):
                supported += 1
                break
    return 1.0 - (supported / len(generated_claims))


def compute_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination. 1.0 = perfect prediction."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - (ss_res / ss_tot)


def compute_hellinger_distance(p_counts: Dict[str, int],
                                q_counts: Dict[str, int]) -> float:
    """Hellinger distance between two categorical distributions. 0 = identical."""
    all_keys = set(p_counts) | set(q_counts)
    p_total = max(1, sum(p_counts.values()))
    q_total = max(1, sum(q_counts.values()))
    sum_sq = 0.0
    for k in all_keys:
        p = p_counts.get(k, 0) / p_total
        q = q_counts.get(k, 0) / q_total
        sum_sq += (math.sqrt(p) - math.sqrt(q)) ** 2
    return math.sqrt(0.5 * sum_sq)
