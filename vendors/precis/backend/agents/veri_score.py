"""© JINAN KORDAB — 2026 AI HYBRID AGENTIC RETRIEVAL-AUGMENTED GENERATION RAG PIPELINE - PERSONAL PROJECT"""

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from backend.core.stemming import PrecisStemmer
from backend.core.tracing import TraceEventType


# ── Known stopwords (NLTK minus content-bearing words + Precis custom) ─
_CONTENT_WORDS_TO_KEEP: Set[str] = {
    "other", "more", "most", "some", "such", "only", "own", "same",
    "very", "just", "both", "few", "each", "every", "any", "all",
    "no", "not", "nor",
}

def _load_veri_stopwords() -> Set[str]:
    """NLTK English stopwords minus content-bearing words."""
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words("english")) - _CONTENT_WORDS_TO_KEEP
    except (ImportError, LookupError, OSError):
        pass
    return {
        "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
        "she", "her", "it", "its", "they", "them", "their", "this", "that",
        "these", "those", "am", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "a", "an", "the",
        "and", "but", "if", "or", "because", "as", "of", "at", "by", "for",
        "with", "about", "between", "into", "through", "during", "before",
        "after", "to", "from", "in", "on", "off", "over", "under",
        "can", "will", "should", "now",
    }

_STOPWORDS: Set[str] = _load_veri_stopwords() | {
    # Precis-specific query-structure words
    "summarize", "summary", "summarise", "explain", "describe",
    "list", "identify", "compare", "contrast", "discuss", "analyze",
    "key", "finding", "findings", "detail", "details", "overview",
    "section", "chapter", "paragraph", "figure", "table", "page",
    "get", "make", "made", "see", "show", "shown", "find", "found",
}


@dataclass
class VeriScoreReport:
    relevancy_score: float = 0.0
    trustworthiness_score: float = 0.0
    exhaustivity_score: float = 0.0
    hallucination_rate: float = 0.0
    citation_coverage: float = 0.0
    per_chunk_scores: List[Dict[str, Any]] = field(default_factory=list)
    flagged_issues: List[str] = field(default_factory=list)
    evaluation_timestamp: str = ""


class VeriScoreEvaluator:
    """Self-evaluation engine. Scores every Precis output on 5 quality dimensions.

    All scoring methods use Porter-stemmed tokens for consistency with the
    retrieval layer (NestedHashIndex).  Stopwords are filtered before scoring.
    """

    def __init__(self) -> None:
        self.min_relevancy = 0.6
        self.min_trustworthiness = 0.5
        self._stemmer = PrecisStemmer()

    # ── Public API ───────────────────────────────────────────────────

    async def evaluate(self, query: str, retrieved_chunks: List[Dict[str, Any]],
                       generated_response: str, citations: List[Dict[str, Any]],
                       trace=None) -> VeriScoreReport:
        """Run all five quality checks **in parallel** and return a VeriScoreReport.

        The five dimensions are fully independent — same inputs, no shared mutable
        state — so we compute them concurrently via asyncio.gather.
        """
        # Launch all five dimension checks + per-chunk scoring concurrently.
        # Each is a sync CPU method offloaded to a thread so the event loop
        # stays free for other work (LLM calls, WebSocket streaming, etc.).
        (rel, trust, exh, hall,
         sentence_count, per_chunk) = await asyncio.gather(
            asyncio.to_thread(self._compute_relevancy, query, retrieved_chunks),
            asyncio.to_thread(self._compute_trustworthiness, retrieved_chunks),
            asyncio.to_thread(self._compute_exhaustivity, query, retrieved_chunks),
            asyncio.to_thread(self._compute_hallucination_rate, generated_response, retrieved_chunks),
            asyncio.to_thread(self._count_sentences, generated_response),
            asyncio.to_thread(self._score_per_chunk, query, retrieved_chunks),
        )

        cit_cov = min(len(citations) / max(1, sentence_count), 1.0)

        # Flagged issues depend on the computed scores — run after gather
        flagged = self._collect_flagged_issues(rel, trust, exh, hall, cit_cov, per_chunk)

        report = VeriScoreReport(
            relevancy_score=rel,
            trustworthiness_score=trust,
            exhaustivity_score=exh,
            hallucination_rate=hall,
            citation_coverage=cit_cov,
            per_chunk_scores=per_chunk,
            flagged_issues=flagged,
            evaluation_timestamp=datetime.now(timezone.utc).isoformat(),
        )

        if trace:
            trace.event(
                TraceEventType.EVALUATION_COMPLETED,
                agent_name="VeriScore",
                message=f"R:{rel:.2f} T:{trust:.2f} H:{hall:.2f}",
                data={
                    "relevancy": rel,
                    "trust": trust,
                    "hallucination": hall,
                    "exhaustivity": exh,
                    "citation_coverage": cit_cov,
                },
            )
        return report

    # ── Dimension 1: Relevancy ───────────────────────────────────────

    def _compute_relevancy(self, query: str, chunks: List[Dict]) -> float:
        """Average Jaccard similarity between stemmed query tokens and each chunk.

        Uses the same PrecisStemmer as the retrieval layer so that "running"
        and "runs" are recognised as the same concept.
        """
        if not chunks:
            return 0.0
        query_stems = self._stem_set(query)
        if not query_stems:
            return 0.0
        scores: List[float] = []
        for c in chunks:
            chunk_stems = self._stem_set(c.get("text", ""))
            if not chunk_stems:
                scores.append(0.0)
                continue
            inter = len(query_stems & chunk_stems)
            union = len(query_stems | chunk_stems)
            scores.append(inter / union if union else 0.0)
        return sum(scores) / len(scores)

    # ── Dimension 2: Trustworthiness ─────────────────────────────────

    def _compute_trustworthiness(self, chunks: List[Dict]) -> float:
        """Score source reliability from chunk metadata.

        Returns 0.0 when there are no chunks (no evidence = no trust),
        rather than a misleading default of 0.5.
        """
        if not chunks:
            return 0.0  # ← was 0.5 — no evidence means no trust
        scores: List[float] = []
        for c in chunks:
            score = 0.5  # Neutral base
            text = c.get("text", "")
            if len(text) > 100:
                score += 0.2
            elif len(text) > 30:
                score += 0.1
            # Has a source document = verifiable
            if c.get("source"):
                score += 0.15
            # Match quality
            mt = c.get("match_type", "")
            if mt == "exact":
                score += 0.15
            elif mt == "subset":
                score += 0.10
            elif mt == "semantic":
                score += 0.05  # semantic matches are fuzzier → lower bonus
            # Structural signals (preserved from MultiToken)
            if c.get("is_title") or c.get("is_header"):
                score += 0.10
            if c.get("token_type") == "contextual":
                score += 0.05
            scores.append(min(score, 1.0))
        return sum(scores) / len(scores)

    # ── Dimension 3: Exhaustivity ────────────────────────────────────

    def _compute_exhaustivity(self, query: str, chunks: List[Dict]) -> float:
        """Fraction of stemmed query tokens that appear in at least one chunk.

        Uses set intersection (word-boundary) rather than naive substring
        matching to avoid false positives (e.g. "in" matching "interesting").
        """
        if not chunks:
            return 0.0
        query_stems = self._stem_set(query)
        if not query_stems:
            return 1.0  # Query had only stopwords → fully covered
        # Build the union of all stemmed tokens across all chunks
        all_stems: Set[str] = set()
        for c in chunks:
            all_stems |= self._stem_set(c.get("text", ""))
        covered = len(query_stems & all_stems)
        return covered / len(query_stems)

    # ── Dimension 4: Hallucination Rate ──────────────────────────────

    def _compute_hallucination_rate(self, response: str, chunks: List[Dict]) -> float:
        """Proportion of response sentences whose *content words* are not
        attested in any retrieved source chunk.

        Uses nltk-style sentence splitting (robust against abbreviations,
        decimal numbers, and bullet lists) and Porter-stemmed token matching
        for consistency with the retrieval layer.

        Returns 0.0 when:
          - Response is empty
          - No source evidence is available (can't assess)
          - Response has no substantive sentences (< 20 chars)
        """
        if not response.strip():
            return 0.0

        # Build evidence: union of all stemmed tokens from source chunks ONLY.
        # IMPORTANT: do NOT include the synthesis/generated text here —
        # otherwise you're comparing the response against itself.
        evidence_stems: Set[str] = set()
        for c in chunks:
            evidence_stems |= self._stem_set(c.get("text", ""))

        if not evidence_stems:
            return 0.0  # No evidence → can't assess

        # Split into sentences (handles ., ?, !, abbreviations, decimals)
        sentences = self._split_sentences(response)
        substantive = [s for s in sentences if len(s.strip()) > 20]
        if not substantive:
            return 0.0

        unsupported = 0
        for sent in substantive:
            sent_stems = self._stem_set(sent)
            # Filter to content words only (len > 4 to avoid noise)
            content_words = {s for s in sent_stems if len(s) > 4}
            if not content_words:
                continue  # Skip sentences with only short/stop words
            # A sentence is "supported" if at least ONE content word
            # appears in the evidence
            if not (content_words & evidence_stems):
                unsupported += 1

        return unsupported / len(substantive) if substantive else 0.0

    # ── Helpers ──────────────────────────────────────────────────────

    def _stem_set(self, text: str) -> Set[str]:
        """Stem every word in *text*, filtering stopwords and empty results.

        Returns a set of stemmed tokens for fast intersection / union ops.
        """
        if not text or not text.strip():
            return set()
        words = text.lower().split()
        stems = self._stemmer.stem_tokens(words)
        return {s for s in stems if s and s not in _STOPWORDS}

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split *text* into sentences, robust against abbreviations and decimals.

        Falls back to simple split if nltk is unavailable.
        """
        try:
            from nltk.tokenize import sent_tokenize
            return sent_tokenize(text)
        except (ImportError, LookupError):
            pass
        # Fallback: split on sentence-ending punctuation followed by space + capital
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z])', text) if s.strip()]

    @staticmethod
    def _count_sentences(text: str) -> int:
        """Count sentences in *text*. Used for citation-coverage denominator."""
        try:
            from nltk.tokenize import sent_tokenize
            return len(sent_tokenize(text))
        except (ImportError, LookupError):
            pass
        return max(1, len(re.findall(r'[.!?]\s', text)) + 1)

    def _score_per_chunk(self, query: str, chunks: List[Dict]) -> List[Dict[str, Any]]:
        """Score each chunk individually and return a list of per-chunk reports."""
        query_stems = self._stem_set(query)
        per_chunk: List[Dict[str, Any]] = []
        for c in chunks:
            text = c.get("text", "")
            chunk_stems = self._stem_set(text)
            inter = len(query_stems & chunk_stems)
            union = len(query_stems | chunk_stems)
            jaccard = inter / union if union else 0.0
            per_chunk.append({
                "text_preview": text[:120],
                "source": c.get("source", ""),
                "page": c.get("page", 1),
                "match_type": c.get("match_type", ""),
                "jaccard": round(jaccard, 3),
                "char_length": len(text),
            })
        return per_chunk

    def _collect_flagged_issues(
        self,
        rel: float, trust: float, exh: float, hall: float,
        cit_cov: float, per_chunk: List[Dict],
    ) -> List[str]:
        """Aggregate issues across all quality dimensions into a human-readable list."""
        issues: List[str] = []
        if rel < self.min_relevancy:
            issues.append(f"Low relevancy ({rel:.2f} < {self.min_relevancy}) — "
                          "query terms not well covered by retrieved chunks")
        if trust < self.min_trustworthiness:
            issues.append(f"Low trustworthiness ({trust:.2f} < {self.min_trustworthiness}) — "
                          "sources may be unreliable or too short")
        if exh < 0.5:
            issues.append(f"Low exhaustivity ({exh:.2f}) — "
                          "many query terms not found in any chunk")
        if hall > 0.3:
            issues.append(f"High hallucination rate ({hall:.0%} > 30%) — "
                          f"many response claims are not supported by source evidence")
        if hall > 0.1:
            issues.append(f"Elevated hallucination rate ({hall:.0%}) — review recommended")
        if cit_cov < 0.2:
            issues.append(f"Low citation coverage ({cit_cov:.0%}) — "
                          "few sources cited relative to response length")
        # Per-chunk issues
        zero_jaccard = [c for c in per_chunk if c.get("jaccard", 0) == 0.0]
        if zero_jaccard and len(zero_jaccard) == len(per_chunk):
            issues.append("All chunks have zero Jaccard overlap with query — "
                          "retrieval may have returned irrelevant content")
        return issues
