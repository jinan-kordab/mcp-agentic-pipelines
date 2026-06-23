# Copyright (c) 2026 Jinan Kordab
# SPDX-License-Identifier: MIT

"""
Stage 2e: Canonical Evidence Mapper [C6]
===========================================
Normalizes heterogeneous search provider formats into a unified
CanonicalEvidence schema.

The Replay Architecture's Canonical Mapping pattern applied to evidence:
  Tavily → CanonicalEvidence
  Serper → CanonicalEvidence
  Google CSE → CanonicalEvidence
  FAISS cache → CanonicalEvidence

All downstream stages (3, 4) operate on CanonicalEvidence ONLY.
No stage-specific format handling. Adding a new provider requires
ONLY a new adapter — zero changes to the verification pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pipeline.stage2.blind_retriever import RawSearchResult


@dataclass
class CanonicalEvidence:
    """
    Unified evidence schema — the single format for all downstream stages.

    Schema [C6]:
      - title: Source title/headline
      - url: Full URL to the source
      - excerpt: Relevant text excerpt from the source
      - source_domain: Domain name (e.g., bbc.com)
      - credibility_score: 0.0–1.0 per-domain credibility [J1b]
      - retrieval_ts: ISO 8601 timestamp of retrieval
      - query_used: The search query that found this source
      - provider: Which search provider returned this (tavily, serper, google_cse, faiss)
      - db_id: PK of the persisted Source row — set by Stage 2 after insert,
               consumed by Stage 3 to populate classifications.source_id (FK).
               Ephemeral, not serialized into the JSONB blob.
    """
    title: str
    url: str
    excerpt: str
    source_domain: str
    credibility_score: float
    retrieval_ts: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    query_used: str = ""
    provider: str = ""
    db_id: Optional[UUID] = None


class CanonicalEvidenceMapper:
    """
    Normalizes raw search results from any provider into CanonicalEvidence.

    This is the ONLY place that handles provider-specific formats.
    Downstream stages (classification, verdict) never see raw provider data.
    """

    async def map_results(
        self,
        raw_results: List[RawSearchResult],
        credibility_scorer=None,  # CredibilityScorer instance
    ) -> List[CanonicalEvidence]:
        """
        Map all raw results to canonical format, attaching credibility scores.

        Args:
            raw_results: Raw results from BlindRetriever.
            credibility_scorer: CredibilityScorer for per-domain scoring.

        Returns:
            List of CanonicalEvidence objects.
        """
        canonical_list = []

        for raw in raw_results:
            # Resolve credibility score
            cred_score = 0.5  # Default
            if credibility_scorer:
                cred_score = await credibility_scorer.score_domain(raw.domain)

            canonical = CanonicalEvidence(
                title=raw.title,
                url=raw.url,
                excerpt=raw.snippet,
                source_domain=raw.domain,
                credibility_score=cred_score,
                query_used=raw.query_used,
                provider=raw.provider,
            )
            canonical_list.append(canonical)

        # Sort by credibility score descending (most credible first)
        canonical_list.sort(key=lambda x: x.credibility_score, reverse=True)

        return canonical_list

    @staticmethod
    def to_dict(evidence: CanonicalEvidence) -> dict:
        """Serialize CanonicalEvidence to dict for JSONB storage."""
        return {
            "title": evidence.title,
            "url": evidence.url,
            "excerpt": evidence.excerpt,
            "source_domain": evidence.source_domain,
            "credibility_score": evidence.credibility_score,
            "retrieval_ts": evidence.retrieval_ts,
            "query_used": evidence.query_used,
            "provider": evidence.provider,
        }

    @staticmethod
    def from_dict(data: dict) -> CanonicalEvidence:
        """Deserialize CanonicalEvidence from JSONB dict."""
        # db_id is not stored in JSONB (it's the row PK itself); strip if present.
        clean = {k: v for k, v in data.items() if k != "db_id"}
        return CanonicalEvidence(**clean)
