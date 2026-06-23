# Copyright (c) 2026 Jinan Kordab
# SPDX-License-Identifier: MIT

"""
Stage 2c: Per-Domain Credibility Scorer [J1b]
================================================
Veracity's second jewel: continuous 0–1 credibility scoring per source domain.
Uses Lin et al. (2023) composite domain quality database.

Unlike Aletheia's binary blacklist, this provides a continuous, auditable metric.
Users see not just WHICH sources were used, but HOW credible each one is.
"""

from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.models import Domain


class CredibilityScorer:
    """
    Scores each source domain on a continuous 0–1 credibility scale.

    Jewel [J1b] — Veracity's per-domain credibility:
    Transforms source transparency from binary (blocked/allowed) into
    a continuous, auditable metric that educates users about source quality
    while holding the system accountable.
    """

    # Domain credibility scores are fetched from PostgreSQL domains table.
    # Unknown domains get a default score of 0.5 (neutral, flagged for review).

    DEFAULT_SCORE: float = 0.5
    UNKNOWN_DOMAIN_SCORE: float = 0.5

    def __init__(self, db: Optional[AsyncSession] = None):
        self.db = db
        self._cache: dict[str, float] = {}  # In-memory cache per session

    async def score_domain(self, domain: str) -> float:
        """
        Get credibility score for a domain.

        Returns:
            0.0–1.0 score, where 1.0 = highest credibility.
        """
        domain = domain.lower().strip()

        # Check in-memory cache
        if domain in self._cache:
            return self._cache[domain]

        # Query PostgreSQL domains table
        if self.db:
            result = await self.db.execute(
                select(Domain.credibility_score).where(
                    Domain.domain_name == domain
                )
            )
            score = result.scalar()
            if score is not None:
                self._cache[domain] = float(score)
                return float(score)

        # Unknown domain → default neutral score
        self._cache[domain] = self.DEFAULT_SCORE
        return self.DEFAULT_SCORE

    async def score_domains(
        self, domains: list[str]
    ) -> dict[str, float]:
        """Batch-score multiple domains."""
        scores = {}
        for domain in domains:
            scores[domain] = await self.score_domain(domain)
        return scores

    async def is_reliable(self, domain: str) -> bool:
        """Quick binary check: is this domain generally reliable?"""
        score = await self.score_domain(domain)
        return score >= 0.6

    def flush_cache(self):
        """Clear in-memory cache (useful between sessions)."""
        self._cache.clear()
