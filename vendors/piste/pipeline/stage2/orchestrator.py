# Copyright (c) 2026 Jinan Kordab
# SPDX-License-Identifier: MIT

"""
Stage 2 Orchestrator — Blind Retrieval
========================================
Coordinates all 5 sub-stages of evidence retrieval:
  2a: Search-Decision Generator [J1] — decide IF search needed
  2b: Blind Retriever [J2] — execute neutral queries (never sees claim)
  2c: Per-Domain Credibility Scorer [J1b] — score each source domain
  2d: Intelligent Query Refiner [J8c] — Loop 1 retry with refined queries
  2e: Canonical Evidence Mapper [C6] — normalize all formats

Manages Loop 1 feedback: if results insufficient → refiner → retry.
Writes APPEND-ONLY stage records to PostgreSQL [C5].
Checks FAISS Tier-1 cache before external search [J7].
Emits SSE events for real-time frontend updates.
"""

import asyncio
import time
import uuid
from typing import Optional, List
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import StageRecord, Source
from app.core.config import settings
from pipeline.stage2.search_decision import search_decision_generator
from pipeline.stage2.blind_retriever import blind_retriever
from pipeline.stage2.credibility_scorer import CredibilityScorer
from pipeline.stage2.query_refiner import QueryRefiner
from pipeline.stage2.canonical_mapper import (
    CanonicalEvidenceMapper, CanonicalEvidence,
)


@dataclass
class Stage2Result:
    """Output of Stage 2 — Blind Retrieval."""
    atomic_claim: str
    needs_search: bool
    search_queries: list[str]
    search_reasoning: str

    # Evidence
    canonical_evidence: list[CanonicalEvidence] = field(default_factory=list)

    # Loop 1 tracking
    retry_count: int = 0
    retry_queries: list[str] = field(default_factory=list)
    insufficient_reason: str = ""

    # If no search was needed (parametric knowledge suffices)
    skipped_search: bool = False


class Stage2Orchestrator:
    """
    Orchestrates Stage 2 of the fact-checking pipeline.

    Flow per atomic claim:
      1. SearchDecisionGenerator: decide if search needed [J1]
      2. If no → skip, return empty evidence
      3. If yes → BlindRetriever: execute NEUTRAL queries [J2]
      4. CredibilityScorer: score each domain [J1b]
      5. CanonicalEvidenceMapper: normalize all formats [C6]
      6. Check sufficiency → if insufficient:
         a. QueryRefiner analyzes gaps [J8c]
         b. BlindRetriever with refined queries (Loop 1)
         c. Repeat up to MAX_RETRY_LOOPS times
      7. Write stage records + sources to PostgreSQL (append-only)
    """

    def __init__(self, sse_callback: Optional[callable] = None):
        self.sse_callback = sse_callback
        self.mapper = CanonicalEvidenceMapper()
        self.refiner = QueryRefiner()
        self._locale: str = "en"  # Default locale, set by caller

    async def process(
        self,
        atomic_claims: List[str],
        db: Optional[AsyncSession] = None,
        locale: str = "en",
        run_id: Optional[uuid.UUID] = None,
    ) -> List[Stage2Result]:
        """
        Run Stage 2 for all atomic claims.

        Args:
            atomic_claims: List of atomic claims from Stage 1.
            db: Optional DB session for audit ledger writes.
            locale: Language locale for search region biasing [C2].
        """
        self._locale = locale

        results: List[Stage2Result] = []
        credibility_scorer = CredibilityScorer(db)

        for claim in atomic_claims:
            await self._emit("stage_2a_start", {
                "atomic_claim": claim,
            })

            result = await self._process_single_claim(
                claim, credibility_scorer, db, run_id
            )
            results.append(result)

            await self._emit("stage_2_complete", {
                "atomic_claim": claim,
                "needs_search": result.needs_search,
                "sources_found": len(result.canonical_evidence),
                "retry_count": result.retry_count,
            })

        return results

    async def _process_single_claim(
        self,
        atomic_claim: str,
        credibility_scorer: CredibilityScorer,
        db: Optional[AsyncSession],
        run_id: Optional[uuid.UUID] = None,
    ) -> Stage2Result:
        """Process one atomic claim through all Stage 2 sub-stages."""

        # --- 2a: Search Decision ---
        t0 = time.monotonic()
        needs_search, search_queries, reasoning = search_decision_generator(
            atomic_claim
        )
        latency_2a = (time.monotonic() - t0) * 1000

        if db:
            db.add(StageRecord(
                run_id=run_id or uuid.UUID("00000000-0000-0000-0000-000000000000"),
                stage_name="stage_2a",
                input_snapshot={"atomic_claim": atomic_claim},
                output_snapshot={
                    "needs_search": needs_search,
                    "search_queries": search_queries,
                    "reasoning": reasoning,
                },
                model_used="dspy/search_decision",
                latency_ms=latency_2a,
                retry_attempt=0,
            ))

        if not needs_search:
            await self._emit("stage_2a_complete", {
                "atomic_claim": atomic_claim,
                "needs_search": False,
                "reasoning": reasoning,
            })
            return Stage2Result(
                atomic_claim=atomic_claim,
                needs_search=False,
                search_queries=[],
                search_reasoning=reasoning,
                skipped_search=True,
            )

        await self._emit("stage_2a_complete", {
            "atomic_claim": atomic_claim,
            "needs_search": True,
            "queries": search_queries,
        })

        # --- 2b + 2c + 2e: Retrieve → Score → Map ---
        all_evidence, retry_count, retry_queries, insufficient = (
            await self._retrieve_with_retry(search_queries, atomic_claim, db, run_id)
        )

        # --- Score credibility ---
        for ev in all_evidence:
            ev.credibility_score = await credibility_scorer.score_domain(
                ev.source_domain
            )

        # --- Write sources to PostgreSQL ---
        # Generate the Source PK up-front and attach it to the in-memory
        # CanonicalEvidence so Stage 3 can populate classifications.source_id
        # (FK to sources.id) without an extra DB roundtrip.
        if db:
            for ev in all_evidence:
                ev.db_id = uuid.uuid4()
                db.add(Source(
                    id=ev.db_id,
                    run_id=run_id or uuid.uuid4(),
                    url=ev.url,
                    domain=ev.source_domain,
                    title=ev.title,
                    snippet=ev.excerpt,
                    credibility_score=ev.credibility_score,
                    canonical_evidence=self.mapper.to_dict(ev),
                ))

        await self._emit("stage_2c_complete", {
            "atomic_claim": atomic_claim,
            "sources_count": len(all_evidence),
            "avg_credibility": (
                sum(e.credibility_score for e in all_evidence) / len(all_evidence)
                if all_evidence else 0.0
            ),
        })

        return Stage2Result(
            atomic_claim=atomic_claim,
            needs_search=True,
            search_queries=search_queries,
            search_reasoning=reasoning,
            canonical_evidence=all_evidence,
            retry_count=retry_count,
            retry_queries=retry_queries,
            insufficient_reason=insufficient,
        )

    async def _retrieve_with_retry(
        self,
        search_queries: List[str],
        atomic_claim: str,
        db: Optional[AsyncSession],
        run_id: Optional[uuid.UUID] = None,
    ) -> tuple:
        """
        Execute retrieval with Loop 1 retry logic.

        Returns:
            (all_evidence, retry_count, retry_queries, insufficient_reason)
        """
        all_evidence: List[CanonicalEvidence] = []
        retry_count = 0
        retry_queries: List[str] = []
        insufficient_reason = ""
        current_queries = list(search_queries)

        for attempt in range(settings.MAX_RETRY_LOOPS + 1):
            # --- 2b: Blind Retrieve ---
            await self._emit("stage_2b_start", {
                "queries": current_queries,
                "attempt": attempt,
            })
            t0 = time.monotonic()

            raw_results = await blind_retriever.search(
                current_queries, locale=getattr(self, "_locale", "en")
            )
            latency_2b = (time.monotonic() - t0) * 1000

            await self._emit("stage_2b_complete", {
                "results_count": len(raw_results),
                "attempt": attempt,
            })

            # --- 2e: Map to Canonical ---
            evidence = await self.mapper.map_results(raw_results)
            all_evidence.extend(evidence)

            if db:
                db.add(StageRecord(
                    run_id=run_id or uuid.uuid4(),
                    stage_name="stage_2b",
                    input_snapshot={"queries": current_queries},
                    output_snapshot={
                        "raw_count": len(raw_results),
                        "canonical_count": len(evidence),
                    },
                    model_used="blind_retriever",
                    latency_ms=latency_2b,
                    retry_attempt=attempt,
                ))

            # Check sufficiency
            if len(all_evidence) >= 3:
                break  # Sufficient evidence found

            # --- 2d: Query Refiner (Loop 1) ---
            insufficient_reason = self.refiner.analyze_insufficiency(
                all_evidence, atomic_claim
            )

            if attempt < settings.MAX_RETRY_LOOPS:
                await self._emit("stage_2d_start", {
                    "insufficient_reason": insufficient_reason,
                })
                refined = self.refiner(
                    original_query=current_queries[0] if current_queries else atomic_claim,
                    insufficient_reason=insufficient_reason,
                )
                retry_queries.extend(refined)
                current_queries = refined
                retry_count += 1

                await self._emit("stage_2d_complete", {
                    "refined_queries": refined,
                    "retry_attempt": retry_count,
                })

        return all_evidence, retry_count, retry_queries, insufficient_reason

    async def _emit(self, event_type: str, data: dict):
        """Emit SSE event via callback if configured."""
        if self.sse_callback:
            await self.sse_callback(event_type, data)


# Singleton
stage2_orchestrator = Stage2Orchestrator()
