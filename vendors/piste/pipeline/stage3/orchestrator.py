# Copyright (c) 2026 Jinan Kordab
# SPDX-License-Identifier: MIT

"""
Stage 3 Orchestrator — Per-Source Classification
===================================================
Runs N classifiers in parallel via asyncio.gather.
Each source independently evaluated → SUPPORTS/REFUTES/UNRELATED.

Emits SSE events per classification for real-time frontend updates.
Writes APPEND-ONLY classification records to PostgreSQL [C5].
"""

import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import StageRecord, Classification
from pipeline.stage3.classifier import source_classifier
from pipeline.stage2.canonical_mapper import CanonicalEvidence


@dataclass
class ClassificationResult:
    """Output of a single source classification."""
    source_index: int
    source_url: str
    source_domain: str
    label: str  # SUPPORTS | REFUTES | UNRELATED
    confidence: float
    rationale: str
    credibility_score: float


@dataclass
class Stage3Result:
    """Output of Stage 3 — Per-Source Classification."""
    atomic_claim: str
    classifications: list[ClassificationResult]
    support_count: int = 0
    refute_count: int = 0
    unrelated_count: int = 0
    total_sources: int = 0


class Stage3Orchestrator:
    """
    Orchestrates Stage 3 — parallel per-source classification.

    Jewel [J3][J8] — Aletheia's structured evaluation:
    Each source classified independently BEFORE aggregation.
    Parallel execution via asyncio.gather reduces latency.
    """

    _semaphore: asyncio.Semaphore = asyncio.Semaphore(200)
    _executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=200)

    def __init__(self, sse_callback: Optional[callable] = None):
        self.sse_callback = sse_callback

    async def process(
        self,
        atomic_claim: str,
        evidence_list: List[CanonicalEvidence],
        db: Optional[AsyncSession] = None,
        run_id: Optional[uuid.UUID] = None,
        locale: str = "en",
    ) -> Stage3Result:
        """
        Classify all evidence sources in parallel against one atomic claim.

        Args:
            atomic_claim: The atomic claim to verify.
            evidence_list: CanonicalEvidence sources from Stage 2.
            db: Optional DB session for audit ledger writes.
            run_id: UUID of the analysis run for audit trail [C5].

        Returns:
            Stage3Result with all per-source classifications.
        """
        await self._emit("stage_3_start", {
            "atomic_claim": atomic_claim,
            "sources_count": len(evidence_list),
        })

        if not evidence_list:
            await self._emit("stage_3_complete", {
                "atomic_claim": atomic_claim,
                "support": 0, "refute": 0, "unrelated": 0,
            })
            return Stage3Result(
                atomic_claim=atomic_claim,
                classifications=[],
                total_sources=0,
            )

        # --- Run N classifiers in parallel ---
        t0 = time.monotonic()

        async def _classify_with_limit(
            idx: int, claim: str, ev: CanonicalEvidence, loc: str
        ) -> ClassificationResult:
            async with self._semaphore:
                return await self._classify_single(idx, claim, ev, loc)

        tasks = [
            _classify_with_limit(i, atomic_claim, evidence, locale)
            for i, evidence in enumerate(evidence_list)
        ]
        results: List[ClassificationResult] = await asyncio.gather(*tasks)

        latency_total = (time.monotonic() - t0) * 1000

        # --- Tally ---
        support_count = sum(1 for r in results if r.label == "SUPPORTS")
        refute_count = sum(1 for r in results if r.label == "REFUTES")
        unrelated_count = sum(1 for r in results if r.label == "UNRELATED")

        # --- Write to PostgreSQL (append-only) ---
        if db:
            for r in results:
                # Stage 2 attached the Source row PK to the in-memory evidence
                # at insertion time; reuse it here as the FK.
                ev = evidence_list[r.source_index]
                db.add(Classification(
                    run_id=run_id or uuid.UUID("00000000-0000-0000-0000-000000000000"),
                    source_id=ev.db_id,
                    label=r.label,
                    confidence=r.confidence,
                    rationale=r.rationale,
                    model_used="dspy/source_classifier",
                ))

            db.add(StageRecord(
                run_id=run_id or uuid.UUID("00000000-0000-0000-0000-000000000000"),
                stage_name="stage_3",
                input_snapshot={
                    "atomic_claim": atomic_claim,
                    "sources_count": len(evidence_list),
                },
                output_snapshot={
                    "support": support_count,
                    "refute": refute_count,
                    "unrelated": unrelated_count,
                    "classifications": [
                        {"label": r.label, "confidence": r.confidence}
                        for r in results
                    ],
                },
                model_used="dspy/source_classifier",
                latency_ms=latency_total,
                retry_attempt=0,
            ))

        # --- Emit SSE events ---
        for r in results:
            await self._emit("source_classified", {
                "source_index": r.source_index,
                "source_url": r.source_url,
                "source_domain": r.source_domain,
                "label": r.label,
                "confidence": r.confidence,
                "credibility_score": r.credibility_score,
            })

        await self._emit("stage_3_complete", {
            "atomic_claim": atomic_claim,
            "support": support_count,
            "refute": refute_count,
            "unrelated": unrelated_count,
        })

        return Stage3Result(
            atomic_claim=atomic_claim,
            classifications=results,
            support_count=support_count,
            refute_count=refute_count,
            unrelated_count=unrelated_count,
            total_sources=len(evidence_list),
        )

    async def _classify_single(
        self,
        index: int,
        atomic_claim: str,
        evidence: CanonicalEvidence,
        locale: str = "en",
    ) -> ClassificationResult:
        """Classify a single source (runs in parallel via asyncio.gather).

        The DSPy source_classifier is synchronous and would block the asyncio
        event loop on every LLM call, forcing serial execution despite the
        gather.  We off-load each call to a thread so the event loop stays
        free and N classifications truly run concurrently.
        """
        loop = asyncio.get_running_loop()
        label, confidence, rationale = await loop.run_in_executor(
            self._executor,
            source_classifier,
            atomic_claim,
            evidence,
            locale,
        )
        return ClassificationResult(
            source_index=index,
            source_url=evidence.url,
            source_domain=evidence.source_domain,
            label=label,
            confidence=confidence,
            rationale=rationale,
            credibility_score=evidence.credibility_score,
        )

    async def _emit(self, event_type: str, data: dict):
        if self.sse_callback:
            await self.sse_callback(event_type, data)


# Singleton
stage3_orchestrator = Stage3Orchestrator()
