# Copyright (c) 2026 Jinan Kordab
# SPDX-License-Identifier: MIT

"""
Stage 4 Orchestrator — Verdict
================================
Coordinates:
  4a: Criticality Gate [C3] — route to automated vs human review
  4b: Verdict Aggregator [J5] — 7-way PolitiFact-aligned verdict
  4c: Editorial Review Panel [C3] — human-in-the-loop for critical claims

Writes APPEND-ONLY verdict records to PostgreSQL [C5].
Emits SSE events: criticality_determined, verdict_complete.
"""

import time
import uuid
from typing import Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import StageRecord, Verdict
from pipeline.stage4.criticality_gate import criticality_gate
from pipeline.stage4.verdict_aggregator import (
    verdict_aggregator, VERDICT_LABELS,
)
from pipeline.stage3.orchestrator import Stage3Result, ClassificationResult


@dataclass
class Stage4Result:
    """Output of Stage 4 — Verdict."""
    atomic_claim: str

    # Criticality Gate
    is_critical: bool
    review_route: str  # "automated" or "human_review"
    criticality_reason: str

    # Verdict
    verdict: str  # One of 7 labels
    confidence: float
    explanation: str
    distribution: Dict[str, float]

    # Human review (if applicable)
    human_review: Optional[dict] = None

    # Stage 3 summary for context
    support_count: int = 0
    refute_count: int = 0
    unrelated_count: int = 0


class Stage4Orchestrator:
    """
    Orchestrates Stage 4 of the fact-checking pipeline.

    Flow:
      1. Criticality Gate: assess if human review needed [C3]
      2a. If automated → VerdictAggregator synthesizes 7-way verdict [J5]
      2b. If critical → queue for Editorial Review Panel [C3]
      3. Write verdict record to PostgreSQL (append-only)
      4. Emit SSE events
    """

    def __init__(self, sse_callback: Optional[callable] = None):
        self.sse_callback = sse_callback

    async def process(
        self,
        atomic_claim: str,
        stage3_result: Stage3Result,
        db: Optional[AsyncSession] = None,
        run_id: Optional[uuid.UUID] = None,
        locale: str = "en",
    ) -> Stage4Result:
        """
        Run Stage 4 processing for one atomic claim.

        Args:
            atomic_claim: The atomic claim to verdict.
            stage3_result: Per-source classifications from Stage 3.
            db: Optional DB session for audit ledger writes.

        Returns:
            Stage4Result with final verdict.
        """
        # --- 4a: Criticality Gate ---
        await self._emit("stage_4a_start", {
            "atomic_claim": atomic_claim,
        })

        assessment = criticality_gate.assess(atomic_claim)

        await self._emit("criticality_determined", {
            "atomic_claim": atomic_claim,
            "is_critical": assessment.is_critical,
            "matched_topics": assessment.matched_topics,
            "recommendation": assessment.recommendation,
        })

        # --- 4b: Verdict Aggregation ---
        if assessment.recommendation == "automated":
            result = await self._automated_verdict(
                atomic_claim, stage3_result, db, run_id, locale
            )
        else:
            result = await self._human_review_verdict(
                atomic_claim, stage3_result, assessment, db, run_id, locale
            )

        # NOTE: Verdict creation moved to pipeline_service.py —
        # one Verdict per run_id, aggregated from all atomic claims.
        # Stage 4 only returns per-atomic-claim results.
        return result

    async def _automated_verdict(
        self,
        atomic_claim: str,
        stage3_result: Stage3Result,
        db: Optional[AsyncSession],
        run_id: Optional[uuid.UUID] = None,
        locale: str = "en",
    ) -> Stage4Result:
        """Run automated VerdictAggregator [J5]."""
        t0 = time.monotonic()

        verdict, confidence, explanation, distribution = verdict_aggregator(
            claim=atomic_claim,
            classifications=stage3_result.classifications,
            locale=locale,
        )

        latency_ms = (time.monotonic() - t0) * 1000

        # Write stage record
        if db:
            db.add(StageRecord(
                run_id=run_id or uuid.uuid4(),
                stage_name="stage_4b",
                input_snapshot={
                    "atomic_claim": atomic_claim,
                    "classifications_count": len(stage3_result.classifications),
                },
                output_snapshot={
                    "verdict": verdict,
                    "confidence": confidence,
                    "distribution": distribution,
                },
                model_used="dspy/verdict_aggregator",
                latency_ms=latency_ms,
                retry_attempt=0,
            ))

        await self._emit("atomic_verdict", {
            "atomic_claim": atomic_claim,
            "verdict": verdict,
            "confidence": confidence,
            "explanation": explanation,
            "distribution": distribution,
            "route": "automated",
        })

        return Stage4Result(
            atomic_claim=atomic_claim,
            is_critical=False,
            review_route="automated",
            criticality_reason="Low-stakes claim — automated verdict.",
            verdict=verdict,
            confidence=confidence,
            explanation=explanation,
            distribution=distribution,
            support_count=stage3_result.support_count,
            refute_count=stage3_result.refute_count,
            unrelated_count=stage3_result.unrelated_count,
        )

    async def _human_review_verdict(
        self,
        atomic_claim: str,
        stage3_result: Stage3Result,
        assessment,
        db: Optional[AsyncSession],
        run_id: Optional[uuid.UUID] = None,
        locale: str = "en",
    ) -> Stage4Result:
        """
        Queue claim for human Editorial Review Panel [C3].

        In production: claim added to review_queue table.
        Human reviewer sees: claim + evidence + per-source classifications.
        Panel votes on final verdict (PolitiFact-style democratic vote).

        MVP: fall back to automated verdict with human_review flag.
        """
        # For MVP, run automated verdict but flag for human review
        t0 = time.monotonic()

        verdict, confidence, explanation, distribution = verdict_aggregator(
            claim=atomic_claim,
            classifications=stage3_result.classifications,
            locale=locale,
        )

        latency_ms = (time.monotonic() - t0) * 1000

        human_review_payload = {
            "status": "pending_review",
            "matched_topics": assessment.matched_topics,
            "auto_verdict": verdict,
            "auto_confidence": confidence,
            "queued_at": datetime.utcnow().isoformat(),
            "reviewer_id": None,
            "final_verdict": None,
            "reason": None,
        }

        if db:
            db.add(StageRecord(
                run_id=run_id or uuid.uuid4(),
                stage_name="stage_4c",
                input_snapshot={
                    "atomic_claim": atomic_claim,
                    "assessment": {
                        "is_critical": assessment.is_critical,
                        "matched_topics": assessment.matched_topics,
                    },
                },
                output_snapshot=human_review_payload,
                model_used="dspy/verdict_aggregator+human_review",
                latency_ms=latency_ms,
                retry_attempt=0,
            ))

        await self._emit("verdict_complete", {
            "atomic_claim": atomic_claim,
            "verdict": verdict,
            "confidence": confidence,
            "explanation": explanation,
            "distribution": distribution,
            "route": "human_review_pending",
            "human_review": human_review_payload,
        })

        return Stage4Result(
            atomic_claim=atomic_claim,
            is_critical=True,
            review_route="human_review",
            criticality_reason=assessment.reason,
            verdict=verdict,
            confidence=confidence,
            explanation=explanation,
            distribution=distribution,
            human_review=human_review_payload,
            support_count=stage3_result.support_count,
            refute_count=stage3_result.refute_count,
            unrelated_count=stage3_result.unrelated_count,
        )

    async def _emit(self, event_type: str, data: dict):
        if self.sse_callback:
            await self.sse_callback(event_type, data)


# Singleton
stage4_orchestrator = Stage4Orchestrator()
