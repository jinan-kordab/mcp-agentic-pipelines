# Copyright (c) 2026 Jinan Kordab
# SPDX-License-Identifier: MIT

"""
Stage 1 Orchestrator
=====================
Coordinates Stage 1a (Check-Worthiness) and Stage 1b (Atomic Decomposition).
Writes APPEND-ONLY stage records to PostgreSQL audit ledger [C5].
Emits SSE events for real-time frontend updates.
"""

import time
import uuid
from typing import Optional
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import StageRecord
from pipeline.stage1.check_worthiness import check_worthiness_detector
from pipeline.stage1.atomic_decomposer import atomic_claim_decomposer


@dataclass
class Stage1Result:
    """Output of Stage 1 — Claim Processing."""
    claim_text: str
    locale: str

    # Stage 1a: Check-Worthiness
    worthiness_label: str  # CFC, UFC, NFC
    worthiness_confidence: float
    worthiness_rationale: str
    worthiness_votes: list[str]

    # Stage 1b: Atomic Decomposition
    atomic_claims: list[str]

    # If not check-worthy, pipeline stops here
    is_check_worthy: bool
    stop_reason: str = ""


class Stage1Orchestrator:
    """
    Orchestrates Stage 1 of the fact-checking pipeline.

    Flow:
      1. Run CheckWorthinessDetector (1a) with voting
      2. If CFC → run AtomicClaimDecomposer (1b)
      3. If UFC or NFC → stop pipeline, return early verdict
      4. Write stage records to PostgreSQL (append-only)
      5. Emit SSE events via callback
    """

    def __init__(self, sse_callback: Optional[callable] = None):
        """
        Args:
            sse_callback: async function(event_type, data) to emit SSE events.
        """
        self.sse_callback = sse_callback

    async def process(
        self,
        claim_text: str,
        locale: str = "en",
        db: Optional[AsyncSession] = None,
        run_id: Optional[uuid.UUID] = None,
        context: str = "",
    ) -> Stage1Result:
        """
        Run Stage 1 processing.

        Args:
            claim_text: The raw claim text submitted by the user.
            locale: Language locale (en, fr, es, ...).
            db: Async database session for audit ledger writes.
            context: Optional additional context to aid the LLM.

        Returns:
            Stage1Result with worthiness classification and atomic claims.
        """
        # Merge context into claim text if provided
        effective_claim = f"Context: {context}\n\nClaim: {claim_text}" if context and context.strip() else claim_text

        # --- Stage 1a: Check-Worthiness Detection ---
        await self._emit("stage_1a_start", {"claim_text": claim_text, "locale": locale, "context": context})
        t0 = time.monotonic()

        label, confidence, rationale, votes = check_worthiness_detector(
            effective_claim, locale
        )

        latency_1a = (time.monotonic() - t0) * 1000

        await self._emit("stage_1a_complete", {
            "label": label,
            "confidence": confidence,
            "votes": votes,
        })

        # Write stage record (append-only)
        if db:
            db.add(StageRecord(
                run_id=run_id or uuid.UUID("00000000-0000-0000-0000-000000000000"),
                stage_name="stage_1a",
                input_snapshot={"claim_text": claim_text, "locale": locale, "context": context or None},
                output_snapshot={
                    "label": label,
                    "confidence": confidence,
                    "rationale": rationale,
                    "votes": votes,
                },
                model_used="dspy/check_worthiness",
                latency_ms=latency_1a,
                retry_attempt=0,
            ))

        # Stop if not check-worthy
        if label != "CFC":
            return Stage1Result(
                claim_text=claim_text,
                locale=locale,
                worthiness_label=label,
                worthiness_confidence=confidence,
                worthiness_rationale=rationale,
                worthiness_votes=votes,
                atomic_claims=[],
                is_check_worthy=False,
                stop_reason=f"Claim classified as {label}: {rationale}",
            )

        # --- Stage 1b: Atomic Claim Decomposition ---
        await self._emit("stage_1b_start", {"claim_text": claim_text})
        t0 = time.monotonic()

        atomic_claims = atomic_claim_decomposer(claim_text)

        latency_1b = (time.monotonic() - t0) * 1000

        await self._emit("stage_1b_complete", {
            "atomic_claims": atomic_claims,
            "count": len(atomic_claims),
        })

        # Write stage record (append-only)
        if db:
            db.add(StageRecord(
                run_id=run_id or uuid.UUID("00000000-0000-0000-0000-000000000000"),
                stage_name="stage_1b",
                input_snapshot={"claim_text": claim_text},
                output_snapshot={"atomic_claims": atomic_claims},
                model_used="dspy/atomic_decomposer",
                latency_ms=latency_1b,
                retry_attempt=0,
            ))

        return Stage1Result(
            claim_text=claim_text,
            locale=locale,
            worthiness_label=label,
            worthiness_confidence=confidence,
            worthiness_rationale=rationale,
            worthiness_votes=votes,
            atomic_claims=atomic_claims,
            is_check_worthy=True,
        )

    async def _emit(self, event_type: str, data: dict):
        """Emit SSE event via callback if configured."""
        if self.sse_callback:
            await self.sse_callback(event_type, data)


# Singleton
stage1_orchestrator = Stage1Orchestrator()
