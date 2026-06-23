# Copyright (c) 2026 Jinan Kordab
# SPDX-License-Identifier: MIT

"""
Unit Tests — Stage 3: Per-Source Classification
=================================================
Tests SourceClassifier [J3], parallel orchestrator, ClassificationResult.

Run: pytest pipeline/stage3/test_stage3.py -v
"""

import pytest


class TestClassificationResult:
    """Test ClassificationResult dataclass."""

    def test_supports_classification(self):
        from pipeline.stage3.orchestrator import ClassificationResult
        r = ClassificationResult(
            source_index=0,
            source_url="https://reuters.com/article",
            source_domain="reuters.com",
            label="SUPPORTS",
            confidence=0.92,
            rationale="The article confirms the claim with primary data.",
            credibility_score=0.95,
        )
        assert r.label == "SUPPORTS"
        assert r.credibility_score == 0.95

    def test_refutes_classification(self):
        from pipeline.stage3.orchestrator import ClassificationResult
        r = ClassificationResult(
            source_index=1,
            source_url="https://example.com",
            source_domain="example.com",
            label="REFUTES",
            confidence=0.78,
            rationale="Data contradicts the claim.",
            credibility_score=0.45,
        )
        assert r.label == "REFUTES"

    def test_unrelated_classification(self):
        from pipeline.stage3.orchestrator import ClassificationResult
        r = ClassificationResult(
            source_index=2,
            source_url="https://other.com",
            source_domain="other.com",
            label="UNRELATED",
            confidence=0.95,
            rationale="Source is about a different topic.",
            credibility_score=0.60,
        )
        assert r.label == "UNRELATED"


class TestStage3Result:
    """Test Stage3Result tallying."""

    def test_tally_counts(self):
        from pipeline.stage3.orchestrator import Stage3Result, ClassificationResult

        classifications = [
            ClassificationResult(0, "a.com", "a.com", "SUPPORTS", 0.9, "...", 0.9),
            ClassificationResult(1, "b.com", "b.com", "SUPPORTS", 0.8, "...", 0.8),
            ClassificationResult(2, "c.com", "c.com", "REFUTES", 0.7, "...", 0.7),
            ClassificationResult(3, "d.com", "d.com", "UNRELATED", 0.9, "...", 0.5),
        ]

        # Manual tally
        support = sum(1 for c in classifications if c.label == "SUPPORTS")
        refute = sum(1 for c in classifications if c.label == "REFUTES")
        unrelated = sum(1 for c in classifications if c.label == "UNRELATED")

        assert support == 2
        assert refute == 1
        assert unrelated == 1

    def test_empty_evidence(self):
        from pipeline.stage3.orchestrator import Stage3Result
        result = Stage3Result(
            atomic_claim="Test",
            classifications=[],
            total_sources=0,
        )
        assert result.total_sources == 0
        assert result.support_count == 0


class TestStage3Orchestrator:
    """Test parallel classification orchestration."""

    def test_sse_callback_stored(self):
        from pipeline.stage3.orchestrator import Stage3Orchestrator
        calls = []
        async def cb(event, data):
            calls.append((event, data))
        orch = Stage3Orchestrator(sse_callback=cb)
        assert orch.sse_callback is not None
