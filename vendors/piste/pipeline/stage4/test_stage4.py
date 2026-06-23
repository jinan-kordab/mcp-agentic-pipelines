# Copyright (c) 2026 Jinan Kordab
# SPDX-License-Identifier: MIT

"""
Unit Tests — Stage 4: Verdict
===============================
Tests CriticalityGate [C3], VerdictAggregator [J5],
7-way verdict labels, Stage4Result.

Run: pytest pipeline/stage4/test_stage4.py -v
"""

import pytest
import json


# ============================================================
# Criticality Gate Tests [C3]
# ============================================================

class TestCriticalityGate:
    """Test routing of claims to automated vs human review."""

    @pytest.fixture
    def gate(self):
        from pipeline.stage4.criticality_gate import CriticalityGate
        return CriticalityGate()

    def test_election_claim_is_critical(self, gate):
        """Claims about elections → human review."""
        result = gate.assess("The election was rigged and the votes were tampered with.")
        assert result.is_critical is True
        assert result.recommendation == "human_review"
        assert "election" in result.matched_topics
        assert "vote" in result.matched_topics

    def test_public_health_claim_is_critical(self, gate):
        """Claims about vaccines/pandemics → human review."""
        result = gate.assess("The vaccine causes severe side effects in children.")
        assert result.is_critical is True
        assert "vaccine" in result.matched_topics

    def test_high_profile_figure_is_critical(self, gate):
        """Claims about presidents → human review."""
        result = gate.assess("The president signed the executive order yesterday.")
        assert result.is_critical is True
        assert result.is_high_profile is True

    def test_neutral_claim_is_automated(self, gate):
        """Non-critical claims → automated verdict."""
        result = gate.assess("Water boils at 100 degrees Celsius at sea level.")
        assert result.is_critical is False
        assert result.recommendation == "automated"

    def test_empty_claim_is_automated(self, gate):
        """Empty claims → automated."""
        result = gate.assess("")
        assert result.is_critical is False


# ============================================================
# Verdict Aggregator Tests [J5]
# ============================================================

class TestVerdictAggregator:
    """Test 7-way verdict normalization and distribution."""

    @pytest.fixture
    def aggregator(self):
        from pipeline.stage4.verdict_aggregator import VerdictAggregator
        return VerdictAggregator()

    def test_normalize_direct_match(self, aggregator):
        """Direct verdict labels pass through unchanged."""
        assert aggregator._normalize_verdict("TRUE") == "TRUE"
        assert aggregator._normalize_verdict("FALSE") == "FALSE"
        assert aggregator._normalize_verdict("UNVERIFIABLE") == "UNVERIFIABLE"
        assert aggregator._normalize_verdict("PANTS_ON_FIRE") == "PANTS_ON_FIRE"

    def test_normalize_with_spaces(self, aggregator):
        """Labels with spaces are normalized."""
        assert aggregator._normalize_verdict("MOSTLY TRUE") == "MOSTLY_TRUE"
        assert aggregator._normalize_verdict("HALF TRUE") == "HALF_TRUE"
        assert aggregator._normalize_verdict("MOSTLY FALSE") == "MOSTLY_FALSE"

    def test_normalize_without_spaces(self, aggregator):
        """Labels without spaces are normalized."""
        assert aggregator._normalize_verdict("MOSTLYTRUE") == "MOSTLY_TRUE"
        assert aggregator._normalize_verdict("HALFTRUE") == "HALF_TRUE"

    def test_normalize_nei(self, aggregator):
        """NEI → UNVERIFIABLE."""
        assert aggregator._normalize_verdict("NEI") == "UNVERIFIABLE"
        assert aggregator._normalize_verdict("NOT ENOUGH INFORMATION") == "UNVERIFIABLE"

    def test_normalize_unknown_fallback(self, aggregator):
        """Unknown labels → UNVERIFIABLE (safe default)."""
        assert aggregator._normalize_verdict("SOMETHING WEIRD") == "UNVERIFIABLE"

    def test_default_distribution_centered(self, aggregator):
        """Default distribution puts all weight on the given verdict."""
        dist = aggregator._default_distribution("TRUE")
        assert dist["TRUE"] == 1.0
        assert dist["FALSE"] == 0.0
        assert sum(dist.values()) == 1.0

    def test_all_verdict_labels_present(self, aggregator):
        """Default distribution includes all 7 labels."""
        from pipeline.stage4.verdict_aggregator import VERDICT_LABELS
        dist = aggregator._default_distribution("MOSTLY_TRUE")
        for label in VERDICT_LABELS:
            assert label in dist
        assert len(dist) == 7

    def test_build_classifications_payload(self, aggregator):
        """Classification payload is valid JSON."""
        from pipeline.stage3.orchestrator import ClassificationResult
        classifications = [
            ClassificationResult(0, "a.com", "a.com", "SUPPORTS", 0.9, "Good", 0.9),
            ClassificationResult(1, "b.com", "b.com", "REFUTES", 0.7, "Bad", 0.5),
        ]
        payload = aggregator._build_classifications_payload("Test", classifications)
        data = json.loads(payload)
        assert len(data) == 2
        assert data[0]["label"] == "SUPPORTS"
        assert data[1]["label"] == "REFUTES"


# ============================================================
# Stage4Result Tests
# ============================================================

class TestStage4Result:
    """Test Stage4Result dataclass."""

    def test_automated_result(self):
        from pipeline.stage4.orchestrator import Stage4Result
        result = Stage4Result(
            atomic_claim="Water boils at 100°C.",
            is_critical=False,
            review_route="automated",
            criticality_reason="Low-stakes claim.",
            verdict="TRUE",
            confidence=0.95,
            explanation="Well-established scientific fact.",
            distribution={"TRUE": 0.95, "MOSTLY_TRUE": 0.05},
            support_count=3,
            refute_count=0,
            unrelated_count=0,
        )
        assert result.verdict == "TRUE"
        assert result.review_route == "automated"
        assert result.human_review is None

    def test_human_review_result(self):
        from pipeline.stage4.orchestrator import Stage4Result
        result = Stage4Result(
            atomic_claim="The election was fraudulent.",
            is_critical=True,
            review_route="human_review",
            criticality_reason="Matches critical topic: election.",
            verdict="UNVERIFIABLE",
            confidence=0.45,
            explanation="Insufficient credible evidence.",
            distribution={"UNVERIFIABLE": 1.0},
            human_review={"status": "pending_review"},
            support_count=0,
            refute_count=0,
            unrelated_count=5,
        )
        assert result.is_critical is True
        assert result.review_route == "human_review"
        assert result.human_review is not None


# ============================================================
# Verdict Labels Tests
# ============================================================

class TestVerdictLabels:
    """Test the 7-way PolitiFact-aligned verdict labels."""

    def test_all_seven_labels(self):
        from pipeline.stage4.verdict_aggregator import VERDICT_LABELS
        assert len(VERDICT_LABELS) == 7
        assert "TRUE" in VERDICT_LABELS
        assert "MOSTLY_TRUE" in VERDICT_LABELS
        assert "HALF_TRUE" in VERDICT_LABELS
        assert "MOSTLY_FALSE" in VERDICT_LABELS
        assert "FALSE" in VERDICT_LABELS
        assert "PANTS_ON_FIRE" in VERDICT_LABELS
        assert "UNVERIFIABLE" in VERDICT_LABELS
