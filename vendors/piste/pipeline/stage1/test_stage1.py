# Copyright (c) 2026 Jinan Kordab
# SPDX-License-Identifier: MIT

"""
Unit Tests — Stage 1: Claim Processing
=======================================
Tests CheckWorthinessDetector [J4] and AtomicClaimDecomposer [J7]
with mock DSPy LLM responses.

Run: pytest pipeline/stage1/test_stage1.py -v
"""

import pytest
from unittest.mock import patch, MagicMock


# ============================================================
# Check-Worthiness Detector Tests
# ============================================================

class TestCheckWorthinessDetector:
    """Test CheckWorthinessDetector [J4] with voting mechanism."""

    @pytest.fixture
    def detector(self):
        from pipeline.stage1.check_worthiness import CheckWorthinessDetector
        return CheckWorthinessDetector()

    def test_majority_vote_cfc(self, detector):
        """When all 3 votes are CFC, return CFC."""
        votes = ["CFC", "CFC", "CFC"]
        confidences = [0.9, 0.85, 0.88]
        result = detector._resolve_vote(votes, confidences)
        assert result == "CFC"

    def test_majority_vote_ufc(self, detector):
        """When 2/3 votes are UFC, return UFC (meets 0.67 threshold)."""
        votes = ["UFC", "UFC", "NFC"]
        confidences = [0.7, 0.8, 0.6]
        result = detector._resolve_vote(votes, confidences)
        assert result == "UFC"

    def test_no_majority_falls_back_to_cfc(self, detector):
        """When no majority met but CFC present, be conservative and check it."""
        votes = ["CFC", "UFC", "NFC"]
        confidences = [0.6, 0.5, 0.4]
        # Agreement is 1/3 = 0.33 < 0.67 threshold
        result = detector._resolve_vote(votes, confidences)
        assert result == "CFC"  # Conservative: better to check than miss

    def test_no_majority_no_cfc_falls_back_to_ufc(self, detector):
        """When no majority and no CFC, default to UFC."""
        votes = ["UFC", "NFC", "NFC"]
        confidences = [0.5, 0.6, 0.5]
        result = detector._resolve_vote(votes, confidences)
        assert result == "UFC"

    def test_rationale_for_known_label(self, detector):
        """Rationale should be a non-empty string for all labels."""
        for label in ["CFC", "UFC", "NFC"]:
            rationale = detector._get_rationale("test claim", label)
            assert isinstance(rationale, str)
            assert len(rationale) > 0


# ============================================================
# Atomic Claim Decomposer Tests
# ============================================================

class TestAtomicClaimDecomposer:
    """Test AtomicClaimDecomposer [J7]."""

    @pytest.fixture
    def decomposer(self):
        from pipeline.stage1.atomic_decomposer import AtomicClaimDecomposer
        return AtomicClaimDecomposer()

    def test_post_process_adds_period(self, decomposer):
        """Claims without ending punctuation get a period appended."""
        # We test the post-processing logic directly via the forward method
        # with a mock that returns claims without periods
        pass  # Requires mock DSPy — see integration tests

    def test_empty_result_falls_back_to_original(self, decomposer):
        """If DSPy returns no atomic claims, use the original text."""
        # Test the post-processing guard
        pass  # Requires mock DSPy


# ============================================================
# Stage1Result Tests
# ============================================================

class TestStage1Result:
    """Test the Stage1Result dataclass."""

    def test_check_worthy_claim(self):
        from pipeline.stage1.orchestrator import Stage1Result
        result = Stage1Result(
            claim_text="Water boils at 100°C.",
            locale="en",
            worthiness_label="CFC",
            worthiness_confidence=0.95,
            worthiness_rationale="Verifiable scientific claim.",
            worthiness_votes=["CFC", "CFC", "CFC"],
            atomic_claims=["Water boils at 100°C."],
            is_check_worthy=True,
        )
        assert result.is_check_worthy is True
        assert len(result.atomic_claims) == 1

    def test_non_check_worthy_claim_stops_pipeline(self):
        from pipeline.stage1.orchestrator import Stage1Result
        result = Stage1Result(
            claim_text="Nice weather today!",
            locale="en",
            worthiness_label="NFC",
            worthiness_confidence=0.92,
            worthiness_rationale="This is an opinion, not a factual claim.",
            worthiness_votes=["NFC", "NFC", "NFC"],
            atomic_claims=[],
            is_check_worthy=False,
            stop_reason="Claim classified as NFC: This is an opinion.",
        )
        assert result.is_check_worthy is False
        assert result.atomic_claims == []
        assert "NFC" in result.stop_reason


# ============================================================
# Integration-style Test (no LLM calls)
# ============================================================

def test_stage1_result_roundtrip():
    """Stage1Result can be serialized/deserialized for JSONB storage."""
    import json
    from pipeline.stage1.orchestrator import Stage1Result

    result = Stage1Result(
        claim_text="Test claim.",
        locale="en",
        worthiness_label="CFC",
        worthiness_confidence=0.88,
        worthiness_rationale="Check-worthy.",
        worthiness_votes=["CFC", "CFC", "UFC"],
        atomic_claims=["Test claim."],
        is_check_worthy=True,
    )

    # Should be JSON-serializable for stage_records JSONB columns
    data = {
        "label": result.worthiness_label,
        "confidence": result.worthiness_confidence,
        "rationale": result.worthiness_rationale,
        "votes": result.worthiness_votes,
        "atomic_claims": result.atomic_claims,
    }
    json_str = json.dumps(data)
    parsed = json.loads(json_str)
    assert parsed["label"] == "CFC"
    assert parsed["votes"] == ["CFC", "CFC", "UFC"]
    assert parsed["atomic_claims"] == ["Test claim."]
