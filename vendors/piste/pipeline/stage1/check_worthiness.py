# Copyright (c) 2026 Jinan Kordab
# SPDX-License-Identifier: MIT

"""
Stage 1a: Check-Worthiness Detector [J4]
==========================================
ClaimBuster's jewel: classify whether a claim is worth fact-checking.
Uses DSPy ChainOfThought with voting (3 completions, majority wins).
If agreement < threshold → safe default (UFC).

Verdict: CFC (Check-worthy Factual Claim), UFC (Unimportant Factual Claim),
         or NFC (Non-Factual Claim).
"""

import dspy
from typing import Literal, Tuple
from pipeline.signatures.signatures import CheckWorthinessSignature
from app.core.config import settings


class CheckWorthinessDetector(dspy.Module):
    """
    DSPy module that classifies whether a claim is worth fact-checking.

    Jewel [J4] — ClaimBuster's pre-filter:
    In the real world, claims are embedded in a firehose of text.
    You must find the needles (check-worthy claims) before examining them.

    Voting mechanism: 3 independent LLM completions, majority wins.
    If no majority meets the threshold → default to UFC (safe).
    """

    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought(CheckWorthinessSignature)
        self.voting_completions: int = settings.VOTING_COMPLETIONS
        self.voting_threshold: float = settings.VOTING_THRESHOLD

    def forward(
        self, claim_text: str, locale: str = "en"
    ) -> Tuple[str, float, str, list[str]]:
        """
        Classify claim check-worthiness with majority voting.

        Returns:
            label: "CFC", "UFC", or "NFC"
            confidence: 0.0–1.0
            rationale: Explanation of the classification
            votes: Raw individual votes for audit trail [C5]
        """
        votes: list[str] = []
        confidences: list[float] = []

        # Run N independent completions
        for _ in range(self.voting_completions):
            result = self.classify(claim_text=claim_text, locale=locale)
            votes.append(result.label.strip().upper())
            confidences.append(float(result.confidence))

        # Majority vote
        final_label = self._resolve_vote(votes, confidences)
        avg_confidence = sum(confidences) / len(confidences)

        # Get rationale from the majority-vote completion
        rationale = self._get_rationale(claim_text, final_label)

        return final_label, avg_confidence, rationale, votes

    def _resolve_vote(
        self, votes: list[str], confidences: list[float]
    ) -> str:
        """Determine final label via majority vote with configurable threshold."""
        from collections import Counter

        counts = Counter(votes)
        most_common_label, most_common_count = counts.most_common(1)[0]
        agreement_ratio = most_common_count / len(votes)

        if agreement_ratio >= self.voting_threshold:
            return most_common_label
        else:
            # No majority met → safe default
            # If any vote was CFC, be conservative and still check it
            if "CFC" in votes:
                return "CFC"
            return "UFC"

    def _get_rationale(self, claim_text: str, label: str) -> str:
        """Generate a concise rationale for the final classification."""
        rationale_map = {
            "CFC": "This claim contains a verifiable factual assertion that warrants evidence-based checking.",
            "UFC": "This claim is factual but trivial or not of public interest — fact-checking resources are better allocated elsewhere.",
            "NFC": "This is an opinion, question, or non-factual statement — there is no verifiable claim to check.",
        }
        # Optionally, re-query LLM for a more specific rationale
        return rationale_map.get(label, "Classification complete.")


# Singleton instance
check_worthiness_detector = CheckWorthinessDetector()
