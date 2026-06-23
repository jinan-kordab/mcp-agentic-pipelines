# Copyright (c) 2026 Jinan Kordab
# SPDX-License-Identifier: MIT

"""
Stage 4b: Verdict Aggregator [J5]
====================================
DSPy-powered synthesis of per-source classifications into a
7-way PolitiFact-aligned verdict.

Jewel [J5] — DSPy typed Signatures with auto-optimization:
  - Weighted by source credibility scores
  - Generates natural language explanation with citations
  - Returns probability distribution over all 7 verdict labels
  - Model-agnostic (swap LLMs without changing code)

Verdict labels:
  TRUE, MOSTLY_TRUE, HALF_TRUE, MOSTLY_FALSE, FALSE, PANTS_ON_FIRE, UNVERIFIABLE
"""

import json
import dspy
from typing import Tuple, Dict
from pipeline.signatures.signatures import VerdictAggregationSignature
from pipeline.stage3.orchestrator import ClassificationResult

# 7-way PolitiFact-aligned verdict labels
VERDICT_LABELS = [
    "TRUE",
    "MOSTLY_TRUE",
    "HALF_TRUE",
    "MOSTLY_FALSE",
    "FALSE",
    "PANTS_ON_FIRE",
    "UNVERIFIABLE",
]


class VerdictAggregator(dspy.Module):
    """
    DSPy module that synthesizes per-source classifications into a
    final 7-way verdict with explanation and probability distribution.

    Jewel [J5] — DSPy Compiler + Framework:
    Typed Signatures, model-agnostic, compiler-optimizable.
    Loop 3: re-optimized with user feedback labels.
    """

    def __init__(self):
        super().__init__()
        self.aggregate = dspy.ChainOfThought(VerdictAggregationSignature)

    def forward(
        self,
        claim: str,
        classifications: list[ClassificationResult],
        locale: str = "en",
    ) -> Tuple[str, float, str, Dict[str, float]]:
        """
        Aggregate per-source classifications into final verdict.

        Args:
            claim: The atomic claim being verified.
            classifications: Per-source classification results from Stage 3.
            locale: Language locale for the response (en, fr).

        Returns:
            verdict: One of the 7 verdict labels
            confidence: 0.0–1.0 overall confidence
            explanation: Natural language explanation with citations
            distribution: Probability weight for each verdict label
        """
        # Build weighted classification summary
        classifications_json = self._build_classifications_payload(
            claim, classifications
        )

        result = self.aggregate(
            claim=claim,
            classifications_json=classifications_json,
            locale=locale,
        )

        # Normalize verdict
        verdict = result.verdict.strip().upper()
        verdict = self._normalize_verdict(verdict)

        # Parse distribution
        try:
            distribution = json.loads(result.distribution_json)
        except (json.JSONDecodeError, TypeError):
            distribution = self._default_distribution(verdict)

        # Ensure all labels are present
        for label in VERDICT_LABELS:
            if label not in distribution:
                distribution[label] = 0.0

        return (
            verdict,
            float(result.confidence),
            result.explanation,
            distribution,
        )

    def _build_classifications_payload(
        self,
        claim: str,
        classifications: list[ClassificationResult],
    ) -> str:
        """Build JSON payload summarizing per-source classifications."""
        summary = []
        for i, c in enumerate(classifications):
            summary.append({
                "source_index": i,
                "source_domain": c.source_domain,
                "credibility_score": c.credibility_score,
                "label": c.label,
                "confidence": c.confidence,
                "rationale": c.rationale,
            })
        return json.dumps(summary, indent=2)

    def _normalize_verdict(self, raw: str) -> str:
        """Normalize LLM output to a valid verdict label."""
        raw_upper = raw.upper().strip()

        # Direct match
        if raw_upper in VERDICT_LABELS:
            return raw_upper

        # Fuzzy match
        mapping = {
            "TRUE": "TRUE",
            "MOSTLY TRUE": "MOSTLY_TRUE",
            "MOSTLYTRUE": "MOSTLY_TRUE",
            "HALF TRUE": "HALF_TRUE",
            "HALFTRUE": "HALF_TRUE",
            "MOSTLY FALSE": "MOSTLY_FALSE",
            "MOSTLYFALSE": "MOSTLY_FALSE",
            "FALSE": "FALSE",
            "PANTS ON FIRE": "PANTS_ON_FIRE",
            "PANTSONFIRE": "PANTS_ON_FIRE",
            "UNVERIFIABLE": "UNVERIFIABLE",
            "NOT ENOUGH INFORMATION": "UNVERIFIABLE",
            "NEI": "UNVERIFIABLE",
        }
        return mapping.get(raw_upper, "UNVERIFIABLE")

    def _default_distribution(self, verdict: str) -> Dict[str, float]:
        """Create a default distribution centered on the given verdict."""
        dist = {label: 0.0 for label in VERDICT_LABELS}
        dist[verdict] = 1.0
        return dist


# Singleton
verdict_aggregator = VerdictAggregator()
