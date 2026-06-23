# Copyright (c) 2026 Jinan Kordab
# SPDX-License-Identifier: MIT

"""
Stage 3: Per-Source Classification [J3][J8]
==============================================
Aletheia's jewel: each evidence source is independently classified
as SUPPORTS, REFUTES, or UNRELATED to the claim BEFORE aggregation.

Run in parallel via asyncio.gather — N classifiers = N simultaneous LLM calls.
This provides:
  - Full audit trail: every source's contribution is explicit
  - Debuggability: errors isolated to single classifications
  - Parallelism: independent evaluations run concurrently
"""

import dspy
from typing import Tuple
from pipeline.signatures.signatures import SourceClassificationSignature
from pipeline.stage2.canonical_mapper import CanonicalEvidence


class SourceClassifier(dspy.Module):
    """
    DSPy module that classifies a single evidence source relative to a claim.

    Jewel [J3] — Aletheia's structured per-source classification:
    - Each source evaluated independently
    - Label: SUPPORTS / REFUTES / UNRELATED
    - Returns confidence + rationale for audit trail [C5]
    """

    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought(SourceClassificationSignature)

    def forward(
        self,
        claim: str,
        evidence: CanonicalEvidence,
        locale: str = "en",
    ) -> Tuple[str, float, str]:
        """
        Classify one source against the claim.

        Args:
            claim: The atomic claim being verified.
            evidence: A single CanonicalEvidence source.
            locale: Language locale for the response (en, fr).

        Returns:
            label: "SUPPORTS", "REFUTES", or "UNRELATED"
            confidence: 0.0–1.0
            rationale: Explanation of the classification
        """
        result = self.classify(
            claim=claim,
            evidence_title=evidence.title,
            evidence_excerpt=evidence.excerpt,
            source_domain=evidence.source_domain,
            credibility_score=evidence.credibility_score,
            locale=locale,
        )

        label = result.label.strip().upper()
        # Normalize to valid labels
        if label not in ("SUPPORTS", "REFUTES", "UNRELATED"):
            if "SUPPORT" in label:
                label = "SUPPORTS"
            elif "REFUT" in label:
                label = "REFUTES"
            else:
                label = "UNRELATED"

        return label, float(result.confidence), result.rationale


# Singleton
source_classifier = SourceClassifier()
