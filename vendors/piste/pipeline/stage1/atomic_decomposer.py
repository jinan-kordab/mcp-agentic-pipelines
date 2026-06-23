# Copyright (c) 2026 Jinan Kordab
# SPDX-License-Identifier: MIT

"""
Stage 1b: Atomic Claim Decomposer [J7]
========================================
FACT5's jewel: decompose compound claims into independent atomic claims.
Each atomic claim focuses on ONE verifiable fact.

Example:
  "We created 800,000 jobs and cut taxes by 20%"
  → ["We created 800,000 jobs.", "We cut taxes by 20%."]
"""

import dspy
from typing import List
from pipeline.signatures.signatures import AtomicClaimDecompositionSignature


class AtomicClaimDecomposer(dspy.Module):
    """
    DSPy module that splits compound claims into atomic, independently
    verifiable claims.

    Jewel [J7] — FACT5's atomization:
    A holistic verdict on a compound claim is meaningless — some parts
    may be true, others false. Atomize first, then verify each sub-claim
    independently.
    """

    def __init__(self):
        super().__init__()
        self.decompose = dspy.ChainOfThought(AtomicClaimDecompositionSignature)

    def forward(self, claim_text: str) -> List[str]:
        """
        Decompose a claim into atomic sub-claims.

        Returns:
            List of independent atomic claims, each a single verifiable fact.
            If the claim is already atomic, returns a single-element list.
        """
        result = self.decompose(claim_text=claim_text)

        # Post-process: ensure each atomic claim is a complete sentence
        atomic_claims = []
        for claim in result.atomic_claims:
            claim = claim.strip()
            if claim and not claim.endswith((".", "!", "?")):
                claim += "."
            atomic_claims.append(claim)

        # If decomposition returned nothing useful, treat original as atomic
        if not atomic_claims:
            atomic_claims = [claim_text.strip()]

        return atomic_claims


# Singleton instance
atomic_claim_decomposer = AtomicClaimDecomposer()
