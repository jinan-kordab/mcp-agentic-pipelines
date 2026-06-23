# Copyright (c) 2026 Jinan Kordab
# SPDX-License-Identifier: MIT

"""
Stage 2a: Search-Decision Generator [J1]
==========================================
Veracity's jewel: the LLM autonomously decides IF web search is needed.
Well-known facts skip search → saves API cost and latency.
If search IS needed → generates NEUTRAL queries (never the original claim).

CRITICAL: This module produces the blind retrieval boundary.
The search queries are factual and neutral — the retriever never sees the claim.
"""

import dspy
from typing import Tuple, List
from pipeline.signatures.signatures import SearchDecisionSignature, QueryGenerationSignature


class SearchDecisionGenerator(dspy.Module):
    """
    DSPy module that decides whether web search is needed and generates
    neutral search queries.

    Jewel [J1] — Veracity's LLM-autonomous search decision:
    - Simple facts ("Water boils at 100°C") → skip search, direct verdict.
    - Complex/current claims → generate neutral queries for blind retrieval.

    Jewel [J2] — Blind Retrieval (Veri-fact.ai):
    - Queries are factual and neutral.
    - NEVER include the original claim text or biased framing.
    - Confirmation bias is prevented at the ARCHITECTURE level.
    """

    def __init__(self):
        super().__init__()
        self.decide = dspy.ChainOfThought(SearchDecisionSignature)
        self.generate_queries = dspy.ChainOfThought(QueryGenerationSignature)

    def forward(self, atomic_claim: str) -> Tuple[bool, List[str], str]:
        """
        Decide if search is needed and generate queries if so.

        Args:
            atomic_claim: A single atomic claim to evaluate.

        Returns:
            needs_search: True if external evidence is needed.
            search_queries: List of neutral search queries (empty if no search).
            reasoning: Why search is or isn't needed.
        """
        # Step 1: Decide if search is needed
        decision = self.decide(atomic_claim=atomic_claim)

        if not decision.needs_search:
            return False, [], decision.reasoning

        # Step 2: Generate neutral search queries
        queries_result = self.generate_queries(atomic_claim=atomic_claim)
        search_queries = [q.strip() for q in queries_result.search_queries if q.strip()]

        if not search_queries:
            return False, [], "Query generation produced no valid queries."

        return True, search_queries, decision.reasoning


# Singleton
search_decision_generator = SearchDecisionGenerator()
