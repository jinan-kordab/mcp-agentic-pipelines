# Copyright (c) 2026 Jinan Kordab
# SPDX-License-Identifier: MIT

"""
DSPy Signatures — Typed Interfaces for the Piste Pipeline
==========================================================
Every LLM call in the pipeline is defined as a typed DSPy Signature.
This makes modules model-agnostic, testable, and compiler-optimizable.
"""

import dspy


# --- Stage 1: Claim Processing ---

class CheckWorthinessSignature(dspy.Signature):
    """Classify whether a claim is worth fact-checking (CFC/UFC/NFC).

    Jewel [J4] — ClaimBuster's pre-filter: find the needles before examining them.
    """

    claim_text: str = dspy.InputField(desc="The raw claim text to evaluate")
    locale: str = dspy.InputField(desc="Language locale of the claim (en, fr, es, ...)")

    label: str = dspy.OutputField(
        desc="CFC (Check-worthy Factual Claim), UFC (Unimportant Factual Claim), or NFC (Non-Factual Claim)"
    )
    confidence: float = dspy.OutputField(desc="Confidence in the classification (0.0–1.0)")
    rationale: str = dspy.OutputField(desc="Brief explanation of why this classification was assigned")


class AtomicClaimDecompositionSignature(dspy.Signature):
    """Decompose a compound claim into independent atomic claims.

    Jewel [J7] — FACT5's atomization: each atomic claim focuses on one verifiable fact.
    """

    claim_text: str = dspy.InputField(desc="The claim text to decompose (may be compound)")

    atomic_claims: list[str] = dspy.OutputField(
        desc="List of independent atomic claims, each a single verifiable fact"
    )


# --- Stage 2: Blind Retrieval ---

class SearchDecisionSignature(dspy.Signature):
    """Decide whether web search is needed for this claim.

    Jewel [J1] — Veracity's LLM-autonomous search decision:
    skip search for well-known facts to save cost and latency.
    """

    atomic_claim: str = dspy.InputField(desc="A single atomic claim to evaluate")

    needs_search: bool = dspy.OutputField(
        desc="True if external evidence is needed; False if answerable from parametric knowledge"
    )
    reasoning: str = dspy.OutputField(desc="Why search is or is not needed")


class QueryGenerationSignature(dspy.Signature):
    """Generate NEUTRAL search queries from a claim.

    Jewel [J2] — Blind Retrieval: queries must be factual and neutral.
    NEVER include the original claim text — prevents confirmation bias.
    """

    atomic_claim: str = dspy.InputField(desc="The atomic claim to search evidence for")

    search_queries: list[str] = dspy.OutputField(
        desc="Neutral, factual search queries. Must NOT contain the claim text or biased framing."
    )


class QueryRefinementSignature(dspy.Signature):
    """Analyze why previous search was insufficient and generate refined queries.

    Jewel [J8c] — ClaimeAI's feedback-driven iterative query refinement.
    Loop 1: seconds-scale retry with intelligent query adjustment.
    """

    original_query: str = dspy.InputField(desc="The query that returned insufficient results")
    insufficient_reason: str = dspy.InputField(desc="Why the previous results were insufficient")

    refined_queries: list[str] = dspy.OutputField(
        desc="New, refined search queries targeting the identified gaps"
    )


# --- Stage 3: Per-Source Classification ---

class SourceClassificationSignature(dspy.Signature):
    """Classify a single evidence source as supporting, refuting, or unrelated to a claim.

    Jewel [J3] — Aletheia's structured per-source classification:
    each source evaluated independently BEFORE aggregation.
    """

    claim: str = dspy.InputField(desc="The atomic claim being verified")
    evidence_title: str = dspy.InputField(desc="Title of the evidence source")
    evidence_excerpt: str = dspy.InputField(desc="Relevant excerpt from the evidence source")
    source_domain: str = dspy.InputField(desc="Domain name of the source (e.g., bbc.com)")
    credibility_score: float = dspy.InputField(desc="Pre-computed domain credibility (0.0–1.0)")
    locale: str = dspy.InputField(desc="Language locale for the response (en, fr, ...)")

    label: str = dspy.OutputField(desc="SUPPORTS, REFUTES, or UNRELATED")
    confidence: float = dspy.OutputField(desc="Confidence in the classification (0.0–1.0)")
    rationale: str = dspy.OutputField(desc="Brief explanation of why this label was assigned")


# --- Stage 4: Verdict Aggregation ---

class VerdictAggregationSignature(dspy.Signature):
    """Synthesize per-source classifications into a final 7-way verdict.

    Jewel [J5] — DSPy-powered aggregation with PolitiFact-aligned granularity.
    Weighted by source credibility scores.
    """

    claim: str = dspy.InputField(desc="The atomic claim being verified")
    classifications_json: str = dspy.InputField(
        desc="JSON array of per-source classifications with labels, confidences, and rationales"
    )
    locale: str = dspy.InputField(desc="Language locale for the response (en, fr, ...)")

    verdict: str = dspy.OutputField(
        desc="TRUE, MOSTLY_TRUE, HALF_TRUE, MOSTLY_FALSE, FALSE, PANTS_ON_FIRE, or UNVERIFIABLE"
    )
    confidence: float = dspy.OutputField(desc="Overall confidence in the verdict (0.0–1.0)")
    explanation: str = dspy.OutputField(
        desc="Natural language explanation of the verdict with source citations"
    )
    distribution_json: str = dspy.OutputField(
        desc="JSON object mapping each verdict label to its probability weight"
    )
