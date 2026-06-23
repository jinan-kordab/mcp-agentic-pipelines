# Copyright (c) 2026 Jinan Kordab
# SPDX-License-Identifier: MIT

"""
VERIFAID Offline Dataset Pipeline [J7]
=========================================
M1: Generate diverse factual claims (LLM, multilingual)
M2: Enrich + Label + FAISS Index

Jewel [J7] — VERIFAID's dataset creation as first-class module:
  - No data bottleneck: system generates its own training data
  - Self-improving: verified claims enrich the knowledge base
  - Freshness: claims about current events generated and indexed immediately
  - Domain expansion: generate claims on any topic, in any language

Scheduled weekly to refresh the FAISS Tier-1 evidence cache.
Loop 2: newly verified claims automatically enrich the offline index.
"""

import dspy
import json
import asyncio
from datetime import datetime
from typing import List
from dataclasses import dataclass, field

import numpy as np
from app.services.vector_store import faiss_store


# ============================================================
# DSPy Signatures
# ============================================================

class ClaimGenerationSignature(dspy.Signature):
    """Generate diverse factual claims for dataset creation."""
    topic: str = dspy.InputField(desc="Topic domain to generate claims about")
    locale: str = dspy.InputField(desc="Language locale for generated claims")
    count: int = dspy.InputField(desc="Number of claims to generate")

    claims: list[str] = dspy.OutputField(desc="Generated factual claims")


class EvidenceLabelingSignature(dspy.Signature):
    """Generate evidence and labels for a claim."""
    claim: str = dspy.InputField(desc="The claim to generate evidence for")

    evidence_text: str = dspy.OutputField(desc="Synthetic evidence text supporting or refuting the claim")
    label: str = dspy.OutputField(desc="TRUE, FALSE, or UNVERIFIABLE")
    explanation: str = dspy.OutputField(desc="Explanation of the label")


# ============================================================
# M1: Claim Generator
# ============================================================

TOPICS = [
    "science", "technology", "health", "politics", "economics",
    "environment", "education", "sports", "entertainment", "history",
]
LOCALES = ["en", "fr", "es"]


class ClaimGenerator(dspy.Module):
    """
    M1: Generate diverse factual claims across topics and languages.

    Uses DSPy to generate claims that cover a wide range of domains,
    ensuring the offline index has broad coverage.
    """

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(ClaimGenerationSignature)

    def forward(self, topic: str, locale: str = "en", count: int = 10) -> List[str]:
        result = self.generate(topic=topic, locale=locale, count=count)
        return [c.strip() for c in result.claims if c.strip()]

    def generate_all(self, topics: List[str] = None, locales: List[str] = None,
                     per_topic: int = 10) -> List[dict]:
        """Generate claims across all topics and locales."""
        topics = topics or TOPICS
        locales = locales or LOCALES
        all_claims = []

        for topic in topics:
            for locale in locales:
                claims = self(topic=topic, locale=locale, count=per_topic)
                for claim_text in claims:
                    all_claims.append({
                        "topic": topic,
                        "locale": locale,
                        "claim_text": claim_text,
                        "generated_at": datetime.utcnow().isoformat(),
                    })

        return all_claims


# ============================================================
# M2: Evidence Enricher
# ============================================================

class EvidenceEnricher(dspy.Module):
    """
    M2: Generate evidence, labels, and FAISS vectors for claims.

    For each generated claim, this module:
      1. Produces synthetic evidence text
      2. Assigns a label (TRUE/FALSE/UNVERIFIABLE)
      3. Creates a FAISS vector for semantic retrieval
      4. Stores metadata in the vector index
    """

    def __init__(self):
        super().__init__()
        self.label = dspy.ChainOfThought(EvidenceLabelingSignature)

    def forward(self, claim: str) -> dict:
        """Generate evidence and label for a single claim."""
        result = self.label(claim=claim)
        return {
            "claim": claim,
            "evidence": result.evidence_text,
            "label": result.label.strip().upper(),
            "explanation": result.explanation,
            "enriched_at": datetime.utcnow().isoformat(),
        }

    async def enrich_batch(self, claims: List[dict], embed_fn=None) -> int:
        """
        Enrich a batch of claims and index them in FAISS.

        Args:
            claims: List of claim dicts from ClaimGenerator.
            embed_fn: Function to convert text → embedding vector (default: random for demo).

        Returns:
            Number of claims indexed.
        """
        indexed = 0
        vectors = []
        metadata_list = []

        for item in claims:
            try:
                enriched = self(item["claim_text"])

                # Generate embedding (placeholder — use real embedding model in production)
                if embed_fn:
                    vector = embed_fn(enriched["claim"])
                else:
                    # Random 1536-dim unit vector as placeholder
                    vector = np.random.randn(1536).astype("float32")
                    vector = vector / np.linalg.norm(vector)

                vectors.append(vector)
                metadata_list.append({
                    "claim_text": enriched["claim"],
                    "evidence": enriched["evidence"],
                    "label": enriched["label"],
                    "explanation": enriched["explanation"],
                    "topic": item.get("topic", ""),
                    "locale": item.get("locale", "en"),
                    "enriched_at": enriched["enriched_at"],
                })
                indexed += 1
            except Exception:
                continue  # Skip failed enrichments

        if vectors:
            faiss_store.add_vectors(
                np.array(vectors, dtype="float32"),
                metadata_list,
            )
            faiss_store.save()

        return indexed


# ============================================================
# Offline Pipeline Scheduler
# ============================================================

class OfflinePipelineScheduler:
    """
    Scheduled execution of the VERIFAID offline pipeline.

    Runs weekly to:
      1. Generate new claims across topics/locales (M1)
      2. Enrich + label + index in FAISS (M2)
      3. Loop 2: ingest newly verified claims from online pipeline
    """

    def __init__(self):
        self.generator = ClaimGenerator()
        self.enricher = EvidenceEnricher()

    async def run_weekly_job(self):
        """Execute the full offline pipeline."""
        print(f"[VERIFAID] Starting weekly dataset pipeline: {datetime.utcnow().isoformat()}")

        # M1: Generate claims
        claims = self.generator.generate_all()
        print(f"[VERIFAID] M1 complete: {len(claims)} claims generated")

        # M2: Enrich + index
        indexed = await self.enricher.enrich_batch(claims)
        print(f"[VERIFAID] M2 complete: {indexed} claims indexed in FAISS")

        # Loop 2: ingest verified claims
        # (Phase 8 — reads from PostgreSQL verdicts table)
        print(f"[VERIFAID] FAISS index size: {len(faiss_store)} vectors")

        return {"claims_generated": len(claims), "claims_indexed": indexed}

    async def ingest_verified_claim(self, claim_text: str, verdict_data: dict):
        """
        Loop 2 integration: ingest a newly verified claim into FAISS.

        Called after each successful pipeline verdict.
        """
        enriched = self.enricher(claim_text)

        vector = np.random.randn(1536).astype("float32")
        vector = vector / np.linalg.norm(vector)

        faiss_store.add_vectors(
            np.array([vector], dtype="float32"),
            [{
                "claim_text": claim_text,
                "evidence": enriched["evidence"],
                "label": verdict_data.get("verdict", enriched["label"]),
                "explanation": verdict_data.get("explanation", enriched["explanation"]),
                "verified": True,
                "run_id": verdict_data.get("run_id"),
            }],
        )

        print(f"[VERIFAID] Loop 2: indexed verified claim in FAISS")


# Singletons
claim_generator = ClaimGenerator()
evidence_enricher = EvidenceEnricher()
offline_scheduler = OfflinePipelineScheduler()
