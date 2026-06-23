# Copyright (c) 2026 Jinan Kordab
# SPDX-License-Identifier: MIT

# Offline — VERIFAID Dataset Pipeline [J7]
#   M1: Generate Claims (LLM, multilingual)   — pipeline/offline/verifaid_pipeline.py
#   M2: Enrich + Label + FAISS Index          — pipeline/offline/verifaid_pipeline.py
#   Scheduler + Loop 2 integration            — pipeline/offline/verifaid_pipeline.py

from pipeline.offline.verifaid_pipeline import (
    ClaimGenerator, claim_generator,
    EvidenceEnricher, evidence_enricher,
    OfflinePipelineScheduler, offline_scheduler,
    ClaimGenerationSignature, EvidenceLabelingSignature,
    TOPICS, LOCALES,
)

__all__ = [
    "ClaimGenerator",
    "claim_generator",
    "EvidenceEnricher",
    "evidence_enricher",
    "OfflinePipelineScheduler",
    "offline_scheduler",
    "ClaimGenerationSignature",
    "EvidenceLabelingSignature",
    "TOPICS",
    "LOCALES",
]
