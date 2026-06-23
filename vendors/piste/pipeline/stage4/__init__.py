# Copyright (c) 2026 Jinan Kordab
# SPDX-License-Identifier: MIT

# Stage 4 — Verdict
#   4a: Criticality Gate [C3]          — pipeline/stage4/criticality_gate.py
#   4b: Verdict Aggregator [J5]        — pipeline/stage4/verdict_aggregator.py
#   4c: Editorial Review Panel [C3]    — pipeline/stage4/orchestrator.py
#   Orchestrator: pipeline/stage4/orchestrator.py

from pipeline.stage4.criticality_gate import (
    CriticalityGate, CriticalityAssessment, criticality_gate,
    CRITICAL_TOPICS, HIGH_PROFILE_INDICATORS,
)
from pipeline.stage4.verdict_aggregator import (
    VerdictAggregator, verdict_aggregator, VERDICT_LABELS,
)
from pipeline.stage4.orchestrator import (
    Stage4Orchestrator, Stage4Result, stage4_orchestrator,
)

__all__ = [
    "CriticalityGate",
    "CriticalityAssessment",
    "criticality_gate",
    "CRITICAL_TOPICS",
    "HIGH_PROFILE_INDICATORS",
    "VerdictAggregator",
    "verdict_aggregator",
    "VERDICT_LABELS",
    "Stage4Orchestrator",
    "Stage4Result",
    "stage4_orchestrator",
]
