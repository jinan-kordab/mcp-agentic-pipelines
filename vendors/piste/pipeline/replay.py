# Copyright (c) 2026 Jinan Kordab
# SPDX-License-Identifier: MIT

"""
Replay Engine [C5] — Re-exports from full implementation.
==========================================================
See pipeline/replay_engine.py for the full ReplayEngine class
with replay_run(), compare_verdicts(), and rollback_pipeline_version().
"""

from pipeline.replay_engine import (
    ReplayEngine, ReplayComparison, StageDiff,
)

__all__ = ["ReplayEngine", "ReplayComparison", "StageDiff"]
