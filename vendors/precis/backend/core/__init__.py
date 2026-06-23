# =============================================================================
# © JINAN KORDAB — 2026 AI HYBRID AGENTIC RETRIEVAL-AUGMENTED GENERATION RAG PIPELINE - PERSONAL PROJECT
# =============================================================================
# backend/core/ — Low-level utilities used across all agents.
# These are pure functions with no external dependencies (other than NLTK/NumPy).
#
# Modules:
#   stemming.py   — NLTK-based stemmer for multi-token normalization
#   multitoken.py — MultiToken extraction from parsed documents
#   hashing.py    — Nested hash utilities for O(1) lookup
#   pmi.py        — Pointwise Mutual Information for relevancy scoring
#   metrics.py    — GenAI evaluation metrics (relevancy, trust, exhaustivity)
# =============================================================================
