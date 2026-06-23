# =============================================================================
# © JINAN KORDAB — 2026 AI HYBRID AGENTIC RETRIEVAL-AUGMENTED GENERATION RAG PIPELINE - PERSONAL PROJECT
# =============================================================================
# All persistent entities in the Precis system.
# Single SQLite database file: data/app.db
#
# Tables:
#   execution_traces  — Complete audit trail per query (JSON events)
#   conversation_turns— Conversation history per session
#   document_index    — Metadata about indexed documents
#   agent_metrics     — Aggregated per-agent performance over time
#
# Related:
#   backend/db/repository.py  — Data access layer
#   backend/core/tracing.py   — TraceCollector (produces trace data)
#   backend/orchestrator/memory.py — Conversation turn storage
# =============================================================================

from datetime import datetime, timezone
from typing import Optional
import uuid

from sqlalchemy import Column, String, Integer, Float, Text, DateTime, ForeignKey, Index
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


# =============================================================================
# ExecutionTrace — The User-Facing Audit Trail
# =============================================================================

class ExecutionTrace(Base):
    """
    Stores the complete audit trail for a single query execution.

    Each row = one user query, with ALL agent decision events stored
    as a JSON array in the events_json column.

    Why one JSON column instead of normalized event rows?
      - Write pattern: All events for a query are generated together.
        One INSERT is faster and simpler than N INSERTs.
      - Read pattern: User always requests the FULL trace for a query.
        No need to join — just SELECT the JSON column.
      - Query pattern: When you DO need to analyze across traces,
        SQLite's json_extract() function can query into the JSON.
    """

    __tablename__ = "execution_traces"

    # Primary key
    trace_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    # Links
    query_id = Column(String(36), nullable=False, index=True)
    session_id = Column(String(36), nullable=True, index=True)

    # Content
    query_text = Column(Text, nullable=False)          # The original user query
    events_json = Column(Text, nullable=False)          # Full trace as JSON string
    status = Column(String(20), nullable=False, default="success")  # success | error

    # Metrics
    duration_ms = Column(Integer, nullable=False, default=0)
    agent_count = Column(Integer, nullable=False, default=0)  # How many agents executed
    event_count = Column(Integer, nullable=False, default=0)  # How many trace events

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    # Index for fast lookup: find all traces for a session
    __table_args__ = (
        Index("idx_traces_session", "session_id", "created_at"),
    )


# =============================================================================
# ConversationTurn — Persistent Conversation History
# =============================================================================

class ConversationTurn(Base):
    """
    Stores a single turn (query + response pair) in a conversation.

    Used by MemoryAgent (backend/orchestrator/memory.py) for Tier 2 persistence.
    Tier 1 is in-memory OrderedDict. Tier 2 is this SQLite table.
    """

    __tablename__ = "conversation_turns"

    turn_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), nullable=False, index=True)
    query_text = Column(Text, nullable=False)
    response_summary = Column(Text, nullable=True)       # LLM-generated compact summary
    trace_id = Column(String(36), nullable=True)          # Link to full execution trace
    plan_used = Column(Text, nullable=True)               # JSON of the ExecutionPlan
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


# =============================================================================
# DocumentIndex — Metadata About Indexed Documents
# =============================================================================

class DocumentIndex(Base):
    """
    Tracks documents that have been ingested into the ExactHash index.

    Used to avoid re-indexing documents and to show the user what's available.
    """

    __tablename__ = "document_index"

    doc_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String(500), nullable=False, unique=True)
    file_hash = Column(String(64), nullable=False)        # SHA-256 of file contents
    page_count = Column(Integer, nullable=False, default=0)
    multitoken_count = Column(Integer, nullable=False, default=0)
    document_text = Column(Text, nullable=True)            # Original text for re-indexing on restart
    corpus_name = Column(String(100), nullable=False, default="default")
    indexed_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


# =============================================================================
# AgentMetrics — Aggregated Performance Over Time
# =============================================================================

class AgentMetrics(Base):
    """
    Aggregated performance metrics per agent, updated periodically.

    Feeds the Evaluation Dashboard (frontend/src/app/evaluate/page.tsx)
    and AgentStatusCard components.
    """

    __tablename__ = "agent_metrics"

    metric_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_name = Column(String(50), nullable=False, index=True)
    metric_name = Column(String(50), nullable=False)       # "avg_latency_ms", "accuracy", etc.
    metric_value = Column(Float, nullable=False)
    sample_count = Column(Integer, nullable=False, default=0)
    computed_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


# =============================================================================
# WorkOrder — Extracted Aviation MRO Work Orders
# =============================================================================

class WorkOrderRecord(Base):
    """Structured work order extracted from aviation maintenance documents."""

    __tablename__ = "work_orders"

    wo_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    source_file = Column(String(500), nullable=False)
    tail_number = Column(String(20), nullable=True, index=True)
    work_order_number = Column(String(50), nullable=True, index=True)
    date = Column(String(30), nullable=True)
    aircraft_model = Column(String(30), nullable=True)
    part_numbers = Column(Text, nullable=True)     # JSON array
    part_descriptions = Column(Text, nullable=True) # JSON array
    serial_numbers = Column(Text, nullable=True)    # JSON array
    mechanic_id = Column(String(30), nullable=True, index=True)
    station = Column(String(10), nullable=True, index=True)
    work_performed = Column(Text, nullable=True)
    hours_worked = Column(String(20), nullable=True)
    ad_sb_references = Column(Text, nullable=True)  # JSON array
    inspector_stamp = Column(String(50), nullable=True)
    extracted_fields_json = Column(Text, nullable=True)  # Full extraction trace
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
