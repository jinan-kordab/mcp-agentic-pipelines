"""© JINAN KORDAB — 2026 AI HYBRID AGENTIC RETRIEVAL-AUGMENTED GENERATION RAG PIPELINE - PERSONAL PROJECT"""

import json
from datetime import datetime, timezone
from typing import Dict, List, Optional
import uuid

from sqlalchemy import create_engine, desc
from sqlalchemy.orm import Session

from backend.db.models import Base, ExecutionTrace, ConversationTurn, DocumentIndex, AgentMetrics
from backend.config import settings

_engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
    echo=False,
)


def init_db() -> None:
    """Create all tables. Safe to call multiple times."""
    Base.metadata.create_all(bind=_engine)


def get_session() -> Session:
    return Session(_engine)


# ── ExecutionTrace ─────────────────────────────────────────────

def save_trace(query_id: str, query_text: str, events_json: str,
               session_id: Optional[str] = None, status: str = "success",
               duration_ms: int = 0, agent_count: int = 0, event_count: int = 0) -> str:
    """Persist a completed execution trace. Returns trace_id."""
    trace_id = str(uuid.uuid4())
    with get_session() as session:
        trace = ExecutionTrace(
            trace_id=trace_id, query_id=query_id, session_id=session_id,
            query_text=query_text, events_json=events_json, status=status,
            duration_ms=duration_ms, agent_count=agent_count, event_count=event_count,
        )
        session.add(trace)
        session.commit()
    return trace_id


def get_trace(trace_id: str) -> Optional[Dict]:
    """Retrieve a trace by ID. Returns dict or None."""
    with get_session() as session:
        row = session.get(ExecutionTrace, trace_id)
        if row is None:
            return None
        return {
            "trace_id": row.trace_id, "query_id": row.query_id,
            "query_text": row.query_text, "status": row.status,
            "duration_ms": row.duration_ms, "agent_count": row.agent_count,
            "event_count": row.event_count,
            "events_json": json.loads(row.events_json) if row.events_json else {},
            "created_at": row.created_at.isoformat() if row.created_at else "",
        }


def get_traces_for_session(session_id: str, limit: int = 50) -> List[Dict]:
    """Get all traces for a session, most recent first."""
    with get_session() as session:
        rows = (session.query(ExecutionTrace)
                .filter(ExecutionTrace.session_id == session_id)
                .order_by(desc(ExecutionTrace.created_at))
                .limit(limit).all())
        return [_trace_summary(r) for r in rows]


def get_recent_traces(limit: int = 20) -> List[Dict]:
    with get_session() as session:
        rows = (session.query(ExecutionTrace)
                .order_by(desc(ExecutionTrace.created_at)).limit(limit).all())
        return [_trace_summary(r) for r in rows]


def _trace_summary(row: ExecutionTrace) -> Dict:
    return {"trace_id": row.trace_id, "query_id": row.query_id,
            "query_text": row.query_text[:200], "status": row.status,
            "duration_ms": row.duration_ms, "agent_count": row.agent_count,
            "created_at": row.created_at.isoformat() if row.created_at else ""}


# ── Conversation ───────────────────────────────────────────────

def save_conversation_turn(session_id: str, query_text: str, response_summary: str,
                           trace_id: Optional[str] = None, plan_used: Optional[str] = None) -> str:
    turn_id = str(uuid.uuid4())
    with get_session() as session:
        turn = ConversationTurn(turn_id=turn_id, session_id=session_id,
                                query_text=query_text, response_summary=response_summary,
                                trace_id=trace_id, plan_used=plan_used)
        session.add(turn)
        session.commit()
    return turn_id


def get_conversation_history(session_id: str, limit: int = 50) -> List[Dict]:
    with get_session() as session:
        rows = (session.query(ConversationTurn)
                .filter(ConversationTurn.session_id == session_id)
                .order_by(desc(ConversationTurn.created_at)).limit(limit).all())
        return [{"turn_id": r.turn_id, "query_text": r.query_text,
                 "response_summary": r.response_summary,
                 "trace_id": r.trace_id, "created_at": r.created_at.isoformat() if r.created_at else ""}
                for r in rows]


# ── Document Index ─────────────────────────────────────────────

def register_document(filename: str, file_hash: str, page_count: int,
                      multitoken_count: int, corpus_name: str = "default") -> str:
    with get_session() as session:
        existing = session.query(DocumentIndex).filter(DocumentIndex.file_hash == file_hash).first()
        if existing:
            return existing.doc_id
        doc_id = str(uuid.uuid4())
        doc = DocumentIndex(doc_id=doc_id, filename=filename, file_hash=file_hash,
                            page_count=page_count, multitoken_count=multitoken_count,
                            corpus_name=corpus_name)
        session.add(doc)
        session.commit()
    return doc_id


def register_document_simple(filename: str, content_hash: str = "", multi_token_count: int = 0,
                             corpus_name: str = "default", document_text: str = "") -> str:
    """Lightweight registration for uploaded docs (no page count needed)."""
    with get_session() as session:
        existing = session.query(DocumentIndex).filter(
            DocumentIndex.filename == filename,
            DocumentIndex.corpus_name == corpus_name
        ).first()
        if existing:
            existing.multitoken_count = multi_token_count
            if document_text:
                existing.document_text = document_text
            session.commit()
            return existing.doc_id
        doc_id = str(uuid.uuid4())
        doc = DocumentIndex(doc_id=doc_id, filename=filename,
                            file_hash=content_hash or filename,
                            page_count=1, multitoken_count=multi_token_count,
                            corpus_name=corpus_name, document_text=document_text)
        session.add(doc)
        session.commit()
    return doc_id


def get_all_documents(corpus_name: str = "default") -> list:
    with get_session() as session:
        rows = (session.query(DocumentIndex)
                .filter(DocumentIndex.corpus_name == corpus_name)
                .order_by(DocumentIndex.indexed_at.desc()).all())
        return [{"doc_id": r.doc_id, "filename": r.filename,
                 "multitoken_count": r.multitoken_count,
                 "page_count": r.page_count,
                 "indexed_at": r.indexed_at.isoformat() if r.indexed_at else ""}
                for r in rows]


def get_all_document_texts(corpus_name: str = "default") -> list:
    """Retrieve all stored document texts for re-indexing on startup."""
    with get_session() as session:
        rows = (session.query(DocumentIndex)
                .filter(DocumentIndex.corpus_name == corpus_name,
                        DocumentIndex.document_text.isnot(None),
                        DocumentIndex.document_text != "")
                .all())
        return [{"filename": r.filename, "text": r.document_text, "multitoken_count": r.multitoken_count}
                for r in rows]


def get_indexed_documents(corpus_name: str = "default") -> List[Dict]:
    with get_session() as session:
        rows = (session.query(DocumentIndex)
                .filter(DocumentIndex.corpus_name == corpus_name)
                .order_by(desc(DocumentIndex.indexed_at)).all())
        return [{"doc_id": r.doc_id, "filename": r.filename, "page_count": r.page_count,
                 "multitoken_count": r.multitoken_count,
                 "indexed_at": r.indexed_at.isoformat() if r.indexed_at else ""} for r in rows]


# ── Agent Metrics ──────────────────────────────────────────────

def record_agent_metric(agent_name: str, metric_name: str, metric_value: float) -> None:
    with get_session() as session:
        m = AgentMetrics(metric_id=str(uuid.uuid4()), agent_name=agent_name,
                         metric_name=metric_name, metric_value=metric_value, sample_count=1)
        session.add(m)
        session.commit()


def get_agent_metrics(agent_name: str, metric_name: str, limit: int = 100) -> List[Dict]:
    with get_session() as session:
        rows = (session.query(AgentMetrics)
                .filter(AgentMetrics.agent_name == agent_name,
                        AgentMetrics.metric_name == metric_name)
                .order_by(desc(AgentMetrics.computed_at)).limit(limit).all())
        return [{"metric_value": r.metric_value, "computed_at": r.computed_at.isoformat() if r.computed_at else ""}
                for r in rows]


# ── Work Orders ─────────────────────────────────────────────────

def save_work_order(wo) -> str:
    """Save an extracted WorkOrder to the database. Returns work_order_id."""
    import json
    from backend.db.models import WorkOrderRecord
    wo_id = str(uuid.uuid4())
    with get_session() as session:
        record = WorkOrderRecord(
            wo_id=wo_id,
            source_file=wo.source_file,
            tail_number=wo.tail_number,
            work_order_number=wo.work_order_number,
            date=wo.date,
            aircraft_model=wo.aircraft_model,
            part_numbers=json.dumps(wo.part_numbers),
            part_descriptions=json.dumps(wo.part_descriptions),
            serial_numbers=json.dumps(wo.serial_numbers),
            mechanic_id=wo.mechanic_id,
            station=wo.station,
            work_performed=wo.work_performed[:2000] if wo.work_performed else None,
            hours_worked=wo.hours_worked,
            ad_sb_references=json.dumps(wo.ad_sb_references),
            inspector_stamp=wo.inspector_stamp,
            extracted_fields_json=json.dumps([{
                "field_name": f.field_name, "raw_label": f.raw_label,
                "value": f.value, "confidence": f.confidence,
                "page": f.page, "line": f.line_number,
            } for f in wo.extracted_fields]),
        )
        session.add(record)
        session.commit()
    return wo_id


def query_work_orders(tail_number: str = "", mechanic_id: str = "", limit: int = 50) -> list:
    """Query work orders by tail number, mechanic ID, or return recent."""
    from backend.db.models import WorkOrderRecord
    import json
    with get_session() as session:
        q = session.query(WorkOrderRecord)
        if tail_number:
            q = q.filter(WorkOrderRecord.tail_number == tail_number)
        if mechanic_id:
            q = q.filter(WorkOrderRecord.mechanic_id == mechanic_id)
        rows = q.order_by(desc(WorkOrderRecord.created_at)).limit(limit).all()
        return [{
            "wo_id": r.wo_id,
            "source_file": r.source_file,
            "tail_number": r.tail_number,
            "work_order_number": r.work_order_number,
            "date": r.date,
            "aircraft_model": r.aircraft_model,
            "part_numbers": json.loads(r.part_numbers) if r.part_numbers else [],
            "mechanic_id": r.mechanic_id,
            "station": r.station,
            "hours_worked": r.hours_worked,
            "ad_sb_references": json.loads(r.ad_sb_references) if r.ad_sb_references else [],
            "inspector_stamp": r.inspector_stamp,
            "work_performed": (r.work_performed or "")[:300],
            "created_at": r.created_at.isoformat() if r.created_at else "",
        } for r in rows]
