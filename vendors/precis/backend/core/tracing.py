"""© JINAN KORDAB — 2026 AI HYBRID AGENTIC RETRIEVAL-AUGMENTED GENERATION RAG PIPELINE - PERSONAL PROJECT"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Dict, List, Optional, Any


class TraceEventType(str, Enum):
    QUERY_STARTED = "query.started"
    QUERY_COMPLETED = "query.completed"
    QUERY_FAILED = "query.failed"
    PLAN_CREATED = "plan.created"
    AGENT_STARTED = "agent.started"
    AGENT_COMPLETED = "agent.completed"
    AGENT_FAILED = "agent.failed"
    DECISION_SEARCH_TYPE = "decision.search_type"
    DECISION_THRESHOLD = "decision.threshold"
    DECISION_PREDICTION = "decision.prediction"
    LLM_CALL_STARTED = "llm.call_started"
    LLM_CALL_COMPLETED = "llm.call_completed"
    LLM_TOKEN_USAGE = "llm.token_usage"
    EVALUATION_COMPLETED = "evaluation.completed"
    GUARDRAIL_ACTION = "guardrail.action"
    RESULT_FOUND = "result.found"
    ANOMALY_FLAGGED = "anomaly.flagged"
    CITATION_ADDED = "citation.added"


@dataclass
class TraceEvent:
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    event_type: TraceEventType = TraceEventType.QUERY_STARTED
    agent_name: str = ""
    span_id: Optional[str] = None
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"event_id": self.event_id, "event_type": self.event_type.value,
                "agent_name": self.agent_name, "span_id": self.span_id,
                "message": self.message, "data": self.data,
                "timestamp": self.timestamp, "duration_ms": self.duration_ms}


@dataclass
class TraceSpan:
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_span_id: Optional[str] = None
    agent_name: str = ""
    operation: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    events: List[TraceEvent] = field(default_factory=list)
    child_spans: List["TraceSpan"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000 if self.end_time and self.start_time else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"span_id": self.span_id, "parent_span_id": self.parent_span_id,
                "agent_name": self.agent_name, "operation": self.operation,
                "duration_ms": self.duration_ms,
                "events": [e.to_dict() for e in self.events],
                "child_spans": [c.to_dict() for c in self.child_spans],
                "metadata": self.metadata}


class TraceCollector:
    """Collects trace events during query execution. Streams to WebSocket, persists to SQLite."""

    def __init__(self, query_id: str, session_id: Optional[str] = None) -> None:
        self.trace_id = str(uuid.uuid4())
        self.query_id = query_id
        self.session_id = session_id
        self._span_stack: List[TraceSpan] = []
        self._root_spans: List[TraceSpan] = []
        self._events: List[TraceEvent] = []
        self._query_start_time = time.time()
        self._status = "running"
        self._stream_callback: Optional[Callable[[Dict[str, Any]], None]] = None

    def span_start(self, agent_name: str, operation: str, metadata: dict = None) -> str:
        span = TraceSpan(agent_name=agent_name, operation=operation,
                         start_time=time.time(), metadata=metadata or {})
        if self._span_stack:
            span.parent_span_id = self._span_stack[-1].span_id
            self._span_stack[-1].child_spans.append(span)
        else:
            self._root_spans.append(span)
        self._span_stack.append(span)
        return span.span_id

    def span_end(self, metadata: dict = None) -> str:
        if not self._span_stack:
            return ""
        span = self._span_stack.pop()
        span.end_time = time.time()
        if metadata:
            span.metadata.update(metadata)
        return span.span_id

    def event(self, event_type: TraceEventType, agent_name: str = "",
              message: str = "", data: Dict[str, Any] = None,
              duration_ms: float = 0.0) -> TraceEvent:
        evt = TraceEvent(event_type=event_type, agent_name=agent_name,
                         message=message, data=data or {}, duration_ms=duration_ms)
        if self._span_stack:
            evt.span_id = self._span_stack[-1].span_id
            self._span_stack[-1].events.append(evt)
        self._events.append(evt)
        if self._stream_callback:
            self._stream_callback(evt.to_dict())
        return evt

    def set_stream_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        self._stream_callback = callback

    def complete(self, status: str = "success") -> None:
        while self._span_stack:
            self.span_end()
        self._status = status
        self.event(TraceEventType.QUERY_COMPLETED if status == "success" else TraceEventType.QUERY_FAILED,
                   message=f"Query {status}", data={"duration_ms": self.get_total_duration_ms()})

    def to_dict(self) -> Dict[str, Any]:
        return {"trace_id": self.trace_id, "query_id": self.query_id,
                "session_id": self.session_id, "status": self._status,
                "duration_ms": self.get_total_duration_ms(),
                "agent_count": len(self._root_spans),
                "event_count": len(self._events),
                "root_spans": [s.to_dict() for s in self._root_spans],
                "events": [e.to_dict() for e in self._events],
                "created_at": datetime.now(timezone.utc).isoformat()}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    def get_event_count(self) -> int:
        return len(self._events)

    def get_total_duration_ms(self) -> float:
        return (time.time() - self._query_start_time) * 1000
