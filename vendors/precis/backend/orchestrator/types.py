"""© JINAN KORDAB — 2026 AI HYBRID AGENTIC RETRIEVAL-AUGMENTED GENERATION RAG PIPELINE - PERSONAL PROJECT"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class TaskType(str, Enum):
    FACTUAL_RETRIEVAL = "factual_retrieval"
    ANOMALY_DETECTION = "anomaly_detection"
    PREDICTION = "prediction"
    DATA_SYNTHESIS = "data_synthesis"
    CREATIVE_REASONING = "creative_reasoning"
    EVALUATION = "evaluation"


@dataclass
class SubTask:
    id: str
    type: TaskType
    query: str
    context: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    depends_on: List[str] = field(default_factory=list)


@dataclass
class ExecutionPlan:
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    original_query: str = ""
    subtasks: List[SubTask] = field(default_factory=list)
    reasoning: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    subtask_id: str = ""
    agent_name: str = ""
    success: bool = True
    data: Any = None
    citations: List[Dict[str, Any]] = field(default_factory=list)
    error_message: str = ""
    execution_time_ms: float = 0.0


@dataclass
class FinalReport:
    query: str = ""
    narrative: str = ""
    agent_results: List[AgentResult] = field(default_factory=list)
    evaluation: Optional[Any] = None
    citations: List[Dict[str, Any]] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)
    execution_plan: Optional[ExecutionPlan] = None
