"""© JINAN KORDAB — 2026 AI HYBRID AGENTIC RETRIEVAL-AUGMENTED GENERATION RAG PIPELINE - PERSONAL PROJECT"""

import json
from typing import Dict, List, Optional

from backend.orchestrator.types import ExecutionPlan, SubTask, TaskType
from backend.llm.base import LLMProvider
from backend.core.tracing import TraceCollector, TraceEventType


class PlannerAgent:
    """Decomposes natural language queries into structured execution plans via LLM."""

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    async def plan(self, query: str, available_tools: Optional[List[Dict[str, str]]] = None,
                   conversation_history: Optional[List[Dict[str, str]]] = None,
                   trace: Optional[TraceCollector] = None) -> ExecutionPlan:
        if trace:
            trace.span_start("Planner", "plan")

        tool_desc = ""
        if available_tools:
            tool_desc = "\n".join(f"- {t['name']}: {t['description']}" for t in available_tools)

        system_prompt = (
            "You are a query planning agent. Decompose the user's question into specific subtasks. "
            "Available specialized agents:\n"
            f"{tool_desc}\n\n"
            "Return ONLY valid JSON:\n"
            '{"subtasks": [{"id": "1", "type": "factual_retrieval|data_synthesis", '
            '"query": "specific sub-query", "priority": 1, "depends_on": []}], "reasoning": "why this plan"}\n'
            'Use "factual_retrieval" to search documents for facts. '
            'Use "data_synthesis" to combine multiple results into an answer.'
        )

        user_prompt = f"Query: {query}\n\nPlan this query into subtasks."
        if conversation_history:
            history_text = "\n".join(f"Q: {t['query']}\nA: {t.get('response_summary', '')}"
                                     for t in conversation_history[-5:])
            user_prompt = f"Conversation history:\n{history_text}\n\n{user_prompt}"

        try:
            response = await self.llm.generate(user_prompt, system_prompt=system_prompt, temperature=0.0, max_tokens=300)
            plan = self._parse_response(response, query)
        except Exception:
            plan = self._fallback_plan(query)

        if trace:
            trace.event(TraceEventType.PLAN_CREATED, agent_name="Planner",
                        message=f"Created plan with {len(plan.subtasks)} subtasks",
                        data={"subtask_count": len(plan.subtasks), "reasoning": plan.reasoning})
            trace.span_end()
        return plan

    def _parse_response(self, response: str, query: str) -> ExecutionPlan:
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            start = response.find("{")
            end = response.rfind("}") + 1
            data = json.loads(response[start:end]) if start >= 0 and end > start else {}

        subtasks = []
        for s in data.get("subtasks", []):
            try:
                ttype = TaskType(s["type"])
            except ValueError:
                ttype = TaskType.FACTUAL_RETRIEVAL
            subtasks.append(SubTask(id=s.get("id", str(len(subtasks))), type=ttype,
                                    query=s.get("query", query), priority=s.get("priority", 1),
                                    depends_on=s.get("depends_on", [])))
        return ExecutionPlan(original_query=query, subtasks=subtasks,
                             reasoning=data.get("reasoning", "LLM-generated plan"))

    def _fallback_plan(self, query: str) -> ExecutionPlan:
        return ExecutionPlan(original_query=query, subtasks=[
            SubTask(id="1", type=TaskType.FACTUAL_RETRIEVAL, query=query),
            SubTask(id="2", type=TaskType.CREATIVE_REASONING, query=query, depends_on=["1"]),
        ], reasoning="Fallback plan: retrieve then reason")
