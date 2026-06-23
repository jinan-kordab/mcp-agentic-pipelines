"""© JINAN KORDAB — 2026 AI HYBRID AGENTIC RETRIEVAL-AUGMENTED GENERATION RAG PIPELINE - PERSONAL PROJECT"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ReportSection:
    section_id: str = ""
    title: str = ""
    content: str = ""
    citations: List[Dict[str, Any]] = field(default_factory=list)
    agent_source: str = ""
    relevance: float = 1.0


class ReportGenerator:
    """Assembles agent results into a structured report with citations."""

    def __init__(self) -> None:
        pass

    async def generate(self, query: str, agent_results: List[Any],
                       veriscore_report: Optional[Any] = None,
                       guardrail_result: Optional[Any] = None) -> dict:
        sections = []
        all_citations = []

        for i, result in enumerate(agent_results):
            if not result.success:
                sections.append({"title": f"Agent: {result.agent_name}",
                                 "content": f"Error: {result.error_message}",
                                 "agent_source": result.agent_name})
                continue

            data = result.data or {}
            
            # Build human-readable content from structured data
            if isinstance(data, dict):
                parts = []
                synth = data.get("synthesis", "")
                if synth:
                    parts.append(str(synth))
                for item in data.get("results", []):
                    if isinstance(item, dict):
                        txt = item.get("text", "")
                        src = item.get("source", "")
                        score = item.get("score", "")
                        mt = item.get("match_type", "")
                        if txt:
                            line = str(txt)
                            if src:
                                line += f"  [{src}]"
                            if score:
                                line += f"  [score={score}, {mt}]"
                            parts.append(line)
                pred = data.get("prediction")
                if pred is not None:
                    parts.append(f"Prediction: {pred}")
                for flag in data.get("flags", []):
                    parts.append(str(flag))
                content = "\n".join(parts) if parts else "(no readable content)"
            else:
                content = str(data)[:1000]

            for citation in getattr(result, "citations", []):
                all_citations.append({**citation, "agent_source": result.agent_name})

            sections.append({"title": f"Findings from {result.agent_name}",
                             "content": content, "agent_source": result.agent_name,
                             "citations": getattr(result, "citations", [])})

        evaluation_summary = None
        if veriscore_report:
            evaluation_summary = {
                "relevancy": getattr(veriscore_report, "relevancy_score", 0),
                "trust": getattr(veriscore_report, "trustworthiness_score", 0),
                "hallucination_rate": getattr(veriscore_report, "hallucination_rate", 0),
                "citation_coverage": getattr(veriscore_report, "citation_coverage", 0),
            }

        guardrail_summary = None
        if guardrail_result:
            guardrail_summary = {
                "action": getattr(guardrail_result, "action", "pass"),
                "issues": getattr(guardrail_result, "issues_found", []),
            }

        return {"query": query, "sections": sections, "citations": all_citations,
                "evaluation": evaluation_summary, "guardrail": guardrail_summary,
                "generated_at": datetime.now().isoformat()}
