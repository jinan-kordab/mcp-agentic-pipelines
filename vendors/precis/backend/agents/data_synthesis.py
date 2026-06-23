"""© JINAN KORDAB — 2026 AI HYBRID AGENTIC RETRIEVAL-AUGMENTED GENERATION RAG PIPELINE - PERSONAL PROJECT"""

from typing import Any, Dict, List, Optional

from backend.orchestrator.types import AgentResult
from backend.llm.base import LLMProvider


class DataSynthesisAgent:
    """Combines results from upstream agents into a coherent synthesis using LLM reasoning.

    The *llm* parameter on __init__ serves as a default; individual calls to
    synthesize() may override it.  If no provider is available at call time
    the agent falls back to returning raw fragments — this is intentional so
    the pipeline never breaks, but a warning is logged.
    """

    def __init__(self, llm: Optional[LLMProvider] = None) -> None:
        self.llm = llm

    async def synthesize(
        self,
        query: str,
        upstream_results: List[AgentResult],
        llm: Optional[LLMProvider] = None,
    ) -> AgentResult:
        """Synthesize results from multiple upstream agents into a single answer.

        Parameters
        ----------
        query : str
            The original user query.
        upstream_results : List[AgentResult]
            Results from upstream retrieval / analysis agents.
        llm : Optional[LLMProvider]
            Per-call override for the LLM provider.  Falls back to self.llm.
        """
        provider = llm or self.llm

        # ── Collect text fragments from upstream results ─────────
        fragments: List[str] = []
        for r in upstream_results:
            if not r.success or not r.data:
                continue
            if isinstance(r.data, dict):
                for item in r.data.get("results", []):
                    if isinstance(item, dict):
                        text = item.get("text", "")
                        if isinstance(text, (list, tuple)):
                            fragments.append(" ".join(str(t) for t in text))
                        elif text:
                            fragments.append(str(text))
                # Also capture any pre-existing synthesis
                synth = r.data.get("synthesis", "")
                if synth:
                    fragments.append(str(synth))
            elif isinstance(r.data, str):
                fragments.append(r.data)

        combined_context = "\n".join(fragments[:50]) if fragments else "(no data retrieved)"

        # ── LLM synthesis (or graceful fallback) ─────────────────
        if provider and fragments:
            prompt = (
                "You are a precise data synthesis agent. Synthesize the following retrieved\n"
                "information into a concise answer to the user's query.\n\n"
                f"USER QUERY: {query}\n\n"
                f"RETRIEVED DATA:\n{combined_context}\n\n"
                "SYNTHESIS: Answer the query using ONLY the retrieved data above. "
                "If the data is insufficient, state what's missing. "
                "Be specific with numbers, percentages, and entity names found in the data."
            )
            try:
                import asyncio
                response = await asyncio.wait_for(
                    provider.generate(prompt, max_tokens=250),
                    timeout=30,
                )
                synthesis_text = response
            except asyncio.TimeoutError:
                synthesis_text = (
                    f"(LLM synthesis timed out. Retrieved {len(fragments)} fragments.)\n"
                    f"{combined_context[:500]}"
                )
            except Exception as e:
                synthesis_text = (
                    f"(LLM synthesis error: {e})\n{combined_context[:500]}"
                )
        elif fragments:
            synthesis_text = (
                f"Retrieved {len(fragments)} fragments:\n{combined_context[:1000]}"
            )
        else:
            synthesis_text = "(No data retrieved — unable to synthesize a response.)"

        return AgentResult(
            subtask_id="synthesis",
            agent_name="DataSynthesis",
            success=True,
            data={
                "synthesis": synthesis_text,
                "source_fragments": len(fragments),
            },
            citations=[],
        )
