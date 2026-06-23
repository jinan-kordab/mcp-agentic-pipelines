"""© JINAN KORDAB — 2026 AI HYBRID AGENTIC RETRIEVAL-AUGMENTED GENERATION RAG PIPELINE - PERSONAL PROJECT"""

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Type

from backend.orchestrator.types import AgentResult, SubTask, TaskType
from backend.llm.base import LLMProvider
from backend.core.tracing import TraceCollector, TraceEventType


@dataclass
class AgentRegistryEntry:
    name: str
    task_type: TaskType
    agent_class: Type
    description: str
    is_external_llm: bool = False
    singleton_instance: object = None  # Pre-built instance to reuse (e.g., seeded index)


class AgentRegistry:
    """Registry mapping TaskType → specialized agent class."""

    def __init__(self) -> None:
        self._registry: Dict[TaskType, AgentRegistryEntry] = {}

    def register(self, entry: AgentRegistryEntry) -> None:
        self._registry[entry.task_type] = entry

    def get(self, task_type: TaskType) -> Optional[AgentRegistryEntry]:
        return self._registry.get(task_type)

    def list_tools(self) -> List[Dict[str, str]]:
        return [{"name": e.name, "description": e.description} for e in self._registry.values()]


class RouterAgent:
    """Routes SubTasks to agents. Handles parallel execution with dependency ordering."""
    
    # ── Section-number helpers ─────────────────────────────────────
    
    @staticmethod
    def _next_section_candidates(tokens: tuple) -> list:
        """Given a section-number token like '3.4', yield likely next
        section numbers: '3.5', then '4'.  Used to find section
        boundaries via the hash index."""
        import re
        for t in tokens:
            m = re.match(r'^(\d+)\.(\d+)$', t)
            if m:
                major, minor = int(m.group(1)), int(m.group(2))
                yield f"{major}.{minor + 1}"   # 3.4 → 3.5
                yield str(major + 1)            # 3.4 → 4
                break

    def __init__(self, registry: AgentRegistry, llm: Optional[LLMProvider] = None) -> None:
        self.registry = registry
        self.llm = llm

    async def execute_subtask(self, subtask: SubTask,
                               source_filter: Optional[List[str]] = None,
                               search_mode: str = "standard",
                               trace: Optional[TraceCollector] = None) -> AgentResult:
        entry = self.registry.get(subtask.type)
        if entry is None:
            return AgentResult(subtask_id=subtask.id, agent_name="router",
                               success=False, error_message=f"No agent for {subtask.type.value}")

        # Data synthesis is handled by main.py with LLM — return a placeholder
        if entry.is_external_llm:
            return AgentResult(subtask_id=subtask.id, agent_name=entry.name,
                               success=True, data={"synthesis": "(pending LLM synthesis)"})

        start = time.time()
        if trace:
            trace.span_start(entry.name, "execute")
            trace.event(TraceEventType.AGENT_STARTED, agent_name=entry.name,
                        message=f"Starting: {subtask.query[:80]}...")

        try:
            # Use singleton instance if provided, otherwise create new
            if entry.singleton_instance is not None:
                agent = entry.singleton_instance
            elif not entry.is_external_llm:
                agent = entry.agent_class()
            else:
                agent = None
            if agent and hasattr(agent, "hybrid_search"):
                # Stem query tokens to match the stemmed index
                from backend.core.stemming import PrecisStemmer
                stemmer = PrecisStemmer()
                query_words = [w for w in subtask.query.lower().split() if len(w) > 1]
                stemmed_tokens = tuple(stemmer.stem_tokens(query_words))
                results = agent.hybrid_search(stemmed_tokens, source_filter=source_filter, trace=trace)
                print(f"[Precis] Router hash search: source_filter={source_filter!r} results={len(results)}")
                
                # Auto-retry with query expansion when 0 results (Thorough mode only)
                if len(results) == 0 and self.llm and search_mode == "thorough":
                    try:
                        from backend.agents.query_expander import QueryExpander
                        expander = QueryExpander(self.llm)
                        expanded_queries = await expander.expand(
                            subtask.query, list(stemmed_tokens)
                        )
                        for eq in expanded_queries:
                            eq_words = [w for w in eq.lower().split() if len(w) > 1]
                            eq_stemmed = tuple(stemmer.stem_tokens(eq_words))
                            retry_results = agent.hybrid_search(eq_stemmed, source_filter=source_filter, trace=trace)
                            if retry_results:
                                if trace:
                                    trace.event(
                                        type("TE", (), {"value": "decision.search_type"})(),
                                        agent_name="QueryExpander",
                                        message=f"Expanded '{subtask.query[:40]}...' → '{eq[:60]}...' → {len(retry_results)} results",
                                        data={"original_query": subtask.query, "expanded_query": eq, "results": len(retry_results)}
                                    )
                                results = retry_results
                                break
                    except Exception as ex:
                        if trace:
                            trace.event(
                                type("TE", (), {"value": "agent.failed"})(),
                                agent_name="QueryExpander",
                                message=f"Expansion failed: {ex}"
                            )
                # Build hash items (CPU work in thread)
                def build_hash_items():
                    items = []
                    for r in results[:40]:
                        text = " ".join(r.multitoken.metadata.get("original_words", r.multitoken.tokens))
                        ctx = agent.get_context(r.multitoken.source_doc, r.multitoken.source_page,
                                                r.multitoken.source_position)
                        items.append({
                            "text": text, "surrounding": ctx["surrounding"],
                            "sentence": ctx["sentence"], "page": ctx["page"],
                            "source": ctx["file"], "score": round(r.relevance_score, 3),
                            "match_type": r.match_type,
                        })
                    return items

                # Run hash build + vector search concurrently (skip vector in Fast mode)
                hash_task = asyncio.to_thread(build_hash_items)
                vec_task = None
                if search_mode != "fast":
                    try:
                        import backend.main as _main
                        if _main._vector_index:
                            vec_task = asyncio.to_thread(_main._vector_index.search, subtask.query, 10, source_filter)
                    except Exception:
                        pass

                if vec_task:
                    raw_items, vec_results = await asyncio.gather(hash_task, vec_task)
                else:
                    raw_items = await hash_task
                    vec_results = []
                print(f"[Precis] After gather: raw_items={len(raw_items)} vec_results={len(vec_results)}")

                # Fuse hash + vector results
                if vec_results:
                    try:
                        from backend.agents.fusion_ranker import FusionRanker
                        fuser = FusionRanker()
                        fused = fuser.fuse({"hash": raw_items, "vector": vec_results}, top_k=15)
                        raw_items = [{
                            "text": f["text"],
                            "surrounding": f.get("surrounding", f["text"]),
                            "sentence": f.get("sentence", f["text"][:200]),
                            "page": f.get("page", 1),
                            "source": f.get("source", ""),
                            "score": f.get("score", 0),
                            "match_type": f.get("match_type", "fusion"),
                        } for f in fused]
                        if trace:
                            trace.event(type("TE",(),{"value":"decision.fusion"})(), agent_name="FusionRanker",
                                message=f"Fused hash+vector: {len(fused)} results",
                                data={"vector_items": len(vec_results), "fused": len(fused)})
                    except Exception:
                        pass
                
                # Dedup by surrounding text (same paragraph = same result)
                seen_texts = {}
                for item in raw_items:
                    key = item["surrounding"][:120]  # First 120 chars of surrounding context
                    if key not in seen_texts or item["score"] > seen_texts[key]["score"]:
                        seen_texts[key] = item
                deduped = sorted(seen_texts.values(), key=lambda x: -x["score"])[:15]
                print(f"[Precis] After dedup: deduped={len(deduped)} first_source={deduped[0].get('source','') if deduped else 'EMPTY'}")

                # Semantic re-ranking: DISABLED — DeepSeek scores every n-gram
                # fragment as 0, adding ~3s latency with zero ranking benefit.
                # Re-enable when using a stronger LLM (GPT-4, Claude) that can
                # actually judge fragment relevance.
                _ENABLE_SEMANTIC_RERANKER = False
                if _ENABLE_SEMANTIC_RERANKER and self.llm and len(deduped) > 3:
                    try:
                        from backend.agents.semantic_reranker import SemanticReRanker
                        reranker = SemanticReRanker(self.llm)
                        reranked = await reranker.rerank(subtask.query, deduped, top_k=5)
                        if reranked:
                            if trace:
                                trace.event(
                                    type("TE", (), {"value": "decision.rerank"})(),
                                    agent_name="SemanticReRanker",
                                    message=f"Re-ranked {len(deduped)} → {len(reranked)} results",
                                    data={"before": len(deduped), "after": len(reranked),
                                          "top_score": reranked[0].get("semantic_score", 0)}
                                )
                            deduped = reranked
                    except Exception:
                        pass  # Re-ranking is best-effort; fall back to hash scores

                # Quality filter: keep results above minimum hash-score threshold.
                # (SemanticReRanker is disabled, so we use the original hash scores.)
                MIN_SCORE = 0.10
                deduped = [d for d in deduped if d.get("score", 0) >= MIN_SCORE]
                print(f"[Precis] After MIN_SCORE: deduped={len(deduped)}")

                # Thorough mode: HASH found the section heading.
                # Both matches are TOC entries — "next heading" is always
                # the adjacent TOC line, so boundary detection fails.
                # Instead: extract a generous window from the BEST match
                # (the one with the most surrounding content).  No char cap —
                # the LLM is smart enough to identify the section body.
                if search_mode == "thorough":
                    print(f"[Precis] Thorough: generous window from hash position")
                    try:
                        import os
                        filter_set = None
                        if source_filter:
                            filter_set = {os.path.basename(str(f).lower().strip()) for f in source_filter}
                        
                        if hasattr(agent, '_doc_texts'):
                            for filename, text in agent._doc_texts.items():
                                if filter_set and os.path.basename(filename.lower().strip()) not in filter_set:
                                    continue
                                lines = text.split("\n")
                                best_excerpt = ""
                                best_pos = 0
                                
                                for r in results[:5]:
                                    pos = r.multitoken.source_position
                                    end = min(len(lines), pos + 300)
                                    excerpt = "\n".join(lines[pos:end])
                                    # Keep the one with the MOST content
                                    if len(excerpt) > len(best_excerpt):
                                        best_excerpt = excerpt
                                        best_pos = pos
                                
                                if len(best_excerpt) > 200:
                                    deduped.append({
                                        "text": best_excerpt,
                                        "source": filename,
                                        "score": 0.7,
                                        "match_type": "section_body",
                                        "page": best_pos // 40 + 1,
                                        "surrounding": best_excerpt,
                                        "sentence": best_excerpt[:500],
                                    })
                                    if trace:
                                        trace.event(
                                            type("TE", (), {"value": "decision.direct_read"})(),
                                            agent_name="SectionExtractor",
                                            message=f"Section window: {len(best_excerpt)} chars from line {best_pos}",
                                            data={"from": best_pos, "chars": len(best_excerpt)},
                                        )
                    except Exception:
                        pass

                result = AgentResult(subtask_id=subtask.id, agent_name=entry.name, success=True,
                                     data={"results": deduped},
                                     citations=[{"source_doc": r.multitoken.source_doc,
                                                 "source_page": r.multitoken.source_page}
                                                for r in results[:20]])
            elif agent and hasattr(agent, "predict"):
                import numpy as np
                pred, contribs = agent.predict(np.array([0.5]), trace=trace)
                result = AgentResult(subtask_id=subtask.id, agent_name=entry.name, success=True,
                                     data={"prediction": float(pred), "contributing_nodes": contribs[:5]})
            elif agent and hasattr(agent, "detect_all"):
                flags = agent.detect_all(trace=trace)
                result = AgentResult(subtask_id=subtask.id, agent_name=entry.name, success=True,
                                     data={"flags": [{"entity": f.entity_id, "type": f.flag_type,
                                                      "severity": f.severity} for f in flags]})
            else:
                result = AgentResult(subtask_id=subtask.id, agent_name=entry.name, success=True,
                                     data={"response": "Agent executed successfully"})
        except Exception as e:
            result = AgentResult(subtask_id=subtask.id, agent_name=entry.name,
                                 success=False, error_message=str(e))

        result.execution_time_ms = (time.time() - start) * 1000
        if trace:
            trace.event(TraceEventType.AGENT_COMPLETED, agent_name=entry.name,
                        message="Completed" if result.success else f"Failed: {result.error_message}",
                        data={"success": result.success, "duration_ms": result.execution_time_ms})
            trace.span_end()
        return result

    async def execute_plan(self, subtasks: List[SubTask], max_parallel: int = 4,
                            source_filter: Optional[List[str]] = None,
                            search_mode: str = "standard",
                            trace: Optional[TraceCollector] = None) -> List[AgentResult]:
        print(f"[Precis] execute_plan: source_filter={source_filter!r} mode={search_mode}")
        results: Dict[str, AgentResult] = {}
        pending = list(subtasks)
        while pending:
            ready = [s for s in pending if all(d in results for d in s.depends_on)]
            if not ready:
                break
            batch = ready[:max_parallel]
            pending = [s for s in pending if s not in batch]
            batch_results = await asyncio.gather(*(self.execute_subtask(s, source_filter, search_mode, trace) for s in batch))
            for s, r in zip(batch, batch_results):
                results[s.id] = r
        return [results.get(s.id, AgentResult(subtask_id=s.id, agent_name="router", success=False,
                                               error_message="Dependency not met")) for s in subtasks]
