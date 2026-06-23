"""© JINAN KORDAB — 2026 AI HYBRID AGENTIC RETRIEVAL-AUGMENTED GENERATION RAG PIPELINE - PERSONAL PROJECT

No embeddings, no vector DB — the LLM reads candidate text and scores relevance 0-100.
Only passes the best candidates to the synthesis step.
"""

from typing import Any, Dict, List, Optional, Tuple
from backend.llm.base import LLMProvider


class SemanticReRanker:
    """Uses LLM semantic understanding to filter hash-retrieved candidates by true relevance."""

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    async def rerank(self, query: str, candidates: List[Dict[str, Any]],
                     top_k: int = 5) -> List[Dict[str, Any]]:
        """Score each candidate by semantic relevance to the query. Returns top_k.

        Each candidate: {"text": str, "source": str, "score": float, "page": int, ...}
        """
        if not candidates:
            return []
        if len(candidates) <= top_k:
            return candidates

        # Build a scoring prompt with numbered candidates (use surrounding context when available)
        items = []
        for i, c in enumerate(candidates[:20]):
            # Prefer surrounding context (full paragraph) over short n-gram text
            text = c.get('surrounding', '') or c.get('sentence', '') or c.get('text', '')
            items.append(f"[{i}] {text[:300]}")

        prompt = f"""You are a precise relevance judge. Score each text chunk below for how well
it answers this query: "{query}"

For each chunk, give a score from 0-100:
  90-100: Directly answers the query with specific facts
  70-89:  Related and useful context
  40-69:  Tangentially related
  0-39:   Not relevant

Text chunks:
{chr(10).join(items)}

Return ONLY a JSON array: [{{"index": 0, "score": 85, "reason": "5 words"}}, ...]
Score ALL chunks. Be strict — only give high scores for truly relevant content."""

        try:
            import asyncio, json
            response = await asyncio.wait_for(
                self.llm.generate(prompt, max_tokens=200, temperature=0.0),
                timeout=20
            )
            # Parse the JSON array
            start = response.find("[")
            end = response.rfind("]") + 1
            if start >= 0 and end > start:
                scores = json.loads(response[start:end])
                # Map scores back to candidates
                scored = []
                for s in scores:
                    idx = s.get("index", 0)
                    if 0 <= idx < len(candidates):
                        candidates[idx]["semantic_score"] = s.get("score", 0)
                        candidates[idx]["relevance_reason"] = s.get("reason", "")
                        scored.append(candidates[idx])
                # Sort by semantic score, return top_k
                scored.sort(key=lambda c: c.get("semantic_score", 0), reverse=True)
                return scored[:top_k]
        except Exception:
            pass

        # Fallback: return top by original hash score
        return sorted(candidates, key=lambda c: c.get("score", 0), reverse=True)[:top_k]


class DirectReader:
    """When hash search finds nothing, ask the LLM to directly read document text and answer."""

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    async def read_and_answer(self, query: str, doc_snippets: List[Dict[str, str]],
                               index) -> Dict[str, Any]:
        """Read document snippets directly and attempt to answer the query.

        doc_snippets: [{"text": "...", "source": "file.pdf", "page": 1}, ...]
        """
        if not doc_snippets:
            return {"found": False, "answer": "No documents have been uploaded yet. Please upload a document to search.", "citation": ""}

        doc_names = ", ".join(d.get("source", "unknown") for d in doc_snippets[:5])

        context = "\n\n".join(
            f"[DOC: {d['source']}, page {d.get('page', 1)}]\n{d['text'][:3000]}"
            for d in doc_snippets[:5]
        )

        prompt = f"""You are a precise document analyst. Read the document excerpts below
and answer this query: "{query}"

Documents available: {doc_names}

DOCUMENTS:
{context[:15000]}

INSTRUCTIONS:
- If the query mentions a section number (like "section 3.4" or just "3.4"), locate
  ALL content under that section heading in the documents and extract the key points.
  Section headings may appear as "3.4 Title", "§3.4", or just "3.4" followed by text.
- Summarize what that section actually says — do not just report that the heading exists.
- If the query asks for "key findings" or a "summary", provide the substantive content
  even if the document doesn't explicitly label anything as "key findings."
- If you genuinely cannot find the section content anywhere in the provided text,
  only then state that it's not found.

Return your answer in this JSON format:
{{"found": true/false, "answer": "your detailed answer", "citation": "document name, page X"}}"""

        try:
            import asyncio, json
            response = await asyncio.wait_for(
                self.llm.generate(prompt, max_tokens=300, temperature=0.2),
                timeout=25
            )
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(response[start:end])
                return result
            return {"found": False, "answer": response[:500], "citation": ""}
        except Exception as e:
            return {"found": False, "answer": f"(LLM reading unavailable: {e})", "citation": ""}
