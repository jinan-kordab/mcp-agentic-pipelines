"""© JINAN KORDAB — 2026 AI HYBRID AGENTIC RETRIEVAL-AUGMENTED GENERATION RAG PIPELINE - PERSONAL PROJECT

Transforms natural language into the domain-specific vocabulary found in indexed documents.
Example: "currency impact" → "foreign exchange rate exposure currency fluctuation risk"
No embeddings, no vector DB — just the LLM's knowledge of financial synonyms.
"""

from typing import Optional
from backend.llm.base import LLMProvider


class QueryExpander:
    """Rewrites queries using LLM knowledge of domain terminology when exact hash search fails."""

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    async def expand(self, original_query: str, failed_tokens: list,
                     document_domain: str = "financial and legal") -> list:
        """Generate alternative queries with different terminology.
        
        Returns a list of expanded query strings, sorted by likely relevance.
        """
        token_str = ", ".join(failed_tokens[:20]) if failed_tokens else original_query
        
        prompt = f"""You are a {document_domain} domain expert. A document search system failed
to find matches for the following query because the EXACT words don't appear in the documents.

ORIGINAL QUERY: {original_query}
FAILED SEARCH TOKENS: {token_str}

The documents contain formal {document_domain} terminology. Rewrite the original query
using alternative words, synonyms, and related {document_domain} terms that are MORE LIKELY
to appear in formal documents.

For example:
- "currency impact" → "foreign exchange rate exposure"
- "money lost" → "financial impairment write-down loss"
- "hacking problem" → "cybersecurity incident data breach unauthorized access"
- "worker shortage" → "talent attrition labor supply constraints headcount reduction"
- "green rules" → "environmental regulation climate compliance carbon emission"

Return ONLY a JSON list of 3 rewritten queries, most likely to match first:
["rewritten query 1", "rewritten query 2", "rewritten query 3"]"""

        try:
            import asyncio
            response = await asyncio.wait_for(
                self.llm.generate(prompt, max_tokens=300, temperature=0.3),
                timeout=20
            )
            # Parse JSON list from response
            import json
            # Find the JSON array in the response
            start = response.find("[")
            end = response.rfind("]") + 1
            if start >= 0 and end > start:
                expansions = json.loads(response[start:end])
                if isinstance(expansions, list):
                    return expansions[:3]
        except Exception:
            pass
        
        # Fallback: simple word-level expansion using common financial synonyms
        return [self._basic_expand(original_query)]

    def _basic_expand(self, query: str) -> str:
        """Simple synonym substitution when LLM is unavailable."""
        synonyms = {
            "currency": "foreign exchange fx rate",
            "money": "capital funds revenue cash",
            "risk": "exposure uncertainty volatility",
            "profit": "earnings income margin return",
            "loss": "impairment write-down decline decrease",
            "revenue": "sales income turnover top-line",
            "cost": "expense expenditure outlay",
            "market": "sector industry segment",
            "growth": "expansion increase appreciation",
            "rule": "regulation compliance requirement policy",
            "problem": "issue incident concern challenge",
            "impact": "effect influence exposure consequence",
        }
        words = query.lower().split()
        expanded = []
        for w in words:
            expanded.append(w)
            if w in synonyms:
                expanded.append(synonyms[w])
        return " ".join(expanded)
