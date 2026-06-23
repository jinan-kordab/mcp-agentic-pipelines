"""© JINAN KORDAB — 2026 AI HYBRID AGENTIC RETRIEVAL-AUGMENTED GENERATION RAG PIPELINE - PERSONAL PROJECT"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple


@dataclass
class MultiToken:
    """Variable-length sequence of stemmed words with source provenance."""
    tokens: Tuple[str, ...]
    token_type: str = "standard"
    source_doc: str = ""
    source_page: int = 0
    source_position: int = 0
    font_size: Optional[float] = None
    is_title: bool = False
    is_header: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """A single retrieval match with relevance, trust, and match type."""
    multitoken: MultiToken
    relevance_score: float
    trustworthiness_score: float
    match_type: str  # "exact" | "subset" | "contextual" | "semantic_fallback"
    matched_tokens: List[str] = field(default_factory=list)


class NestedHashIndex:
    """Core retrieval engine: nested dict tree → O(m) exact lookup, no embeddings."""

    def __init__(self) -> None:
        self.index: Dict[str, Any] = {}
        self.multitoken_count: int = 0
        self.unique_tokens: set = set()
        self._access_counts: Dict[str, int] = {}
        self._doc_texts: Dict[str, str] = {}  # filename → full original text
        from backend.core.pmi import PMIScorer
        self.pmi: PMIScorer = PMIScorer()

    # ── Insert ─────────────────────────────────────────────────

    def insert(self, mt: MultiToken) -> None:
        current = self.index
        for token in mt.tokens:
            if token not in current:
                current[token] = {}
            current = current[token]
        if "_items" not in current:
            current["_items"] = []
        current["_items"].append(mt)
        self.multitoken_count += 1
        self.unique_tokens.update(mt.tokens)
        # Feed PMI scorer for token rarity weighting
        try:
            from backend.core.pmi import get_pmi_scorer
            get_pmi_scorer().ingest_tokens(list(mt.tokens))
        except Exception:
            pass
        self.pmi.ingest_tokens(list(mt.tokens))  # Build PMI statistics

    def insert_batch(self, multitokens: List[MultiToken]) -> int:
        for mt in multitokens:
            self.insert(mt)
        return len(multitokens)

    # ── Search ─────────────────────────────────────────────────

    def exact_search(self, query_tokens: Tuple[str, ...]) -> List[RetrievalResult]:
        current = self.index
        for token in query_tokens:
            if token not in current:
                return []
            current = current[token]
        items = current.get("_items", [])
        self._record_access(items)
        return [RetrievalResult(multitoken=mt, relevance_score=1.0,
                trustworthiness_score=self._trust_score(mt), match_type="exact")
                for mt in items]

    def subset_search(self, query_tokens: Tuple[str, ...],
                      min_match_ratio: float = 0.75) -> List[RetrievalResult]:
        results: List[RetrievalResult] = []
        query_set = set(query_tokens)
        for leaf in self._iter_leaves():
            items = leaf.get("_items", [])
            if not items:
                continue
            leaf_tokens = set(items[0].tokens)
            overlap = len(query_set & leaf_tokens)
            ratio = overlap / len(query_set) if query_set else 0.0
            if ratio >= min_match_ratio:
                score = ratio * (overlap / len(leaf_tokens)) if leaf_tokens else ratio
                self._record_access(items)
                for mt in items:
                    results.append(RetrievalResult(multitoken=mt, relevance_score=score,
                        trustworthiness_score=self._trust_score(mt), match_type="subset",
                        matched_tokens=list(query_set & leaf_tokens)))
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results

    def contextual_search(self, query_tokens: Tuple[str, ...]) -> List[RetrievalResult]:
        results: List[RetrievalResult] = []
        query_set = set(query_tokens)
        for leaf in self._iter_leaves():
            items = leaf.get("_items", [])
            if not items:
                continue
            for mt in items:
                if mt.token_type == "contextual" or mt.is_title or mt.is_header:
                    overlap = len(query_set & set(mt.tokens))
                    if overlap > 0:
                        score = min(overlap / len(query_set) * 1.5, 1.0)
                        results.append(RetrievalResult(multitoken=mt, relevance_score=score,
                            trustworthiness_score=self._trust_score(mt), match_type="contextual",
                            matched_tokens=list(query_set & set(mt.tokens))))
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results

    def hybrid_search(self, query_tokens: Tuple[str, ...],
                      include_semantic_fallback: bool = False,
                      source_filter: Optional[List[str]] = None,
                      trace=None) -> List[RetrievalResult]:
        """Multi-tier search across all indexed documents.

        Parameters
        ----------
        source_filter : Optional[List[str]]
            If provided, only return results whose source_doc is in this list.
            Case-insensitive basename matching (e.g. ``["report.pdf"]``).
        """
        # Tier 1: exact
        results = self.exact_search(query_tokens)
        if trace:
            trace.event(type("TE", (), {"value": "decision.search_type"})(), agent_name="ExactHash",
                        message=f"Exact search: {len(results)} results", data={"tier": 1, "count": len(results)})
        # Tier 2: subset fallback (lowered threshold to 0.5 for better recall)
        if len(results) < 5 and len(query_tokens) >= 2:
            subset = self.subset_search(query_tokens, min_match_ratio=0.5)
            if trace:
                trace.event(type("TE", (), {"value": "decision.search_type"})(), agent_name="ExactHash",
                            message=f"Subset fallback: {len(subset)} results", data={"tier": 2, "count": len(subset)})
            results.extend(subset)
        # Tier 3: contextual boost (titles/headers)
        ctx = self.contextual_search(query_tokens)
        results.extend(ctx)
        # Tier 4: broad sweep — any token overlap at all
        if len(results) < 3:
            broad = self.broad_search(query_tokens)
            if trace:
                trace.event(type("TE", (), {"value": "decision.search_type"})(), agent_name="ExactHash",
                            message=f"Broad sweep: {len(broad)} results", data={"tier": 4, "count": len(broad)})
            results.extend(broad)
        # Tier 5: PMI re-ranking — rare-token matches boosted, boilerplate suppressed
        try:
            from backend.core.pmi import get_pmi_scorer
            pmi = get_pmi_scorer()
            for r in results:
                pmi_score = pmi.score(list(query_tokens), list(r.multitoken.tokens))
                pmi_norm = pmi.normalize_score(pmi_score)
                # Blend: 70% structural match + 30% token rarity
                r.relevance_score = round(r.relevance_score * 0.7 + pmi_norm * 0.3, 4)
        except Exception:
            pass

        # ── Document-scope filter ────────────────────────────────
        if source_filter:
            import os
            filter_set = set()
            for f in source_filter:
                f = str(f).lower().strip()
                f = os.path.basename(f)
                if f:
                    filter_set.add(f)
            before = len(results)
            if filter_set:
                results = [r for r in results
                           if os.path.basename(r.multitoken.source_doc.lower().strip()) in filter_set]
            print(f"[Precis] Hash filter: source_filter={source_filter!r} filter_set={filter_set!r} before={before} after={len(results)}")

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results

    def broad_search(self, query_tokens: Tuple[str, ...]) -> List[RetrievalResult]:
        """Last-resort search: any leaf with meaningful token overlap (≥30% of query)."""
        results: List[RetrievalResult] = []
        query_set = set(query_tokens)
        min_overlap = max(1, int(len(query_tokens) * 0.3))  # Require ≥30% query token match
        for leaf in self._iter_leaves():
            items = leaf.get("_items", [])
            if not items:
                continue
            for mt in items:
                overlap = len(query_set & set(mt.tokens))
                if overlap >= min_overlap:
                    score = min(overlap / len(query_set) * 0.8, 0.9)
                    results.append(RetrievalResult(multitoken=mt, relevance_score=score,
                        trustworthiness_score=self._trust_score(mt), match_type="broad",
                        matched_tokens=list(query_set & set(mt.tokens))))
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:30]  # Cap broad results

    # ── Maintenance ─────────────────────────────────────────────

    def auto_distill(self, min_access_count: int = 3) -> int:
        removed = 0
        for leaf in list(self._iter_leaves()):
            items = leaf.get("_items", [])
            if not items:
                continue
            key = self._leaf_key(items[0])
            if self._access_counts.get(key, 0) < min_access_count:
                leaf["_items"] = []
                self.multitoken_count -= len(items)
                removed += len(items)
        return removed

    def index_document(self, text: str, source: str = "uploaded_document") -> int:
        """Parse raw text into MultiTokens and insert into the index. Returns count of tokens indexed."""
        # Store original text for context retrieval
        self._doc_texts[source] = text

        from backend.core.multitoken import MultiTokenExtractor

        extractor = MultiTokenExtractor(max_token_length=7, min_token_length=2)

        lines = text.strip().split("\n")
        parsed = [{"page_number": 1, "elements": []}]
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            parsed[0]["elements"].append({
                "text": stripped,
                "is_title": stripped.isupper() and len(stripped) < 80,
                "is_header": stripped.isupper() and len(stripped) < 60,
                "font_size": 14.0 if stripped.isupper() else 10.0,
            })

        # Use the proper MultiTokenExtractor API: index_document(filename, parsed, self)
        return extractor.index_document(source, parsed, self)

    def get_statistics(self) -> Dict[str, Any]:
        depth = self._compute_depth(self.index)
        return {"multitoken_count": self.multitoken_count, "unique_tokens": len(self.unique_tokens),
                "index_depth": depth, "memory_estimate_mb": self.multitoken_count * 0.002,
                "cached_documents": len(self._doc_texts)}

    def get_context(self, source_doc: str, source_page: int, source_position: int,
                    window: int = 5) -> Dict[str, Any]:
        """Retrieve surrounding lines from the original document for a match position.

        source_position is the LINE NUMBER in the full document (set during indexing).
        Uses raw lines without merging so the index stays accurate.
        """
        text = self._doc_texts.get(source_doc, "")
        if not text:
            return {"sentence": "(source text not cached)", "surrounding": "", "page": source_page, "file": source_doc}
        
        lines = text.split("\n")
        # Filter out fully empty lines but keep line numbering intact
        non_empty = [(i, l.strip()) for i, l in enumerate(lines) if l.strip()]
        
        if not non_empty:
            return {"sentence": "", "surrounding": "", "page": source_page, "file": source_doc}
        
        # Find the closest non-empty line to source_position
        idx = 0
        for i, (line_no, _) in enumerate(non_empty):
            if line_no >= source_position:
                idx = i
                break
        else:
            idx = len(non_empty) - 1
        
        start = max(0, idx - window)
        end = min(len(non_empty), idx + window + 1)
        
        surrounding = "\n".join(l for _, l in non_empty[start:end])
        sentence = non_empty[idx][1] if idx < len(non_empty) else ""
        
        return {
            "sentence": sentence,
            "surrounding": surrounding,
            "page": source_page,
            "file": source_doc,
        }

    # ── Internal ────────────────────────────────────────────────

    def _iter_leaves(self) -> Iterator[Dict[str, Any]]:
        def recurse(node):
            if "_items" in node:
                yield node
            for k, v in node.items():
                if k != "_items" and isinstance(v, dict):
                    yield from recurse(v)
        yield from recurse(self.index)

    def _trust_score(self, mt: MultiToken) -> float:
        score = 0.5
        if mt.is_title:
            score += 0.2
        if mt.is_header:
            score += 0.1
        if mt.font_size and mt.font_size > 12:
            score += 0.1
        if mt.token_type == "contextual":
            score += 0.15
        return min(score, 1.0)

    def _record_access(self, items: List[MultiToken]) -> None:
        for mt in items:
            key = self._leaf_key(mt)
            self._access_counts[key] = self._access_counts.get(key, 0) + 1

    @staticmethod
    def _leaf_key(mt: MultiToken) -> str:
        return f"{mt.source_doc}|{mt.source_page}|{mt.source_position}"

    @staticmethod
    def _compute_depth(node: dict) -> int:
        if not isinstance(node, dict) or not node:
            return 0
        return 1 + max((NestedHashIndex._compute_depth(v) for k, v in node.items() if k != "_items"), default=0)
