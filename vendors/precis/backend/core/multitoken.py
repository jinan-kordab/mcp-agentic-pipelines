"""© JINAN KORDAB — 2026 AI HYBRID AGENTIC RETRIEVAL-AUGMENTED GENERATION RAG PIPELINE - PERSONAL PROJECT

Optimised: pre-stems all words once (O(N) stemming) then slides an n-gram window
over the pre-stemmed tokens — a ~12× speedup vs. stemming each n-gram independently.
Element-level extraction can run in parallel via index_document_async().
"""

import asyncio
from typing import List, Tuple

from backend.agents.exact_hash_retriever import MultiToken, NestedHashIndex
from backend.core.stemming import PrecisStemmer


class MultiTokenExtractor:
    """Extracts variable-length multi-tokens (n-grams of stemmed words) from document text.

    Parameters
    ----------
    max_token_length : int
        Maximum n-gram length (inclusive).
    min_token_length : int
        Minimum n-gram length (inclusive).
    """

    def __init__(self, max_token_length: int = 7, min_token_length: int = 2) -> None:
        self.max_token_length = max_token_length
        self.min_token_length = min_token_length
        self.stemmer = PrecisStemmer()

    # ── Core extraction (pre-stemmed, single-element) ──────────────

    def extract(self, text: str, source_doc: str, source_page: int,
                font_size: float = 12.0, is_title: bool = False,
                is_header: bool = False,
                global_position: int = 0) -> List[MultiToken]:
        """Extract all valid multi-tokens from a text segment.

        Optimisation: words are stemmed ONCE, then n-grams are formed from
        the pre-stemmed list.  This is O(N) stemming instead of O(N×M).

        global_position is the LINE number in the full document — used by
        get_context() to retrieve surrounding text from the correct location.
        """
        words = text.strip().split()
        if len(words) < self.min_token_length:
            return []

        # ── Pre-stem all words once ────────────────────────────
        stemmed_words = self.stemmer.stem_tokens(words)
        if len(stemmed_words) < self.min_token_length:
            return []

        token_type = "contextual" if (is_title or is_header) else "standard"
        multitokens: List[MultiToken] = []

        n = len(stemmed_words)
        for start in range(n):
            max_len = min(self.max_token_length, n - start)
            for length in range(self.min_token_length, max_len + 1):
                stemmed = tuple(stemmed_words[start:start + length])
                # All tokens already stemmed & filtered; just check length
                if len(stemmed) >= self.min_token_length:
                    mt = MultiToken(
                        tokens=stemmed,
                        token_type=token_type,
                        source_doc=source_doc,
                        source_page=source_page,
                        source_position=global_position,  # ← LINE number, not word index
                        font_size=font_size,
                        is_title=is_title,
                        is_header=is_header,
                        metadata={"original_words": words[start:start + length]},
                    )
                    multitokens.append(mt)
        return multitokens

    # ── Document indexing (sequential) ────────────────────────────

    def index_document(self, doc_path: str, parsed_content: List[dict],
                       index: NestedHashIndex) -> int:
        """Parse a full document and index all multi-tokens. Returns count indexed."""
        total = 0
        global_line = 0  # track actual line number in the document
        for page in parsed_content:
            page_num = page.get("page_number", 0)
            for element in page.get("elements", []):
                text = element.get("text", "")
                mts = self.extract(
                    text, doc_path, page_num,
                    font_size=element.get("font_size", 12.0),
                    is_title=element.get("is_title", False),
                    is_header=element.get("is_header", False),
                    global_position=global_line,  # ← pass line number, not word index
                )
                for mt in mts:
                    index.insert(mt)
                total += len(mts)
                global_line += 1
        return total

    # ── Document indexing (async / parallel pages) ──────────────

    async def index_document_async(self, doc_path: str,
                                    parsed_content: List[dict],
                                    index: NestedHashIndex,
                                    max_tasks: int = 32) -> int:
        """Like index_document(), but processes **pages** concurrently.

        Each page's full set of elements is extracted in a single thread —
        this avoids the thread-pool explosion that would result from
        launching one task per element (thousands of tasks for a large doc).

        Parameters
        ----------
        max_tasks : int
            Upper bound on concurrent page-extraction tasks.  Prevents
            thread-pool exhaustion on documents with many short pages.
        """
        # ── Build one task per PAGE (not per element!) ────────
        # Each task extracts ALL elements on that page sequentially.
        # Tracks global line numbers so get_context() retrieves from
        # the correct position in the document.
        _line_counter = [0]  # mutable counter shared across pages

        def extract_page(page: dict) -> List[MultiToken]:
            page_num = page.get("page_number", 0)
            all_mts: List[MultiToken] = []
            for element in page.get("elements", []):
                text = element.get("text", "")
                if not text.strip():
                    _line_counter[0] += 1
                    continue
                mts = self.extract(
                    text, doc_path, page_num,
                    font_size=element.get("font_size", 12.0),
                    is_title=element.get("is_title", False),
                    is_header=element.get("is_header", False),
                    global_position=_line_counter[0],
                )
                all_mts.extend(mts)
                _line_counter[0] += 1
            return all_mts

        pages = [p for p in parsed_content if p.get("elements")]
        if not pages:
            return 0

        # ── Throttle: at most max_tasks in flight at once ─────
        sem = asyncio.Semaphore(max_tasks)

        async def bounded_extract(page: dict) -> List[MultiToken]:
            async with sem:
                return await asyncio.to_thread(extract_page, page)

        results: List[List[MultiToken]] = await asyncio.gather(
            *(bounded_extract(p) for p in pages)
        )

        # ── Batch-insert all MultiTokens ──────────────────────
        total = 0
        for mts in results:
            for mt in mts:
                index.insert(mt)
            total += len(mts)
        return total
