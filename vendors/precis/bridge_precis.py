"""
Precis Bridge — stdin/stdout JSON worker for MCP server.

Usage: python bridge_precis.py

Reads JSON requests from stdin, processes them using the real Precis backend,
writes JSON responses to stdout.

Protocol:
  Input:  {"id": 1, "action": "query", "params": {"query": "...", "search_mode": "standard"}}
  Output: {"id": 1, "result": {...}} or {"id": 1, "error": "message"}

Actions:
  - query:         Full RAG pipeline
  - list_documents: List indexed documents
  - debug_stem:     Show stemmer output for a query
  - debug_search:   Direct hybrid search result
  - health:         Returns {"status": "ok"}
"""

import sys, importlib, json, asyncio, os

# ── Verify dependencies (installed by MCP server on startup) ─────
REQUIRED = {'fastapi': 'fastapi', 'uvicorn': 'uvicorn', 'pydantic': 'pydantic',
            'numpy': 'numpy', 'nltk': 'nltk', 'sqlalchemy': 'sqlalchemy',
            'httpx': 'httpx', 'dotenv': 'python-dotenv'}
_missing = [mod for mod in REQUIRED if not importlib.util.find_spec(mod)]
if _missing:
    sys.stderr.write(f'[precis] FATAL: missing packages: {", ".join(_missing)}. '
                     f'The MCP server should have installed them.\n')
    sys.stderr.flush()
    sys.exit(1)

# Ensure the precis-agentic-pipeline directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress excessive logging
os.environ.setdefault('PRECIS_LOG_LEVEL', 'WARNING')

# ── Global state (initialized lazily) ──────────────────────────────────
_app = None
_loop = None

def get_loop():
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
    return _loop

def init_app():
    """Initialize the Precis FastAPI app and its lifespan (DB, indexes, LLM)."""
    global _app
    if _app is not None:
        return _app

    from backend.main import app, lifespan
    _app = app

    loop = get_loop()
    # Run the lifespan startup
    async def startup():
        async with lifespan(app) as gen:
            await gen.__anext__()
    loop.run_until_complete(startup())

    return _app


def handle_query(params):
    """Execute a full RAG query through the real Precis pipeline."""
    from backend.main import process_query

    loop = get_loop()
    query_dict = {
        "query": params.get("query", ""),
        "session_id": params.get("session_id", None),
        "source_filter": params.get("source_filter", None),
        "search_mode": params.get("search_mode", "standard"),
    }

    result = loop.run_until_complete(process_query(query_dict))
    return result


def handle_list_documents(params):
    """List indexed documents from the database."""
    from backend.db.repository import get_all_documents
    docs = get_all_documents()
    return docs


def handle_debug_stem(params):
    """Show how the PrecisStemmer processes a query."""
    from backend.core.stemming import PrecisStemmer
    stemmer = PrecisStemmer()
    raw = params.get("q", "").lower().split()
    stemmed = stemmer.stem_tokens(raw)
    return {"raw_tokens": raw, "stemmed_tokens": list(stemmed)}


def handle_debug_search(params):
    """Run a direct hybrid search bypassing the planner."""
    from backend.core.stemming import PrecisStemmer
    import backend.main as _main

    stemmer = PrecisStemmer()
    raw = params.get("q", "").lower().split()
    stemmed = tuple(stemmer.stem_tokens(raw))
    index = _main._demo_index
    results = index.hybrid_search(stemmed)

    return {
        "query": params.get("q", ""),
        "stemmed_tokens": list(stemmed),
        "result_count": len(results),
        "results": [
            {
                "tokens": list(r.multitoken.tokens) if hasattr(r, 'multitoken') else [],
                "source": r.multitoken.source_doc if hasattr(r, 'multitoken') else "",
                "score": r.relevance_score if hasattr(r, 'relevance_score') else 0,
                "match_type": r.match_type if hasattr(r, 'match_type') else "",
            }
            for r in results[:10]
        ],
    }


# ── Action dispatcher ─────────────────────────────────────────────────

ACTIONS = {
    "query": handle_query,
    "list_documents": handle_list_documents,
    "debug_stem": handle_debug_stem,
    "debug_search": handle_debug_search,
    "health": lambda p: {"status": "ok", "backend": "precis"},
}


def main():
    # Send ready signal
    sys.stdout.write("__READY__\n")
    sys.stdout.flush()

    # Initialize app on first request, not at startup (faster initial ready signal)
    _initialized = False

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        req_id = request.get("id")
        action = request.get("action", "")
        params = request.get("params", {})

        handler = ACTIONS.get(action)
        if not handler:
            result = {"id": req_id, "error": f"Unknown action: {action}"}
            sys.stdout.write(json.dumps(result) + "\n")
            sys.stdout.flush()
            continue

        try:
            # Lazy init
            if not _initialized and action != "health":
                init_app()
                _initialized = True

            result_data = handler(params)
            response = {"id": req_id, "result": result_data}
        except Exception as e:
            response = {"id": req_id, "error": str(e)}

        sys.stdout.write(json.dumps(response, default=str) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
