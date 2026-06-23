"""
Piste Bridge — stdin/stdout JSON worker for MCP server.

Usage: python bridge_piste.py

Reads JSON requests from stdin, processes them using the real Piste DSPy pipeline,
writes JSON responses to stdout.

IMPORTANT: This bridge uses the REAL pipeline modules (pipeline/stage1-4) directly.
It does NOT require PostgreSQL, Redis, or Docker. The pipeline stages use DSPy + LiteLLM
for LLM calls and Tavily/Serper for web search. Results are returned as JSON.

Protocol:
  Input:  {"id": 1, "action": "fact_check", "params": {"claim_text": "...", "locale": "en"}}
  Output: {"id": 1, "result": {...}} or {"id": 1, "error": "message"}

Actions:
  - fact_check:  Run the full 4-stage fact-checking pipeline
  - health:      Returns {"status": "ok"}
"""

import sys, importlib, json, asyncio, os

# ── Verify dependencies (installed by MCP server on startup) ─────
REQUIRED = {'dspy': 'dspy-ai', 'litellm': 'litellm', 'dotenv': 'python-dotenv'}
_missing = [mod for mod in REQUIRED if not importlib.util.find_spec(mod)]
if _missing:
    sys.stderr.write(f'[piste] FATAL: missing packages: {", ".join(_missing)}. '
                     f'The MCP server should have installed them.\n')
    sys.stderr.flush()
    sys.exit(1)

# Ensure piste directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

os.environ.setdefault('PISTE_LOG_LEVEL', 'WARNING')

# ── Pipeline imports (lazy) ────────────────────────────────────────────

_initialized = False
_loop = None

def get_loop():
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
    return _loop


def init_pipeline():
    """Configure DSPy with the API key from environment."""
    global _initialized
    if _initialized:
        return

    from pipeline.compiler import configure_dspy
    configure_dspy()
    _initialized = True


async def run_fact_check(claim_text: str, locale: str = "en", context: str = None) -> dict:
    """
    Run the full 4-stage Piste fact-checking pipeline.
    Uses REAL DSPy modules from the cloned piste repo.
    """
    import dspy
    import uuid

    run_id = str(uuid.uuid4())[:12]

    # ── Stage 1: Check-Worthiness + Atomic Decomposition ──────────
    from pipeline.stage1.check_worthiness import CheckWorthinessDetector
    from pipeline.stage1.atomic_decomposer import AtomicClaimDecomposer

    cw_detector = CheckWorthinessDetector()
    decomposer = AtomicClaimDecomposer()

    # Check if the claim is worth fact-checking
    cw_result = cw_detector(claim_text)
    cw_label = getattr(cw_result, 'label', 'CFC') if hasattr(cw_result, 'label') else str(cw_result)
    cw_score = getattr(cw_result, 'score', 1.0) if hasattr(cw_result, 'score') else 1.0

    # Decompose into atomic claims
    atomic_claims = []
    try:
        decomp_result = decomposer(claim_text)
        if hasattr(decomp_result, 'claims'):
            atomic_claims = decomp_result.claims
        elif isinstance(decomp_result, list):
            atomic_claims = decomp_result
    except Exception:
        atomic_claims = [claim_text]

    # ── Stage 2: Blind Web Retrieval ──────────────────────────────
    from pipeline.stage2.search_decision import SearchDecisionGenerator
    from pipeline.stage2.blind_retriever import BlindRetriever

    sd_generator = SearchDecisionGenerator()
    retriever = BlindRetriever()

    all_sources = []
    search_needed = True

    try:
        sd_result = sd_generator(claim_text)
        search_needed = getattr(sd_result, 'search_needed', True) if hasattr(sd_result, 'search_needed') else True

        if search_needed:
            # Generate neutral search queries (blind — never sees the claim)
            queries = []
            if hasattr(sd_result, 'queries'):
                queries = sd_result.queries
            elif hasattr(sd_result, 'search_queries'):
                queries = sd_result.search_queries

            if not queries:
                # Fallback: generate a simple neutral query
                queries = [f"fact check {locale} political claim"]

            for query in queries[:5]:  # Max 5 queries
                try:
                    results = retriever(query, locale=locale)
                    if hasattr(results, 'sources'):
                        all_sources.extend(results.sources)
                    elif isinstance(results, list):
                        all_sources.extend(results)
                except Exception:
                    pass
    except Exception:
        pass

    # ── Stage 3: Per-Source Classification ────────────────────────
    from pipeline.stage3.classifier import SourceClassifier

    classifier = SourceClassifier()
    classifications = []

    for source in all_sources[:20]:  # Max 20 sources
        try:
            # Extract source text
            if isinstance(source, dict):
                source_text = source.get('text', '') or source.get('content', '') or source.get('title', '')
                source_url = source.get('url', '') or source.get('link', '')
                source_title = source.get('title', '') or source_url
            elif isinstance(source, str):
                source_text = source
                source_url = ''
                source_title = source[:100]
            else:
                source_text = str(source)
                source_url = ''
                source_title = str(source)[:100]

            if not source_text.strip():
                continue

            classification = classifier(claim_text, source_text)
            label = getattr(classification, 'label', 'UNRELATED') if hasattr(classification, 'label') else 'UNRELATED'
            confidence = getattr(classification, 'confidence', 0.5) if hasattr(classification, 'confidence') else 0.5
            rationale = getattr(classification, 'rationale', '') if hasattr(classification, 'rationale') else ''

            classifications.append({
                "url": source_url,
                "title": source_title[:200],
                "classification": str(label).upper(),
                "confidence": float(confidence),
                "rationale": str(rationale)[:500],
            })
        except Exception:
            classifications.append({
                "url": source.get('url', '') if isinstance(source, dict) else '',
                "title": source.get('title', '')[:200] if isinstance(source, dict) else '',
                "classification": "UNRELATED",
                "confidence": 0.0,
                "rationale": "Classification failed",
            })

    # ── Stage 4: Verdict Aggregation ──────────────────────────────
    from pipeline.stage4.verdict_aggregator import VerdictAggregator

    aggregator = VerdictAggregator()

    try:
        verdict = aggregator(classifications, claim_text)
        verdict_label = getattr(verdict, 'label', 'UNVERIFIABLE') if hasattr(verdict, 'label') else 'UNVERIFIABLE'
        verdict_explanation = getattr(verdict, 'explanation', '') if hasattr(verdict, 'explanation') else ''
        verdict_distribution = getattr(verdict, 'distribution', {}) if hasattr(verdict, 'distribution') else {}
    except Exception:
        # Fallback verdict
        supports = sum(1 for c in classifications if c['classification'] == 'SUPPORTS')
        refutes = sum(1 for c in classifications if c['classification'] == 'REFUTES')
        total = len(classifications) or 1

        if supports > refutes and supports > total * 0.5:
            verdict_label = 'TRUE' if supports > total * 0.8 else 'MOSTLY_TRUE'
        elif refutes > supports and refutes > total * 0.5:
            verdict_label = 'FALSE' if refutes > total * 0.8 else 'MOSTLY_FALSE'
        elif supports == refutes and total > 0:
            verdict_label = 'HALF_TRUE'
        else:
            verdict_label = 'UNVERIFIABLE'

        verdict_explanation = f"Based on {supports} supporting and {refutes} refuting sources out of {total} total."
        verdict_distribution = {"TRUE": 0, "MOSTLY_TRUE": 0, "HALF_TRUE": 0, "MOSTLY_FALSE": 0, "FALSE": 0, "PANTS_ON_FIRE": 0, "UNVERIFIABLE": 0}

    return {
        "run_id": run_id,
        "claim_id": run_id,
        "verdict": {
            "label": str(verdict_label).upper(),
            "distribution": verdict_distribution if isinstance(verdict_distribution, dict) else {},
            "explanation": str(verdict_explanation)[:2000],
            "sources": classifications[:15],
        },
        "stage1": {
            "check_worthy": str(cw_label),
            "score": float(cw_score),
            "atomic_claims": [str(c) for c in atomic_claims[:5]],
        },
        "stage2": {
            "search_needed": bool(search_needed),
            "sources_found": len(all_sources),
        },
        "audit_url": f"piste://claims/{run_id}",
        "elapsed_ms": 0,
    }


# ── Action Handlers ───────────────────────────────────────────────────

def handle_fact_check(params):
    """Run the full fact-checking pipeline."""
    claim_text = params.get("claim_text", "")
    locale = params.get("locale", "en")
    context = params.get("context", None)

    if not claim_text or len(claim_text) < 10:
        return {"error": "Claim text must be at least 10 characters"}

    loop = get_loop()
    init_pipeline()

    result = loop.run_until_complete(run_fact_check(claim_text, locale, context))
    return result


# ── Dispatcher ────────────────────────────────────────────────────────

ACTIONS = {
    "fact_check": handle_fact_check,
    "health": lambda p: {"status": "ok", "backend": "piste"},
}


def main():
    # Send ready signal
    sys.stdout.write("__READY__\n")
    sys.stdout.flush()

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
            result_data = handler(params)
            response = {"id": req_id, "result": result_data}
        except Exception as e:
            import traceback
            response = {"id": req_id, "error": f"{type(e).__name__}: {str(e)}"}

        sys.stdout.write(json.dumps(response, default=str) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
