"""© JINAN KORDAB — 2026 AI HYBRID AGENTIC RETRIEVAL-AUGMENTED GENERATION RAG PIPELINE - PERSONAL PROJECT"""

import asyncio
import json
import uuid
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Dict, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from backend.config import settings
from backend.db.repository import init_db


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    print(f"[Precis] Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"[Precis] LLM: {settings.DEFAULT_LLM_PROVIDER} | Docs: http://localhost:8000/docs")

    from backend.llm.factory import LLMFactory
    from backend.llm.deepseek_provider import DeepSeekProvider

    LLMFactory.register("deepseek", DeepSeekProvider)
    print(f"[Precis] LLM provider: deepseek")

    init_db()
    print("[Precis] Database initialized")

    # Initialize vector index (FAISS) alongside hash index
    try:
        from backend.agents.vector_index import VectorIndex, HAS_FAISS
        if HAS_FAISS:
            vector_index = VectorIndex()
            print(f"[Precis] Vector index ready (FAISS, {vector_index.dimension}-dim)")
        else:
            vector_index = None
            print("[Precis] Vector index skipped (faiss-cpu not installed)")
    except Exception as e:
        vector_index = None
        print(f"[Precis] Vector index skipped: {e}")

    # Create empty index + re-index persisted documents from DB
    try:
        from backend.agents.exact_hash_retriever import NestedHashIndex
        from backend.core.multitoken import MultiTokenExtractor
        idx = NestedHashIndex()
        extractor = MultiTokenExtractor()

        from backend.db.repository import get_all_document_texts
        stored = get_all_document_texts()
        for doc in stored:
            # Build parsed content for async extraction
            text = doc["text"]
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

            # ── Hash + vector indexing in PARALLEL per document ─
            async def hash_doc():
                return await extractor.index_document_async(doc["filename"], parsed, idx)
            async def vec_doc():
                if vector_index:
                    return vector_index.index_text(text, source=doc["filename"])
                return 0

            added, chunks = await asyncio.gather(hash_doc(), vec_doc())
            idx._doc_texts[doc["filename"]] = text
            print(f"[Precis] Hash-indexed: {doc['filename']} ({added} tokens)")
            if vector_index:
                print(f"[Precis] Vector-indexed: {doc['filename']} ({chunks} chunks)")
        if stored:
            print(f"[Precis] {len(stored)} persisted documents restored")
    except Exception as e:
        print(f"[Precis] Seed error: {e}")

    # Store singletons for query access
    import backend.main as _main
    _main._vector_index = vector_index
    _main._demo_index = idx
    yield
    print("[Precis] Shutting down...")


app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION,
              description="Precis — Multi-agent orchestration platform with LLM-based planning and deterministic execution.",
              docs_url="/docs", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# Track active WebSocket connections for trace streaming
_active_ws: Dict[str, WebSocket] = {}


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the lightweight Precis UI."""
    ui_path = Path(__file__).parent.parent / "frontend" / "src" / "app" / "index.html"
    if ui_path.exists():
        return HTMLResponse(ui_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Precis API</h1><p>UI not found. Use /docs for API reference.</p>")


@app.get("/health", tags=["System"])
async def health_check():
    return JSONResponse({"status": "healthy", "app": settings.APP_NAME,
                         "version": settings.APP_VERSION})


@app.post("/query", tags=["Query"])
async def process_query(query: dict):
    """Execute a query through the full agent pipeline. Streams trace to WebSocket."""
    from backend.orchestrator.types import TaskType
    from backend.orchestrator.planner import PlannerAgent
    from backend.orchestrator.router import RouterAgent, AgentRegistry, AgentRegistryEntry
    from backend.llm.factory import LLMFactory
    from backend.core.tracing import TraceCollector, TraceEventType
    from backend.agents.exact_hash_retriever import NestedHashIndex
    from backend.agents.veri_score import VeriScoreEvaluator
    from backend.agents.guardrail import GuardrailAgent, GuardrailAction
    from backend.agents.report_generator import ReportGenerator

    user_query = query.get("query", "")
    session_id = query.get("session_id", str(uuid.uuid4())[:8])
    source_filter = query.get("source_filter", None)  # Optional[List[str]] — scope search to specific docs
    search_mode = query.get("search_mode", "standard")  # "fast" | "standard" | "thorough"
    print(f"[Precis] QUERY source_filter={source_filter!r} mode={search_mode} query={user_query[:60]!r}")
    trace = TraceCollector(query_id=f"q_{uuid.uuid4().hex[:8]}", session_id=session_id)

    # Stream trace events to connected WebSocket clients
    def stream_trace(event_dict: dict) -> None:
        for ws in list(_active_ws.values()):
            asyncio.create_task(ws.send_json({"type": "trace_event", "event": event_dict}))
    trace.set_stream_callback(stream_trace)

    try:
        llm = LLMFactory.create_default()
        planner = PlannerAgent(llm)

        tools = [{"name": "ExactHash", "description": "Multi-source retrieval — keyword hash + semantic vector fusion"},
                 {"name": "DataSynthesis", "description": "LLM-based synthesis of retrieved data into a coherent answer"}]
        plan = await planner.plan(user_query, available_tools=tools, trace=trace)

        import backend.main as _main
        demo_index = _main._demo_index

        registry = AgentRegistry()
        registry.register(AgentRegistryEntry("ExactHash", TaskType.FACTUAL_RETRIEVAL, NestedHashIndex,
                                             "Multi-source retrieval (keyword + vector)", singleton_instance=demo_index))
        registry.register(AgentRegistryEntry("DataSynthesis", TaskType.DATA_SYNTHESIS, None,
                                             "LLM synthesis of multi-source data", is_external_llm=True))

        router = RouterAgent(registry, llm=llm)
        agent_results = await router.execute_plan(plan.subtasks, source_filter=source_filter, search_mode=search_mode, trace=trace)

        # Run data_synthesis subtasks through the synthesis agent
        from backend.agents.data_synthesis import DataSynthesisAgent
        synth = DataSynthesisAgent(llm)
        for i, subtask in enumerate(plan.subtasks):
            if subtask.type == TaskType.DATA_SYNTHESIS:
                trace.span_start("DataSynthesis", "synthesize")
                trace.event(TraceEventType.AGENT_STARTED, agent_name="DataSynthesis",
                            message=f"Synthesizing from {len(subtask.depends_on)} upstream results...")
                upstream = [r for r in agent_results if r.subtask_id in subtask.depends_on]
                synth_result = await synth.synthesize(subtask.query, upstream, llm=llm)
                synth_result.subtask_id = subtask.id
                trace.event(TraceEventType.AGENT_COMPLETED, agent_name="DataSynthesis",
                            message=f"Synthesis complete: {len(synth_result.data.get('synthesis', ''))} chars",
                            data={"success": True, "fragments": synth_result.data.get("source_fragments", 0)})
                trace.span_end()
                # Replace the placeholder or append
                replaced = False
                for j, ar in enumerate(agent_results):
                    if ar.subtask_id == subtask.id:
                        agent_results[j] = synth_result
                        replaced = True
                        break
                if not replaced:
                    agent_results.append(synth_result)

        # If no synthesis subtask was planned, create a verdict from top results
        if not any(s.type == TaskType.DATA_SYNTHESIS for s in plan.subtasks):
            trace.span_start("DataSynthesis", "verdict")
            upstream = [r for r in agent_results if r.success and r.agent_name == "ExactHash"]
            if upstream:
                synth_result = await synth.synthesize(user_query, upstream, llm=llm)
                synth_result.subtask_id = "verdict"
                synth_result.agent_name = "DataSynthesis"
                agent_results.append(synth_result)
                trace.event(TraceEventType.AGENT_COMPLETED, agent_name="DataSynthesis",
                            message=f"Verdict generated")
            trace.span_end()

        evaluator = VeriScoreEvaluator()

        # ── Build separate lists for evaluation ─────────────────
        # source_chunks  = ONLY retrieved source text (no synthesis)
        # synthesis_text = ONLY the LLM-generated synthesis
        # citations      = extracted citation metadata per source
        # This separation is CRITICAL: hallucination detection must
        # compare the LLM output against source evidence, NOT against itself.
        source_chunks: list = []
        synthesis_parts: list = []
        all_citations: list = []

        for r in agent_results:
            if not r.success:
                continue
            data = r.data
            if not data:
                continue

            if isinstance(data, dict):
                # ── Retrieved source chunks ─────────────────
                for item in data.get("results", []):
                    if not isinstance(item, dict):
                        continue
                    # Prefer surrounding context (full paragraph) over n-gram text
                    # so VeriScore has meaningful evidence to compare against.
                    txt = item.get("surrounding", "") or item.get("sentence", "") or item.get("text", "")
                    if not txt:
                        continue
                    source_chunks.append({
                        "text": str(txt),
                        "source": item.get("source", ""),
                        "page": item.get("page", 1),
                        "score": item.get("score", 0),
                        "match_type": item.get("match_type", "broad"),
                        # Preserve structural metadata from MultiToken
                        "is_title": item.get("is_title", False),
                        "is_header": item.get("is_header", False),
                        "token_type": item.get("token_type", "standard"),
                        "font_size": item.get("font_size"),
                    })
                    # ── Citations ───────────────────────────
                    src = item.get("source", "")
                    if src:
                        all_citations.append({
                            "source": src,
                            "page": item.get("page", 1),
                            "text_preview": str(txt)[:200],
                            "match_type": item.get("match_type", ""),
                            "score": item.get("score", 0),
                        })

                # ── LLM synthesis (separate from source chunks!) ─
                synth = data.get("synthesis", "")
                if synth:
                    synthesis_parts.append(str(synth))

            elif isinstance(data, str):
                source_chunks.append({
                    "text": data,
                    "source": r.agent_name,
                    "match_type": "broad",
                })

        # ── Build evaluation inputs ──────────────────────────
        # generated_response = ONLY the LLM synthesis (not source chunks!)
        # retrieved_chunks   = ONLY source evidence (not synthesis!)
        generated_text = " ".join(synthesis_parts) if synthesis_parts else ""
        if not generated_text:
            # Fallback: if no synthesis was produced, use concatenated sources
            generated_text = " ".join(c["text"] for c in source_chunks) if source_chunks else "(no results)"
        generated_text = generated_text[:2000]

        if not source_chunks:
            source_chunks = [{"text": generated_text, "source": "fallback", "match_type": "broad"}]

        eval_report = await evaluator.evaluate(
            user_query,
            source_chunks,       # ← ONLY source evidence
            generated_text,      # ← ONLY synthesis (or fallback)
            all_citations,       # ← actual citations, not []
            trace=trace,
        )

        # ── Guardrail ────────────────────────────────────────
        guard = GuardrailAgent()
        guard_result = await guard.validate(
            generated_text,
            source_chunks,
            user_query,
            eval_report,
        )

        # ── Honour the guardrail action ──────────────────────
        if guard_result.action.value == "block":
            trace.complete("blocked")
            blocked_result = {
                "status": "blocked",
                "trace_id": trace.trace_id,
                "guardrail": {
                    "action": "block",
                    "issues": guard_result.issues_found,
                },
                "message": "Response blocked by safety guardrail.",
            }
            for ws in list(_active_ws.values()):
                asyncio.create_task(ws.send_json({"type": "report_ready", "data": blocked_result}))
            return JSONResponse(blocked_result, status_code=422)

        # ── Report ───────────────────────────────────────────
        # If guardrail redacted PII, swap in the scrubbed text so the
        # final report never contains raw PII.
        if guard_result.action == GuardrailAction.REDACT and guard_result.redacted_response:
            for r in agent_results:
                if r.agent_name == "DataSynthesis" and r.data and isinstance(r.data, dict):
                    r.data["synthesis"] = guard_result.redacted_response
                    break

        report_gen = ReportGenerator()
        report = await report_gen.generate(user_query, agent_results, eval_report, guard_result)

        trace.complete("success")
        from backend.db.repository import save_trace
        trace_id = save_trace(trace.query_id, user_query, trace.to_json(),
                              session_id=session_id, duration_ms=int(trace.get_total_duration_ms()),
                              agent_count=len(plan.subtasks), event_count=trace.get_event_count())

        # Helper to extract human-readable summary from AgentResult data
        def readable_data(r):
            if not r.data:
                return r.error_message or ""
            if isinstance(r.data, str):
                return r.data[:500]
            if isinstance(r.data, dict):
                parts = []
                for item in r.data.get("results", []):
                    if isinstance(item, dict) and item.get("text"):
                        line = item["text"]
                        src = item.get("source", "")
                        pg = item.get("page", "")
                        sc = item.get("score", "")
                        ctx = item.get("surrounding", "")
                        if ctx:
                            line = ctx  # Show surrounding context as primary text
                        if src:
                            line += f"\n  └─ {src}, page {pg} [score: {sc}]"
                        parts.append(line)
                synth = r.data.get("synthesis", "")
                if synth:
                    parts.append(str(synth))
                if parts:
                    return "\n\n".join(parts)[:1000]
                return str(r.data)[:500]
            return str(r.data)[:500]

        result = {"status": "success", "trace_id": trace_id,
                  "plan": {"subtasks": [{"id": s.id, "type": s.type.value, "query": s.query} for s in plan.subtasks],
                           "reasoning": plan.reasoning},
                  "results": [{"agent": r.agent_name, "success": r.success,
                               "data": readable_data(r),
                               "duration_ms": r.execution_time_ms} for r in agent_results],
                  "report": report,
                  "evaluation": {"relevancy": eval_report.relevancy_score,
                                 "trust": eval_report.trustworthiness_score,
                                 "exhaustivity": eval_report.exhaustivity_score,
                                 "hallucination_rate": eval_report.hallucination_rate,
                                 "citation_coverage": eval_report.citation_coverage,
                                 "flagged_issues": eval_report.flagged_issues},
                  "guardrail": {"action": guard_result.action.value,
                                "issues": guard_result.issues_found,
                                "requires_human_review": guard_result.requires_human_review}}

        # Push final result to WebSocket
        for ws in list(_active_ws.values()):
            asyncio.create_task(ws.send_json({"type": "report_ready", "data": result}))

        return result

    except Exception as e:
        trace.complete("error")
        error_result = {"status": "error", "error": str(e), "trace_id": trace.trace_id}
        for ws in list(_active_ws.values()):
            asyncio.create_task(ws.send_json({"type": "report_ready", "data": error_result}))
        return error_result


@app.post("/upload", tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """Upload a TXT or PDF file and index it into both retrieval indexes in parallel."""
    from backend.agents.exact_hash_retriever import NestedHashIndex
    from backend.core.multitoken import MultiTokenExtractor

    content = await file.read()
    filename = file.filename or "uploaded_document"

    # Parse based on file type
    text = _parse_upload_content(content, filename)
    if isinstance(text, JSONResponse):
        return text

    if not text.strip():
        return JSONResponse({"status": "error", "message": "File is empty or could not be parsed"}, status_code=400)

    # Build parsed content (lines → elements) for MultiToken extraction
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

    # ── Run hash indexing + vector indexing in PARALLEL ──────
    import backend.main as _main
    index = _main._demo_index
    extractor = MultiTokenExtractor()

    async def do_hash_index():
        return await extractor.index_document_async(filename, parsed, index)

    async def do_vector_index():
        try:
            if _main._vector_index:
                return _main._vector_index.index_text(text, source=filename)
        except Exception:
            pass
        return 0

    hash_count_task = do_hash_index()
    vec_count_task = do_vector_index()
    count, vec_chunks = await asyncio.gather(hash_count_task, vec_count_task)

    # Also store full text for context retrieval
    index._doc_texts[filename] = text

    # Register in DB (with original text for persistence)
    try:
        from backend.db.repository import register_document_simple
        register_document_simple(filename, multi_token_count=count, document_text=text)
    except Exception:
        pass

    return JSONResponse({"status": "ok", "filename": filename,
                         "multi_tokens_indexed": count, "vector_chunks_indexed": vec_chunks})


@app.post("/upload/batch", tags=["Documents"])
async def upload_documents_batch(files: List[UploadFile] = File(...)):
    """Upload multiple TXT/PDF files concurrently.

    Each file is parsed and indexed in parallel — N files = N concurrent
    hash+vector indexing operations.
    """
    if not files:
        return JSONResponse({"status": "error", "message": "No files provided"}, status_code=400)

    async def process_one(file: UploadFile) -> dict:
        """Process a single file — same logic as /upload but returns dict."""
        try:
            from backend.core.multitoken import MultiTokenExtractor

            content = await file.read()
            filename = file.filename or "uploaded_document"
            text = _parse_upload_content(content, filename)
            if isinstance(text, JSONResponse):
                return {"filename": filename, "status": "error",
                        "error": text.body.decode() if hasattr(text, 'body') else str(text)}

            if not text.strip():
                return {"filename": filename, "status": "error", "error": "Empty file"}

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

            import backend.main as _main
            index = _main._demo_index
            extractor = MultiTokenExtractor()

            async def do_hash(): return await extractor.index_document_async(filename, parsed, index)
            async def do_vec():
                try:
                    if _main._vector_index:
                        return _main._vector_index.index_text(text, source=filename)
                except Exception:
                    pass
                return 0

            count, vec_chunks = await asyncio.gather(do_hash(), do_vec())
            index._doc_texts[filename] = text

            try:
                from backend.db.repository import register_document_simple
                register_document_simple(filename, multi_token_count=count, document_text=text)
            except Exception:
                pass

            return {"filename": filename, "status": "ok",
                    "multi_tokens_indexed": count, "vector_chunks_indexed": vec_chunks}
        except Exception as e:
            return {"filename": file.filename or "unknown", "status": "error", "error": str(e)}

    # ── Process ALL files concurrently ────────────────────────
    results = await asyncio.gather(*(process_one(f) for f in files))

    ok_count = sum(1 for r in results if r.get("status") == "ok")
    return JSONResponse({
        "status": "ok",
        "total": len(results),
        "succeeded": ok_count,
        "failed": len(results) - ok_count,
        "results": results,
    })


# ── Shared helper: parse raw bytes → text ───────────────────────────

def _parse_upload_content(content: bytes, filename: str):
    """Parse uploaded bytes into text. Returns str or JSONResponse on error."""
    text = ""
    if filename.lower().endswith(".pdf"):
        try:
            import fitz
            doc = fitz.open(stream=content, filetype="pdf")
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
        except ImportError:
            return JSONResponse(
                {"status": "error", "message": "PDF parsing unavailable (pymupdf not installed)"},
                status_code=500,
            )
    elif filename.lower().endswith(".txt"):
        text = content.decode("utf-8", errors="replace")
    else:
        return JSONResponse(
            {"status": "error", "message": "Only .txt and .pdf files accepted"},
            status_code=400,
        )
    return text


@app.post("/work-orders/extract", tags=["Work Orders"])
async def extract_work_order(file: UploadFile = File(...)):
    """Upload a work order PDF/TXT and extract structured fields into the database."""
    from backend.agents.work_order_extractor import WorkOrderExtractor
    import json

    content = await file.read()
    filename = file.filename or "work_order"

    text = ""
    if filename.lower().endswith(".pdf"):
        try:
            import fitz, io
            doc = fitz.open(stream=content, filetype="pdf")
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
        except ImportError:
            return JSONResponse({"status": "error", "message": "PDF parsing unavailable"}, status_code=500)
    elif filename.lower().endswith(".txt"):
        text = content.decode("utf-8", errors="replace")
    else:
        return JSONResponse({"status": "error", "message": "Only .txt and .pdf accepted"}, status_code=400)

    if not text.strip():
        return JSONResponse({"status": "error", "message": "Empty file"}, status_code=400)

    extractor = WorkOrderExtractor()
    wo = extractor.extract(text, source_file=filename)

    # Save to database
    from backend.db.repository import save_work_order
    wo_id = save_work_order(wo)

    return JSONResponse({
        "status": "ok",
        "work_order_id": wo_id,
        "tail_number": wo.tail_number,
        "work_order_number": wo.work_order_number,
        "date": wo.date,
        "aircraft_model": wo.aircraft_model,
        "part_numbers": wo.part_numbers,
        "mechanic_id": wo.mechanic_id,
        "station": wo.station,
        "hours_worked": wo.hours_worked,
        "inspector_stamp": wo.inspector_stamp,
        "ad_sb_references": wo.ad_sb_references,
        "fields_extracted": len(wo.extracted_fields),
        "fields_detail": [{"field": f.field_name, "value": f.value, "confidence": f.confidence}
                         for f in wo.extracted_fields[:20]],
    })


@app.get("/work-orders", tags=["Work Orders"])
async def list_work_orders(tail_number: str = "", mechanic_id: str = "", limit: int = 50):
    """Query extracted work orders by tail number, mechanic, or list all."""
    from backend.db.repository import query_work_orders
    results = query_work_orders(tail_number=tail_number, mechanic_id=mechanic_id, limit=limit)
    return JSONResponse(results)


@app.get("/documents", tags=["Documents"])
async def list_documents():
    """List all documents currently indexed."""
    from backend.db.repository import get_all_documents
    docs = get_all_documents()
    return JSONResponse(docs)


@app.get("/debug/stem", tags=["Debug"])
async def debug_stem(q: str = ""):
    """Show how a query is stemmed (for debugging 0-result issues)."""
    from backend.core.stemming import PrecisStemmer
    stemmer = PrecisStemmer()
    raw = q.lower().split()
    stemmed = stemmer.stem_tokens(raw)
    return JSONResponse({"raw_tokens": raw, "stemmed_tokens": stemmed})


@app.get("/debug/search", tags=["Debug"])
async def debug_search(q: str = ""):
    """Run a direct hybrid_search and return results (bypasses planner)."""
    from backend.core.stemming import PrecisStemmer
    stemmer = PrecisStemmer()
    raw = q.lower().split()
    stemmed = tuple(stemmer.stem_tokens(raw))
    import backend.main as _main
    index = _main._demo_index
    results = index.hybrid_search(stemmed)
    return JSONResponse({
        "query": q,
        "stemmed_tokens": list(stemmed),
        "result_count": len(results),
        "results": [{"tokens": list(r.multitoken.tokens),
                     "source": r.multitoken.source_doc,
                     "score": r.relevance_score,
                     "match_type": r.match_type}
                    for r in results[:10]]
    })


@app.websocket("/ws")
async def ws_handler(websocket: WebSocket):
    await websocket.accept()
    ws_id = str(uuid.uuid4())[:8]
    _active_ws[ws_id] = websocket
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("action") == "query":
                asyncio.create_task(process_query(data))
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        _active_ws.pop(ws_id, None)
