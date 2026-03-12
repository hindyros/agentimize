"""
FastAPI dashboard API for Agentimize.

Serves the static dashboard and provides REST endpoints for
trace data, optimization results, and session management.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import networkx as nx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

# Locate the static directory relative to this file
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Agentimize Dashboard",
    description="AI Agent Cost Optimization Dashboard",
    version="0.1.0",
)

# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class JudgeRequest(BaseModel):
    task_description: str


class AnalyzeRequest(BaseModel):
    trace_file: str


class TraceEventInput(BaseModel):
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    tool_name: str | None = None
    tool_calls: list[dict] = []
    event_type: str = "llm_call"


class OptimizeRequest(BaseModel):
    events: list[TraceEventInput]
    task_description: str | None = None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _is_read_only_env() -> bool:
    """Return True when running in a read-only/serverless environment (e.g. Vercel)."""
    return os.getenv("VERCEL") == "1" or os.getenv("VERCEL_ENV") is not None


def _get_search_dir() -> Path:
    """Return the directory to search for .jsonl trace files."""
    return Path(os.getenv("TRACE_SEARCH_DIR", ".")).resolve()


def _load_all_sessions() -> dict[str, Any]:
    """Load all sessions from all .jsonl files in the search dir."""
    if _is_read_only_env():
        return {}

    from agentimize.tracer.graph_builder import load_all_traces, parse_trace

    search_dir = _get_search_dir()
    raw_sessions = load_all_traces(search_dir)

    sessions: dict[str, Any] = {}
    for session_id, raw_events in raw_sessions.items():
        trace = parse_trace(session_id, raw_events)
        sessions[session_id] = trace

    return sessions


def _get_optimization(session_id: str) -> dict[str, Any] | None:
    """Run optimization on a session and return the result dict."""
    from agentimize.tracer.graph_builder import build_graph, load_all_traces, parse_trace
    from agentimize.optimizer.solver import optimize_trace

    search_dir = _get_search_dir()
    raw_sessions = load_all_traces(search_dir)

    if session_id not in raw_sessions:
        return None

    trace = parse_trace(session_id, raw_sessions[session_id])
    graph = build_graph(trace)
    result = optimize_trace(trace, graph)

    return result.model_dump()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard() -> FileResponse:
    """Serve the main dashboard HTML."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard index.html not found")
    return FileResponse(str(index_path))


@app.get("/api/sessions")
async def list_sessions() -> JSONResponse:
    """List all sessions from all .jsonl trace files."""
    sessions = _load_all_sessions()

    result = []
    for session_id, trace in sessions.items():
        result.append({
            "session_id": session_id,
            "event_count": len(trace.events),
            "total_cost_usd": trace.total_cost_usd,
            "total_tokens": trace.total_tokens,
            "task_description": trace.task_description,
            "task_success": trace.task_success,
            "quality_score": trace.quality_score,
            "start_time": trace.start_time,
            "duration_seconds": trace.duration_seconds,
            "models_used": list({e.model for e in trace.events}),
        })

    # Sort by start_time descending (newest first)
    result.sort(key=lambda s: s.get("start_time") or 0, reverse=True)
    return JSONResponse(content=result)


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str) -> JSONResponse:
    """Get full trace details for a session."""
    sessions = _load_all_sessions()

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    trace = sessions[session_id]
    events_data = []
    for event in trace.events:
        events_data.append({
            "node_id": event.node_id,
            "timestamp": event.timestamp,
            "event_type": event.event_type,
            "model": event.model,
            "prompt_tokens": event.prompt_tokens,
            "completion_tokens": event.completion_tokens,
            "total_tokens": event.total_tokens,
            "cost_usd": event.cost_usd,
            "latency_ms": event.latency_ms,
            "tool_name": event.tool_name,
            "tool_calls": event.tool_calls,
            "completion_text": event.completion_text[:500] if event.completion_text else "",
            "success": event.success,
        })

    return JSONResponse(content={
        "session_id": session_id,
        "total_cost_usd": trace.total_cost_usd,
        "total_tokens": trace.total_tokens,
        "event_count": len(trace.events),
        "task_success": trace.task_success,
        "task_description": trace.task_description,
        "quality_score": trace.quality_score,
        "judge_reasoning": trace.judge_reasoning,
        "start_time": trace.start_time,
        "duration_seconds": trace.duration_seconds,
        "events": events_data,
    })


@app.get("/api/sessions/{session_id}/optimization")
async def get_session_optimization(session_id: str) -> JSONResponse:
    """Get optimization recommendations for a session."""
    result = _get_optimization(session_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return JSONResponse(content=result)


@app.post("/api/sessions/{session_id}/judge")
async def judge_session(session_id: str, request: JudgeRequest) -> JSONResponse:
    """Run LLM judge on a session to assess task success."""
    from agentimize.tracer.graph_builder import load_all_traces, parse_trace
    from agentimize.judge.llm_judge import LLMJudge

    search_dir = _get_search_dir()
    raw_sessions = load_all_traces(search_dir)

    if session_id not in raw_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    trace = parse_trace(session_id, raw_sessions[session_id])
    trace.task_description = request.task_description

    try:
        judge = LLMJudge()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    updated_trace = judge.judge_and_update_trace(trace, request.task_description)

    return JSONResponse(content={
        "session_id": session_id,
        "task_success": updated_trace.task_success,
        "quality_score": updated_trace.quality_score,
        "reasoning": updated_trace.judge_reasoning,
    })


@app.post("/api/analyze")
async def analyze_trace(request: AnalyzeRequest) -> JSONResponse:
    """Analyze a trace file and return optimization results for all sessions."""
    from agentimize.tracer.graph_builder import build_graph, load_trace_file, parse_trace
    from agentimize.optimizer.solver import optimize_trace

    trace_path = Path(request.trace_file)
    if not trace_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Trace file not found: {request.trace_file}"
        )

    raw_sessions = load_trace_file(trace_path)
    results = []

    for session_id, raw_events in raw_sessions.items():
        trace = parse_trace(session_id, raw_events)
        graph = build_graph(trace)
        opt_result = optimize_trace(trace, graph)
        results.append({
            "session_id": session_id,
            "trace_summary": {
                "event_count": len(trace.events),
                "total_cost_usd": trace.total_cost_usd,
                "total_tokens": trace.total_tokens,
            },
            "optimization": opt_result.model_dump(),
        })

    return JSONResponse(content={"results": results, "trace_file": str(trace_path)})


@app.get("/api/summary")
async def get_summary() -> JSONResponse:
    """Get aggregate statistics across all sessions."""
    from agentimize.tracer.graph_builder import build_graph
    from agentimize.optimizer.solver import optimize_trace

    sessions = _load_all_sessions()

    total_runs = len(sessions)
    total_cost = sum(t.total_cost_usd for t in sessions.values())
    total_tokens = sum(t.total_tokens for t in sessions.values())

    # Run optimization on each session to get savings estimates
    total_savings = 0.0
    savings_pcts: list[float] = []

    for session_id, trace in sessions.items():
        try:
            graph = build_graph(trace)
            opt_result = optimize_trace(trace, graph)
            total_savings += opt_result.savings_usd
            if opt_result.savings_pct > 0:
                savings_pcts.append(opt_result.savings_pct)
        except Exception:
            pass

    avg_savings_pct = sum(savings_pcts) / len(savings_pcts) if savings_pcts else 0.0

    successful = sum(1 for t in sessions.values() if t.task_success is True)
    failed = sum(1 for t in sessions.values() if t.task_success is False)
    unknown = total_runs - successful - failed

    return JSONResponse(content={
        "total_runs": total_runs,
        "total_cost_usd": round(total_cost, 6),
        "total_tokens": total_tokens,
        "total_savings_usd": round(total_savings, 6),
        "avg_savings_pct": round(avg_savings_pct, 2),
        "task_outcomes": {
            "successful": successful,
            "failed": failed,
            "unknown": unknown,
        },
    })


@app.post("/api/optimize")
async def optimize_events(request: OptimizeRequest) -> JSONResponse:
    """
    Public optimization endpoint for join39.org and external callers.

    Accepts a list of LLM call events and returns cost optimization recommendations.
    No file system access required — pass events directly in the request body.
    """
    import time, uuid
    from agentimize.tracer.models import Trace, TraceEvent
    from agentimize.tracer.graph_builder import build_graph
    from agentimize.optimizer.solver import optimize_trace

    if not request.events:
        raise HTTPException(status_code=400, detail="events list cannot be empty")

    session_id = str(uuid.uuid4())
    events = []
    for i, e in enumerate(request.events):
        events.append(TraceEvent(
            session_id=session_id,
            timestamp=time.time() + i,
            event_type=e.event_type,
            model=e.model,
            prompt_tokens=e.prompt_tokens,
            completion_tokens=e.completion_tokens,
            cost_usd=e.cost_usd,
            latency_ms=e.latency_ms,
            tool_name=e.tool_name,
            tool_calls=e.tool_calls,
            completion_text="",
        ))

    trace = Trace(session_id=session_id, events=events, task_description=request.task_description)
    trace.recalculate()
    graph = build_graph(trace)
    result = optimize_trace(trace, graph)
    d = result.model_dump()

    return JSONResponse(content={
        "session_id": session_id,
        "original_cost_usd": d["original_cost_usd"],
        "optimized_cost_usd": d["optimized_cost_usd"],
        "savings_usd": d["savings_usd"],
        "savings_pct": d["savings_pct"],
        "recommendations": d["recommendations"],
        "loop_recommendations": d["loop_recommendations"],
        "summary": d["summary"],
        "formulation": d["formulation"],
    })


@app.get("/api/health")
async def health_check() -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse(content={"status": "ok", "service": "agentimize-dashboard"})
