"""
Converts trace events from a JSONL file into a NetworkX DAG
and detects optimization opportunities (loops, over-expensive models).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx

from agentimize.tracer.models import Trace, TraceEvent


def _message_hash(event: TraceEvent) -> str:
    """Create a hash of the model + tool_name combination for loop detection."""
    key = f"{event.model}::{event.tool_name or 'none'}"
    return hashlib.md5(key.encode()).hexdigest()[:8]


def load_trace_file(trace_file: str | Path) -> dict[str, list[dict[str, Any]]]:
    """
    Load a JSONL trace file and group events by session_id.
    Returns a dict mapping session_id -> list of raw event dicts.
    """
    sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)

    path = Path(trace_file)
    if not path.exists():
        return dict(sessions)

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                event_data = json.loads(line)
                session_id = event_data.get("session_id", "unknown")
                sessions[session_id].append(event_data)
            except json.JSONDecodeError as e:
                pass  # skip malformed lines

    return dict(sessions)


def load_all_traces(search_dir: str | Path = ".") -> dict[str, list[dict[str, Any]]]:
    """
    Load all .jsonl files in the given directory and merge session data.
    """
    all_sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    search_path = Path(search_dir)

    for jsonl_file in search_path.glob("*.jsonl"):
        file_sessions = load_trace_file(jsonl_file)
        for session_id, events in file_sessions.items():
            all_sessions[session_id].extend(events)

    # Sort each session's events by timestamp
    for session_id in all_sessions:
        all_sessions[session_id].sort(key=lambda e: e.get("timestamp", 0))

    return dict(all_sessions)


def parse_trace(session_id: str, raw_events: list[dict[str, Any]]) -> Trace:
    """
    Parse raw event dicts into a Trace object.
    """
    events: list[TraceEvent] = []
    for raw in raw_events:
        try:
            event = TraceEvent(**raw)
            events.append(event)
        except Exception:
            # Try a more lenient parse with only required fields
            try:
                event = TraceEvent(
                    session_id=raw.get("session_id", session_id),
                    timestamp=float(raw.get("timestamp", 0)),
                    event_type=raw.get("event_type", "llm_call"),
                    model=raw.get("model", "unknown"),
                    prompt_tokens=int(raw.get("prompt_tokens", 0)),
                    completion_tokens=int(raw.get("completion_tokens", 0)),
                    cost_usd=float(raw.get("cost_usd", 0.0)),
                    latency_ms=float(raw.get("latency_ms", 0.0)),
                    tool_name=raw.get("tool_name"),
                    completion_text=raw.get("completion_text", ""),
                )
                events.append(event)
            except Exception:
                pass

    trace = Trace(session_id=session_id, events=events)
    trace.recalculate()
    return trace


def detect_loops(events: list[TraceEvent]) -> list[dict[str, Any]]:
    """
    Detect repeated patterns that might indicate infinite or excessive loops.
    Returns a list of loop annotations: {start_idx, end_idx, pattern, count, events}.
    """
    loops: list[dict[str, Any]] = []
    LOOP_THRESHOLD = 3

    i = 0
    while i < len(events):
        # Check for consecutive runs of the same (model, tool_name) pair
        pattern = _message_hash(events[i])
        j = i + 1
        while j < len(events) and _message_hash(events[j]) == pattern:
            j += 1

        count = j - i
        if count >= LOOP_THRESHOLD:
            loops.append({
                "start_idx": i,
                "end_idx": j - 1,
                "pattern": f"{events[i].model}::{events[i].tool_name or 'none'}",
                "count": count,
                "node_ids": [e.node_id for e in events[i:j]],
            })
            i = j
        else:
            i += 1

    return loops


def build_graph(trace: Trace) -> nx.DiGraph:
    """
    Build a NetworkX DiGraph from a Trace.

    Nodes represent TraceEvent objects with their attributes.
    Edges represent temporal sequence (event i -> event i+1).
    Loop nodes are marked with a 'is_loop' attribute.
    """
    G = nx.DiGraph()

    if not trace.events:
        return G

    # Detect loops before building graph
    loops = detect_loops(trace.events)
    loop_node_ids: set[str] = set()
    for loop in loops:
        for nid in loop["node_ids"]:
            loop_node_ids.add(nid)

    # Add nodes
    for idx, event in enumerate(trace.events):
        node_id = event.node_id
        G.add_node(
            node_id,
            event=event,
            idx=idx,
            model=event.model,
            event_type=event.event_type,
            tool_name=event.tool_name,
            prompt_tokens=event.prompt_tokens,
            completion_tokens=event.completion_tokens,
            cost_usd=event.cost_usd,
            latency_ms=event.latency_ms,
            is_loop=node_id in loop_node_ids,
            timestamp=event.timestamp,
        )

    # Add edges: temporal sequence
    node_ids = [e.node_id for e in trace.events]
    for i in range(len(node_ids) - 1):
        G.add_edge(
            node_ids[i],
            node_ids[i + 1],
            edge_type="temporal",
        )

    # Annotate loop info on the graph itself
    G.graph["loops"] = loops
    G.graph["session_id"] = trace.session_id

    return G


def detect_expensive_model_choices(
    trace: Trace,
    graph: nx.DiGraph,
) -> list[dict[str, Any]]:
    """
    Detect nodes where a cheaper model would likely suffice.

    Heuristics:
    - Calls with prompt_tokens < 500 and no tool calls -> can use cheapest model
    - Calls with prompt_tokens < 2000 and no tool calls -> can use gpt-4o-mini
    - Calls using gpt-4o/gpt-4-turbo for simple classification or short tasks
    """
    EXPENSIVE_MODELS = {"gpt-4o", "gpt-4-turbo", "gpt-4-turbo-preview", "claude-3-opus-20240229"}
    recommendations: list[dict[str, Any]] = []

    for event in trace.events:
        if event.model not in EXPENSIVE_MODELS:
            continue

        is_simple = event.prompt_tokens < 500 and not event.tool_calls
        is_medium = event.prompt_tokens < 2000 and not event.tool_calls

        if is_simple:
            recommendations.append({
                "node_id": event.node_id,
                "current_model": event.model,
                "suggested_model": "gpt-4o-mini",
                "reason": f"Simple call ({event.prompt_tokens} tokens, no tools) — cheapest model sufficient",
                "prompt_tokens": event.prompt_tokens,
                "completion_tokens": event.completion_tokens,
            })
        elif is_medium:
            recommendations.append({
                "node_id": event.node_id,
                "current_model": event.model,
                "suggested_model": "gpt-4o-mini",
                "reason": f"Medium complexity ({event.prompt_tokens} tokens, no tools) — gpt-4o-mini can handle this",
                "prompt_tokens": event.prompt_tokens,
                "completion_tokens": event.completion_tokens,
            })

    return recommendations


def build_trace_from_file(
    trace_file: str | Path,
    session_id: str | None = None,
) -> list[tuple[Trace, nx.DiGraph]]:
    """
    Load a trace file and return (Trace, DiGraph) pairs for each session.
    If session_id is given, only return that session.
    """
    sessions = load_trace_file(trace_file)

    if session_id:
        sessions = {k: v for k, v in sessions.items() if k == session_id}

    results: list[tuple[Trace, nx.DiGraph]] = []
    for sid, raw_events in sessions.items():
        trace = parse_trace(sid, raw_events)
        graph = build_graph(trace)
        results.append((trace, graph))

    return results
