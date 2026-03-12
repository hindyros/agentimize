"""Pydantic v2 models for trace events and aggregated traces."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class TraceEvent(BaseModel):
    """Represents a single LLM API call captured by the proxy."""

    session_id: str
    timestamp: float
    event_type: str  # "llm_call" | "tool_call"
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float
    tool_name: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    success: bool | None = None  # filled in by judge after the run
    path: str = "/v1/chat/completions"
    raw_request: dict[str, Any] = Field(default_factory=dict)
    messages_summary: dict[str, Any] = Field(default_factory=dict)
    raw_response: dict[str, Any] = Field(default_factory=dict)
    completion_text: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def node_id(self) -> str:
        """Unique node identifier for graph building."""
        return f"{self.session_id}_{int(self.timestamp * 1000)}"


class Trace(BaseModel):
    """Aggregated trace for a complete agent session."""

    session_id: str
    events: list[TraceEvent] = Field(default_factory=list)
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    task_success: bool | None = None
    task_description: str | None = None
    quality_score: float | None = None
    judge_reasoning: str | None = None

    @model_validator(mode="after")
    def compute_totals(self) -> "Trace":
        """Recompute totals from events if not explicitly set."""
        if self.events and self.total_cost_usd == 0.0:
            self.total_cost_usd = sum(e.cost_usd for e in self.events)
        if self.events and self.total_tokens == 0:
            self.total_tokens = sum(e.total_tokens for e in self.events)
        return self

    def recalculate(self) -> None:
        """Recalculate totals from events."""
        self.total_cost_usd = round(sum(e.cost_usd for e in self.events), 8)
        self.total_tokens = sum(e.total_tokens for e in self.events)

    @property
    def start_time(self) -> float | None:
        if not self.events:
            return None
        return min(e.timestamp for e in self.events)

    @property
    def end_time(self) -> float | None:
        if not self.events:
            return None
        return max(e.timestamp for e in self.events)

    @property
    def duration_seconds(self) -> float:
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time

    @property
    def final_completion(self) -> str:
        """Return the last completion text in the trace."""
        for event in reversed(self.events):
            if event.completion_text:
                return event.completion_text
        return ""
