"""
LLM-based judge that evaluates task success and generates optimization reports.

Uses GPT-4o to assess whether an agent successfully completed its task,
and to write human-readable optimization summaries.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from agentimize.optimizer.solver import OptimizationResult
from agentimize.tracer.models import Trace

load_dotenv()

JUDGE_SYSTEM_PROMPT = """You are an objective evaluator of AI agent outputs.
Given a task description and the final output of an AI agent, determine if the task was completed successfully.

Return ONLY valid JSON with exactly these keys:
- success: boolean (true if the task was completed successfully, false otherwise)
- quality_score: number between 0.0 and 1.0 (0.0 = completely failed, 1.0 = perfectly completed)
- reasoning: string explaining your assessment

Be strict but fair. A task is successful if the agent's output meaningfully addresses the task requirements."""

OPTIMIZATION_SUMMARY_PROMPT = """You are an expert in AI infrastructure cost optimization.
Given a trace of an AI agent's LLM API calls and the optimization analysis results,
write a clear, actionable optimization report in 3-5 paragraphs.

Focus on:
1. What the agent was doing (based on the trace)
2. Where costs are highest and why
3. Specific actionable recommendations
4. Expected impact of implementing the recommendations

Be specific and technical but accessible. Mention actual model names, token counts, and dollar amounts."""


class LLMJudge:
    """Uses GPT-4o to evaluate task success and generate optimization summaries."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable."
            )
        self.model = model
        self._client = OpenAI(api_key=self.api_key)

    def judge_task(
        self,
        task_description: str,
        final_response: str,
        max_response_chars: int = 4000,
    ) -> dict[str, Any]:
        """
        Judge whether a task was completed successfully.

        Args:
            task_description: What the agent was supposed to do
            final_response: The agent's final output (last completion in trace)
            max_response_chars: Truncate final_response to this length to control costs

        Returns:
            Dict with keys: success (bool), quality_score (float), reasoning (str)
        """
        # Truncate very long responses to control judge costs
        truncated_response = final_response[:max_response_chars]
        if len(final_response) > max_response_chars:
            truncated_response += f"\n\n[... truncated, {len(final_response) - max_response_chars} more chars ...]"

        user_content = f"""TASK DESCRIPTION:
{task_description}

AGENT'S FINAL OUTPUT:
{truncated_response}

Evaluate whether the agent successfully completed the task. Return valid JSON only."""

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=500,
            )

            result_text = response.choices[0].message.content or "{}"
            result = json.loads(result_text)

            # Normalize the result
            return {
                "success": bool(result.get("success", False)),
                "quality_score": float(result.get("quality_score", 0.0)),
                "reasoning": str(result.get("reasoning", "No reasoning provided")),
            }

        except json.JSONDecodeError as e:
            return {
                "success": False,
                "quality_score": 0.0,
                "reasoning": f"Judge response could not be parsed as JSON: {e}",
            }
        except Exception as e:
            return {
                "success": False,
                "quality_score": 0.0,
                "reasoning": f"Judge evaluation failed: {e}",
            }

    def generate_optimization_summary(
        self,
        trace: Trace,
        optimization_result: OptimizationResult,
    ) -> str:
        """
        Generate a human-readable optimization report using GPT-4o.

        Args:
            trace: The agent's execution trace
            optimization_result: The optimization analysis result

        Returns:
            A multi-paragraph optimization report string
        """
        # Build a concise trace summary for the judge
        trace_summary = _build_trace_summary(trace)
        opt_summary = _build_optimization_summary(optimization_result)

        user_content = f"""AGENT TRACE SUMMARY:
{trace_summary}

OPTIMIZATION ANALYSIS:
{opt_summary}

Write a clear, actionable optimization report for this agent's LLM usage."""

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": OPTIMIZATION_SUMMARY_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.3,
                max_tokens=800,
            )
            return response.choices[0].message.content or optimization_result.summary

        except Exception as e:
            # Fall back to the programmatically generated summary
            return f"{optimization_result.summary}\n\n(Note: GPT-4o summary generation failed: {e})"

    def judge_and_update_trace(
        self,
        trace: Trace,
        task_description: str,
    ) -> Trace:
        """
        Run the judge on a trace and update it with the results.

        Returns the updated Trace with task_success, quality_score, and judge_reasoning set.
        """
        final_response = trace.final_completion
        if not final_response:
            # Use all completion texts joined
            final_response = " | ".join(
                e.completion_text for e in trace.events if e.completion_text
            )

        result = self.judge_task(
            task_description=task_description,
            final_response=final_response,
        )

        trace.task_success = result["success"]
        trace.quality_score = result["quality_score"]
        trace.judge_reasoning = result["reasoning"]
        trace.task_description = task_description

        return trace


def _build_trace_summary(trace: Trace) -> str:
    """Build a concise text summary of a trace for the judge prompt."""
    lines = [
        f"Session: {trace.session_id[:8]}...",
        f"Total events: {len(trace.events)}",
        f"Total cost: ${trace.total_cost_usd:.6f}",
        f"Total tokens: {trace.total_tokens}",
        f"Duration: {trace.duration_seconds:.1f}s",
        "",
        "Events breakdown:",
    ]

    from collections import Counter
    model_counts = Counter(e.model for e in trace.events)
    for model, count in model_counts.most_common():
        model_cost = sum(e.cost_usd for e in trace.events if e.model == model)
        lines.append(f"  {model}: {count} calls, ${model_cost:.6f}")

    tool_events = [e for e in trace.events if e.tool_name]
    if tool_events:
        tool_counts = Counter(e.tool_name for e in tool_events)
        lines.append("")
        lines.append("Tool calls:")
        for tool, count in tool_counts.most_common():
            lines.append(f"  {tool}: {count} calls")

    return "\n".join(lines)


def _build_optimization_summary(result: OptimizationResult) -> str:
    """Build a concise optimization result summary for the judge prompt."""
    lines = [
        f"Original cost: ${result.original_cost_usd:.6f}",
        f"Optimized cost: ${result.optimized_cost_usd:.6f}",
        f"Potential savings: ${result.savings_usd:.6f} ({result.savings_pct:.1f}%)",
        "",
    ]

    if result.recommendations:
        lines.append(f"Top recommendations ({len(result.recommendations)} total):")
        for rec in result.recommendations[:5]:
            lines.append(
                f"  - {rec['original_model']} → {rec['recommended_model']}: "
                f"${rec['savings_usd']:.6f} savings ({rec['reason']})"
            )

    if result.loop_recommendations:
        lines.append("")
        lines.append(f"Loop issues detected: {len(result.loop_recommendations)}")
        for lr in result.loop_recommendations:
            lines.append(
                f"  - {lr['pattern']}: {lr['detected_iterations']} iterations, "
                f"recommend cap at {lr['recommended_max']}"
            )

    return "\n".join(lines)
