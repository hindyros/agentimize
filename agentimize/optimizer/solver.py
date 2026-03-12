"""
Cost-minimization optimizer for agent LLM calls.

Uses a greedy heuristic approach to assign the cheapest model that
meets the complexity requirements of each LLM call node.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
from pydantic import BaseModel

from agentimize.tracer.models import Trace, TraceEvent

# Pricing table: cost per 1M tokens
MODEL_PRICES: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4-turbo-preview": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "gpt-3.5-turbo-0125": {"input": 0.50, "output": 1.50},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-20240620": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
}

# Ordered from cheapest to most expensive (capability tiers)
MODEL_TIERS: list[str] = [
    "gpt-4o-mini",
    "gpt-3.5-turbo",
    "claude-3-haiku-20240307",
    "gpt-4o",
    "claude-3-5-sonnet-20241022",
    "gpt-4-turbo",
    "claude-3-opus-20240229",
]

# Minimum tier index required per complexity level
COMPLEXITY_MIN_TIER: dict[str, int] = {
    "simple": 0,   # gpt-4o-mini
    "medium": 0,   # gpt-4o-mini (it's quite capable)
    "complex": 3,  # gpt-4o minimum
}


def model_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost in USD for a given model and token counts."""
    price = MODEL_PRICES.get(model)
    if price is None:
        # Try partial match
        for key in MODEL_PRICES:
            if model.startswith(key[:10]):
                price = MODEL_PRICES[key]
                break
    if price is None:
        price = MODEL_PRICES["gpt-4o"]  # conservative fallback

    return (prompt_tokens * price["input"] + completion_tokens * price["output"]) / 1_000_000


def model_tier_index(model: str) -> int:
    """Return the tier index of a model, or a middle-ground default."""
    if model in MODEL_TIERS:
        return MODEL_TIERS.index(model)
    # Try to find a close match
    for i, tier in enumerate(MODEL_TIERS):
        if model.startswith(tier[:8]):
            return i
    # Default to gpt-4o tier (index 3) for unknown models
    return 3


def classify_call_complexity(event: TraceEvent) -> str:
    """
    Classify an LLM call as simple, medium, or complex.

    Rules:
    - complex: has tool_calls OR prompt_tokens > 2000
    - simple: prompt_tokens < 500 AND no tool_calls
    - medium: everything in between
    """
    has_tools = bool(event.tool_calls)
    if has_tools or event.prompt_tokens > 2000:
        return "complex"
    elif event.prompt_tokens < 500:
        return "simple"
    else:
        return "medium"


class OptimizationResult(BaseModel):
    """Result of the optimization analysis."""

    original_cost_usd: float
    optimized_cost_usd: float
    savings_usd: float
    savings_pct: float
    recommendations: list[dict[str, Any]]  # per-node recommendations
    loop_recommendations: list[dict[str, Any]]  # loop reduction recommendations
    summary: str  # human-readable summary
    formulation: dict[str, Any] = {}  # mathematical optimization formulation


def optimize_trace(
    trace: Trace,
    graph: nx.DiGraph,
    force_no_downgrade: bool = False,
) -> OptimizationResult:
    """
    Run the greedy cost optimization on a trace.

    For each node:
    1. Classify complexity (simple / medium / complex)
    2. Assign cheapest model meeting the complexity threshold
    3. Calculate savings vs original

    If task_success is known to be False, do not downgrade any model.
    """
    # If the task failed, don't recommend any downgrades
    if trace.task_success is False or force_no_downgrade:
        return OptimizationResult(
            original_cost_usd=round(trace.total_cost_usd, 6),
            optimized_cost_usd=round(trace.total_cost_usd, 6),
            savings_usd=0.0,
            savings_pct=0.0,
            recommendations=[],
            loop_recommendations=[],
            summary=(
                "Optimization skipped: task was marked as failed. "
                "Fix the agent's task completion before optimizing costs."
            ),
        )

    recommendations: list[dict[str, Any]] = []
    original_total = 0.0
    optimized_total = 0.0

    for event in trace.events:
        complexity = classify_call_complexity(event)
        min_tier_idx = COMPLEXITY_MIN_TIER[complexity]

        # Find the cheapest model that meets the complexity requirement
        best_model = MODEL_TIERS[min_tier_idx]
        best_cost = model_cost(best_model, event.prompt_tokens, event.completion_tokens)

        original_cost = model_cost(event.model, event.prompt_tokens, event.completion_tokens)
        # Use original cost (not recalculated) for accurate comparison
        # But recalculate to be consistent
        original_total += original_cost
        optimized_total += best_cost

        savings = original_cost - best_cost
        current_tier = model_tier_index(event.model)

        # Only recommend if we'd actually downgrade
        if current_tier > min_tier_idx and savings > 0.000001:
            recommendations.append({
                "node_id": event.node_id,
                "original_model": event.model,
                "recommended_model": best_model,
                "complexity": complexity,
                "reason": _build_reason(event, complexity, best_model),
                "original_cost_usd": round(original_cost, 8),
                "optimized_cost_usd": round(best_cost, 8),
                "savings_usd": round(savings, 8),
                "prompt_tokens": event.prompt_tokens,
                "completion_tokens": event.completion_tokens,
                "timestamp": event.timestamp,
                "tool_name": event.tool_name,
            })

    # Loop recommendations from graph metadata
    loop_recommendations: list[dict[str, Any]] = []
    loops = graph.graph.get("loops", [])
    # Build a lookup of node_id -> recommended model from model recommendations
    rec_model_by_node = {r["node_id"]: r["recommended_model"] for r in recommendations}

    for loop in loops:
        # Estimate savings from capping iterations
        detected = loop["count"]
        recommended_max = max(2, detected // 2)  # cap at half the detected count

        # Calculate loop cost at the OPTIMIZED model price (not original) to avoid double-counting
        loop_optimized_cost = 0.0
        for node_id in loop["node_ids"]:
            if node_id in graph.nodes:
                node_data = graph.nodes[node_id]
                event = node_data.get("event")
                if event:
                    effective_model = rec_model_by_node.get(node_id, event.model)
                    loop_optimized_cost += model_cost(
                        effective_model, event.prompt_tokens, event.completion_tokens
                    )

        saved_iterations = detected - recommended_max
        loop_savings = loop_optimized_cost * (saved_iterations / detected) if detected > 0 else 0.0

        loop_recommendations.append({
            "node_id": loop["node_ids"][0] if loop["node_ids"] else "",
            "pattern": loop["pattern"],
            "detected_iterations": detected,
            "recommended_max": recommended_max,
            "savings_usd": round(loop_savings, 8),
            "description": (
                f"Loop detected: '{loop['pattern']}' repeated {detected} times. "
                f"Consider adding early-exit logic or capping at {recommended_max} iterations."
            ),
        })
        # Subtract loop savings from optimized total (at optimized model cost, safe from going negative)
        optimized_total = max(0.0, optimized_total - loop_savings)

    savings_usd = max(0.0, original_total - optimized_total)
    savings_pct = (savings_usd / original_total * 100) if original_total > 0 else 0.0

    summary = _build_summary(
        original_total=original_total,
        optimized_total=optimized_total,
        savings_usd=savings_usd,
        savings_pct=savings_pct,
        recommendations=recommendations,
        loop_recommendations=loop_recommendations,
        trace=trace,
    )

    formulation = _build_formulation(
        trace=trace,
        recommendations=recommendations,
        loop_recommendations=loop_recommendations,
        original_total=original_total,
        optimized_total=optimized_total,
    )

    return OptimizationResult(
        original_cost_usd=round(original_total, 6),
        optimized_cost_usd=round(max(0.0, optimized_total), 6),
        savings_usd=round(savings_usd, 6),
        savings_pct=round(savings_pct, 2),
        recommendations=recommendations,
        loop_recommendations=loop_recommendations,
        summary=summary,
        formulation=formulation,
    )


def _build_reason(event: TraceEvent, complexity: str, recommended_model: str) -> str:
    """Build a human-readable reason for the model recommendation."""
    parts = []
    if complexity == "simple":
        parts.append(f"Simple call ({event.prompt_tokens} tokens, no tool calls)")
        parts.append(f"{recommended_model} is fully capable for this complexity")
    elif complexity == "medium":
        parts.append(f"Medium complexity ({event.prompt_tokens} tokens, no tool calls)")
        parts.append(f"{recommended_model} handles this well at much lower cost")
    else:
        parts.append(f"Complex call ({event.prompt_tokens} tokens")
        if event.tool_calls:
            parts.append(f", {len(event.tool_calls)} tool call(s))")
        else:
            parts.append(")")
        parts.append(f"Minimum required tier: {recommended_model}")

    return ". ".join(parts) + "."


def _build_summary(
    original_total: float,
    optimized_total: float,
    savings_usd: float,
    savings_pct: float,
    recommendations: list[dict],
    loop_recommendations: list[dict],
    trace: Trace,
) -> str:
    """Build a concise human-readable optimization summary."""
    lines = [
        f"Optimization Analysis for Session {trace.session_id[:8]}...",
        "",
        f"Original cost: ${original_total:.6f}",
        f"Optimized cost: ${max(0.0, optimized_total):.6f}",
        f"Estimated savings: ${savings_usd:.6f} ({savings_pct:.1f}%)",
        "",
    ]

    if recommendations:
        lines.append(f"Model Recommendations ({len(recommendations)} calls to optimize):")
        for r in recommendations[:5]:  # top 5
            lines.append(
                f"  - Replace {r['original_model']} → {r['recommended_model']} "
                f"(saves ${r['savings_usd']:.6f}): {r['reason']}"
            )
        if len(recommendations) > 5:
            lines.append(f"  ... and {len(recommendations) - 5} more recommendations")
    else:
        lines.append("No model downgrade recommendations — models are already well-sized.")

    if loop_recommendations:
        lines.append("")
        lines.append(f"Loop Recommendations ({len(loop_recommendations)} patterns detected):")
        for lr in loop_recommendations:
            lines.append(f"  - {lr['description']}")

    if savings_pct > 50:
        lines.append("")
        lines.append(
            "SIGNIFICANT savings opportunity detected! "
            "Implementing these recommendations could cut your agent costs in half."
        )
    elif savings_pct > 20:
        lines.append("")
        lines.append(
            "Good optimization opportunity. "
            "These changes would meaningfully reduce your operating costs."
        )

    return "\n".join(lines)


def _build_formulation(
    trace: Trace,
    recommendations: list[dict[str, Any]],
    loop_recommendations: list[dict[str, Any]],
    original_total: float,
    optimized_total: float,
) -> dict[str, Any]:
    """
    Build a structured representation of the optimization problem for display in the UI.
    Includes the mathematical formulation, decision variables, objective, constraints,
    and a plain-English explanation.
    """
    n = len(trace.events)
    decision_variables = []
    for i, event in enumerate(trace.events):
        complexity = classify_call_complexity(event)
        min_tier = COMPLEXITY_MIN_TIER[complexity]
        feasible_models = MODEL_TIERS[min_tier:]
        decision_variables.append({
            "index": i,
            "node_id": event.node_id,
            "var_name": f"m_{i}",
            "description": f"Model choice for call #{i+1} ({event.event_type}, {event.prompt_tokens} prompt tokens)",
            "domain": feasible_models,
            "current_value": event.model,
            "assigned_value": next(
                (r["recommended_model"] for r in recommendations if r["node_id"] == event.node_id),
                event.model,
            ),
            "complexity": complexity,
            "prompt_tokens": event.prompt_tokens,
            "completion_tokens": event.completion_tokens,
            "tool_name": event.tool_name,
        })

    # Build the objective function terms
    objective_terms = []
    for dv in decision_variables:
        price = MODEL_PRICES.get(dv["assigned_value"], MODEL_PRICES["gpt-4o"])
        term_cost = (dv["prompt_tokens"] * price["input"] + dv["completion_tokens"] * price["output"]) / 1_000_000
        objective_terms.append({
            "var": dv["var_name"],
            "model": dv["assigned_value"],
            "prompt_tokens": dv["prompt_tokens"],
            "completion_tokens": dv["completion_tokens"],
            "input_price_per_1m": price["input"],
            "output_price_per_1m": price["output"],
            "cost": round(term_cost, 8),
            "latex": (
                f"({dv['prompt_tokens']} \\times {price['input']} "
                f"+ {dv['completion_tokens']} \\times {price['output']}) / 10^6"
            ),
        })

    # Constraints
    constraints = []
    for dv in decision_variables:
        min_tier = COMPLEXITY_MIN_TIER[dv["complexity"]]
        min_model = MODEL_TIERS[min_tier]
        constraints.append({
            "type": "model_tier",
            "var": dv["var_name"],
            "description": f"m_{dv['index']} ∈ {{{', '.join(MODEL_TIERS[min_tier:])}}}",
            "natural_language": (
                "Call #{} is classified as '{}' ({}) → minimum model tier: {}".format(
                    dv["index"] + 1,
                    dv["complexity"],
                    "has tool calls" if dv["tool_name"] else "{} tokens, no tools".format(dv["prompt_tokens"]),
                    min_model,
                )
            ),
            "satisfied": True,
        })

    for lr in loop_recommendations:
        constraints.append({
            "type": "loop_cap",
            "var": f"k_{lr['node_id'][:6]}",
            "description": f"k ≤ {lr['recommended_max']}  (loop pattern: {lr['pattern']})",
            "natural_language": (
                f"Loop '{lr['pattern']}' ran {lr['detected_iterations']} times. "
                f"Cap iterations at {lr['recommended_max']} to eliminate redundant calls."
            ),
            "satisfied": True,
        })

    # Plain-English explanation
    n_simple   = sum(1 for dv in decision_variables if dv["complexity"] == "simple")
    n_medium   = sum(1 for dv in decision_variables if dv["complexity"] == "medium")
    n_complex  = sum(1 for dv in decision_variables if dv["complexity"] == "complex")
    n_loops    = len(loop_recommendations)
    n_changed  = sum(1 for dv in decision_variables if dv["current_value"] != dv["assigned_value"])

    explanation = (
        f"This run produced {n} LLM calls costing ${original_total:.6f} total. "
        f"The optimizer classified each call by complexity: "
        f"{n_simple} simple, {n_medium} medium, {n_complex} complex. "
        f"For each call, it selects the cheapest model from the feasible tier set — "
        f"simple calls can use any model (minimum: gpt-4o-mini), "
        f"complex calls require at least gpt-4o. "
        f"{n_changed} of {n} calls were assigned a cheaper model. "
    )
    if n_loops:
        explanation += (
            f"Additionally, {n_loops} loop pattern(s) were detected and capped at reduced iteration counts. "
        )
    explanation += (
        f"The optimized total is ${max(0.0, optimized_total):.6f}, "
        f"a reduction of ${max(0.0, original_total - optimized_total):.6f}."
    )

    return {
        "problem_type": "Integer Linear Program (greedy relaxation)",
        "n_variables": n,
        "n_constraints": len(constraints),
        "objective": {
            "sense": "minimize",
            "description": "Minimize total cost of all LLM API calls",
            "latex": "\\min \\sum_{i=1}^{N} \\left( p_i^{in} \\cdot t_i^{in} + p_i^{out} \\cdot t_i^{out} \\right) / 10^6",
            "natural_language": (
                "Choose the cheapest viable model for each call such that the sum of "
                "(prompt_tokens × input_price + completion_tokens × output_price) across all calls is minimized."
            ),
            "terms": objective_terms,
        },
        "decision_variables": decision_variables,
        "constraints": constraints,
        "explanation": explanation,
        "solver": "Greedy tier assignment (optimal for independent call cost minimization)",
        "original_cost": round(original_total, 8),
        "optimized_cost": round(max(0.0, optimized_total), 8),
    }
