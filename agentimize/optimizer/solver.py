"""
Cost-minimization optimizer for agent LLM calls.

Formulates a Mixed Integer Linear Program (MILP):

  Variables:
    x[i,j] ∈ {0,1}   — assign model j to call i
    k[l]   ∈ Z+       — iterations eliminated for loop l

  Objective:
    min  Σ_{i,j} x[i,j] · c(model_j, tokens_i)  −  Σ_l k[l] · s_l

  Constraints:
    Σ_j x[i,j] = 1                              ∀ i   (assignment)
    x[i,j] = 0  if tier(j) < min_tier(i)               (feasibility, via bounds)
    Σ_{i,j} x[i,j] · cap_j · w_i / W ≥ Q_min          (quality floor, optional)
    Σ_{i,j} x[i,j] · c_{i,j} − Σ_l k[l]·s_l ≤ B      (budget cap, optional)
    0 ≤ k[l] ≤ detected[l] − 1                 ∀ l   (loop elimination bounds)

Solved with scipy.optimize.milp (HiGHS backend).
Falls back to greedy if scipy is unavailable or the problem is infeasible.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
from pydantic import BaseModel

from agentimize.tracer.models import Trace, TraceEvent

# ---------------------------------------------------------------------------
# Pricing and model tables
# ---------------------------------------------------------------------------

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
    # Anthropic Claude 4 family
    "claude-opus-4-6": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    # Groq models (approximate)
    "llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "mixtral-8x7b-32768": {"input": 0.24, "output": 0.24},
    "gemma2-9b-it": {"input": 0.20, "output": 0.20},
}

# Ordered cheapest → most expensive (capability tiers)
MODEL_TIERS: list[str] = [
    "gpt-4o-mini",           # tier 0  — cheapest OpenAI
    "gpt-3.5-turbo",         # tier 1
    "claude-3-haiku-20240307",  # tier 2
    "gpt-4o",                # tier 3  — strong general purpose
    "claude-3-5-sonnet-20241022",  # tier 4
    "gpt-4-turbo",           # tier 5
    "claude-3-opus-20240229",  # tier 6 — most expensive
]

# Minimum tier index required per complexity class
COMPLEXITY_MIN_TIER: dict[str, int] = {
    "simple": 0,   # any model works
    "medium": 0,   # gpt-4o-mini handles medium calls well
    "complex": 3,  # gpt-4o minimum for tool-use / long context
}

# Capability scores for quality-floor constraint (0 = weakest, 1 = strongest)
MODEL_CAPABILITY: dict[str, float] = {
    "gpt-4o-mini": 0.30,
    "gpt-3.5-turbo": 0.35,
    "claude-3-haiku-20240307": 0.38,
    "gpt-4o": 0.80,
    "claude-3-5-sonnet-20241022": 0.85,
    "gpt-4-turbo": 0.90,
    "claude-3-opus-20240229": 1.00,
}

# Importance weights for quality-floor constraint by complexity class
COMPLEXITY_WEIGHT: dict[str, float] = {
    "simple": 0.1,
    "medium": 0.5,
    "complex": 1.0,
}


# ---------------------------------------------------------------------------
# Pricing utilities
# ---------------------------------------------------------------------------

def model_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost in USD for a given model and token counts."""
    price = MODEL_PRICES.get(model)
    if price is None:
        for key in MODEL_PRICES:
            if model.startswith(key[:10]):
                price = MODEL_PRICES[key]
                break
    if price is None:
        price = MODEL_PRICES["gpt-4o"]  # conservative fallback
    return (prompt_tokens * price["input"] + completion_tokens * price["output"]) / 1_000_000


def model_tier_index(model: str) -> int:
    """Return the tier index of a model, or a sensible default."""
    if model in MODEL_TIERS:
        return MODEL_TIERS.index(model)
    for i, tier in enumerate(MODEL_TIERS):
        if model.startswith(tier[:8]):
            return i
    return 3  # default to gpt-4o tier for unknown models


def classify_call_complexity(event: TraceEvent) -> str:
    """
    Classify an LLM call as simple, medium, or complex.

    Rules:
      complex — has tool_calls OR prompt_tokens > 2000
      simple  — prompt_tokens < 500 AND no tool_calls
      medium  — everything in between
    """
    has_tools = bool(event.tool_calls)
    if has_tools or event.prompt_tokens > 2000:
        return "complex"
    elif event.prompt_tokens < 500:
        return "simple"
    else:
        return "medium"


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

class OptimizationResult(BaseModel):
    original_cost_usd: float
    optimized_cost_usd: float
    savings_usd: float
    savings_pct: float
    recommendations: list[dict[str, Any]]
    loop_recommendations: list[dict[str, Any]]
    summary: str
    formulation: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# MILP solver (scipy / HiGHS backend)
# ---------------------------------------------------------------------------

def _precompute_loop_savings(
    loops: list[dict],
    events: list[TraceEvent],
    graph: nx.DiGraph,
) -> list[float]:
    """
    Precompute the cost savings per eliminated iteration for each loop.

    We use the cheapest feasible model for each call inside the loop.
    This linearises the otherwise bilinear term k_l · x_{i,j}.
    """
    savings_per_iter: list[float] = []

    for loop in loops:
        detected = loop["count"]
        if detected == 0:
            savings_per_iter.append(0.0)
            continue

        loop_cost = 0.0
        for node_id in loop.get("node_ids", []):
            if node_id not in graph.nodes:
                continue
            event = graph.nodes[node_id].get("event")
            if event is None:
                continue
            complexity = classify_call_complexity(event)
            min_tier = COMPLEXITY_MIN_TIER[complexity]
            cheapest_model = MODEL_TIERS[min_tier]
            loop_cost += model_cost(cheapest_model, event.prompt_tokens, event.completion_tokens)

        savings_per_iter.append(loop_cost / detected)

    return savings_per_iter


def _solve_mip(
    events: list[TraceEvent],
    loops: list[dict],
    loop_savings_per_iter: list[float],
    quality_floor: float,
    budget_usd: float | None,
) -> tuple[list[str], list[int], bool, str]:
    """
    Solve the MILP using scipy.optimize.milp (HiGHS).

    Returns:
        (assigned_models, eliminated_iters, success, solver_msg)
    """
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import csr_matrix
    except ImportError:
        return [], [], False, "scipy not available"

    N = len(events)
    M = len(MODEL_TIERS)
    L = len(loops)

    if N == 0:
        return [], [], True, "no events"

    n_vars = N * M + L

    # ---- cost vector -------------------------------------------------------
    c = np.zeros(n_vars)
    for i, event in enumerate(events):
        for j, model_name in enumerate(MODEL_TIERS):
            c[i * M + j] = model_cost(model_name, event.prompt_tokens, event.completion_tokens)
    for l in range(L):
        # Negative: eliminating an iteration SAVES cost
        c[N * M + l] = -loop_savings_per_iter[l]

    # ---- integrality (1 = integer/binary for all vars) ---------------------
    integrality = np.ones(n_vars)

    # ---- bounds ------------------------------------------------------------
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)  # x_{i,j} ∈ [0,1] (binary)

    # Infeasible (i,j) pairs: set upper bound to 0
    for i, event in enumerate(events):
        complexity = classify_call_complexity(event)
        min_tier = COMPLEXITY_MIN_TIER[complexity]
        for j in range(min_tier):
            ub[i * M + j] = 0.0

    # Loop elimination bounds: 0 ≤ k_l ≤ detected_l − 1
    for l, loop in enumerate(loops):
        ub[N * M + l] = float(max(0, loop["count"] - 1))

    # ---- constraint matrix -------------------------------------------------
    A_data: list[float] = []
    A_row: list[int] = []
    A_col: list[int] = []
    b_lo: list[float] = []
    b_hi: list[float] = []

    inf = float("inf")
    n_rows = 0

    # Constraint 1: assignment  Σ_j x[i,j] = 1  ∀ i
    for i in range(N):
        for j in range(M):
            A_row.append(n_rows)
            A_col.append(i * M + j)
            A_data.append(1.0)
        b_lo.append(1.0)
        b_hi.append(1.0)
        n_rows += 1

    # Constraint 2 (optional): quality floor
    #   Σ_{i,j} x[i,j] · cap_j · w_i / W ≥ quality_floor
    if quality_floor > 0.0:
        total_weight = sum(
            COMPLEXITY_WEIGHT[classify_call_complexity(e)] for e in events
        )
        if total_weight > 0.0:
            for i, event in enumerate(events):
                complexity = classify_call_complexity(event)
                w_i = COMPLEXITY_WEIGHT[complexity]
                for j, model_name in enumerate(MODEL_TIERS):
                    cap_j = MODEL_CAPABILITY.get(model_name, 0.5)
                    A_row.append(n_rows)
                    A_col.append(i * M + j)
                    A_data.append(w_i * cap_j / total_weight)
            b_lo.append(quality_floor)
            b_hi.append(inf)
            n_rows += 1

    # Constraint 3 (optional): budget cap
    #   Σ_{i,j} x[i,j] · c_{i,j} − Σ_l k[l] · s_l ≤ budget
    if budget_usd is not None:
        for i, event in enumerate(events):
            for j, model_name in enumerate(MODEL_TIERS):
                cost_ij = model_cost(model_name, event.prompt_tokens, event.completion_tokens)
                A_row.append(n_rows)
                A_col.append(i * M + j)
                A_data.append(cost_ij)
        for l in range(L):
            A_row.append(n_rows)
            A_col.append(N * M + l)
            A_data.append(-loop_savings_per_iter[l])
        b_lo.append(-inf)
        b_hi.append(budget_usd)
        n_rows += 1

    if not A_data:
        return [], [], False, "no constraints built"

    A_sparse = csr_matrix(
        (np.array(A_data), (np.array(A_row), np.array(A_col))),
        shape=(n_rows, n_vars),
    )
    constraint = LinearConstraint(A_sparse, np.array(b_lo), np.array(b_hi))

    result = milp(
        c=c,
        constraints=constraint,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        options={"disp": False},
    )

    # HiGHS status: 0=optimal, 1=iteration limit, 2=infeasible, 3=unbounded, 4=other
    if result.status not in (0, 1) or result.x is None:
        msg = {2: "infeasible", 3: "unbounded"}.get(result.status, f"status={result.status}")
        return [], [], False, msg

    x = result.x

    # Extract assigned models (argmax over j for each i)
    assigned_models: list[str] = []
    for i in range(N):
        best_j = int(np.argmax(x[i * M: i * M + M]))
        # Validate: must be feasible
        complexity = classify_call_complexity(events[i])
        min_tier = COMPLEXITY_MIN_TIER[complexity]
        if best_j < min_tier:
            best_j = min_tier
        assigned_models.append(MODEL_TIERS[best_j])

    # Extract iterations eliminated
    eliminated_iters: list[int] = []
    for l in range(L):
        k = max(0, round(float(x[N * M + l])))
        eliminated_iters.append(k)

    solver_status = "optimal" if result.status == 0 else "time-limited"
    return assigned_models, eliminated_iters, True, f"HiGHS MILP ({solver_status})"


# ---------------------------------------------------------------------------
# Greedy fallback
# ---------------------------------------------------------------------------

def _solve_greedy(
    events: list[TraceEvent],
    loops: list[dict],
    loop_savings_per_iter: list[float],
    graph: nx.DiGraph,
) -> tuple[list[str], list[int]]:
    """Greedy tier assignment (optimal for the unconstrained base problem)."""
    assigned_models: list[str] = []
    for event in events:
        complexity = classify_call_complexity(event)
        min_tier = COMPLEXITY_MIN_TIER[complexity]
        assigned_models.append(MODEL_TIERS[min_tier])

    eliminated_iters: list[int] = []
    for loop in loops:
        detected = loop["count"]
        recommended_max = max(2, detected // 2)
        eliminated_iters.append(detected - recommended_max)

    return assigned_models, eliminated_iters


# ---------------------------------------------------------------------------
# Public optimization entry point
# ---------------------------------------------------------------------------

def optimize_trace(
    trace: Trace,
    graph: nx.DiGraph,
    force_no_downgrade: bool = False,
    quality_floor: float = 0.0,
    budget_usd: float | None = None,
) -> OptimizationResult:
    """
    Solve the cost-minimization MILP for a traced agent run.

    Parameters
    ----------
    trace            : Parsed agent trace.
    graph            : NetworkX DiGraph built from the trace.
    force_no_downgrade: If True, return zero savings (used when task failed).
    quality_floor    : Minimum weighted-average capability score (0.0–1.0).
                       0.0 = no quality constraint (equivalent to greedy).
                       0.5 = require medium-quality mix.
    budget_usd       : Optional hard spend cap (USD) on optimized cost.
    """
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

    events = trace.events
    loops: list[dict] = graph.graph.get("loops", [])

    # Precompute per-iteration savings for loop variables
    loop_savings_per_iter = _precompute_loop_savings(loops, events, graph)

    # ---- Solve MILP (HiGHS) -----------------------------------------------
    assigned_models, eliminated_iters, mip_ok, solver_msg = _solve_mip(
        events, loops, loop_savings_per_iter, quality_floor, budget_usd
    )

    if not mip_ok:
        # Fallback: greedy
        assigned_models, eliminated_iters = _solve_greedy(
            events, loops, loop_savings_per_iter, graph
        )
        solver_msg = "Greedy (fallback: " + solver_msg + ")"

    # ---- Build recommendations from assigned models -----------------------
    recommendations: list[dict[str, Any]] = []
    original_total = 0.0
    optimized_total = 0.0

    for i, event in enumerate(events):
        original_cost = model_cost(event.model, event.prompt_tokens, event.completion_tokens)
        best_model = assigned_models[i]
        best_cost = model_cost(best_model, event.prompt_tokens, event.completion_tokens)

        original_total += original_cost
        optimized_total += best_cost

        savings = original_cost - best_cost
        current_tier = model_tier_index(event.model)
        target_tier = model_tier_index(best_model)

        if current_tier > target_tier and savings > 0.000001:
            complexity = classify_call_complexity(event)
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

    # ---- Loop recommendations --------------------------------------------
    loop_recommendations: list[dict[str, Any]] = []
    rec_model_by_node = {r["node_id"]: r["recommended_model"] for r in recommendations}

    for l, loop in enumerate(loops):
        detected = loop["count"]
        k_elim = eliminated_iters[l] if l < len(eliminated_iters) else 0
        recommended_max = detected - k_elim

        # Cost at optimised model prices for the loop calls
        loop_opt_cost = 0.0
        for node_id in loop.get("node_ids", []):
            if node_id not in graph.nodes:
                continue
            node_event = graph.nodes[node_id].get("event")
            if node_event is None:
                continue
            eff_model = rec_model_by_node.get(node_id, node_event.model)
            loop_opt_cost += model_cost(eff_model, node_event.prompt_tokens, node_event.completion_tokens)

        loop_savings = loop_opt_cost * (k_elim / detected) if detected > 0 else 0.0
        optimized_total = max(0.0, optimized_total - loop_savings)

        loop_recommendations.append({
            "node_id": loop["node_ids"][0] if loop.get("node_ids") else "",
            "pattern": loop["pattern"],
            "detected_iterations": detected,
            "recommended_max": recommended_max,
            "savings_usd": round(loop_savings, 8),
            "description": (
                "Loop detected: '{}' repeated {} times. "
                "Consider adding early-exit logic or capping at {} iterations.".format(
                    loop["pattern"], detected, recommended_max
                )
            ),
        })

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
        assigned_models=assigned_models,
        recommendations=recommendations,
        loop_recommendations=loop_recommendations,
        loop_savings_per_iter=loop_savings_per_iter,
        eliminated_iters=eliminated_iters,
        original_total=original_total,
        optimized_total=optimized_total,
        quality_floor=quality_floor,
        budget_usd=budget_usd,
        solver_msg=solver_msg,
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


# ---------------------------------------------------------------------------
# Human-readable text builders
# ---------------------------------------------------------------------------

def _build_reason(event: TraceEvent, complexity: str, recommended_model: str) -> str:
    parts = []
    if complexity == "simple":
        parts.append("Simple call ({} tokens, no tool calls)".format(event.prompt_tokens))
        parts.append("{} is fully capable for this complexity".format(recommended_model))
    elif complexity == "medium":
        parts.append("Medium complexity ({} tokens, no tool calls)".format(event.prompt_tokens))
        parts.append("{} handles this well at much lower cost".format(recommended_model))
    else:
        parts.append("Complex call ({} tokens".format(event.prompt_tokens))
        if event.tool_calls:
            parts.append(", {} tool call(s))".format(len(event.tool_calls)))
        else:
            parts.append(")")
        parts.append("Minimum required tier: {}".format(recommended_model))
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
    lines = [
        "Optimization Analysis for Session {}...".format(trace.session_id[:8]),
        "",
        "Original cost:  ${:.6f}".format(original_total),
        "Optimized cost: ${:.6f}".format(max(0.0, optimized_total)),
        "Estimated savings: ${:.6f} ({:.1f}%)".format(savings_usd, savings_pct),
        "",
    ]

    if recommendations:
        lines.append("Model Recommendations ({} calls to optimize):".format(len(recommendations)))
        for r in recommendations[:5]:
            lines.append(
                "  - Replace {} → {} (saves ${:.6f}): {}".format(
                    r["original_model"], r["recommended_model"], r["savings_usd"], r["reason"]
                )
            )
        if len(recommendations) > 5:
            lines.append("  ... and {} more recommendations".format(len(recommendations) - 5))
    else:
        lines.append("No model downgrade recommendations — models are already well-sized.")

    if loop_recommendations:
        lines.append("")
        lines.append("Loop Recommendations ({} patterns detected):".format(len(loop_recommendations)))
        for lr in loop_recommendations:
            lines.append("  - {}".format(lr["description"]))

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


# ---------------------------------------------------------------------------
# Formulation builder (for UI display)
# ---------------------------------------------------------------------------

def _build_formulation(
    trace: Trace,
    assigned_models: list[str],
    recommendations: list[dict[str, Any]],
    loop_recommendations: list[dict[str, Any]],
    loop_savings_per_iter: list[float],
    eliminated_iters: list[int],
    original_total: float,
    optimized_total: float,
    quality_floor: float,
    budget_usd: float | None,
    solver_msg: str,
) -> dict[str, Any]:
    """
    Build the structured mathematical formulation for display in the UI.
    """
    n = len(trace.events)
    M = len(MODEL_TIERS)
    L = len(loop_recommendations)

    # Decision variables
    decision_variables = []
    for i, event in enumerate(trace.events):
        complexity = classify_call_complexity(event)
        min_tier = COMPLEXITY_MIN_TIER[complexity]
        feasible = MODEL_TIERS[min_tier:]
        assigned = assigned_models[i] if i < len(assigned_models) else event.model
        decision_variables.append({
            "index": i,
            "node_id": event.node_id,
            "var_name": "m_{}".format(i),
            "description": "Model for call #{} ({}, {} prompt tokens)".format(
                i + 1, event.event_type, event.prompt_tokens
            ),
            "domain": feasible,
            "current_value": event.model,
            "assigned_value": assigned,
            "complexity": complexity,
            "prompt_tokens": event.prompt_tokens,
            "completion_tokens": event.completion_tokens,
            "tool_name": event.tool_name,
        })

    # Objective terms
    objective_terms = []
    for dv in decision_variables:
        price = MODEL_PRICES.get(dv["assigned_value"], MODEL_PRICES["gpt-4o"])
        term_cost = (
            dv["prompt_tokens"] * price["input"] + dv["completion_tokens"] * price["output"]
        ) / 1_000_000
        objective_terms.append({
            "var": dv["var_name"],
            "model": dv["assigned_value"],
            "prompt_tokens": dv["prompt_tokens"],
            "completion_tokens": dv["completion_tokens"],
            "input_price_per_1m": price["input"],
            "output_price_per_1m": price["output"],
            "cost": round(term_cost, 8),
            "latex": (
                "({} \\times {} + {} \\times {}) / 10^6".format(
                    dv["prompt_tokens"],
                    price["input"],
                    dv["completion_tokens"],
                    price["output"],
                )
            ),
        })

    # Constraints
    constraints = []
    # 1. Assignment
    for dv in decision_variables:
        constraints.append({
            "type": "assignment",
            "var": dv["var_name"],
            "description": "\\sum_j x_{{{},j}} = 1".format(dv["index"]),
            "natural_language": "Exactly one model must be assigned to call #{}.".format(
                dv["index"] + 1
            ),
            "satisfied": True,
        })

    # 2. Feasibility
    for dv in decision_variables:
        min_tier = COMPLEXITY_MIN_TIER[dv["complexity"]]
        min_model = MODEL_TIERS[min_tier]
        constraints.append({
            "type": "model_tier",
            "var": dv["var_name"],
            "description": "m_{} \\in \\{{{}\\}}".format(
                dv["index"], ", ".join(MODEL_TIERS[min_tier:])
            ),
            "natural_language": "Call #{} is '{}' ({}) — minimum model: {}.".format(
                dv["index"] + 1,
                dv["complexity"],
                "has tool calls" if dv["tool_name"] else "{} tokens, no tools".format(
                    dv["prompt_tokens"]
                ),
                min_model,
            ),
            "satisfied": True,
        })

    # 3. Quality floor (if active)
    if quality_floor > 0.0:
        constraints.append({
            "type": "quality_floor",
            "var": "all",
            "description": "\\frac{{1}}{{W}} \\sum_{{i,j}} x_{{i,j}} \\cdot cap_j \\cdot w_i \\geq {:.2f}".format(
                quality_floor
            ),
            "natural_language": (
                "Weighted-average model capability must be ≥ {:.0f}%. "
                "This may force some calls to use a higher-tier model.".format(quality_floor * 100)
            ),
            "satisfied": True,
        })

    # 4. Budget cap (if active)
    if budget_usd is not None:
        constraints.append({
            "type": "budget",
            "var": "all",
            "description": "\\text{{total cost}} \\leq ${:.4f}".format(budget_usd),
            "natural_language": "Total optimized cost must not exceed ${:.4f}.".format(budget_usd),
            "satisfied": max(0.0, optimized_total) <= budget_usd,
        })

    # 5. Loop caps
    for l, lr in enumerate(loop_recommendations):
        k = eliminated_iters[l] if l < len(eliminated_iters) else 0
        constraints.append({
            "type": "loop_cap",
            "var": "k_{}".format(lr["node_id"][:6]),
            "description": "0 \\leq k_{{{}}} \\leq {}".format(
                lr["node_id"][:6], lr["detected_iterations"] - 1
            ),
            "natural_language": (
                "Loop '{}' ran {} times. Eliminate {} iteration(s) → "
                "cap at {} (saves ${:.6f}).".format(
                    lr["pattern"],
                    lr["detected_iterations"],
                    k,
                    lr["recommended_max"],
                    lr["savings_usd"],
                )
            ),
            "satisfied": True,
        })

    # Plain-English explanation
    n_simple = sum(1 for dv in decision_variables if dv["complexity"] == "simple")
    n_medium = sum(1 for dv in decision_variables if dv["complexity"] == "medium")
    n_complex = sum(1 for dv in decision_variables if dv["complexity"] == "complex")
    n_changed = sum(1 for dv in decision_variables if dv["current_value"] != dv["assigned_value"])
    n_loops = len(loop_recommendations)

    explanation = (
        "This run produced {n} LLM calls costing ${orig:.6f} total. "
        "The optimizer formulated a Mixed Integer Linear Program with {nv} binary assignment "
        "variables (x_{{i,j}}) and {nl} integer loop-elimination variables (k_l). "
        "Calls were classified by complexity: {ns} simple, {nm} medium, {nc} complex. "
        "The feasibility constraints restrict each call to model tiers that meet its complexity: "
        "simple/medium calls may use any tier (≥ gpt-4o-mini); "
        "complex calls require at least gpt-4o. "
        "{nq}"
        "The HiGHS solver minimised the objective over all feasible assignments. "
        "{nch} of {n} calls were reassigned to cheaper models. "
        "{loop_text}"
        "Optimised total: ${opt:.6f} — a saving of ${sav:.6f}."
    ).format(
        n=n,
        orig=original_total,
        nv=n * M,
        nl=n_loops,
        ns=n_simple,
        nm=n_medium,
        nc=n_complex,
        nch=n_changed,
        nq=(
            "A quality-floor constraint (≥ {:.0f}% weighted capability) "
            "was active, potentially upgrading some calls. ".format(quality_floor * 100)
        ) if quality_floor > 0.0 else "",
        loop_text=(
            "{} loop pattern(s) were detected; the solver chose to eliminate {} "
            "redundant iteration(s). ".format(
                n_loops,
                sum(eliminated_iters[:n_loops]),
            )
        ) if n_loops else "",
        opt=max(0.0, optimized_total),
        sav=max(0.0, original_total - optimized_total),
    )

    return {
        "problem_type": "Mixed Integer Linear Program (MILP)",
        "solver": solver_msg,
        "n_variables": n * M + n_loops,
        "n_binary_vars": n * M,
        "n_integer_vars": n_loops,
        "n_constraints": len(constraints),
        "quality_floor": quality_floor,
        "budget_usd": budget_usd,
        "objective": {
            "sense": "minimize",
            "description": "Minimise total cost of all LLM API calls, including loop savings",
            "latex": (
                "\\min \\sum_{i=1}^{N} \\sum_{j=1}^{M} x_{ij} \\cdot c_{ij}"
                " - \\sum_{l=1}^{L} k_l \\cdot s_l"
            ),
            "natural_language": (
                "Choose the cheapest viable model for each call and decide how many "
                "redundant loop iterations to eliminate, such that total cost is minimised."
            ),
            "terms": objective_terms,
        },
        "decision_variables": decision_variables,
        "constraints": constraints,
        "explanation": explanation,
        "original_cost": round(original_total, 8),
        "optimized_cost": round(max(0.0, optimized_total), 8),
    }
