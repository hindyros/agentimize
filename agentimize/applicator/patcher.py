"""
Generates optimized agent configuration patches based on OptimizationResult.

Produces environment variable patches, config file snippets, and
code-level recommendations that can be applied to agent implementations.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from agentimize.optimizer.solver import OptimizationResult


class ConfigPatch(dict):
    """A dict subclass representing a configuration patch."""

    def to_env_vars(self) -> str:
        """Return the patch as shell export statements."""
        lines = ["# Agentimize Optimized Configuration", "# Apply these environment variables to your agent"]
        for key, value in self.items():
            if isinstance(value, str):
                lines.append(f'export {key}="{value}"')
            else:
                lines.append(f"export {key}={json.dumps(value)}")
        return "\n".join(lines)

    def to_dotenv(self) -> str:
        """Return the patch as .env file format."""
        lines = ["# Agentimize Optimized Configuration"]
        for key, value in self.items():
            if isinstance(value, str):
                lines.append(f'{key}="{value}"')
            else:
                lines.append(f"{key}={json.dumps(value)}")
        return "\n".join(lines)

    def to_json(self) -> str:
        """Return the patch as pretty-printed JSON."""
        return json.dumps(dict(self), indent=2)


def generate_patch(result: OptimizationResult) -> ConfigPatch:
    """
    Generate a configuration patch from an OptimizationResult.

    Returns a ConfigPatch containing:
    - Model substitution rules
    - Loop iteration limits
    - Environment variable suggestions
    - Code snippets
    """
    patch = ConfigPatch()

    # Count how many times each original model appears and what it maps to
    model_mappings: dict[str, Counter] = {}
    for rec in result.recommendations:
        orig = rec["original_model"]
        recommended = rec["recommended_model"]
        if orig not in model_mappings:
            model_mappings[orig] = Counter()
        model_mappings[orig][recommended] += 1

    # Determine the consensus mapping for each original model
    primary_model_map: dict[str, str] = {}
    for orig, counter in model_mappings.items():
        # Pick the most-recommended replacement
        primary_model_map[orig] = counter.most_common(1)[0][0]

    patch["MODEL_SUBSTITUTIONS"] = primary_model_map
    patch["ESTIMATED_SAVINGS_USD"] = result.savings_usd
    patch["ESTIMATED_SAVINGS_PCT"] = result.savings_pct

    # Generate loop iteration limits
    loop_limits: dict[str, int] = {}
    for loop_rec in result.loop_recommendations:
        pattern = loop_rec.get("pattern", "")
        loop_limits[pattern] = loop_rec.get("recommended_max", 3)

    if loop_limits:
        patch["LOOP_ITERATION_LIMITS"] = loop_limits

    # Generate primary model override (if most calls use one model)
    all_originals = [r["original_model"] for r in result.recommendations]
    if all_originals:
        most_common_original = Counter(all_originals).most_common(1)[0][0]
        recommended = primary_model_map.get(most_common_original)
        if recommended:
            patch["AGENTIMIZE_PRIMARY_MODEL"] = recommended

    return patch


def generate_code_snippet(result: OptimizationResult) -> str:
    """
    Generate a Python code snippet that applies the optimizations.

    Returns Python code showing how to configure the OpenAI client
    with the optimized settings.
    """
    if not result.recommendations:
        return "# No model changes recommended — your agent is already cost-efficient!\n"

    # Build model substitution map
    model_map: dict[str, str] = {}
    for rec in result.recommendations:
        model_map[rec["original_model"]] = rec["recommended_model"]

    lines = [
        "# Agentimize Optimized Configuration",
        "# Generated based on your agent's trace analysis",
        f"# Estimated savings: ${result.savings_usd:.6f} ({result.savings_pct:.1f}%)",
        "",
        "from openai import OpenAI",
        "",
        "# Model substitution map (original -> optimized)",
        f"MODEL_MAP = {json.dumps(model_map, indent=4)}",
        "",
        "def get_optimized_model(model: str) -> str:",
        '    """Return the cost-optimized model for a given model name."""',
        "    return MODEL_MAP.get(model, model)",
        "",
        "# Usage example:",
        "client = OpenAI(",
        '    base_url="http://localhost:7453/v1",  # Keep proxy for continued monitoring',
        ")",
        "",
        "# Replace model calls like:",
        "# response = client.chat.completions.create(",
        '#     model="gpt-4o",  # Original',
        "#     ...",
        "# )",
        "# With:",
        "# response = client.chat.completions.create(",
        '#     model=get_optimized_model("gpt-4o"),  # Optimized',
        "#     ...",
        "# )",
    ]

    if result.loop_recommendations:
        lines.extend([
            "",
            "# Loop iteration recommendations:",
        ])
        for lr in result.loop_recommendations:
            lines.append(
                f"# Pattern '{lr['pattern']}': cap at {lr['recommended_max']} iterations "
                f"(detected {lr['detected_iterations']})"
            )

    return "\n".join(lines)


def generate_full_report(
    result: OptimizationResult,
    session_id: str,
) -> dict[str, Any]:
    """
    Generate a comprehensive optimization report dict.
    Suitable for serialization and display in the dashboard.
    """
    patch = generate_patch(result)
    code_snippet = generate_code_snippet(result)

    return {
        "session_id": session_id,
        "summary": result.summary,
        "savings": {
            "original_cost_usd": result.original_cost_usd,
            "optimized_cost_usd": result.optimized_cost_usd,
            "savings_usd": result.savings_usd,
            "savings_pct": result.savings_pct,
        },
        "model_recommendations": result.recommendations,
        "loop_recommendations": result.loop_recommendations,
        "config_patch": {
            "env_vars": patch.to_env_vars(),
            "dotenv": patch.to_dotenv(),
            "json": patch.to_json(),
        },
        "code_snippet": code_snippet,
    }
