"""
NYC Apartment Search Agent — Agentimize Test Agent

A realistic multi-step agent that simulates finding the best 20 apartment
rentals in NYC for a $3000/month budget. Uses OpenAI function calling with
simulated tool implementations.

Features intentional inefficiencies for Agentimize to detect and optimize:
- Uses gpt-4o for simple classification tasks (should use gpt-4o-mini)
- Makes redundant tool calls for the same data
- Has a loop pattern that runs more iterations than necessary

Usage:
  python -m test_agent.nyc_apartment_agent [--proxy http://localhost:7453]
  python -m test_agent.nyc_apartment_agent --no-proxy  # Direct API access
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ─── Tool Definitions ────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_listings",
            "description": "Search rental listings on a given site for apartments in NYC.",
            "parameters": {
                "type": "object",
                "properties": {
                    "site": {
                        "type": "string",
                        "description": "Listing site to search (e.g., 'streeteasy', 'zillow', 'apartments_com', 'craigslist')",
                        "enum": ["streeteasy", "zillow", "apartments_com", "craigslist", "renthop"],
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query (neighborhood, apartment type, etc.)",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (1-50)",
                        "minimum": 1,
                        "maximum": 50,
                    },
                },
                "required": ["site", "query", "max_results"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_listings",
            "description": "Filter a list of rental listings by specific criteria.",
            "parameters": {
                "type": "object",
                "properties": {
                    "listing_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of listing IDs to filter",
                    },
                    "criteria": {
                        "type": "object",
                        "description": "Filter criteria: max_price, min_bedrooms, neighborhoods, max_commute_min, pets_allowed, laundry",
                        "properties": {
                            "max_price": {"type": "number"},
                            "min_bedrooms": {"type": "integer"},
                            "neighborhoods": {"type": "array", "items": {"type": "string"}},
                            "pets_allowed": {"type": "boolean"},
                            "laundry": {"type": "string", "enum": ["in_unit", "in_building", "any"]},
                        },
                    },
                },
                "required": ["listing_ids", "criteria"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_listing_details",
            "description": "Get detailed information about a specific rental listing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "listing_id": {
                        "type": "string",
                        "description": "The unique identifier of the listing",
                    },
                },
                "required": ["listing_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank_listings",
            "description": "Rank a list of rental listings by value for money and quality.",
            "parameters": {
                "type": "object",
                "properties": {
                    "listing_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Listing IDs to rank",
                    },
                    "preferences": {
                        "type": "object",
                        "description": "Ranking preferences",
                        "properties": {
                            "budget": {"type": "number"},
                            "priority": {
                                "type": "string",
                                "enum": ["price", "size", "location", "amenities", "balanced"],
                            },
                            "must_have": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    },
                },
                "required": ["listing_ids", "preferences"],
            },
        },
    },
]

# ─── Simulated Tool Implementations ──────────────────────────────────────────

NEIGHBORHOODS = [
    "Williamsburg", "Astoria", "Long Island City", "Bushwick", "Crown Heights",
    "Park Slope", "Greenpoint", "Ridgewood", "Jackson Heights", "Sunnyside",
    "Upper East Side", "Upper West Side", "Harlem", "Chelsea", "Hell's Kitchen",
    "Prospect Heights", "Fort Greene", "Bed-Stuy", "Flatbush", "Inwood",
    "Washington Heights", "Morningside Heights", "East Village", "Flatbush",
]

LISTING_POOL: list[dict[str, Any]] = []


def _generate_listing_pool() -> None:
    """Generate a pool of realistic NYC rental listings."""
    global LISTING_POOL
    if LISTING_POOL:
        return

    random.seed(42)
    listing_count = 120

    for i in range(listing_count):
        neighborhood = random.choice(NEIGHBORHOODS)
        bedrooms = random.choices([0, 1, 2, 3], weights=[15, 40, 35, 10])[0]
        base_price = {0: 1600, 1: 2200, 2: 2800, 3: 3800}[bedrooms]
        price = int(base_price + random.gauss(0, 300))
        price = max(1200, min(4500, price))

        sqft = (bedrooms + 1) * random.randint(180, 280)
        score = random.uniform(3.0, 5.0)
        floor = random.randint(1, 20)
        has_elevator = floor > 3 or random.random() > 0.4
        has_doorman = random.random() > 0.7
        has_gym = random.random() > 0.6
        has_laundry = random.choices(["in_unit", "in_building", "none"], weights=[25, 60, 15])[0]
        pets_allowed = random.random() > 0.4
        has_outdoor_space = random.random() > 0.6
        subway_lines = random.sample(["A", "C", "E", "B", "D", "F", "M", "N", "Q", "R", "1", "2", "3", "4", "5", "6", "L", "J", "Z", "G"], k=random.randint(1, 3))
        walk_to_subway = random.randint(2, 15)
        year_built = random.randint(1920, 2022)
        available_date = f"2024-0{random.randint(1,9)}-{random.randint(1,28):02d}"

        LISTING_POOL.append({
            "id": f"NYC-{i+1:04d}",
            "address": f"{random.randint(100, 999)} {random.choice(['Broadway', 'Amsterdam Ave', 'Columbus Ave', 'Lexington Ave', 'Park Ave', 'Madison Ave', '5th Ave', 'Riverside Dr', 'West End Ave', 'Central Park West'])}",
            "neighborhood": neighborhood,
            "borough": random.choice(["Manhattan", "Brooklyn", "Queens"]) if neighborhood in ["Williamsburg", "Astoria", "Long Island City", "Bushwick", "Crown Heights", "Park Slope", "Greenpoint", "Ridgewood", "Jackson Heights", "Sunnyside", "Prospect Heights", "Fort Greene", "Bed-Stuy", "Flatbush"] else "Manhattan",
            "bedrooms": bedrooms,
            "bathrooms": max(1, bedrooms),
            "sqft": sqft,
            "price": price,
            "price_per_sqft": round(price / sqft, 2),
            "floor": floor,
            "total_floors": max(floor, random.randint(4, 25)),
            "elevator": has_elevator,
            "doorman": has_doorman,
            "gym": has_gym,
            "laundry": has_laundry,
            "pets_allowed": pets_allowed,
            "outdoor_space": has_outdoor_space,
            "subway_lines": subway_lines,
            "walk_to_subway_min": walk_to_subway,
            "year_built": year_built,
            "available_date": available_date,
            "rating": round(score, 1),
            "review_count": random.randint(0, 150),
            "broker_fee": random.choice([0, price * 0.5, price]),
            "utilities_included": random.choice(["none", "heat", "heat_hot_water", "all"]),
            "description": f"Charming {bedrooms}BR in {neighborhood}. {sqft}sqft, {floor}th floor. "
                           f"{'Elevator building. ' if has_elevator else ''}"
                           f"{'Doorman. ' if has_doorman else ''}"
                           f"{'Gym in building. ' if has_gym else ''}"
                           f"{'In-unit laundry. ' if has_laundry == 'in_unit' else 'Laundry in building. ' if has_laundry == 'in_building' else ''}"
                           f"{'Pets welcome. ' if pets_allowed else ''}"
                           f"Near {'/'.join(subway_lines)} train, {walk_to_subway} min walk.",
            "site": random.choice(["streeteasy", "zillow", "apartments_com", "craigslist", "renthop"]),
        })


def tool_search_listings(site: str, query: str, max_results: int) -> dict[str, Any]:
    """Simulate searching for listings on a given site."""
    _generate_listing_pool()

    # Filter by site
    site_listings = [l for l in LISTING_POOL if l["site"] == site]

    # Simple keyword matching on neighborhood and description
    query_lower = query.lower()
    scored: list[tuple[float, dict]] = []
    for listing in site_listings:
        score = 0.0
        if any(word in listing["neighborhood"].lower() for word in query_lower.split()):
            score += 2.0
        if any(word in listing["description"].lower() for word in query_lower.split()):
            score += 1.0
        # Add some randomness for realism
        score += random.uniform(0, 0.5)
        scored.append((score, listing))

    # Sort by score, return top N
    scored.sort(key=lambda x: -x[0])
    results = [l for _, l in scored[:max_results]]

    return {
        "site": site,
        "query": query,
        "total_found": len(site_listings),
        "returned": len(results),
        "listings": [
            {
                "id": l["id"],
                "address": l["address"],
                "neighborhood": l["neighborhood"],
                "bedrooms": l["bedrooms"],
                "price": l["price"],
                "sqft": l["sqft"],
                "available_date": l["available_date"],
                "rating": l["rating"],
            }
            for l in results
        ],
    }


def tool_filter_listings(listing_ids: list[str], criteria: dict) -> dict[str, Any]:
    """Filter listings by criteria."""
    _generate_listing_pool()

    listing_map = {l["id"]: l for l in LISTING_POOL}
    listings = [listing_map[lid] for lid in listing_ids if lid in listing_map]

    filtered = []
    for listing in listings:
        if "max_price" in criteria and listing["price"] > criteria["max_price"]:
            continue
        if "min_bedrooms" in criteria and listing["bedrooms"] < criteria["min_bedrooms"]:
            continue
        if "neighborhoods" in criteria and criteria["neighborhoods"]:
            if listing["neighborhood"] not in criteria["neighborhoods"]:
                continue
        if "pets_allowed" in criteria and criteria["pets_allowed"] and not listing["pets_allowed"]:
            continue
        if "laundry" in criteria and criteria["laundry"] != "any":
            if criteria["laundry"] == "in_unit" and listing["laundry"] != "in_unit":
                continue
            if criteria["laundry"] == "in_building" and listing["laundry"] == "none":
                continue
        filtered.append(listing)

    return {
        "original_count": len(listings),
        "filtered_count": len(filtered),
        "criteria_applied": criteria,
        "listing_ids": [l["id"] for l in filtered],
        "listings": [
            {
                "id": l["id"],
                "address": l["address"],
                "neighborhood": l["neighborhood"],
                "bedrooms": l["bedrooms"],
                "price": l["price"],
                "pets_allowed": l["pets_allowed"],
                "laundry": l["laundry"],
            }
            for l in filtered[:20]
        ],
    }


def tool_get_listing_details(listing_id: str) -> dict[str, Any]:
    """Get full details for a listing."""
    _generate_listing_pool()
    listing_map = {l["id"]: l for l in LISTING_POOL}

    if listing_id not in listing_map:
        return {"error": f"Listing {listing_id} not found", "listing_id": listing_id}

    return listing_map[listing_id]


def tool_rank_listings(listing_ids: list[str], preferences: dict) -> dict[str, Any]:
    """Rank listings by value."""
    _generate_listing_pool()
    listing_map = {l["id"]: l for l in LISTING_POOL}

    budget = preferences.get("budget", 3000)
    priority = preferences.get("priority", "balanced")

    scored: list[tuple[float, dict]] = []
    for lid in listing_ids:
        if lid not in listing_map:
            continue
        listing = listing_map[lid]

        # Score components
        price_score = max(0, (budget - listing["price"]) / budget * 40)
        size_score = min(listing["sqft"] / 50, 30)
        location_score = (10 - listing["walk_to_subway_min"]) * 2
        amenity_score = (
            (5 if listing["elevator"] else 0) +
            (5 if listing["doorman"] else 0) +
            (5 if listing["gym"] else 0) +
            (8 if listing["laundry"] == "in_unit" else 4 if listing["laundry"] == "in_building" else 0) +
            (3 if listing["outdoor_space"] else 0)
        )
        rating_score = listing["rating"] * 4

        if priority == "price":
            total = price_score * 2 + size_score + location_score * 0.5 + amenity_score * 0.5 + rating_score
        elif priority == "size":
            total = price_score + size_score * 2 + location_score + amenity_score + rating_score
        elif priority == "location":
            total = price_score + size_score * 0.5 + location_score * 3 + amenity_score + rating_score
        elif priority == "amenities":
            total = price_score + size_score + location_score + amenity_score * 2 + rating_score
        else:  # balanced
            total = price_score + size_score + location_score + amenity_score + rating_score

        # Bonus for must-haves
        must_have = preferences.get("must_have", [])
        for feature in must_have:
            if feature == "pets" and listing["pets_allowed"]:
                total += 10
            if feature == "laundry" and listing["laundry"] != "none":
                total += 8
            if feature == "gym" and listing["gym"]:
                total += 5

        scored.append((total, listing))

    scored.sort(key=lambda x: -x[0])

    return {
        "ranked_count": len(scored),
        "preferences": preferences,
        "rankings": [
            {
                "rank": i + 1,
                "listing_id": l["id"],
                "address": l["address"],
                "neighborhood": l["neighborhood"],
                "bedrooms": l["bedrooms"],
                "price": l["price"],
                "sqft": l["sqft"],
                "score": round(score, 1),
                "key_features": [
                    f for f, v in [
                        ("In-unit laundry", l["laundry"] == "in_unit"),
                        ("Doorman", l["doorman"]),
                        ("Elevator", l["elevator"]),
                        ("Gym", l["gym"]),
                        ("Outdoor space", l["outdoor_space"]),
                        ("Pets allowed", l["pets_allowed"]),
                    ] if v
                ],
            }
            for i, (score, l) in enumerate(scored[:20])
        ],
    }


# ─── Tool Dispatcher ──────────────────────────────────────────────────────────

def execute_tool(tool_name: str, tool_args: dict[str, Any]) -> str:
    """Execute a tool and return its result as a JSON string."""
    if tool_name == "search_listings":
        result = tool_search_listings(**tool_args)
    elif tool_name == "filter_listings":
        result = tool_filter_listings(**tool_args)
    elif tool_name == "get_listing_details":
        result = tool_get_listing_details(**tool_args)
    elif tool_name == "rank_listings":
        result = tool_rank_listings(**tool_args)
    else:
        result = {"error": f"Unknown tool: {tool_name}"}

    return json.dumps(result, indent=2)


# ─── Agent ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert NYC real estate agent helping a client find the best apartment rentals.

Your task: Find the top 20 best flat rental options in NYC for a budget of $3,000/month.

Use the available tools to:
1. Search multiple listing sites for available apartments
2. Filter by budget and requirements
3. Get details on promising listings
4. Rank them by overall value

Target criteria:
- Maximum rent: $3,000/month
- Prefer: good transit access, laundry, modern building
- Nice to have: pets allowed, gym, doorman, outdoor space

Work systematically: search broadly first, then filter and refine, then rank.
Present the final top 20 with key details for each.

Be thorough but efficient. When you have enough data to produce a high-quality ranking, present the results."""


def run_agent(
    proxy_url: str | None = "http://localhost:7453",
    max_iterations: int = 20,
    verbose: bool = True,
) -> str:
    """
    Run the NYC apartment search agent.

    Args:
        proxy_url: URL of the Agentimize proxy (None for direct API access)
        max_iterations: Maximum number of agent loop iterations
        verbose: Print progress to stdout

    Returns:
        The agent's final summary response
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Configure client — use proxy if specified
    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if proxy_url:
        client_kwargs["base_url"] = f"{proxy_url}/v1"
        if verbose:
            print(f"\n[Agent] Using proxy: {proxy_url}")
    else:
        if verbose:
            print("\n[Agent] Direct API access (no proxy)")

    client = OpenAI(**client_kwargs)

    if verbose:
        print("[Agent] Starting NYC Apartment Search Agent")
        print("[Agent] Task: Find best 20 flats in NYC under $3000/month\n")

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Please find me the best 20 apartment rental options in NYC "
                "with a budget of $3,000 per month. I need a comprehensive list "
                "with details on each option."
            ),
        },
    ]

    iteration = 0
    all_collected_ids: list[str] = []
    final_response = ""

    # ── INTENTIONAL INEFFICIENCY 1: Use gpt-4o for a simple classification task ──
    # This is something Agentimize should detect and suggest using gpt-4o-mini
    if verbose:
        print("[Agent] Step 0: Classifying the task complexity (gpt-4o — intentional inefficiency)")

    classification = client.chat.completions.create(
        model="gpt-4o",  # Could use gpt-4o-mini for this simple task
        messages=[
            {"role": "system", "content": "Classify the complexity of this search task as 'simple', 'medium', or 'complex'. Reply with just the word."},
            {"role": "user", "content": "Find best 20 flats in NYC under $3000/month using multiple listing sites"},
        ],
        max_tokens=10,
        temperature=0,
    )
    task_complexity = classification.choices[0].message.content.strip()
    if verbose:
        print(f"[Agent] Task classified as: {task_complexity}")

    # ── INTENTIONAL INEFFICIENCY 2: Redundant search on same site ──
    # Search StreetEasy twice with slightly different queries
    if verbose:
        print("[Agent] Step 1a: Searching StreetEasy (first pass)")

    while iteration < max_iterations:
        iteration += 1
        if verbose:
            print(f"\n[Agent] Iteration {iteration}/{max_iterations}")

        # Alternate between gpt-4o and gpt-4o-mini
        # Using gpt-4o for all calls — intentional inefficiency for some steps
        model_to_use = "gpt-4o" if iteration <= 3 else "gpt-4o-mini"

        try:
            response = client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=4000,
                temperature=0.1,
            )
        except Exception as e:
            if verbose:
                print(f"[Agent] API error: {e}")
            break

        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        if verbose:
            print(f"[Agent]   model={model_to_use}, finish_reason={finish_reason}")
            if message.content:
                preview = message.content[:100].replace('\n', ' ')
                print(f"[Agent]   content preview: {preview}...")

        # Add assistant message to conversation
        assistant_msg: dict[str, Any] = {"role": "assistant"}
        if message.content:
            assistant_msg["content"] = message.content
        else:
            assistant_msg["content"] = None

        if message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        messages.append(assistant_msg)

        # If done, capture final response
        if finish_reason == "stop":
            final_response = message.content or ""
            if verbose:
                print("\n[Agent] Agent completed task!")
            break

        # Process tool calls
        if not message.tool_calls:
            if verbose:
                print("[Agent] No tool calls and not stop — ending")
            final_response = message.content or ""
            break

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            if verbose:
                print(f"[Agent]   Tool: {tool_name}({', '.join(f'{k}={repr(v)[:30]}' for k,v in tool_args.items())})")

            # Execute the tool
            tool_result = execute_tool(tool_name, tool_args)

            # Collect listing IDs for tracking
            try:
                result_data = json.loads(tool_result)
                if "listings" in result_data:
                    for listing in result_data.get("listings", []):
                        if "id" in listing and listing["id"] not in all_collected_ids:
                            all_collected_ids.append(listing["id"])
                elif "listing_ids" in result_data:
                    for lid in result_data.get("listing_ids", []):
                        if lid not in all_collected_ids:
                            all_collected_ids.append(lid)
                elif "rankings" in result_data:
                    for r in result_data.get("rankings", []):
                        if "listing_id" in r and r["listing_id"] not in all_collected_ids:
                            all_collected_ids.append(r["listing_id"])
            except (json.JSONDecodeError, KeyError):
                pass

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result,
            })

        # ── INTENTIONAL INEFFICIENCY 3: Extra gpt-4o classification after each tool batch ──
        # This creates a loop pattern: every iteration does the same "check progress" call
        if iteration % 2 == 0 and iteration < 8:
            if verbose:
                print("[Agent]   Extra: Checking search progress (gpt-4o, intentional loop pattern)")

            progress_check = client.chat.completions.create(
                model="gpt-4o",  # Intentionally using expensive model for simple check
                messages=[
                    {"role": "system", "content": "You are a progress tracker. Reply with just a brief status."},
                    {"role": "user", "content": f"We have collected {len(all_collected_ids)} listings so far. Is this enough to find top 20? Reply: 'yes' or 'need more'."},
                ],
                max_tokens=10,
                temperature=0,
            )
            progress_status = progress_check.choices[0].message.content.strip()
            if verbose:
                print(f"[Agent]   Progress check: {progress_status}")

    if not final_response:
        # Force a final summary if we ran out of iterations
        if verbose:
            print("\n[Agent] Generating final summary...")

        summary_messages = messages.copy()
        summary_messages.append({
            "role": "user",
            "content": (
                f"Based on all the listings we've collected ({len(all_collected_ids)} total), "
                "please provide your final ranking of the top 20 best NYC apartment options "
                "for a $3000/month budget. Format as a numbered list with key details."
            ),
        })

        final_call = client.chat.completions.create(
            model="gpt-4o-mini",  # Appropriate model for summarization
            messages=summary_messages,
            max_tokens=3000,
            temperature=0.2,
        )
        final_response = final_call.choices[0].message.content or ""

    if verbose:
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)
        print(final_response)
        print("="*70)
        print(f"\n[Agent] Total listings explored: {len(all_collected_ids)}")
        print("[Agent] Done!")

    return final_response


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NYC Apartment Search Agent — Agentimize Test Agent"
    )
    parser.add_argument(
        "--proxy",
        default="http://localhost:7453",
        help="Agentimize proxy URL (default: http://localhost:7453)",
    )
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="Bypass the proxy and call OpenAI directly",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=15,
        help="Maximum agent loop iterations (default: 15)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    args = parser.parse_args()

    proxy_url = None if args.no_proxy else args.proxy

    try:
        result = run_agent(
            proxy_url=proxy_url,
            max_iterations=args.max_iterations,
            verbose=not args.quiet,
        )
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n[Agent] Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[Agent] Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
