"""
aiohttp-based HTTP proxy server that intercepts OpenAI-compatible API calls,
records trace events, and forwards requests to the upstream provider.
"""

import asyncio
import json
import os
import time
import uuid
from typing import Any

import aiohttp
from aiohttp import web
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.table import Table

load_dotenv()

console = Console()

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

DEFAULT_PRICE = {"input": 2.50, "output": 10.00}  # fallback to gpt-4o pricing


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate the cost in USD for a given model and token counts."""
    # Try exact match first, then prefix match
    price = MODEL_PRICES.get(model)
    if price is None:
        for key in MODEL_PRICES:
            if model.startswith(key) or key.startswith(model.split("-")[0]):
                price = MODEL_PRICES[key]
                break
    if price is None:
        price = DEFAULT_PRICE

    cost = (prompt_tokens * price["input"] + completion_tokens * price["output"]) / 1_000_000
    return round(cost, 8)


class AgentimizeProxy:
    """HTTP proxy that intercepts and records LLM API calls."""

    def __init__(
        self,
        port: int = 7453,
        upstream_base_url: str = "https://api.openai.com",
        trace_file: str = "agentimize_trace.jsonl",
    ):
        self.port = port
        self.upstream_base_url = upstream_base_url.rstrip("/")
        self.trace_file = trace_file
        self.session_id = str(uuid.uuid4())
        self.app = web.Application()
        self.app.router.add_route("*", "/{path_info:.*}", self.handle_request)
        self._session: aiohttp.ClientSession | None = None
        self._event_count = 0

        console.print(f"[bold green]Agentimize Proxy[/bold green] session_id=[cyan]{self.session_id}[/cyan]")
        console.print(f"Upstream: [yellow]{self.upstream_base_url}[/yellow]")
        console.print(f"Trace file: [yellow]{self.trace_file}[/yellow]")

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(ssl=True)
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    def _build_upstream_url(self, request: web.Request) -> str:
        """Build the full upstream URL from the incoming request."""
        path = request.path
        query = request.query_string
        url = f"{self.upstream_base_url}{path}"
        if query:
            url = f"{url}?{query}"
        return url

    def _sanitize_headers(self, headers: dict) -> dict:
        """Remove hop-by-hop headers that shouldn't be forwarded."""
        hop_by_hop = {
            "connection", "keep-alive", "proxy-authenticate",
            "proxy-authorization", "te", "trailers", "transfer-encoding",
            "upgrade", "host",
        }
        return {k: v for k, v in headers.items() if k.lower() not in hop_by_hop}

    def _extract_tool_calls(self, response_data: dict) -> list[dict]:
        """Extract tool calls from the LLM response if present."""
        tool_calls = []
        try:
            choices = response_data.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                raw_tool_calls = message.get("tool_calls", [])
                for tc in raw_tool_calls:
                    tool_calls.append({
                        "id": tc.get("id", ""),
                        "type": tc.get("type", "function"),
                        "function": tc.get("function", {}),
                    })
        except (KeyError, IndexError, TypeError):
            pass
        return tool_calls

    async def _record_trace_event(
        self,
        request_data: dict,
        response_data: dict,
        latency_ms: float,
        path: str,
    ) -> None:
        """Parse and record a trace event to the JSONL file."""
        model = request_data.get("model", "unknown")
        usage = response_data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        cost_usd = calculate_cost(model, prompt_tokens, completion_tokens)

        tool_calls = self._extract_tool_calls(response_data)
        tool_name: str | None = None
        if tool_calls:
            # Use first tool call name as representative
            fn = tool_calls[0].get("function", {})
            tool_name = fn.get("name")

        # Determine event type
        if tool_calls:
            event_type = "tool_call"
        else:
            event_type = "llm_call"

        event = {
            "session_id": self.session_id,
            "timestamp": time.time(),
            "event_type": event_type,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost_usd,
            "latency_ms": round(latency_ms, 2),
            "tool_name": tool_name,
            "tool_calls": tool_calls,
            "success": None,
            "path": path,
            "raw_request": {
                k: v for k, v in request_data.items()
                if k not in ("messages",)  # exclude full messages to keep trace compact
            },
            "messages_summary": {
                "count": len(request_data.get("messages", [])),
                "last_role": (request_data.get("messages") or [{}])[-1].get("role", ""),
            },
            "raw_response": {
                k: v for k, v in response_data.items()
                if k not in ("choices",)
            },
            "completion_text": self._extract_completion_text(response_data),
        }

        # Write to JSONL file
        with open(self.trace_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

        self._event_count += 1
        console.print(
            f"[dim][[bold]#{self._event_count}[/bold]][/dim] "
            f"[cyan]{model}[/cyan] "
            f"[yellow]{prompt_tokens}+{completion_tokens} tok[/yellow] "
            f"[green]${cost_usd:.6f}[/green] "
            f"[dim]{latency_ms:.0f}ms[/dim]"
            + (f" [magenta]tool={tool_name}[/magenta]" if tool_name else "")
        )

    def _extract_completion_text(self, response_data: dict) -> str:
        """Extract the completion text from the response."""
        try:
            choices = response_data.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                return message.get("content", "") or ""
        except (KeyError, IndexError, TypeError):
            pass
        return ""

    async def handle_request(self, request: web.Request) -> web.Response:
        """Handle an incoming HTTP request by proxying it upstream."""
        path = request.path
        upstream_url = self._build_upstream_url(request)

        # Read request body
        body = await request.read()

        # Parse JSON body for LLM calls
        request_data: dict[str, Any] = {}
        is_llm_call = path.startswith("/v1/") and body
        if is_llm_call:
            try:
                request_data = json.loads(body)
            except (json.JSONDecodeError, ValueError):
                is_llm_call = False

        # Build headers to forward
        forward_headers = self._sanitize_headers(dict(request.headers))
        # Ensure host matches upstream
        from urllib.parse import urlparse
        parsed = urlparse(self.upstream_base_url)
        forward_headers["host"] = parsed.netloc

        start_time = time.monotonic()

        try:
            session = await self._get_session()
            async with session.request(
                method=request.method,
                url=upstream_url,
                headers=forward_headers,
                data=body,
                allow_redirects=False,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as upstream_response:
                latency_ms = (time.monotonic() - start_time) * 1000
                response_body = await upstream_response.read()

                # Record trace for LLM calls only
                if is_llm_call and upstream_response.status == 200:
                    try:
                        response_data = json.loads(response_body)
                        # Only trace completion endpoints with usage data
                        if "usage" in response_data or "choices" in response_data:
                            await self._record_trace_event(
                                request_data=request_data,
                                response_data=response_data,
                                latency_ms=latency_ms,
                                path=path,
                            )
                    except (json.JSONDecodeError, ValueError):
                        pass
                elif upstream_response.status != 200 and is_llm_call:
                    console.print(
                        f"[red]Upstream error {upstream_response.status}[/red] for {path}"
                    )

                # Build response to client
                response_headers = self._sanitize_headers(dict(upstream_response.headers))
                return web.Response(
                    status=upstream_response.status,
                    headers=response_headers,
                    body=response_body,
                )

        except aiohttp.ClientConnectorError as e:
            console.print(f"[red]Connection error:[/red] {e}")
            return web.Response(
                status=502,
                text=json.dumps({"error": {"message": f"Proxy upstream connection failed: {e}", "type": "proxy_error"}}),
                content_type="application/json",
            )
        except asyncio.TimeoutError:
            console.print("[red]Upstream request timed out[/red]")
            return web.Response(
                status=504,
                text=json.dumps({"error": {"message": "Proxy upstream request timed out", "type": "proxy_error"}}),
                content_type="application/json",
            )
        except Exception as e:
            console.print(f"[red]Proxy error:[/red] {e}")
            return web.Response(
                status=500,
                text=json.dumps({"error": {"message": f"Proxy internal error: {e}", "type": "proxy_error"}}),
                content_type="application/json",
            )

    async def cleanup(self, app: web.Application) -> None:
        """Cleanup resources on shutdown."""
        if self._session and not self._session.closed:
            await self._session.close()

    def run(self) -> None:
        """Start the proxy server."""
        self.app.on_cleanup.append(self.cleanup)
        console.print(f"\n[bold green]Starting Agentimize Proxy on port {self.port}[/bold green]")
        console.print("Configure your agent with:")
        console.print(f"  [cyan]OPENAI_BASE_URL=http://localhost:{self.port}/v1[/cyan]")
        console.print("  [cyan]OPENAI_API_KEY=<your-key>[/cyan]")
        console.print("\n[dim]Press Ctrl+C to stop[/dim]\n")
        web.run_app(self.app, host="0.0.0.0", port=self.port, print=None)


def start_proxy(
    port: int | None = None,
    upstream: str | None = None,
    trace_file: str | None = None,
) -> None:
    """Entry point to start the proxy server."""
    port = port or int(os.getenv("PROXY_PORT", "7453"))
    upstream = upstream or os.getenv("UPSTREAM_BASE_URL", "https://api.openai.com")
    trace_file = trace_file or os.getenv("TRACE_FILE", "agentimize_trace.jsonl")

    proxy = AgentimizeProxy(port=port, upstream_base_url=upstream, trace_file=trace_file)
    proxy.run()


if __name__ == "__main__":
    start_proxy()
