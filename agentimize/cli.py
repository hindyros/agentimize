"""
Agentimize CLI entry point.

Commands:
  proxy start     Start the local HTTP proxy
  proxy stop      Stop a running proxy (PID-based)
  analyze         Analyze a trace file
  judge           Run LLM judge on a session
  dashboard       Start the web dashboard
  report          Generate a full optimization report
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

load_dotenv()

app = typer.Typer(
    name="agentimize",
    help="AI Agent Cost Optimizer — intercept, analyze, and optimize LLM API calls.",
    no_args_is_help=True,
)
proxy_app = typer.Typer(help="Proxy server commands.")
app.add_typer(proxy_app, name="proxy")

console = Console()

PIDFILE = Path("/tmp/agentimize_proxy.pid")


# ─── proxy commands ──────────────────────────────────────────────────────────

@proxy_app.command("start")
def proxy_start(
    port: int = typer.Option(None, "--port", "-p", help="Proxy port (default: PROXY_PORT env or 7453)"),
    upstream: str = typer.Option(None, "--upstream", "-u", help="Upstream base URL"),
    trace_file: str = typer.Option(None, "--trace-file", "-t", help="Trace output file"),
    background: bool = typer.Option(False, "--background", "-b", help="Run in background (fork)"),
) -> None:
    """Start the Agentimize proxy server."""
    resolved_port = port or int(os.getenv("PROXY_PORT", "7453"))
    resolved_upstream = upstream or os.getenv("UPSTREAM_BASE_URL", "https://api.openai.com")
    resolved_trace = trace_file or os.getenv("TRACE_FILE", "agentimize_trace.jsonl")

    console.print(Panel.fit(
        f"[bold green]Starting Agentimize Proxy[/bold green]\n"
        f"Port: [cyan]{resolved_port}[/cyan]\n"
        f"Upstream: [yellow]{resolved_upstream}[/yellow]\n"
        f"Trace: [yellow]{resolved_trace}[/yellow]",
        title="Agentimize",
        border_style="green",
    ))

    if background:
        # Fork the process
        pid = os.fork()
        if pid > 0:
            PIDFILE.write_text(str(pid))
            console.print(f"[green]Proxy started in background (PID {pid})[/green]")
            console.print(f"Stop with: [cyan]agentimize proxy stop[/cyan]")
            return
        # Child process: redirect stdout/stderr
        sys.stdout = open("/tmp/agentimize_proxy.log", "w")
        sys.stderr = sys.stdout

    from agentimize.proxy.server import start_proxy
    start_proxy(port=resolved_port, upstream=resolved_upstream, trace_file=resolved_trace)


@proxy_app.command("stop")
def proxy_stop() -> None:
    """Stop a background proxy server."""
    if not PIDFILE.exists():
        console.print("[yellow]No PID file found. Is the proxy running in background?[/yellow]")
        raise typer.Exit(1)

    pid = int(PIDFILE.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        PIDFILE.unlink()
        console.print(f"[green]Proxy (PID {pid}) stopped.[/green]")
    except ProcessLookupError:
        console.print(f"[yellow]Process {pid} not found (already stopped?)[/yellow]")
        PIDFILE.unlink()
    except PermissionError:
        console.print(f"[red]Permission denied to stop PID {pid}[/red]")
        raise typer.Exit(1)


# ─── analyze command ─────────────────────────────────────────────────────────

@app.command("analyze")
def analyze(
    trace_file: str = typer.Argument(
        None,
        help="Path to trace .jsonl file (default: agentimize_trace.jsonl)",
    ),
    session_id: Optional[str] = typer.Option(None, "--session", "-s", help="Analyze specific session ID only"),
) -> None:
    """Analyze a trace file and show optimization recommendations."""
    from agentimize.tracer.graph_builder import build_trace_from_file
    from agentimize.optimizer.solver import optimize_trace

    resolved_file = trace_file or os.getenv("TRACE_FILE", "agentimize_trace.jsonl")
    path = Path(resolved_file)

    if not path.exists():
        console.print(f"[red]Trace file not found: {resolved_file}[/red]")
        raise typer.Exit(1)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Loading trace...", total=None)
        trace_graphs = build_trace_from_file(path, session_id=session_id)
        progress.update(task, description=f"Loaded {len(trace_graphs)} session(s)")

    if not trace_graphs:
        console.print("[yellow]No sessions found in trace file.[/yellow]")
        raise typer.Exit(0)

    for trace, graph in trace_graphs:
        console.print(f"\n[bold]Session:[/bold] [cyan]{trace.session_id}[/cyan]")
        console.print(f"  Events: {len(trace.events)} · Cost: [purple]${trace.total_cost_usd:.6f}[/purple] · Tokens: {trace.total_tokens}")

        result = optimize_trace(trace, graph)

        # Summary panel
        savings_color = "green" if result.savings_pct > 20 else "yellow" if result.savings_pct > 5 else "white"
        console.print(Panel(
            f"Original: [purple]${result.original_cost_usd:.6f}[/purple]  →  "
            f"Optimized: [blue]${result.optimized_cost_usd:.6f}[/blue]  "
            f"Savings: [{savings_color}]${result.savings_usd:.6f} ({result.savings_pct:.1f}%)[/{savings_color}]",
            title="Optimization Summary",
            border_style=savings_color,
        ))

        # Recommendations table
        if result.recommendations:
            rec_table = Table(title="Model Recommendations", show_header=True, header_style="bold magenta")
            rec_table.add_column("Original Model", style="red")
            rec_table.add_column("Recommended", style="green")
            rec_table.add_column("Complexity")
            rec_table.add_column("Savings", style="green")
            rec_table.add_column("Reason")

            for rec in result.recommendations:
                rec_table.add_row(
                    rec["original_model"],
                    rec["recommended_model"],
                    rec["complexity"],
                    f"${rec['savings_usd']:.6f}",
                    rec["reason"][:60] + "…" if len(rec["reason"]) > 60 else rec["reason"],
                )
            console.print(rec_table)

        # Loop recommendations
        if result.loop_recommendations:
            console.print("\n[bold yellow]Loop Recommendations:[/bold yellow]")
            for lr in result.loop_recommendations:
                console.print(f"  ⚠ {lr['description']}")
                console.print(f"    Estimated savings: [green]${lr['savings_usd']:.6f}[/green]")


# ─── judge command ────────────────────────────────────────────────────────────

@app.command("judge")
def judge(
    session_id: str = typer.Argument(..., help="Session ID to judge"),
    task: str = typer.Option(..., "--task", "-t", help="Task description for the judge"),
    trace_file: Optional[str] = typer.Option(None, "--trace-file", help="Trace file path"),
) -> None:
    """Run the LLM judge on a session to assess task success."""
    from agentimize.tracer.graph_builder import load_all_traces, parse_trace
    from agentimize.judge.llm_judge import LLMJudge

    resolved_file = trace_file or os.getenv("TRACE_FILE", "agentimize_trace.jsonl")
    search_dir = Path(resolved_file).parent if trace_file else Path(".")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        t = progress.add_task("Loading trace...", total=None)
        raw_sessions = load_all_traces(search_dir)
        progress.update(t, description="Running LLM judge...")

        if session_id not in raw_sessions:
            console.print(f"[red]Session {session_id} not found.[/red]")
            raise typer.Exit(1)

        trace = parse_trace(session_id, raw_sessions[session_id])

        try:
            judge_instance = LLMJudge()
        except ValueError as e:
            console.print(f"[red]Judge error: {e}[/red]")
            raise typer.Exit(1)

        updated_trace = judge_instance.judge_and_update_trace(trace, task)
        progress.update(t, description="Done!")

    success_style = "green" if updated_trace.task_success else "red"
    console.print(Panel(
        f"Task Success: [{success_style}]{'Yes' if updated_trace.task_success else 'No'}[/{success_style}]\n"
        f"Quality Score: {(updated_trace.quality_score or 0) * 100:.0f}%\n"
        f"Reasoning: {updated_trace.judge_reasoning}",
        title=f"Judge Result for {session_id[:12]}…",
        border_style=success_style,
    ))


# ─── dashboard command ────────────────────────────────────────────────────────

@app.command("dashboard")
def dashboard(
    port: int = typer.Option(None, "--port", "-p", help="Dashboard port (default: DASHBOARD_PORT env or 8080)"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
) -> None:
    """Start the Agentimize web dashboard."""
    import uvicorn

    resolved_port = port or int(os.getenv("DASHBOARD_PORT", "8080"))

    console.print(Panel.fit(
        f"[bold blue]Starting Agentimize Dashboard[/bold blue]\n"
        f"Open: [cyan]http://localhost:{resolved_port}[/cyan]",
        title="Dashboard",
        border_style="blue",
    ))

    uvicorn.run(
        "agentimize.dashboard.api:app",
        host="0.0.0.0",
        port=resolved_port,
        reload=reload,
        log_level="info",
    )


# ─── report command ───────────────────────────────────────────────────────────

@app.command("report")
def report(
    session_id: str = typer.Argument(..., help="Session ID to generate report for"),
    trace_file: Optional[str] = typer.Option(None, "--trace-file", help="Trace file path"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Save report to JSON file"),
    ai_summary: bool = typer.Option(False, "--ai-summary", help="Generate AI-powered summary (requires API key)"),
) -> None:
    """Generate a comprehensive optimization report for a session."""
    from agentimize.tracer.graph_builder import build_graph, load_all_traces, parse_trace
    from agentimize.optimizer.solver import optimize_trace
    from agentimize.applicator.patcher import generate_full_report

    search_dir = Path(trace_file).parent if trace_file else Path(".")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        t = progress.add_task("Loading trace...", total=None)
        raw_sessions = load_all_traces(search_dir)

        if session_id not in raw_sessions:
            console.print(f"[red]Session {session_id} not found.[/red]")
            raise typer.Exit(1)

        progress.update(t, description="Building graph...")
        trace = parse_trace(session_id, raw_sessions[session_id])
        graph = build_graph(trace)

        progress.update(t, description="Running optimizer...")
        opt_result = optimize_trace(trace, graph)

        if ai_summary:
            progress.update(t, description="Generating AI summary...")
            try:
                from agentimize.judge.llm_judge import LLMJudge
                judge_instance = LLMJudge()
                ai_text = judge_instance.generate_optimization_summary(trace, opt_result)
                opt_result.summary = ai_text
            except Exception as e:
                console.print(f"[yellow]AI summary failed: {e}[/yellow]")

        progress.update(t, description="Generating report...")
        full_report = generate_full_report(opt_result, session_id)

    # Print to console
    console.print(Panel(
        f"[bold]Session:[/bold] {session_id}\n"
        f"[bold]Events:[/bold] {len(trace.events)}\n"
        f"[bold]Original Cost:[/bold] [purple]${opt_result.original_cost_usd:.6f}[/purple]\n"
        f"[bold]Optimized Cost:[/bold] [blue]${opt_result.optimized_cost_usd:.6f}[/blue]\n"
        f"[bold]Savings:[/bold] [green]${opt_result.savings_usd:.6f} ({opt_result.savings_pct:.1f}%)[/green]",
        title="Optimization Report",
        border_style="blue",
    ))

    console.print("\n[bold]Summary:[/bold]")
    console.print(opt_result.summary)

    if full_report.get("config_patch"):
        console.print("\n[bold]Config Patch (dotenv):[/bold]")
        console.print(full_report["config_patch"]["dotenv"])

    if output:
        Path(output).write_text(json.dumps(full_report, indent=2))
        console.print(f"\n[green]Report saved to: {output}[/green]")


if __name__ == "__main__":
    app()
