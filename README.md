# Agentimize

**Plug-and-play AI agent cost optimizer.** Intercepts LLM API calls via a local HTTP proxy, traces execution, builds a workflow graph, runs a cost-minimization optimizer, and shows savings in a web dashboard.

## How It Works

```
Your Agent → Agentimize Proxy (port 7453) → OpenAI / Anthropic API
                     ↓
            agentimize_trace.jsonl
                     ↓
         Optimizer (greedy heuristic)
                     ↓
         Dashboard (FastAPI + Chart.js)
```

1. **Proxy**: Your agent points its `OPENAI_BASE_URL` at `http://localhost:7453`. The proxy forwards all requests transparently and records each LLM call to a `.jsonl` trace file.
2. **Tracer**: Parses the trace file, groups by session, builds a NetworkX DAG.
3. **Optimizer**: Classifies each call's complexity (simple/medium/complex) and recommends the cheapest model that meets requirements.
4. **Dashboard**: Shows cost charts, session details, optimization recommendations, and loop detection results.

## Quick Start

### 1. Install

```bash
pip install -e .
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Start the proxy

```bash
agentimize proxy start
# Proxy running at http://localhost:7453
```

### 4. Run your agent through the proxy

```bash
# Configure your agent to use the proxy:
export OPENAI_BASE_URL=http://localhost:7453/v1

# Or run the included test agent:
python -m test_agent.nyc_apartment_agent --proxy http://localhost:7453
```

### 5. Analyze the trace

```bash
agentimize analyze agentimize_trace.jsonl
```

### 6. Open the dashboard

```bash
agentimize dashboard
# Open http://localhost:8080
```

## CLI Reference

```
agentimize proxy start [--port 7453] [--upstream https://api.openai.com]
agentimize proxy stop
agentimize analyze [trace_file] [--session SESSION_ID]
agentimize judge SESSION_ID --task "Find best 20 flats in NYC"
agentimize dashboard [--port 8080]
agentimize report SESSION_ID [--output report.json] [--ai-summary]
```

## Model Pricing

| Model | Input ($/1M) | Output ($/1M) |
|-------|-------------|--------------|
| gpt-4o-mini | $0.15 | $0.60 |
| gpt-3.5-turbo | $0.50 | $1.50 |
| claude-3-haiku | $0.25 | $1.25 |
| gpt-4o | $2.50 | $10.00 |
| claude-3-5-sonnet | $3.00 | $15.00 |
| gpt-4-turbo | $10.00 | $30.00 |
| claude-3-opus | $15.00 | $75.00 |

## Optimization Logic

Each LLM call is classified by complexity:
- **Simple**: < 500 prompt tokens, no tool calls → assign `gpt-4o-mini`
- **Medium**: < 2000 prompt tokens, no tool calls → assign `gpt-4o-mini`
- **Complex**: > 2000 tokens OR has tool calls → assign minimum `gpt-4o`

Loop detection: if the same `(model, tool)` pattern repeats 3+ consecutive times, it's flagged as a potential infinite loop with iteration cap recommendations.

## Dashboard API

| Endpoint | Description |
|----------|-------------|
| `GET /api/sessions` | List all sessions |
| `GET /api/sessions/{id}` | Full trace for a session |
| `GET /api/sessions/{id}/optimization` | Optimization recommendations |
| `POST /api/sessions/{id}/judge` | Run LLM judge |
| `POST /api/analyze` | Analyze a trace file |
| `GET /api/summary` | Aggregate stats |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | OpenAI API key (required) |
| `UPSTREAM_BASE_URL` | `https://api.openai.com` | Upstream LLM provider |
| `PROXY_PORT` | `7453` | Proxy listen port |
| `DASHBOARD_PORT` | `8080` | Dashboard listen port |
| `TRACE_FILE` | `agentimize_trace.jsonl` | Trace output file |

## Test Agent

The included test agent (`test_agent/nyc_apartment_agent.py`) simulates finding the best 20 NYC apartment rentals under $3000/month. It includes **intentional inefficiencies** for Agentimize to detect:

- Uses `gpt-4o` for simple 10-token classification tasks
- Makes redundant tool calls
- Has a loop pattern (progress checks every 2 iterations)

Run it:
```bash
python -m test_agent.nyc_apartment_agent
```
