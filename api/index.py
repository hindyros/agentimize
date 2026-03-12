import sys
import os

# Ensure the repo root is on the path so agentimize package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agentimize.dashboard.api import app  # noqa: F401 — Vercel picks up `app`
