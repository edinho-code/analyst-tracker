#!/usr/bin/env python3
"""Launcher that sets cwd before starting Streamlit (needed for preview sandbox)."""
import os
import sys

PROJECT = "/Users/edersonsouto/Documents/analyst-tracker"
os.chdir(PROJECT)

sys.argv = [
    "streamlit", "run",
    os.path.join(PROJECT, "dashboard.py"),
    "--server.port", os.environ.get("PORT", "8501"),
    "--server.headless", "true",
]

from streamlit.web.cli import main  # noqa: E402
main(prog_name="streamlit")
