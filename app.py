"""Shiny app for Project Truth — minimal interactive UI.

Usage: run with `python app.py` (this will run a quick import smoke test).
To run the Shiny dev server, install `shiny` and run `shiny run --reload app.py`.
"""
from __future__ import annotations

import os
import json
from shiny import App, ui, render, reactive
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv is optional; server can still use exported env vars
    pass

import logging
logger = logging.getLogger("project_truth")
logging.basicConfig(level=logging.INFO)
from os import environ
if environ.get("GEMINI_API_KEY"):
    logger.info("GEMINI_API_KEY present in process (will use SDK)")
else:
    logger.info("GEMINI_API_KEY NOT present in process")
import pandas as pd

from data_ingestion import passthrough_text, fetch_from_url, fetch_from_youtube
from analysis_engine import extract_facts, check_facts, classify_harari


def make_ui():
    return ui.page_fluid(
        ui.layout_sidebar(
            ui.sidebar(
                ui.h3("Project Truth — Harari Fact Analyzer"),
                ui.input_text_area("text_in", "Paste text", rows=6),
                ui.input_text("url_in", "Or enter URL", value=""),
                ui.input_text("youtube_in", "Or YouTube URL", value=""),
                ui.input_action_button("analyze", "Analyze Content"),
            ),
            ui.div(
                ui.output_table("results_table")
            )
        )
    )


app_ui = make_ui()


def server(input, output, session):
    @reactive.event(input.analyze)
    def run_analysis():
        text = input.text_in()
        url = input.url_in()
        youtube = input.youtube_in()

        # Check for API key early and return a user-friendly message
        from os import environ
        if not environ.get("GEMINI_API_KEY"):
            import pandas as _pd
            return _pd.DataFrame([
                {
                    "assertion": "(GEMINI_API_KEY not set)",
                    "harari_tier": None,
                    "rating": "UNAVAILABLE",
                    "summary": "The environment variable GEMINI_API_KEY is not set in the Shiny server process.\nSet it and restart the server, or restart this session with the key exported.",
                    "source_url": None,
                }
            ])

        if url:
            text = fetch_from_url(url)
        elif youtube:
            text = fetch_from_youtube(youtube)
        else:
            text = passthrough_text(text)

        assertions = extract_facts(text)
        if not assertions:
            return pd.DataFrame([{"assertion": "(no assertions found)"}])

        checked = check_facts(assertions)
        classified = classify_harari(checked)

        rows = []
        for item in classified:
            rows.append({
                "assertion": item.get("assertion"),
                "harari_tier": item.get("harari_tier"),
                "rating": item.get("rating"),
                "summary": item.get("summary"),
                "source_url": item.get("source_url"),
            })

        return pd.DataFrame(rows)

    @output
    @render.table
    def results_table():
        return run_analysis()

    return server


app = App(app_ui, server)


if __name__ == "__main__":
    # Quick smoke import/run check (does not start a Shiny server)
    print("App module loaded. To run the Shiny app: `shiny run --reload app.py`")