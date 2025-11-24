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
                ui.input_action_button("analyze", "Extract Assertions"),
            ),
            ui.div(
                ui.output_ui("assertions_ui"),
                ui.output_table("results_table")
            )
        )
    )

    # Inject a small client-side script to help debug clicks.
    # This will show a browser alert when the 'analyze' button is clicked
    # and log to the console. It does not change server behavior.
    # Note: Shiny may render elements asynchronously; we listen on the
    # document and check event targets.
    ui.tags.script(
        "(function() {\n"
        "  function attach() {\n"
        "    try {\n"
        "      var analyze = document.getElementById('analyze');\n"
        "      if (analyze && !analyze.dataset._debugAttached) {\n"
        "        analyze.dataset._debugAttached = '1';\n"
        "        analyze.addEventListener('click', function(e){\n"
        "          console.log('Client(Debug): analyze clicked (direct)');\n"
        "          try { alert('Client: Extract Assertions clicked'); } catch(_){}\n"
        "        });\n"
        "        console.log('Client(Debug): attached analyze handler');\n"
        "      }\n"
        "      var submit = document.getElementById('submit_checks');\n"
        "      if (submit && !submit.dataset._debugAttached) {\n"
        "        submit.dataset._debugAttached = '1';\n"
        "        submit.addEventListener('click', function(e){\n"
        "          console.log('Client(Debug): submit_checks clicked (direct)');\n"
        "          try { alert('Client: Fact-check selected clicked'); } catch(_){}\n"
        "        });\n"
        "        console.log('Client(Debug): attached submit_checks handler');\n"
        "      }\n"
        "    } catch(err) { console.error('attach error', err); }\n"
        "  }\n"
        "  // try immediately and then poll a few times in case Shiny renders later\n"
        "  attach();\n"
        "  var tries = 0;\n"
        "  var iv = setInterval(function(){ tries++; attach(); if (tries>20) clearInterval(iv); }, 250);\n"
        "  // also capture bubbled clicks (fallback)\n"
        "  document.addEventListener('click', function(e){\n"
        "    try {\n"
        "      var btn = e.target.closest && e.target.closest('#analyze, #submit_checks');\n"
        "      if (btn) {\n"
        "        console.log('Client(Debug): bubbled click on', btn.id);\n"
        "      }\n"
        "    } catch(_){}\n"
        "  }, true);\n"
        "})();"
    )


app_ui = make_ui()


def server(input, output, session):
    # reactive holders for assertions and results
    assertions_rv = reactive.Value(None)
    results_rv = reactive.Value(None)

    # Step 1: extract assertions and present them to the user as choices
    @reactive.event(input.analyze)
    def extract_assertions():
        logger.info("extract_assertions() triggered")
        text = input.text_in()
        url = input.url_in()
        youtube = input.youtube_in()

        # If no GEMINI key present we still allow extraction (spaCy only)
        if url:
            text = fetch_from_url(url)
        elif youtube:
            text = fetch_from_youtube(youtube)
        else:
            text = passthrough_text(text)

        assertions = extract_facts(text)
        logger.info(f"extract_facts returned {len(assertions) if assertions else 0} assertions")
        if not assertions:
            assertions_rv.set([])
            results_rv.set(pd.DataFrame([{"assertion": "(no assertions found)"}]))
            return

        assertions_rv.set(assertions)
        # clear previous results
        results_rv.set(None)

    # Render the assertions as a checkbox group + submit button
    @output
    @render.ui
    def assertions_ui():
        logger.info("rendering assertions_ui")
        assertions = assertions_rv.get()
        if assertions is None:
            return ui.HTML("<p>Paste text and click 'Extract Assertions' to begin.</p>")
        if not assertions:
            return ui.HTML("<p>No assertions detected.</p>")

        # choices: (value, label) where value is index as string
        choices = [(str(i), a) for i, a in enumerate(assertions)]
        return ui.tag_list(
            ui.input_checkbox_group("selected_assertions", "Select assertions to fact-check", choices=choices),
            ui.input_action_button("submit_checks", "Fact-check selected")
        )

    # Step 2: when user submits selected assertions, run LLM check + Harari classification
    @reactive.event(input.submit_checks)
    def run_selected_checks():
        logger.info("run_selected_checks() triggered")
        # Check API key presence; if missing, show friendly message
        from os import environ
        if not environ.get("GEMINI_API_KEY"):
            results_rv.set(pd.DataFrame([
                {
                    "assertion": "(GEMINI_API_KEY not set)",
                    "harari_tier": None,
                    "rating": "UNAVAILABLE",
                    "summary": "The environment variable GEMINI_API_KEY is not set in the Shiny server process. Set it and restart the server.",
                    "source_url": None,
                }
            ]))
            return

        sel = input.selected_assertions()
        if not sel:
            results_rv.set(pd.DataFrame([{"assertion": "(no assertions selected)"}]))
            return

        assertions = assertions_rv.get() or []
        # map selected indices back to assertions
        try:
            to_check = [assertions[int(i)] for i in sel]
        except Exception:
            results_rv.set(pd.DataFrame([{"assertion": "(invalid selection)"}]))
            return

        checked = check_facts(to_check)
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

        results_rv.set(pd.DataFrame(rows))
        logger.info(f"run_selected_checks set results with {len(rows)} rows")

    @output
    @render.table
    def results_table():
        # Return the latest results DataFrame stored in the reactive value.
        df = results_rv.get()
        if df is None:
            # show empty table placeholder
            return pd.DataFrame([])
        return df

    # Do not return the server function; Shiny expects the handlers to be
    # registered by defining them in this scope. Ending the function allows
    # Shiny to use it as the server callback.


app = App(app_ui, server)


if __name__ == "__main__":
    # Quick smoke import/run check (does not start a Shiny server)
    print("App module loaded. To run the Shiny app: `shiny run --reload app.py`")