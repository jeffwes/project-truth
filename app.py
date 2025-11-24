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
    # Build UI and include client-side debug script inside the page so
    # it will actually be delivered to the browser.
    debug_script = ui.tags.script(
        "(function() {\n"
        "  function attach() {\n"
        "    try {\n"
        "      var analyze = document.getElementById('analyze');\n"
        "      if (analyze && !analyze.dataset._debugAttached) {\n"
        "        analyze.dataset._debugAttached = '1';\n"
        "        analyze.addEventListener('click', function(e){\n"
        "          console.log('Client(Debug): analyze clicked (direct)');\n"
        "        });\n"
        "        console.log('Client(Debug): attached analyze handler');\n"
        "      }\n"
        "      var submit = document.getElementById('submit_checks');\n"
        "      if (submit && !submit.dataset._debugAttached) {\n"
        "        submit.dataset._debugAttached = '1';\n"
        "        submit.addEventListener('click', function(e){\n"
        "          console.log('Client(Debug): submit_checks clicked (direct)');\n"
        "        });\n"
        "        console.log('Client(Debug): attached submit_checks handler');\n"
        "      }\n"
        "    } catch(err) { console.error('attach error', err); }\n"
        "  }\n"
        "  attach();\n"
        "  var tries = 0;\n"
        "  var iv = setInterval(function(){ tries++; attach(); if (tries>20) clearInterval(iv); }, 250);\n"
        "  document.addEventListener('click', function(e){\n"
        "    try {\n"
        "      var btn = e.target.closest && e.target.closest('#analyze, #submit_checks');\n"
        "      if (btn) {\n"
        "        console.log('Client(Debug): bubbled click on', btn.id);\n"
        "      }\n"
        "    } catch(_){}\n"
        "  }, true);\n"
        # probe: also set a custom input value using Shiny.setInputValue when analyze is clicked
        "  try {\n"
        "    if (window.Shiny && Shiny.setInputValue) {\n"
        "      var el = document.getElementById('analyze');\n"
        "      if (el) {\n"
        "        el.addEventListener('click', function(){\n"
        "          try {\n"
        "            Shiny.setInputValue('probe_analyze', Date.now(), {priority: 'event'});\n"
        "            console.log('Client(Debug): Shiny.setInputValue probe_analyze fired');\n"
        "          } catch(e){ console.error('probe setInput error', e); }\n"
        "        });\n"
        "      }\n"
        "    }\n"
        "  } catch(e){ console.error('probe attach err', e); }\n"
        "})();"
    )

    return ui.page_fluid(
        ui.layout_sidebar(
            ui.sidebar(
                ui.h3("Project Truth — Harari Fact Analyzer"),
                ui.input_text_area("text_in", "Paste text", rows=6),
                ui.input_text("url_in", "Or enter URL", value=""),
                ui.input_text("youtube_in", "Or YouTube URL", value=""),
                ui.input_action_button("analyze", "Extract Assertions"),
                # debug: show the raw numeric value of the action button
                ui.output_text("debug_analyze"),
                # debug probe value (set via Shiny.setInputValue from the client)
                ui.output_text("debug_probe"),
                ui.output_text("debug_selected"),
                ui.output_text("debug_submit"),
            ),
            ui.div(
                ui.output_ui("assertions_ui"),
                ui.output_table("results_table")
            )
        ),
        debug_script,
    )


app_ui = make_ui()


def server(input, output, session):
    # reactive holders for assertions and results
    assertions_rv = reactive.Value(None)
    results_rv = reactive.Value(None)
    # simple per-session counter to detect new clicks from an Effect
    _last_analyze = reactive.Value(0)

    # Step 1: extract assertions and present them to the user as choices
    # Use an Effect to observe `input.analyze()` and run extraction when
    # the counter increments. This is more reliable across some Shiny
    # client versions and avoids mysterious decorator-related misses.
    @reactive.Effect
    def _watch_analyze():
        try:
            v = input.analyze()
        except Exception:
            return
        # only run when the button count increases
        if not v or v == _last_analyze.get():
            return
        _last_analyze.set(v)
        logger.info("watch_analyze triggered")

        text = input.text_in()
        url = input.url_in()
        youtube = input.youtube_in()

        # If url/youtube provided, fetch; else use passthrough_text
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

        # choices: use a mapping of value->label (both strings)
        # to avoid passing tuples which may be treated as tag
        # attributes and cause type errors.
        choices = {str(i): str(a) for i, a in enumerate(assertions)}
        # Use a simple container div rather than tag_list (some Shiny
        # versions don't provide `tag_list`). This keeps the UI simple
        # and avoids attribute errors in the browser.
        return ui.div(
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

    # Debug output: logs the raw value of the `analyze` action button when it changes.
    @output
    @render.text
    def debug_analyze():
        try:
            v = input.analyze()
            logger.info(f"debug_input_analyze value: {v}")
            # If the reactive event handler didn't run for some reason,
            # invoke the extraction directly when we detect a new click
            # and there are no assertions yet. This is a temporary
            # debugging fallback to determine whether events are being
            # lost before they reach the server-side handler.
            # NOTE: removed fallback extraction call here — calling
            # `extract_assertions()` directly from a render function can
            # cause Shiny client/server output-state errors. Keep this
            # render purely observational (logs the click value) so we
            # can diagnose without interfering with Shiny's lifecycle.
            return str(v)
        except Exception:
            logger.exception("debug_analyze error")
            return "(error)"

    @output
    @render.text
    def debug_probe():
        try:
            v = input.probe_analyze()
            logger.info(f"debug_probe_analyze value: {v}")
            return str(v)
        except Exception:
            # No probe value yet
            return "(no probe)"

    @output
    @render.text
    def debug_selected():
        try:
            sel = input.selected_assertions()
            logger.info(f"debug_selected_assertions value: {sel}")
            return str(sel)
        except Exception:
            return "(no selection)"

    @output
    @render.text
    def debug_submit():
        try:
            v = input.submit_checks()
            logger.info(f"debug_input_submit value: {v}")
            return str(v)
        except Exception:
            return "(no submit)"

    # Do not return the server function; Shiny expects the handlers to be
    # registered by defining them in this scope. Ending the function allows
    # Shiny to use it as the server callback.


app = App(app_ui, server)


if __name__ == "__main__":
    # Quick smoke import/run check (does not start a Shiny server)
    print("App module loaded. To run the Shiny app: `shiny run --reload app.py`")