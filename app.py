"""
The Resonance Engine
Desktop application for analyzing media through academic frameworks.
"""
import os
import sys
import time
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from shiny import App, ui, render, reactive
import pandas as pd

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.gemini_client import GeminiClient
from src.content_ingestion import ContentIngester
from src.reality_taxonomy import RealityTaxonomyAnalyzer
from src.moral_foundations import MoralFoundationsAnalyzer
from src.tribal_resonance import TribalResonancePredictor


# UI Definition
app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.style("""
            .foundation-card {
                padding: 15px;
                margin: 10px 0;
                border-radius: 8px;
                border-left: 4px solid #007bff;
                background: #f8f9fa;
            }
            .tribe-card {
                padding: 12px;
                margin: 8px 0;
                border-radius: 6px;
                background: #fff;
                border: 1px solid #dee2e6;
            }
            .score-bar {
                height: 20px;
                background: #007bff;
                border-radius: 4px;
                transition: width 0.3s;
            }
            .high-score { background: #28a745; }
            .medium-score { background: #ffc107; }
            .low-score { background: #6c757d; }
            /* Loading overlay */
            #loading-overlay { position: fixed; top:0; left:0; width:100%; height:100%; display:flex; flex-direction:column; align-items:center; justify-content:center; background:rgba(255,255,255,0.88); z-index:10000; }
            .spinner { width:54px; height:54px; border:6px solid #e0e0e0; border-top-color:#007bff; border-radius:50%; animation: spin 0.8s linear infinite; margin-bottom:18px; }
            @keyframes spin { to { transform: rotate(360deg); } }
            .loading-msg { font-size:1.1em; color:#333; }
        """)
        ,
        # Plotly for visualizations
        ui.tags.script(src="https://cdn.plot.ly/plotly-2.35.2.min.js")
    ),
    
    ui.panel_title("The Resonance Engine", "Deconstruct Persuasive Media"),
    
    ui.layout_sidebar(
        ui.sidebar(
            ui.h4("Input"),
            ui.input_radio_buttons(
                "input_mode",
                "Content Source:",
                {
                    "text": "Paste Text",
                    "url": "URL/Browser Tab",
                    "youtube": "YouTube"
                },
                selected="text"
            ),
            
            ui.output_ui("input_ui"),
            
            ui.input_slider(
                "max_assertions",
                "Max Assertions to Extract:",
                min=5,
                max=25,
                value=12,
                step=1
            ),

            ui.input_radio_buttons(
                "analysis_mode",
                "Analysis Depth:",
                {"quick": "Quick (single prompt)", "deep": "Deep (full reasoning)"},
                selected="deep"
            ),

            ui.input_checkbox("use_cache", "Use caching (speeds repeat analyses)", True),
            ui.input_action_button("clear_cache", "Clear Cache", class_="btn-outline-secondary w-100"),
            
            ui.input_action_button(
                "analyze_btn",
                "Analyze Content",
                class_="btn-primary btn-lg w-100"
            ),
            ui.br(),
            
            ui.hr(),

            ui.tags.div(
                ui.output_ui("status_display"),
                style="margin-top: 10px;"
            ),
            
            width=350
        ),
        
        ui.navset_tab(
            ui.nav_panel(
                "Overview",
                ui.div(
                    ui.h3("Analysis Summary"),
                    ui.output_ui("overview_ui"),
                )
            ),
            
            ui.nav_panel(
                "Reality Taxonomy",
                ui.div(
                    ui.h3("Harari's Reality Classification"),
                    ui.p("Classification of assertions by reality type: Objective, Subjective, or Intersubjective."),
                    ui.output_ui("taxonomy_ui"),
                )
            ),
            
            ui.nav_panel(
                "Moral Foundations",
                ui.div(
                    ui.h3("Moral Foundations Theory (Haidt)"),
                    ui.p("Analysis of which psychological triggers are activated."),
                    ui.output_ui("foundations_ui"),
                )
            ),
            
            ui.nav_panel(
                "Tribal Resonance",
                ui.div(
                    ui.h3("Predicted Social Tribe Resonance"),
                    ui.p("Which groups will find this content compelling."),
                    ui.output_ui("tribes_ui"),
                )
            ),
            
            ui.nav_panel(
                "Export",
                ui.div(
                    ui.h3("Export Analysis"),
                    ui.p("Download analysis results in various formats."),
                    ui.download_button("download_json", "Download JSON", class_="btn-success"),
                    ui.br(), ui.br(),
                    ui.download_button("download_csv", "Download CSV", class_="btn-info"),
                    ui.br(), ui.br(),
                    ui.output_text("export_info")
                )
            )
        )
    ),
    # Loading overlay output
    ui.output_ui("loading_overlay")
)


# Server Logic
def server(input, output, session):
    # Reactive values
    analysis_results = reactive.Value(None)
    processing = reactive.Value(False)
    error_message = reactive.Value(None)
    perf_metrics = reactive.Value(None)
    progress_phase = reactive.Value("")  # Track current analysis phase
    cancel_requested = reactive.Value(False)
    canceled = reactive.Value(False)

    async def _wait_for_cancel():
        """Await until user requests cancellation."""
        # Poll reactive flag at short intervals to react quickly
        while not cancel_requested.get():
            await asyncio.sleep(0.1)
        return True

    def _detach_task(task: asyncio.Task):
        """Prevent 'Task was destroyed but it is pending' noise for background work we abandon."""
        def _ignore(_):
            try:
                task.result()
            except Exception:
                # Ignore any exception after abandonment
                pass
        task.add_done_callback(_ignore)

    # Simple in-memory cache
    cache_store: dict = {}
    
    # Initialize clients (lazy)
    gemini_client = None
    
    def get_client():
        nonlocal gemini_client
        if gemini_client is None:
            try:
                gemini_client = GeminiClient()
            except ValueError as e:
                error_message.set(str(e))
                return None
        return gemini_client
    
    # Dynamic input UI based on mode
    @output
    @render.ui
    def input_ui():
        mode = input.input_mode()
        
        if mode == "text":
            return ui.input_text_area(
                "content_input",
                "Paste article text or transcript:",
                rows=10,
                placeholder="Enter the content you want to analyze..."
            )
        elif mode == "url":
            return ui.input_text(
                "url_input",
                "Enter URL:",
                placeholder="https://example.com/article"
            )
        elif mode == "youtube":
            return ui.input_text(
                "youtube_input",
                "Enter YouTube URL:",
                placeholder="https://www.youtube.com/watch?v=..."
            )
    
    # Status display
    @output
    @render.ui
    def status_display():
        """Render status - always returns visible content."""
        base_style = "padding: 12px; background: #f8f9fa; border-radius: 4px; font-weight: 500; min-height: 50px; border: 1px solid #dee2e6;"
        
        if error_message.get():
            return ui.div(
                ui.tags.strong("Error: ", style="color: #dc3545;"),
                ui.span(str(error_message.get()), style="color: #dc3545;"),
                style=base_style
            )

        if processing.get():
            phase = progress_phase.get()
            msg = f"Analyzing... {phase}" if phase else "Analyzing content..."
            return ui.div(
                ui.tags.strong(msg, style="color: #0066cc; font-size: 1.05em;"),
                style=base_style + " background: #e7f3ff;"
            )

        results = analysis_results.get()
        if results:
            pm = perf_metrics.get() or {}
            total = pm.get("total", 0.0)
            mode = pm.get("mode", "")
            return ui.div(
                ui.tags.strong(f"✓ {mode.title()} analysis complete in {total:.1f}s", style="color: #28a745; font-size: 1.05em;"),
                style=base_style + " background: #d4edda;"
            )

        # Show canceled message if last operation was canceled
        if canceled.get():
            return ui.div(
                ui.tags.strong("Analysis canceled", style="color: #6c757d; font-size: 1.05em;"),
                style=base_style
            )

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return ui.div(
                ui.tags.strong("⚠ GEMINI_API_KEY not set", style="color: #856404; font-size: 1.05em;"),
                style=base_style + " background: #fff3cd;"
            )

        return ui.div(
            ui.tags.strong("Ready to analyze content", style="color: #6c757d; font-size: 1.05em;"),
            style=base_style
        )

    # Loading overlay
    @output
    @render.ui
    def loading_overlay():
        """Show loading overlay only when processing."""
        if not processing.get():
            # Return an empty HTML stub to ensure output exists and can toggle reliably
            return ui.HTML("")

        phase = progress_phase.get() or ""
        main_msg = phase if phase else "Text analysis in progress…"

        return ui.div(
            ui.div(class_="spinner"),
            ui.div(main_msg, class_="loading-msg", style="font-weight:600; font-size:1.3em;"),
            ui.div("This may take a few seconds depending on depth.", style="font-size:0.95em; color:#666; margin-top:10px;"),
            id="loading-overlay"
        )

    # Cancel button UI (only show while processing)
    @output
    @render.ui
    def cancel_button():
        # Sidebar cancel button disabled; using overlay button instead
        return ui.HTML("")

    # Clear cache handler
    @reactive.Effect
    @reactive.event(input.clear_cache)
    def _clear_cache():
        cache_store.clear()
        error_message.set(None)
        analysis_results.set(None)
        perf_metrics.set(None)
        canceled.set(False)
        cancel_requested.set(False)

    # Cancel handler removed per request; cancel control no longer exposed
    
    # Main analysis pipeline
    @reactive.Effect
    @reactive.event(input.analyze_btn)
    async def run_analysis():
        processing.set(True)
        error_message.set(None)
        perf_metrics.set(None)
        progress_phase.set("")
        canceled.set(False)
        cancel_requested.set(False)
        # Ensure UI updates immediately
        await reactive.flush()

        t0 = time.perf_counter()
        try:
            client = get_client()
            if not client:
                processing.set(False)
                return

            # Ingest
            progress_phase.set("Ingesting content...")
            await reactive.flush()
            ingest_start = time.perf_counter()
            ingester = ContentIngester()
            mode = input.input_mode()
            if mode == "text":
                content = input.content_input()
                ingestion = ingester.ingest_text(content)
            elif mode == "url":
                ingestion = ingester.ingest_url(input.url_input())
            elif mode == "youtube":
                ingestion = ingester.ingest_youtube(input.youtube_input())
            else:
                error_message.set("Invalid input mode")
                processing.set(False)
                return
            ingest_end = time.perf_counter()

            if ingestion.get("error"):
                error_message.set(ingestion["error"])
                processing.set(False)
                return
            content = ingestion["content"]

            # Generate a concise neutral summary (2 sentences) of the submitted content
            content_summary = None
            try:
                summary_prompt = f"""Summarize the following content neutrally in 2 sentences capturing its central claims/themes and overall tone. Avoid evaluative judgment. Return ONLY JSON: {{\n  \"summary\": \"text\"\n}}\n\nCONTENT:\n{content[:12000]}"""
                summary_resp = await asyncio.to_thread(client.generate_json, summary_prompt, 55)
                if summary_resp.get("ok") and isinstance(summary_resp.get("data"), dict):
                    data = summary_resp.get("data", {})
                    content_summary = (data.get("summary") or data.get("Summary") or "").strip()
            except Exception:
                content_summary = None
            # Fallback heuristic if model summary failed or empty
            if not content_summary:
                import re
                sentences = re.split(r'[.!?]+\s+', content.strip())
                content_summary = " ".join(sentences[:2]).strip()[:600]

            # Check for cancel
            if cancel_requested.get():
                print("[ResonanceEngine] Canceled after ingestion")
                canceled.set(True)
                processing.set(False)
                progress_phase.set("")
                await reactive.flush()
                return

            # Cache key
            analysis_mode = input.analysis_mode()
            max_assertions = input.max_assertions()
            cache_key = hashlib.sha256((content + f"|{analysis_mode}|{max_assertions}").encode()).hexdigest()
            if input.use_cache() and cache_key in cache_store:
                cached = cache_store[cache_key]
                analysis_results.set(cached["results"])
                perf_metrics.set(cached["perf"])
                processing.set(False)
                return

            # QUICK MODE combined prompt path
            if analysis_mode == "quick":
                progress_phase.set("Running quick analysis (combined prompt)...")
                await reactive.flush()
                quick_start = time.perf_counter()
                combined_prompt = f"""You are the Resonance Engine. Perform a QUICK integrated analysis of the following content.

Return ONLY valid JSON with this exact structure:
{{
  "assertions": [
    {{"assertion": "statement text", "classification": "objective|subjective|intersubjective"}},
    ...
  ],
  "foundations": {{
    "care_harm": {{"triggered": bool, "intensity": 0.0-1.0, "valence": "positive|neutral|negative", "explanation": "text", "triggers": ["phrase1", "phrase2"]}},
    "fairness_cheating": {{...}},
    "loyalty_betrayal": {{...}},
                    # If canceled before starting, stop
                    if cancel_requested.get():
                        print("[ResonanceEngine] Canceled before quick call")
                        canceled.set(True)
                        processing.set(False)
                        progress_phase.set("")
                        await reactive.flush()
                        return

                    combined_resp = await asyncio.to_thread(client.generate_json, combined_prompt, 90)
    "sanctity_degradation": {{...}},
                    # If canceled during call, stop and drop result
                    if cancel_requested.get():
                        canceled.set(True)
                        processing.set(False)
                        progress_phase.set("")
                        await reactive.flush()
                        return
    "liberty_oppression": {{...}},
    "overall_profile": "brief summary of moral signature"
  }},
  "tribes": {{
    "predictions": [
      {{"name": "tribe_name", "resonance_score": 0.0-1.0, "sentiment": "positive|neutral|negative", "reasoning": "text", "hooks": ["hook1", "hook2"]}},
      ...
    ],
    "polarization_risk": "low|medium|high",
    "tribal_signature": "brief phrase"
  }}
}}

DEFINITIONS:
- Objective: Facts independent of human consciousness (gravity, DNA, chemical reactions)
- Subjective: Individual feelings/experiences (personal pain, preferences)
- Intersubjective: Shared social constructs (money, laws, national borders)

TASK: Extract up to {min(8, max_assertions)} key assertions from the text and classify each one. Then analyze moral foundations and tribal resonance.

TEXT:
{content[:8000]}
"""
                # Run model call in background and race against cancel signal
                gen_task = asyncio.create_task(asyncio.to_thread(client.generate_json, combined_prompt, 90))
                cancel_task = asyncio.create_task(_wait_for_cancel())
                done, pending = await asyncio.wait({gen_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED)
                if cancel_task in done and cancel_requested.get():
                    # Abandon the running call; it will finish in background
                    _detach_task(gen_task)
                    print("[ResonanceEngine] Cancelled during quick call (abandoning result)")
                    canceled.set(True)
                    processing.set(False)
                    progress_phase.set("")
                    await reactive.flush()
                    return
                # Otherwise, get the generation result
                cancel_task.cancel()
                try:
                    combined_resp = gen_task.result()
                except Exception as e:
                    combined_resp = {"ok": False, "error": str(e)}
                quick_end = time.perf_counter()
                if combined_resp.get("ok"):
                    data = combined_resp.get("data", {})
                    # Normalize taxonomy
                    assertions = data.get("assertions", [])
                    norm_assertions = []
                    for a in assertions:
                        if isinstance(a, dict):
                            norm_assertions.append({
                                "assertion": a.get("assertion", ""),
                                "classification": a.get("classification", "unknown"),
                                "confidence": float(a.get("confidence", 0.8)),
                                "reasoning": a.get("reasoning", "") or "(quick mode - no detailed reasoning)",
                                "error": None
                            })
                        elif isinstance(a, str):
                            norm_assertions.append({
                                "assertion": a,
                                "classification": "unknown",
                                "confidence": 0.0,
                                "reasoning": "(quick mode - classification missing)",
                                "error": None
                            })
                    
                    # Compute taxonomy summary if not provided
                    taxonomy_summary = data.get("taxonomy_summary", {})
                    if not taxonomy_summary or not taxonomy_summary.get("total"):
                        summary = {"objective": 0, "subjective": 0, "intersubjective": 0, "unknown": 0}
                        for a in norm_assertions:
                            cls = a.get("classification", "unknown")
                            summary[cls] = summary.get(cls, 0) + 1
                        summary["total"] = len(norm_assertions)
                        taxonomy_summary = summary
                    
                    taxonomy_result = {"assertions": norm_assertions, "summary": taxonomy_summary, "error": None}
                    
                    # Normalize foundations
                    foundations_data = data.get("foundations", {})
                    if not isinstance(foundations_data, dict) or "overall_profile" not in foundations_data:
                        # Add overall_profile if missing
                        if isinstance(foundations_data, dict):
                            foundations_data["overall_profile"] = "Quick moral signature analysis"
                    foundations_result = {"foundations": foundations_data, "overall_profile": foundations_data.get("overall_profile", "Quick analysis"), "error": None}
                    
                    # Normalize tribes
                    tribes_block = data.get("tribes", {})
                    if isinstance(tribes_block, dict):
                        tribes_result = tribes_block
                    else:
                        tribes_result = {"predictions": tribes_block, "polarization_risk": "unknown", "tribal_signature": "", "error": None}
                    # Store
                    results_obj = {
                        "content": content,
                        "content_summary": content_summary,
                        "ingestion": ingestion,
                        "taxonomy": taxonomy_result,
                        "foundations": foundations_result,
                        "tribes": tribes_result
                    }
                    analysis_results.set(results_obj)
                    total_time = time.perf_counter() - t0
                    perf_obj = {
                        "mode": "quick",
                        "ingestion": ingest_end - ingest_start,
                        "combined": quick_end - quick_start,
                        "total": total_time
                    }
                    perf_metrics.set(perf_obj)
                    if input.use_cache():
                        cache_store[cache_key] = {"results": results_obj, "perf": perf_obj}
                    progress_phase.set("")
                    progress_phase.set("")
                    processing.set(False)
                    return
                # Fallback to deep if quick failed

            # DEEP MODE (or quick fallback) with parallel stages
            progress_phase.set("Extracting assertions and analyzing foundations...")
            await reactive.flush()
            taxonomy_analyzer = RealityTaxonomyAnalyzer(client)
            foundations_analyzer = MoralFoundationsAnalyzer(client)
            tribal_predictor = TribalResonancePredictor(client)

            taxonomy_start = time.perf_counter()
            foundations_start = taxonomy_start

            async def run_taxonomy():
                return await asyncio.to_thread(taxonomy_analyzer.analyze_content, content, max_assertions, analysis_mode == "quick")

            async def run_foundations():
                return await asyncio.to_thread(foundations_analyzer.analyze_foundations, content)

            if cancel_requested.get():
                print("[ResonanceEngine] Canceled before parallel taxonomy/foundations")
                canceled.set(True)
                processing.set(False)
                progress_phase.set("")
                await reactive.flush()
                return

            # Run deep mode taxonomy + foundations in parallel and race with cancel
            tax_task = asyncio.create_task(run_taxonomy())
            fnd_task = asyncio.create_task(run_foundations())
            cancel_task = asyncio.create_task(_wait_for_cancel())
            done, pending = await asyncio.wait({tax_task, fnd_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED)
            if cancel_task in done and cancel_requested.get():
                # Abandon in-flight work
                _detach_task(tax_task)
                _detach_task(fnd_task)
                print("[ResonanceEngine] Cancelled during taxonomy/foundations (abandoning results)")
                canceled.set(True)
                processing.set(False)
                progress_phase.set("")
                await reactive.flush()
                return
            # Otherwise, ensure both results are available
            cancel_task.cancel()
            taxonomy_result = await tax_task
            foundations_result = await fnd_task
            taxonomy_end = time.perf_counter()
            foundations_end = taxonomy_end

            if cancel_requested.get():
                print("[ResonanceEngine] Canceled after taxonomy/foundations")
                canceled.set(True)
                processing.set(False)
                progress_phase.set("")
                await reactive.flush()
                return

            progress_phase.set("Predicting tribal resonance...")
            await reactive.flush()
            tribal_start = time.perf_counter()
            # Run tribal prediction and race with cancel
            tri_task = asyncio.create_task(asyncio.to_thread(tribal_predictor.predict_resonance, content, foundations_result, taxonomy_result))
            cancel_task = asyncio.create_task(_wait_for_cancel())
            done, pending = await asyncio.wait({tri_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED)
            if cancel_task in done and cancel_requested.get():
                _detach_task(tri_task)
                print("[ResonanceEngine] Cancelled during tribal prediction (abandoning result)")
                canceled.set(True)
                processing.set(False)
                progress_phase.set("")
                await reactive.flush()
                return
            cancel_task.cancel()
            tribal_result = await tri_task
            tribal_end = time.perf_counter()

            if cancel_requested.get():
                print("[ResonanceEngine] Canceled after tribal prediction")
                canceled.set(True)
                processing.set(False)
                progress_phase.set("")
                await reactive.flush()
                return

            results_obj = {
                "content": content,
                "content_summary": content_summary,
                "ingestion": ingestion,
                "taxonomy": taxonomy_result,
                "foundations": foundations_result,
                "tribes": tribal_result
            }
            analysis_results.set(results_obj)

            total_time = time.perf_counter() - t0
            perf_obj = {
                "mode": analysis_mode,
                "ingestion": ingest_end - ingest_start,
                "taxonomy": taxonomy_end - taxonomy_start,
                "foundations": foundations_end - foundations_start,
                "tribal": tribal_end - tribal_start,
                "total": total_time
            }
            perf_metrics.set(perf_obj)
            if input.use_cache():
                cache_store[cache_key] = {"results": results_obj, "perf": perf_obj}

        except Exception as e:
            error_message.set(f"Analysis failed: {str(e)}")
        finally:
            progress_phase.set("")
            processing.set(False)
    
    # Overview UI
    @output
    @render.ui
    def overview_ui():
        results = analysis_results.get()
        if not results:
            return ui.p("No analysis yet. Enter content and click 'Analyze Content'.")
        
        taxonomy = results["taxonomy"]
        foundations = results["foundations"]
        tribes = results["tribes"]
        perf = perf_metrics.get() or {}
        content_summary = results.get("content_summary") or "(summary unavailable)"
        
        # Build summary cards
        cards = []
        # Content summary card (placed first)
        cards.append(ui.div(
            ui.h4("Content Summary"),
            ui.p(content_summary),
            class_="foundation-card"
        ))
        
        # Reality taxonomy summary
        summary = taxonomy.get("summary", {})
        if summary:
            total = summary.get("total", 0)
            obj = summary.get("objective", 0)
            subj = summary.get("subjective", 0)
            inter = summary.get("intersubjective", 0)
            
            cards.append(ui.div(
                ui.h4("Reality Distribution"),
                ui.p(f"Total assertions: {total}"),
                ui.tags.ul(
                    ui.tags.li(f"Objective: {obj} ({obj/total*100:.0f}%)" if total > 0 else "Objective: 0"),
                    ui.tags.li(f"Subjective: {subj} ({subj/total*100:.0f}%)" if total > 0 else "Subjective: 0"),
                    ui.tags.li(f"Intersubjective: {inter} ({inter/total*100:.0f}%)" if total > 0 else "Intersubjective: 0")
                ),
                class_="foundation-card"
            ))

        # Note: additional enrichment summaries are intentionally omitted from Overview per request.
        
        # Moral foundations summary
        profile = foundations.get("overall_profile", "")
        if profile:
            cards.append(ui.div(
                ui.h4("Moral Signature"),
                ui.p(profile),
                class_="foundation-card"
            ))
        
        # Tribal resonance summary
        signature = tribes.get("tribal_signature", "")
        polarization = tribes.get("polarization_risk", "unknown")
        if signature:
            cards.append(ui.div(
                ui.h4("Tribal Appeal"),
                ui.p(signature),
                ui.p(f"Polarization Risk: {polarization.title()}", style=f"color: {'red' if polarization == 'high' else 'orange' if polarization == 'medium' else 'green'};"),
                class_="foundation-card"
            ))
        
        # Performance card
        if perf:
            cards.append(ui.div(
                ui.h4("Performance"),
                ui.tags.ul(
                    ui.tags.li(f"Mode: {perf.get('mode','')}"),
                    ui.tags.li(f"Total: {perf.get('total',0.0):.1f}s"),
                    ui.tags.li(f"Ingestion: {perf.get('ingestion',0.0):.2f}s"),
                    ui.tags.li(f"Taxonomy: {perf.get('taxonomy', perf.get('combined',0.0)):.2f}s"),
                    ui.tags.li(f"Foundations: {perf.get('foundations',0.0):.2f}s"),
                    ui.tags.li(f"Tribal: {perf.get('tribal',0.0):.2f}s") if perf.get('tribal') is not None else ""
                ),
                class_="foundation-card"
            ))

        return ui.div(*cards)
    
    # Reality Taxonomy UI
    @output
    @render.ui
    def taxonomy_ui():
        results = analysis_results.get()
        if not results:
            return ui.p("No analysis available.")
        
        taxonomy = results["taxonomy"]
        assertions = taxonomy.get("assertions", [])
        
        if not assertions:
            return ui.p("No assertions extracted.")
        
        # Build Sankey chart at top
        import json as _json
        labels = []
        index = {}
        def idx(label: str) -> int:
            if label not in index:
                index[label] = len(labels)
                labels.append(label)
            return index[label]

        sources = []
        targets = []
        values = []

        # Top-level nodes
        n_obj = idx("Objective")
        n_inter = idx("Intersubjective")
        n_subj = idx("Subjective")

        # Objective -> Fact-check statuses
        fact_counts = {"verified": 0, "partially_verified": 0, "disputed": 0, "unclear": 0}
        for a in assertions:
            if a.get("classification") == "objective":
                fc = (a.get("fact_check") or {}).get("status")
                if fc in fact_counts:
                    fact_counts[fc] += 1
        for k, v in fact_counts.items():
            if v > 0:
                child = idx(f"Fact: {k.replace('_',' ').title()}")
                sources.append(n_obj)
                targets.append(child)
                values.append(v)

        # Intersubjective -> Stability -> Myth
        stability_keys = ["naturalized", "contested", "ambiguous"]
        myth_keys = ["tribal_national", "legal_bureaucratic", "economic", "divine_ideological", "other"]
        joint_st_myth = {s: {m: 0 for m in myth_keys} for s in stability_keys}
        st_totals = {s: 0 for s in stability_keys}
        for a in assertions:
            if a.get("classification") == "intersubjective":
                si = (a.get("stability_index") or {}).get("status")
                mt = (a.get("myth_taxonomy") or {}).get("category")
                if si in stability_keys:
                    st_totals[si] += 1
                    if mt in myth_keys:
                        joint_st_myth[si][mt] += 1
        st_nodes = {}
        for s, count in st_totals.items():
            if count > 0:
                st_node = idx(f"Stability: {s.title()}")
                st_nodes[s] = st_node
                sources.append(n_inter)
                targets.append(st_node)
                values.append(count)
        for s, myth_map in joint_st_myth.items():
            s_node = st_nodes.get(s)
            if s_node is None:
                continue
            for m, c in myth_map.items():
                if c > 0:
                    m_node = idx(f"Myth: {m.replace('_',' ').title()}")
                    sources.append(s_node)
                    targets.append(m_node)
                    values.append(c)

        # Subjective -> Arousal -> Empathy
        arousal_keys = ["high", "neutral", "low"]
        empathy_keys = ["one_sided", "balanced", "unclear"]
        joint_ar_emp = {a: {e: 0 for e in empathy_keys} for a in arousal_keys}
        ar_totals = {a: 0 for a in arousal_keys}
        for a in assertions:
            if a.get("classification") == "subjective":
                va = (a.get("viral_arousal") or {}).get("category")
                es = (a.get("empathy_span") or {}).get("focus_bias")
                if va in arousal_keys:
                    ar_totals[va] += 1
                    if es in empathy_keys:
                        joint_ar_emp[va][es] += 1
        ar_nodes = {}
        for a_key, count in ar_totals.items():
            if count > 0:
                a_node = idx(f"Arousal: {a_key.title()}")
                ar_nodes[a_key] = a_node
                sources.append(n_subj)
                targets.append(a_node)
                values.append(count)
        for a_key, emp_map in joint_ar_emp.items():
            a_node = ar_nodes.get(a_key)
            if a_node is None:
                continue
            for e_key, c in emp_map.items():
                if c > 0:
                    e_node = idx(f"Empathy: {e_key.replace('_',' ').title()}")
                    sources.append(a_node)
                    targets.append(e_node)
                    values.append(c)

        # Build Sankey chart UI component
        sankey_html = ""
        if values:
            sankey_data = {
                "type": "sankey",
                "orientation": "h",
                "node": {"label": labels, "pad": 15, "thickness": 20},
                "link": {"source": sources, "target": targets, "value": values}
            }
            layout = {"margin": {"l": 10, "r": 10, "t": 10, "b": 10}, "paper_bgcolor": "#ffffff"}
            fig_json = _json.dumps({"data": [sankey_data], "layout": layout})
            sankey_html = f"""
            <div id="sankey_rt" style="width:100%;height:520px;margin-bottom:24px;"></div>
            <script>
              (function(){{
                var spec = {fig_json};
                if (window.Plotly && document.getElementById('sankey_rt')) {{
                  Plotly.newPlot('sankey_rt', spec.data, spec.layout, {{displayModeBar:false}});
                }}
              }})();
            </script>
        """
        
        # Group by classification
        grouped = {"objective": [], "subjective": [], "intersubjective": [], "unknown": []}
        for item in assertions:
            classification = item.get("classification", "unknown")
            grouped[classification].append(item)
        
        sections = []
        for class_type, items in grouped.items():
            if items:
                section_items = []
                for item in items:
                    assertion = item.get("assertion", "")
                    reasoning = item.get("reasoning", "")
                    confidence = item.get("confidence", 0.0)
                    # Stack each metadata block in its own div for clean formatting
                    meta_blocks = []
                    meta_blocks.append(ui.div(ui.tags.small(f"Confidence: {confidence:.1%}")))
                    if reasoning:
                        meta_blocks.append(ui.div(ui.tags.small(reasoning)))

                    if class_type == "objective":
                        fc = item.get("fact_check") or {}
                        status = fc.get('status')
                        if status:
                            meta_blocks.append(ui.div(ui.tags.small(f"Fact-Check Status: {status.title()} (conf {fc.get('verification_confidence',0.0):.0%})")))
                        ev = fc.get('evidence') or []
                        if ev:
                            meta_blocks.append(ui.div(
                                ui.tags.small("Evidence:"),
                                ui.tags.ul(*[ui.tags.li(e) for e in ev[:3]])
                            ))
                        srcs = fc.get('sources') or []
                        if srcs:
                            meta_blocks.append(ui.div(
                                ui.tags.small("Sources:"),
                                ui.tags.ul(*[ui.tags.li(ui.tags.a((s.get('title') or s.get('url','Source')), href=(s.get('url') or '#'), target="_blank")) for s in srcs[:3]])
                            ))
                        notes = fc.get('notes')
                        if notes:
                            meta_blocks.append(ui.div(ui.tags.small(f"Notes: {notes}")))

                    elif class_type == "intersubjective":
                        si = item.get("stability_index") or {}
                        mt = item.get("myth_taxonomy") or {}
                        meta_blocks.append(ui.div(ui.tags.small(f"Stability: {si.get('status','ambiguous').title()}")))
                        si_reason = si.get('reasoning')
                        if si_reason:
                            meta_blocks.append(ui.div(ui.tags.small(f"Stability Rationale: {si_reason}")))
                        cues = si.get('cues') or []
                        if cues:
                            meta_blocks.append(ui.div(
                                ui.tags.small("Stability Cues:"),
                                ui.tags.ul(*[ui.tags.li(c) for c in cues[:6]])
                            ))
                        mt_cat = mt.get('category')
                        if mt_cat:
                            meta_blocks.append(ui.div(ui.tags.small(f"Myth: {mt_cat.replace('_',' ').title()} ({mt.get('confidence',0.0):.0%})")))
                        mt_reason = mt.get('reasoning')
                        if mt_reason:
                            meta_blocks.append(ui.div(ui.tags.small(f"Myth Rationale: {mt_reason}")))

                    elif class_type == "subjective":
                        va = item.get("viral_arousal") or {}
                        es = item.get("empathy_span") or {}
                        meta_blocks.append(ui.div(ui.tags.small(f"Arousal: {va.get('category','neutral').title()} ({va.get('arousal_score',0.0):.0%})")))
                        etags = va.get('emotion_tags') or []
                        if etags:
                            meta_blocks.append(ui.div(
                                ui.tags.small("Emotion Tags:"),
                                ui.tags.ul(*[ui.tags.li(t) for t in etags[:6]])
                            ))
                        meta_blocks.append(ui.div(ui.tags.small(f"Empathy Bias: {es.get('focus_bias','unclear').replace('_',' ').title()}")))
                        sides = es.get('sides_described') or []
                        if sides:
                            meta_blocks.append(ui.div(
                                ui.tags.small("Sides Described:"),
                                ui.tags.ul(*[ui.tags.li(s) for s in sides[:6]])
                            ))
                        ents = es.get('entities_with_emotion')
                        if isinstance(ents, int):
                            meta_blocks.append(ui.div(ui.tags.small(f"Entities With Emotion: {ents}")))

                    section_items.append(ui.div(
                        ui.tags.strong(assertion),
                        ui.br(),
                        *meta_blocks,
                        style="margin: 12px 0; padding: 14px; background: #f8f9fa; border-radius: 6px;"
                    ))
                
                sections.append(ui.div(
                    ui.h4(f"{class_type.title()} Reality ({len(items)})"),
                    *section_items,
                    style="margin: 20px 0;"
                ))
        
        # Add disclaimer / mode hint
        # Show enrichment disclaimer if any enriched fields present; otherwise hint that deep mode provides more detail
        has_enrichment = any(
            any(
                itm.get('fact_check') or itm.get('stability_index') or itm.get('myth_taxonomy') or itm.get('viral_arousal') or itm.get('empathy_span')
                for itm in grouped[k]
            ) for k in grouped
        )
        if has_enrichment:
            sections.insert(0, ui.div(
                ui.p("Enriched metrics (fact-check, stability, arousal, empathy) are LLM-assisted. Verify sources independently.", style="font-size:0.75em; color:#666;"),
                style="margin-bottom:12px;"
            ))
        else:
            sections.insert(0, ui.div(
                ui.p("Tip: Switch to Deep mode to see fact-check, stability, myth taxonomy, arousal, and empathy details.", style="font-size:0.75em; color:#666;"),
                style="margin-bottom:12px;"
            ))
        
        # Prepend Sankey chart if available
        if sankey_html:
            sections.insert(0, ui.HTML(sankey_html))
        
        return ui.div(*sections)
    
    # Moral Foundations UI
    @output
    @render.ui
    def foundations_ui():
        results = analysis_results.get()
        if not results:
            return ui.p("No analysis available.")
        
        foundations = results["foundations"].get("foundations", {})
        
        if not foundations:
            return ui.p("No moral foundations data.")
        
        foundation_cards = []
        for name, data in foundations.items():
            if not isinstance(data, dict):
                continue
            
            triggered = data.get("triggered", False)
            intensity = data.get("intensity", 0.0)
            valence = data.get("valence", "neutral")
            explanation = data.get("explanation", "")
            triggers_list = data.get("triggers", [])
            
            if triggered:
                # Color based on valence
                if valence == "positive":
                    score_class = "high-score"  # green
                elif valence == "negative":
                    score_class = "low-score"   # grey, will override to red
                    bar_color = "#dc3545"  # red for negative
                else:
                    score_class = "low-score"  # grey for neutral
                    bar_color = "#6c757d"
                
                # Override color for negative valence
                bar_style = f"width: {intensity*100}%; background: {bar_color if valence == 'negative' else ''};"
                
                foundation_cards.append(ui.div(
                    ui.h5(name.replace("_", " ").title()),
                    ui.div(
                        ui.div(class_=f"score-bar {score_class if valence != 'negative' else ''}", style=bar_style)
                    ),
                    ui.p(f"Intensity: {intensity:.1%} | Valence: {valence.title()}"),
                    ui.p(explanation, style="font-size: 0.9em; color: #666;"),
                    ui.tags.ul(*[ui.tags.li(t) for t in triggers_list[:5]]) if triggers_list else "",
                    class_="foundation-card"
                ))
        
        return ui.div(*foundation_cards) if foundation_cards else ui.p("No significant moral foundations triggered.")
    
    # Tribal Resonance UI
    @output
    @render.ui
    def tribes_ui():
        results = analysis_results.get()
        if not results:
            return ui.p("No analysis available.")
        
        tribes = results["tribes"]
        predictions = tribes.get("predictions", [])
        
        if not predictions:
            return ui.p("No tribal predictions.")
        
        tribe_cards = []
        for tribe in predictions[:7]:  # Top 7 tribes
            name = tribe.get("name", "").replace("_", " ").title()
            score = tribe.get("resonance_score", 0.0)
            sentiment = tribe.get("sentiment", "neutral")
            reasoning = tribe.get("reasoning", "")
            hooks = tribe.get("hooks", [])
            
            sentiment_color = {"positive": "success", "negative": "danger", "neutral": "secondary"}.get(sentiment, "secondary")
            
            # Color bar based on sentiment
            if sentiment == "positive":
                bar_color = "#28a745"  # green
            elif sentiment == "negative":
                bar_color = "#dc3545"  # red
            else:
                bar_color = "#6c757d"  # grey
            
            tribe_cards.append(ui.div(
                ui.div(
                    ui.tags.strong(name),
                    ui.tags.span(f" {sentiment.title()}", class_=f"badge bg-{sentiment_color}", style="margin-left: 10px;"),
                ),
                ui.div(
                    ui.div(style=f"height: 20px; background: {bar_color}; border-radius: 4px; width: {score*100}%; transition: width 0.3s;")
                ),
                ui.p(f"Resonance: {score:.1%}", style="margin: 5px 0;"),
                ui.p(reasoning, style="font-size: 0.9em; color: #666;"),
                ui.tags.ul(*[ui.tags.li(h) for h in hooks[:3]]) if hooks else "",
                class_="tribe-card"
            ))
        
        # Add polarization info
        polarization_exp = tribes.get("polarization_explanation", "")
        if polarization_exp:
            tribe_cards.insert(0, ui.div(
                ui.h5("Polarization Analysis"),
                ui.p(polarization_exp),
                class_="foundation-card"
            ))
        
        return ui.div(*tribe_cards)
    
    # Export handlers
    @output
    @render.text
    def export_info():
        results = analysis_results.get()
        if not results:
            return "No analysis to export. Run an analysis first."
        return "Analysis ready for export."
    
    @session.download(filename="resonance_analysis.json")
    def download_json():
        import json
        results = analysis_results.get()
        if not results:
            return ""
        return json.dumps(results, indent=2)
    
    @session.download(filename="resonance_analysis.csv")
    def download_csv():
        results = analysis_results.get()
        if not results:
            return ""
        
        # Create CSV from assertions
        taxonomy = results["taxonomy"]
        assertions = taxonomy.get("assertions", [])
        
        rows = []
        for item in assertions:
            rows.append({
                "assertion": item.get("assertion", ""),
                "reality_type": item.get("classification", ""),
                "confidence": item.get("confidence", 0.0),
                "reasoning": item.get("reasoning", "")
            })
        
        df = pd.DataFrame(rows)
        return df.to_csv(index=False)


# Create app
app = App(app_ui, server)


if __name__ == "__main__":
    print("Run with: shiny run app.py")
    print("Make sure GEMINI_API_KEY is set in your .env file or environment.")
