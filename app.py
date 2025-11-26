"""
The Resonance Engine
Desktop application for analyzing media through academic frameworks.
"""
import os, sys, time, hashlib, asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from shiny import App, ui, render, reactive
import pandas as pd

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
from src.linguistic_analysis import LinguisticAnalyzer

app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.style("""
            .foundation-card { padding:15px; margin:10px 0; border-radius:8px; border-left:4px solid #007bff; background:#f8f9fa; }
            .tribe-card { padding:12px; margin:8px 0; border-radius:6px; background:#fff; border:1px solid #dee2e6; }
            .score-bar { height:20px; background:#007bff; border-radius:4px; transition:width .3s; }
            .high-score { background:#28a745; }
            .medium-score { background:#ffc107; }
            .low-score { background:#6c757d; }
            #loading-overlay { position:fixed; top:0; left:0; width:100%; height:100%; display:flex; flex-direction:column; align-items:center; justify-content:center; background:rgba(255,255,255,0.88); z-index:10000; }
            .spinner { width:54px; height:54px; border:6px solid #e0e0e0; border-top-color:#007bff; border-radius:50%; animation: spin .8s linear infinite; margin-bottom:18px; }
            @keyframes spin { to { transform: rotate(360deg); } }
            .loading-msg { font-size:1.1em; color:#333; }
        """),
        ui.tags.script(src="https://cdn.plot.ly/plotly-2.35.2.min.js")
    ),
    ui.panel_title("The Resonance Engine", "Deconstruct Persuasive Media"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.h4("Input"),
            ui.input_radio_buttons(
                "input_mode", "Content Source:",
                {"text": "Paste Text", "url": "URL/Browser Tab", "youtube": "YouTube"}, selected="text"
            ),
            ui.output_ui("input_ui"),
            ui.input_slider("max_assertions", "Max Assertions to Extract:", min=5, max=25, value=12, step=1),
            ui.input_radio_buttons("analysis_mode", "Analysis Depth:", {"quick": "Quick", "deep": "Deep"}, selected="deep"),
            ui.input_checkbox("use_cache", "Use caching (speeds repeat analyses)", True),
            ui.input_action_button("clear_cache", "Clear Cache", class_="btn-outline-secondary w-100"),
            ui.input_action_button("analyze_btn", "Analyze Content", class_="btn-primary btn-lg w-100"),
            ui.br(), ui.hr(),
            ui.tags.div(ui.output_ui("status_display"), style="margin-top:10px;"),
            width=350
        ),
        ui.navset_tab(
            ui.nav_panel("Overview", ui.div(ui.h3("Analysis Summary"), ui.output_ui("overview_ui"))),
            ui.nav_panel("Reality Taxonomy", ui.div(ui.h3("Harari's Reality Classification"), ui.p("Classification by reality type. Detailed description of the theory and categories can be found at the bottom of this page."), ui.output_ui("taxonomy_ui"))),
            ui.nav_panel("Moral Foundations", ui.div(ui.h3("Moral Foundations (Haidt)"), ui.p("Detailed description of the theory and foundations can be found at the bottom of this page."), ui.output_ui("foundations_ui"))),
            ui.nav_panel("Tribal Resonance", ui.div(ui.h3("Predicted Social Tribe Resonance"), ui.p("Detailed description of the theory and tribes can be found at the bottom of this page."), ui.output_ui("tribes_ui"))),
            ui.nav_panel("Linguistic Analysis", ui.div(ui.h3("Linguistic Forensics"), ui.p("Detailed description of the framework and dimensions can be found at the bottom of this page."), ui.output_ui("linguistic_ui"))),
            ui.nav_panel("Export", ui.div(
                ui.h3("Export Analysis"),
                ui.download_button("download_json", "Download JSON", class_="btn-success"), ui.br(), ui.br(),
                ui.download_button("download_csv", "Download CSV", class_="btn-info"), ui.br(), ui.br(),
                ui.output_text("export_info")
            ))
        )
    ),
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
                        "tribes": tribes_result,
                        "linguistic": None  # Quick mode skips linguistic (can add later if needed)
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
            linguistic_analyzer = LinguisticAnalyzer(client)

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

            progress_phase.set("Analyzing linguistic patterns...")
            await reactive.flush()
            linguistic_start = time.perf_counter()
            ling_task = asyncio.create_task(asyncio.to_thread(linguistic_analyzer.analyze_content, content))
            cancel_task = asyncio.create_task(_wait_for_cancel())
            done, pending = await asyncio.wait({ling_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED)
            if cancel_task in done and cancel_requested.get():
                _detach_task(ling_task)
                print("[ResonanceEngine] Cancelled during linguistic analysis (abandoning result)")
                canceled.set(True)
                processing.set(False)
                progress_phase.set("")
                await reactive.flush()
                return
            cancel_task.cancel()
            linguistic_result = await ling_task
            linguistic_end = time.perf_counter()

            if cancel_requested.get():
                print("[ResonanceEngine] Canceled after linguistic analysis")
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
                "tribes": tribal_result,
                "linguistic": linguistic_result
            }
            analysis_results.set(results_obj)

            total_time = time.perf_counter() - t0
            perf_obj = {
                "mode": analysis_mode,
                "ingestion": ingest_end - ingest_start,
                "taxonomy": taxonomy_end - taxonomy_start,
                "foundations": foundations_end - foundations_start,
                "tribal": tribal_end - tribal_start,
                "linguistic": linguistic_end - linguistic_start,
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
        
        taxonomy = results.get("taxonomy", {})
        foundations = results.get("foundations", {})
        tribes = results.get("tribes", {})
        linguistic = results.get("linguistic") or {}
        perf = perf_metrics.get() or {}
        content_summary = results.get("content_summary") or "(summary unavailable)"

        import json as _json

        cards = []

        # 0) Content Summary
        cards.append(ui.div(
            ui.h4("Content Summary"),
            ui.p(content_summary, style="font-size:0.95em; color:#333;"),
            class_="foundation-card"
        ))

        # 1) Reality Mix Donut
        summary = taxonomy.get("summary", {}) or {}
        total = int(summary.get("total", 0) or 0)
        obj = int(summary.get("objective", 0) or 0)
        subj = int(summary.get("subjective", 0) or 0)
        inter = int(summary.get("intersubjective", 0) or 0)
        if total > 0 and (obj + subj + inter) > 0:
            donut_spec = {
                "data": [{
                    "type": "pie",
                    "labels": ["Objective", "Subjective", "Intersubjective"],
                    "values": [obj, subj, inter],
                    "hole": 0.6,
                    "marker": {"colors": ["#007bff", "#6c757d", "#17a2b8"]},
                    "textinfo": "label+percent",
                    "hoverinfo": "label+value+percent"
                }],
                "layout": {
                    "margin": {"l": 10, "r": 10, "t": 10, "b": 10},
                    "height": 220,
                    "showlegend": False,
                    "paper_bgcolor": "#f8f9fa"
                }
            }
            donut_html = f"""
            <div id=\"overview_reality_donut\" style=\"width:100%;height:220px;\"></div>
            <script>
              (function(){{
                var spec = {_json.dumps(donut_spec)};
                if (window.Plotly && document.getElementById('overview_reality_donut')) {{
                  Plotly.newPlot('overview_reality_donut', spec.data, spec.layout, {{displayModeBar:false}});
                }}
              }})();
            </script>
            """
        else:
            donut_html = "<div style='color:#666;font-size:0.9em;'>No assertions detected.</div>"

        cards.append(ui.div(
            ui.h4("Reality Distribution"),
            ui.p(f"Total assertions: {total}", style="margin-top:-6px; color:#555;"),
            ui.HTML(donut_html),
            class_="foundation-card"
        ))

        # 2) Moral Foundations Mini Radar
        fnd = foundations.get("foundations", {}) or {}
        axes = ["care", "fairness", "loyalty", "authority", "sanctity", "liberty"]
        labels = ["Care", "Fairness", "Loyalty", "Authority", "Sanctity", "Liberty"]
        intensities = [float((fnd.get(k) or {}).get("intensity", 0.0) or 0.0) for k in axes]
        valences = [str((fnd.get(k) or {}).get("valence", "neutral") or "neutral").lower() for k in axes]
        
        # Map valence to color
        def valence_color(v):
            if v == "positive":
                return "#28a745"
            elif v == "negative":
                return "#dc3545"
            else:
                return "#6c757d"
        
        marker_colors = [valence_color(v) for v in valences]
        # Close radar loop by repeating first value
        marker_colors_closed = marker_colors + [marker_colors[0] if marker_colors else "#6c757d"]
        
        radar_spec = {
            "data": [{
                "type": "scatterpolar",
                "r": intensities + [intensities[0] if intensities else 0],
                "theta": labels + [labels[0]],
                "fill": "toself",
                "fillcolor": "rgba(108, 117, 125, 0.1)",
                "line": {"color": "#6c757d", "width": 1.5},
                "marker": {"color": marker_colors_closed, "size": 8, "line": {"width": 1, "color": "#fff"}},
                "name": "Intensity"
            }],
            "layout": {
                "polar": {"radialaxis": {"visible": True, "range": [0, 1]}},
                "margin": {"l": 40, "r": 40, "t": 20, "b": 20},
                "height": 240,
                "paper_bgcolor": "#f8f9fa",
                "showlegend": False
            }
        }
        radar_html = f"""
        <div id=\"overview_moral_radar\" style=\"width:100%;height:240px;\"></div>
        <script>
          (function(){{
            var spec = {_json.dumps(radar_spec)};
            if (window.Plotly && document.getElementById('overview_moral_radar')) {{
              Plotly.newPlot('overview_moral_radar', spec.data, spec.layout, {{displayModeBar:false}});
            }}
          }})();
        </script>
        """
        cards.append(ui.div(
            ui.h4("Moral Foundations (Intensity)"),
            ui.HTML(radar_html),
            class_="foundation-card"
        ))

        # 3) Tribal Resonance Bars (All tribes, sorted)
        preds = tribes.get("predictions", []) or []
        try:
            sorted_preds = sorted(preds, key=lambda x: x.get("resonance_score", 0.0), reverse=True)
        except Exception:
            sorted_preds = []
        tribe_names = [p.get("name", "unknown") for p in sorted_preds]
        tribe_scores = [float(p.get("resonance_score", 0.0) or 0.0) for p in sorted_preds]
        sentiments = [str(p.get("sentiment", "neutral") or "neutral").lower() for p in sorted_preds]
        colors = [("#28a745" if s == "positive" else "#6c757d" if s == "neutral" else "#dc3545") for s in sentiments]
        if sorted_preds:
            height = max(180, 40 * len(sorted_preds))
            tribe_bar = {
                "data": [{
                    "type": "bar",
                    "orientation": "h",
                    "y": tribe_names[::-1],
                    "x": tribe_scores[::-1],
                    "marker": {"color": colors[::-1]},
                    "text": [f"{v:.2f}" for v in tribe_scores[::-1]],
                    "textposition": "outside"
                }],
                "layout": {
                    "margin": {"l": 220, "r": 40, "t": 10, "b": 30},
                    "height": height,
                    "xaxis": {"title": "Resonance", "range": [0, 1]},
                    "paper_bgcolor": "#f8f9fa",
                    "showlegend": False
                }
            }
            tribe_bar_html = f"""
            <div id=\"overview_tribe_all\" style=\"width:100%;height:{height}px;\"></div>
            <script>
              (function(){{
                var spec = {_json.dumps(tribe_bar)};
                if (window.Plotly && document.getElementById('overview_tribe_all')) {{
                  Plotly.newPlot('overview_tribe_all', spec.data, spec.layout, {{displayModeBar:false}});
                }}
              }})();
            </script>
            """
        else:
            tribe_bar_html = "<div style='color:#666;font-size:0.9em;'>No tribe predictions.</div>"

        pol_risk = tribes.get("polarization_risk", "unknown")
        pol_color = 'red' if pol_risk == 'high' else 'orange' if pol_risk == 'medium' else '#28a745'

        cards.append(ui.div(
            ui.h4("Tribal Resonance (All)"),
            ui.HTML(tribe_bar_html),
            ui.p(f"Polarization Risk: {pol_risk.title()}", style=f"color:{pol_color}; margin-top:6px;"),
            class_="foundation-card"
        ))

        # 4) Linguistic Signals Strip
        # Defaults if linguistic not available (e.g., quick mode)
        agency = (linguistic.get("agency_analysis") or {}) if linguistic else {}
        passive = int(agency.get("passive_voice_percent", 0) or 0)
        polar = (linguistic.get("polarization_metrics") or {}) if linguistic else {}
        us_them = float(polar.get("us_vs_them_ratio", 0.0) or 0.0)
        cert = (linguistic.get("certainty_metrics") or {}) if linguistic else {}
        dogma = int(cert.get("dogmatism_score", 0) or 0)
        quant = (linguistic.get("quantifier_vagueness") or {}) if linguistic else {}
        vag_ratio = float(quant.get("vagueness_ratio", 0.0) or 0.0)
        vag_score = int(round(vag_ratio * 100))
        read = (linguistic.get("readability") or {}) if linguistic else {}
        grade = float(read.get("grade_level", 0.0) or 0.0)
        pers = (linguistic.get("persuasion_signature") or {}) if linguistic else {}
        density = float(pers.get("rhetorical_density_score", 0.0) or 0.0)
        classification = pers.get("classification", "Unknown")
        valence = float(pers.get("net_valence_score", 0.0) or 0.0)

        def color_gauge(val, green_thresh, yellow_thresh):
            # return color based on thresholds (lower is better)
            return "#28a745" if val < green_thresh else ("#ffc107" if val < yellow_thresh else "#dc3545")

        dogma_color = color_gauge(dogma, 40, 70)
        other_color = color_gauge(us_them, 1.0, 2.0)
        vag_color = color_gauge(vag_score, 30, 70)
        passive_color = color_gauge(passive, 20, 40)
        density_color = color_gauge(density, 10, 25)
        val_color = "#28a745" if valence > 0.1 else ("#6c757d" if abs(valence) <= 0.1 else "#dc3545")

        badge_css = "display:inline-block; min-width:120px; padding:8px 10px; margin:4px; border-radius:8px; background:#eef3f7;"
        label_css = "display:block; font-size:0.75em; color:#555;"
        value_css = "font-weight:700; font-size:1.05em;"

        metric_strip = ui.div(
            ui.h4("Linguistic Signals"),
            ui.div(
                # Dogmatism
                ui.div(
                    ui.span("Dogmatism", style=label_css),
                    ui.span(str(dogma), style=f"{value_css} color:{dogma_color};"),
                    style=badge_css
                ),
                # Othering
                ui.div(
                    ui.span("Othering Ratio", style=label_css),
                    ui.span(f"{us_them:.2f}", style=f"{value_css} color:{other_color};"),
                    style=badge_css
                ),
                # Vagueness
                ui.div(
                    ui.span("Vagueness Score", style=label_css),
                    ui.span(f"{vag_score}", style=f"{value_css} color:{vag_color};"),
                    style=badge_css
                ),
                # Passive Voice
                ui.div(
                    ui.span("Passive Voice %", style=label_css),
                    ui.span(str(passive), style=f"{value_css} color:{passive_color};"),
                    style=badge_css
                ),
                # Readability
                ui.div(
                    ui.span("Grade Level", style=label_css),
                    ui.span(f"{grade:.1f}", style=f"{value_css} color:#007bff;"),
                    style=badge_css
                ),
                # Rhetorical Density
                ui.div(
                    ui.span("Devices /1k words", style=label_css),
                    ui.span(f"{density:.1f}", style=f"{value_css} color:{density_color};"),
                    style=badge_css
                ),
                style="display:flex; flex-wrap:wrap; align-items:center;"
            ),
            class_="foundation-card"
        )
        cards.append(metric_strip)

        # 5) Rhetorical Signature Badge
        signature_card = ui.div(
            ui.h4("Rhetorical Signature"),
            ui.p(
                ui.tags.span(
                    classification,
                    style=f"font-weight:700; padding:4px 8px; border-radius:6px; background:#eef3f7; color:#333; margin-right:8px;"
                ),
                ui.tags.span(
                    f"Net Valence: {valence:+.2f}",
                    style=f"font-weight:700; padding:4px 8px; border-radius:6px; background:#eef3f7; color:{val_color};"
                ),
                style="font-size:0.95em;"
            ),
            class_="foundation-card"
        )
        cards.append(signature_card)

        # 6) Performance (unchanged)
        if perf:
            cards.append(ui.div(
                ui.h4("Performance"),
                ui.tags.ul(
                    ui.tags.li(f"Mode: {perf.get('mode','')}"),
                    ui.tags.li(f"Total: {perf.get('total',0.0):.1f}s"),
                    ui.tags.li(f"Ingestion: {perf.get('ingestion',0.0):.2f}s"),
                    ui.tags.li(f"Taxonomy: {perf.get('taxonomy', perf.get('combined',0.0)):.2f}s"),
                    ui.tags.li(f"Foundations: {perf.get('foundations',0.0):.2f}s"),
                    ui.tags.li(f"Tribal: {perf.get('tribal',0.0):.2f}s") if perf.get('tribal') is not None else "",
                    ui.tags.li(f"Linguistic: {perf.get('linguistic',0.0):.2f}s") if perf.get('linguistic') is not None else ""
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
        # Collect example sentences for each node label to use in tooltips
        node_examples = {}
        def add_example(label: str, text: str):
            if not text:
                return
            arr = node_examples.setdefault(label, [])
            # keep a few short examples, avoid duplicates
            if text not in arr:
                arr.append(text)
                # cap examples per node to 3
                if len(arr) > 3:
                    del arr[0]

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
                child_label = f"Fact: {k.replace('_',' ').title()}"
                child = idx(child_label)
                sources.append(n_obj)
                targets.append(child)
                values.append(v)
        # Attach examples for objective children
        for a in assertions:
            if a.get("classification") == "objective":
                fc = (a.get("fact_check") or {}).get("status")
                if fc in fact_counts:
                    add_example(f"Fact: {fc.replace('_',' ').title()}", a.get("assertion", ""))

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
                    m_label = f"Myth: {m.replace('_',' ').title()}"
                    m_node = idx(m_label)
                    sources.append(s_node)
                    targets.append(m_node)
                    values.append(c)
        # Attach examples for intersubjective children
        for a in assertions:
            if a.get("classification") == "intersubjective":
                si = (a.get("stability_index") or {}).get("status")
                mt = (a.get("myth_taxonomy") or {}).get("category")
                if si in stability_keys:
                    add_example(f"Stability: {si.title()}", a.get("assertion", ""))
                if mt in myth_keys:
                    add_example(f"Myth: {mt.replace('_',' ').title()}", a.get("assertion", ""))

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
                    e_label = f"Empathy: {e_key.replace('_',' ').title()}"
                    e_node = idx(e_label)
                    sources.append(a_node)
                    targets.append(e_node)
                    values.append(c)
        # Attach examples for subjective children
        for a in assertions:
            if a.get("classification") == "subjective":
                va = (a.get("viral_arousal") or {}).get("category")
                es = (a.get("empathy_span") or {}).get("focus_bias")
                if va in arousal_keys:
                    add_example(f"Arousal: {va.title()}", a.get("assertion", ""))
                if es in empathy_keys:
                    add_example(f"Empathy: {es.replace('_',' ').title()}", a.get("assertion", ""))

        # Build two Sunburst charts side-by-side
        sunburst_html_1 = ""
        sunburst_html_2 = ""
        if values:
            # Chart 1: original (Arousal & Stability outer layer)
            sb_labels = ["Reality"]
            sb_parents = [""]
            sb_values = [sum(values)]
            sb_customdata = ["\n".join(node_examples.get("Reality", [])) or "Overall taxonomy"]

            def push(label, parent, value, examples_key=None):
                sb_labels.append(label)
                sb_parents.append(parent)
                sb_values.append(value)
                # Custom tooltip logic
                if parent == "Objective":
                    # Show actual facts for fact nodes
                    facts = []
                    for a in assertions:
                        if a.get("classification") == "objective":
                            fc = (a.get("fact_check") or {}).get("status")
                            fact_label = f"Fact: {fc.replace('_',' ').title()}" if fc else ""
                            if fact_label == label:
                                assertion_text = a.get("assertion", "")
                                if assertion_text:
                                    facts.append(assertion_text)
                    sb_customdata.append("\n\n".join(facts) if facts else f"Count: {value}")
                elif label.startswith("Empathy"):
                    # Show each assertion with empathy bias details
                    items = []
                    for a in assertions:
                        if a.get("classification") == "subjective":
                            es = (a.get("empathy_span") or {})
                            if es.get("focus_bias") and label.endswith(es.get("focus_bias").replace('_',' ').title()):
                                assertion_text = a.get("assertion", "")
                                sides = es.get("sides_described", [])
                                entities = es.get("entities_with_emotion")
                                item_parts = [assertion_text]
                                if sides:
                                    item_parts.append(f"  Sides: {', '.join(sides)}")
                                if entities:
                                    item_parts.append(f"  Entities: {entities}")
                                items.append("\n".join(item_parts))
                    sb_customdata.append("\n\n".join(items) if items else label)
                elif label.startswith("Arousal"):
                    # Show each assertion with arousal details
                    items = []
                    for a in assertions:
                        if a.get("classification") == "subjective":
                            va = (a.get("viral_arousal") or {})
                            if va.get("category") and label.endswith(va.get("category").title()):
                                assertion_text = a.get("assertion", "")
                                score = va.get("arousal_score")
                                tags = va.get("emotion_tags", [])
                                item_parts = [assertion_text]
                                if score is not None:
                                    item_parts.append(f"  Arousal Score: {int(score*100)}%")
                                if tags:
                                    item_parts.append(f"  Emotion Tags: {', '.join(tags)}")
                                items.append("\n".join(item_parts))
                    sb_customdata.append("\n\n".join(items) if items else label)
                elif label.startswith("Stability"):
                    # Show each assertion with stability details
                    items = []
                    for a in assertions:
                        if a.get("classification") == "intersubjective":
                            si = (a.get("stability_index") or {})
                            if si.get("status") and label.endswith(si.get("status").title()):
                                assertion_text = a.get("assertion", "")
                                status = si.get("status")
                                cues = si.get("cues", [])
                                reasoning = si.get("reasoning")
                                item_parts = [assertion_text]
                                if status:
                                    item_parts.append(f"  Status: {status}")
                                if cues:
                                    item_parts.append(f"  Cues: {', '.join(cues)}")
                                if reasoning:
                                    item_parts.append(f"  Reasoning: {reasoning}")
                                items.append("\n".join(item_parts))
                    sb_customdata.append("\n\n".join(items) if items else label)
                elif label.startswith("Myth"):
                    # Show each assertion with myth details
                    items = []
                    for a in assertions:
                        if a.get("classification") == "intersubjective":
                            mt = (a.get("myth_taxonomy") or {})
                            if mt.get("category") and label.endswith(mt.get("category").replace('_',' ').title()):
                                assertion_text = a.get("assertion", "")
                                category = mt.get("category")
                                confidence = mt.get("confidence")
                                reasoning = mt.get("reasoning")
                                item_parts = [assertion_text]
                                if category:
                                    item_parts.append(f"  Category: {category}")
                                if confidence is not None:
                                    item_parts.append(f"  Confidence: {int(confidence*100)}%")
                                if reasoning:
                                    item_parts.append(f"  Reasoning: {reasoning}")
                                items.append("\n".join(item_parts))
                    sb_customdata.append("\n\n".join(items) if items else label)
                else:
                    examples = node_examples.get(examples_key or label, [])
                    sb_customdata.append("\n".join(examples) or label)

            # Classification counts
            class_counts = {"Objective": 0, "Intersubjective": 0, "Subjective": 0}
            for a in assertions:
                c = a.get("classification")
                if c == "objective":
                    class_counts["Objective"] += 1
                elif c == "intersubjective":
                    class_counts["Intersubjective"] += 1
                elif c == "subjective":
                    class_counts["Subjective"] += 1
            for label, cnt in class_counts.items():
                if cnt > 0:
                    push(label, "Reality", cnt, label)

            # Objective children
            for k, v in fact_counts.items():
                if v > 0:
                    push(f"Fact: {k.replace('_',' ').title()}", "Objective", v)

            # Intersubjective children
            for st, count in st_totals.items():
                if count > 0:
                    st_label = f"Stability: {st.title()}"
                    push(st_label, "Intersubjective", count)
                    for m, c in joint_st_myth[st].items():
                        if c > 0:
                            push(f"Myth: {m.replace('_',' ').title()}", st_label, c)

            # Subjective children
            for a_key, count in ar_totals.items():
                if count > 0:
                    a_label = f"Arousal: {a_key.title()}"
                    push(a_label, "Subjective", count)
                    for e_key, c in joint_ar_emp[a_key].items():
                        if c > 0:
                            push(f"Empathy: {e_key.replace('_',' ').title()}", a_label, c)

            sunburst_data_1 = {
                "type": "sunburst",
                "labels": sb_labels,
                "parents": sb_parents,
                "values": sb_values,
                "customdata": sb_customdata,
                "branchvalues": "total",
                "marker": {"line": {"width": 1}},
                "hovertemplate": "<b>%{label}</b><br>%{customdata}<extra></extra>",
                "maxdepth": 3
            }
            layout_1 = {
                "margin": {"l": 0, "r": 0, "t": 30, "b": 0},
                "paper_bgcolor": "#ffffff",
                "height": 540
            }
            fig_json_1 = _json.dumps({"data": [sunburst_data_1], "layout": layout_1})
            sunburst_html_1 = f"""
            <div id=\"sunburst_rt\" style=\"width:100%;height:540px;margin-bottom:24px;border:1px solid #e0e0e0;border-radius:4px;\"></div>
            <script>
              (function(){{
                var spec = {fig_json_1};
                if (window.Plotly && document.getElementById('sunburst_rt')) {{
                  Plotly.newPlot('sunburst_rt', spec.data, spec.layout, {{displayModeBar:false}});
                  document.getElementById('sunburst_rt').on('plotly_click', function(e) {{
                    var pt = e.points[0];
                    var label = pt.label;
                    var details = pt.customdata;
                    var modal = document.getElementById('sunburst_modal');
                    var modal_content = document.getElementById('sunburst_modal_content');
                    modal_content.innerHTML = '<h4>' + label + '</h4><pre style="white-space:pre-wrap;font-size:1em;">' + details + '</pre>';
                    modal.style.display = 'block';
                  }});
                }}
              }})();
            </script>
            """

            # Chart 2: empathy bias & myth outer layer
            sb2_labels = ["Reality"]
            sb2_parents = [""]
            sb2_values = [sum(values)]
            sb2_customdata = ["\n".join(node_examples.get("Reality", [])) or "Overall taxonomy"]

            def push2(label, parent, value, examples_key=None):
                sb2_labels.append(label)
                sb2_parents.append(parent)
                sb2_values.append(value)
                # Custom tooltip logic for second chart
                if parent == "Objective":
                    # Show actual facts for fact nodes
                    facts = []
                    for a in assertions:
                        if a.get("classification") == "objective":
                            fc = (a.get("fact_check") or {}).get("status")
                            fact_label = f"Fact: {fc.replace('_',' ').title()}" if fc else ""
                            if fact_label == label:
                                assertion_text = a.get("assertion", "")
                                if assertion_text:
                                    facts.append(assertion_text)
                    sb2_customdata.append("\n\n".join(facts) if facts else f"Count: {value}")
                elif label.startswith("Empathy Bias"):
                    # Show each assertion with empathy bias details
                    items = []
                    for a in assertions:
                        if a.get("classification") == "subjective":
                            es = (a.get("empathy_span") or {})
                            if es.get("focus_bias") and label.endswith(es.get("focus_bias").replace('_',' ').title()):
                                assertion_text = a.get("assertion", "")
                                sides = es.get("sides_described", [])
                                entities = es.get("entities_with_emotion")
                                item_parts = [assertion_text]
                                if sides:
                                    item_parts.append(f"  Sides: {', '.join(sides)}")
                                if entities:
                                    item_parts.append(f"  Entities: {entities}")
                                items.append("\n".join(item_parts))
                    sb2_customdata.append("\n\n".join(items) if items else label)
                elif label.startswith("Arousal"):
                    # Show each assertion with arousal details
                    items = []
                    for a in assertions:
                        if a.get("classification") == "subjective":
                            va = (a.get("viral_arousal") or {})
                            if va.get("category") and label.endswith(va.get("category").title()):
                                assertion_text = a.get("assertion", "")
                                score = va.get("arousal_score")
                                tags = va.get("emotion_tags", [])
                                item_parts = [assertion_text]
                                if score is not None:
                                    item_parts.append(f"  Arousal Score: {int(score*100)}%")
                                if tags:
                                    item_parts.append(f"  Emotion Tags: {', '.join(tags)}")
                                items.append("\n".join(item_parts))
                    sb2_customdata.append("\n\n".join(items) if items else label)
                elif label.startswith("Stability"):
                    # Show each assertion with stability details
                    items = []
                    for a in assertions:
                        if a.get("classification") == "intersubjective":
                            si = (a.get("stability_index") or {})
                            if si.get("status") and label.endswith(si.get("status").title()):
                                assertion_text = a.get("assertion", "")
                                status = si.get("status")
                                cues = si.get("cues", [])
                                reasoning = si.get("reasoning")
                                item_parts = [assertion_text]
                                if status:
                                    item_parts.append(f"  Status: {status}")
                                if cues:
                                    item_parts.append(f"  Cues: {', '.join(cues)}")
                                if reasoning:
                                    item_parts.append(f"  Reasoning: {reasoning}")
                                items.append("\n".join(item_parts))
                    sb2_customdata.append("\n\n".join(items) if items else label)
                elif label.startswith("Myth"):
                    # Show each assertion with myth details
                    items = []
                    for a in assertions:
                        if a.get("classification") == "intersubjective":
                            mt = (a.get("myth_taxonomy") or {})
                            if mt.get("category") and label.endswith(mt.get("category").replace('_',' ').title()):
                                assertion_text = a.get("assertion", "")
                                category = mt.get("category")
                                confidence = mt.get("confidence")
                                reasoning = mt.get("reasoning")
                                item_parts = [assertion_text]
                                if category:
                                    item_parts.append(f"  Category: {category}")
                                if confidence is not None:
                                    item_parts.append(f"  Confidence: {int(confidence*100)}%")
                                if reasoning:
                                    item_parts.append(f"  Reasoning: {reasoning}")
                                items.append("\n".join(item_parts))
                    sb2_customdata.append("\n\n".join(items) if items else label)
                else:
                    examples = node_examples.get(examples_key or label, [])
                    sb2_customdata.append("\n".join(examples) or label)

            # Classification counts
            for label, cnt in class_counts.items():
                if cnt > 0:
                    push2(label, "Reality", cnt, label)

            # Objective children (repeat same layers or blank)
            for k, v in fact_counts.items():
                if v > 0:
                    push2(f"Fact: {k.replace('_',' ').title()}", "Objective", v)

            # Intersubjective children (myth outer layer)
            # Reality → Intersubjective → Myth (outer) → Stability (inner)
            myth_totals = {m: 0 for m in myth_keys}
            joint_myth_st = {m: {s: 0 for s in stability_keys} for m in myth_keys}
            for a in assertions:
                if a.get("classification") == "intersubjective":
                    si = (a.get("stability_index") or {}).get("status")
                    mt = (a.get("myth_taxonomy") or {}).get("category")
                    if mt in myth_keys:
                        myth_totals[mt] += 1
                        if si in stability_keys:
                            joint_myth_st[mt][si] += 1
            myth_nodes = {}
            for m, count in myth_totals.items():
                if count > 0:
                    m_label = f"Myth: {m.replace('_',' ').title()}"
                    myth_node = m_label
                    push2(m_label, "Intersubjective", count)
                    myth_nodes[m] = myth_node
                    for s, c in joint_myth_st[m].items():
                        if c > 0:
                            s_label = f"Stability: {s.title()}"
                            push2(s_label, m_label, c)

            # Subjective children (empathy bias outer layer)
            # Reality → Subjective → Empathy Bias (outer) → Arousal (inner)
            emp_totals = {e: 0 for e in empathy_keys}
            joint_emp_ar = {e: {a: 0 for a in arousal_keys} for e in empathy_keys}
            for a in assertions:
                if a.get("classification") == "subjective":
                    va = (a.get("viral_arousal") or {}).get("category")
                    es = (a.get("empathy_span") or {}).get("focus_bias")
                    if es in empathy_keys:
                        emp_totals[es] += 1
                        if va in arousal_keys:
                            joint_emp_ar[es][va] += 1
            emp_nodes = {}
            for e, count in emp_totals.items():
                if count > 0:
                    e_label = f"Empathy Bias: {e.replace('_',' ').title()}"
                    emp_node = e_label
                    push2(e_label, "Subjective", count)
                    emp_nodes[e] = emp_node
                    for a, c in joint_emp_ar[e].items():
                        if c > 0:
                            a_label = f"Arousal: {a.title()}"
                            push2(a_label, e_label, c)

            sunburst_data_2 = {
                "type": "sunburst",
                "labels": sb2_labels,
                "parents": sb2_parents,
                "values": sb2_values,
                "customdata": sb2_customdata,
                "branchvalues": "total",
                "marker": {"line": {"width": 1}},
                "hovertemplate": "<b>%{label}</b><br>%{customdata}<extra></extra>",
                "maxdepth": 3
            }
            layout_2 = {
                "margin": {"l": 0, "r": 0, "t": 30, "b": 0},
                "paper_bgcolor": "#ffffff",
                "height": 540
            }
            fig_json_2 = _json.dumps({"data": [sunburst_data_2], "layout": layout_2})
            sunburst_html_2 = f"""
            <div id=\"sunburst_rt2\" style=\"width:100%;height:540px;margin-bottom:24px;border:1px solid #e0e0e0;border-radius:4px;\"></div>
            <script>
              (function(){{
                var spec = {fig_json_2};
                if (window.Plotly && document.getElementById('sunburst_rt2')) {{
                  Plotly.newPlot('sunburst_rt2', spec.data, spec.layout, {{displayModeBar:false}});
                  document.getElementById('sunburst_rt2').on('plotly_click', function(e) {{
                    var pt = e.points[0];
                    var label = pt.label;
                    var details = pt.customdata;
                    var modal = document.getElementById('sunburst_modal');
                    var modal_content = document.getElementById('sunburst_modal_content');
                    modal_content.innerHTML = '<h4>' + label + '</h4><pre style="white-space:pre-wrap;font-size:1em;">' + details + '</pre>';
                    modal.style.display = 'block';
                  }});
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
        
        # Prepend both Sunburst charts side-by-side if available
        if sunburst_html_1 and sunburst_html_2:
            chart_row = ui.HTML(f"""
            <div style='display:flex;flex-direction:row;gap:24px;'>
              <div style='flex:1'>{sunburst_html_1}</div>
              <div style='flex:1'>{sunburst_html_2}</div>
            </div>
            <div id='sunburst_modal' onclick="if(event.target.id==='sunburst_modal'){{this.style.display='none'}}" style='display:none;position:fixed;z-index:9999;left:0;top:0;width:100vw;height:100vh;background:rgba(0,0,0,0.4);cursor:pointer;'>
              <div id='sunburst_modal_content' style='background:#fff;max-width:600px;max-height:80vh;overflow-y:auto;margin:80px auto;padding:32px;border-radius:8px;box-shadow:0 2px 16px #333;position:relative;cursor:default;'>
                <button onclick="document.getElementById('sunburst_modal').style.display='none';event.stopPropagation();" style='position:absolute;top:12px;right:16px;font-size:1.5em;background:none;border:none;cursor:pointer;color:#666;line-height:1;'>×</button>
              </div>
            </div>
            """)
            sections.insert(0, chart_row)
        elif sunburst_html_1:
            chart_row = sunburst_html_1 + """
            <div id='sunburst_modal' onclick="if(event.target.id==='sunburst_modal'){{this.style.display='none'}}" style='display:none;position:fixed;z-index:9999;left:0;top:0;width:100vw;height:100vh;background:rgba(0,0,0,0.4);cursor:pointer;'>
              <div id='sunburst_modal_content' style='background:#fff;max-width:600px;max-height:80vh;overflow-y:auto;margin:80px auto;padding:32px;border-radius:8px;box-shadow:0 2px 16px #333;position:relative;cursor:default;'>
                <button onclick="document.getElementById('sunburst_modal').style.display='none';event.stopPropagation();" style='position:absolute;top:12px;right:16px;font-size:1.5em;background:none;border:none;cursor:pointer;color:#666;line-height:1;'>×</button>
              </div>
            </div>
            """
            sections.insert(0, ui.HTML(chart_row))
        
        # Add framework description at the bottom
        framework_description = ui.div(
            ui.tags.hr(style="margin: 40px 0 30px 0; border-top: 2px solid #ccc;"),
            ui.h3("Harari's Reality Classification Framework", style="margin-top: 30px;"),
            
            ui.h4("Core Principles", style="margin-top: 20px;"),
            ui.p("Yuval Noah Harari's framework categorizes assertions about reality into three fundamental tiers based on their dependence on human consciousness and agreement:"),
            
            ui.tags.strong("1. Objective Reality"),
            ui.p("Exists independently of human beliefs or consciousness. Facts remain true regardless of what anyone thinks. Example: \"The Earth orbits the Sun\" is objectively verifiable through physical observation."),
            
            ui.tags.strong("2. Subjective Reality"),
            ui.p("Exists within individual consciousness and varies from person to person. These are personal feelings, sensations, and emotional experiences. Example: \"I feel anxious about climate change\" reflects an individual's internal state."),
            
            ui.tags.strong("3. Intersubjective Reality"),
            ui.p("Exists in the shared imagination and collective beliefs of groups. These \"social facts\" only exist because many people believe in them and act accordingly. Example: \"Money has value\" is true only because we collectively agree it is."),
            
            ui.tags.hr(style="margin: 30px 0;"),
            
            ui.h4("Objective Reality Sub-Categories"),
            ui.tags.strong("Fact-Check Status"),
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("Verified:"), " Claims supported by strong empirical evidence and expert consensus"),
                ui.tags.li(ui.tags.strong("Disputed:"), " Claims with conflicting evidence or significant expert disagreement"),
                ui.tags.li(ui.tags.strong("Unclear:"), " Claims lacking sufficient evidence for verification or refutation"),
                ui.tags.li(ui.tags.strong("Partially Verified:"), " Claims with some supporting evidence but important caveats or limitations")
            ),
            ui.p(ui.tags.em("Indicates: The empirical reliability and evidential support for factual claims"), style="font-style: italic; color: #555;"),
            
            ui.tags.hr(style="margin: 30px 0;"),
            
            ui.h4("Subjective Reality Sub-Categories"),
            ui.h5("Viral Arousal (Emotional Intensity)", style="margin-top: 15px;"),
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("High:"), " Content evoking strong emotions (anger, fear, outrage, excitement)"),
                ui.tags.li(ui.tags.strong("Neutral:"), " Content with minimal emotional charge"),
                ui.tags.li(ui.tags.strong("Low:"), " Content evoking mild or subdued emotional response")
            ),
            ui.p(ui.tags.em("Indicates: The emotional energy and potential for content to spread through emotional contagion"), style="font-style: italic; color: #555;"),
            
            ui.h5("Empathy Span (Perspective Bias)", style="margin-top: 15px;"),
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("One-sided:"), " Narrative focuses on emotions/experiences of a single group or perspective"),
                ui.tags.li(ui.tags.strong("Balanced:"), " Narrative acknowledges multiple perspectives with emotional consideration"),
                ui.tags.li(ui.tags.strong("Unclear:"), " Difficult to determine whose experiences are centered")
            ),
            ui.p(ui.tags.em("Indicates: Which groups' subjective experiences are represented and validated in the narrative"), style="font-style: italic; color: #555;"),
            
            ui.tags.hr(style="margin: 30px 0;"),
            
            ui.h4("Intersubjective Reality Sub-Categories"),
            ui.h5("Stability Index (Social Consensus)", style="margin-top: 15px;"),
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("Naturalized:"), " Beliefs so deeply embedded they appear as objective facts; invisible and unquestioned"),
                ui.tags.li(ui.tags.strong("Contested:"), " Beliefs under active challenge with competing narratives and power struggles"),
                ui.tags.li(ui.tags.strong("Ambiguous:"), " Beliefs in flux with unclear boundaries or mixed acceptance")
            ),
            ui.p(ui.tags.em("Indicates: How firmly a shared belief is anchored in collective consciousness and its resistance to change"), style="font-style: italic; color: #555;"),
            
            ui.h5("Myth Taxonomy (Social Institution Type)", style="margin-top: 15px;"),
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("Tribal/National:"), " Identity-based narratives (ethnicity, nationality, heritage)"),
                ui.tags.li(ui.tags.strong("Legal/Bureaucratic:"), " Rules, laws, rights, and organizational structures"),
                ui.tags.li(ui.tags.strong("Economic:"), " Money, markets, corporations, and financial systems"),
                ui.tags.li(ui.tags.strong("Divine/Ideological:"), " Religious beliefs, political ideologies, and moral frameworks"),
                ui.tags.li(ui.tags.strong("Other:"), " Myths not fitting primary categories")
            ),
            ui.p(ui.tags.em("Indicates: Which type of imagined order gives structure and meaning to the shared reality"), style="font-style: italic; color: #555;"),
            
            ui.tags.hr(style="margin: 30px 0;"),
            
            ui.h4("Key Insight"),
            ui.p("The framework reveals that much of what humans consider \"real\" exists only through collective agreement. While objective facts can be empirically tested, intersubjective realities—despite being \"imagined\"—have enormous power to shape behavior, institutions, and history. Understanding which tier a claim belongs to helps evaluate evidence standards, recognize whose perspectives are centered, and identify the social forces maintaining or challenging shared beliefs."),
            
            style="margin: 30px 0; padding: 30px; background: #f9f9f9; border-radius: 8px; border: 1px solid #ddd;"
        )
        
        sections.append(framework_description)
        
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
        
        
        theory_description = ui.div(
            ui.tags.hr(style="margin: 40px 0 30px 0; border-top: 2px solid #ccc;"),
            ui.h3("Moral Foundations Theory (Jonathan Haidt)", style="margin-top: 30px;"),
            
            ui.h4("Core Principles", style="margin-top: 20px;"),
            ui.p("Moral Foundations Theory, developed by psychologist Jonathan Haidt and colleagues, proposes that human moral reasoning is built upon six innate psychological foundations. These foundations are universal across cultures but are prioritized differently by individuals and groups, explaining diverse moral perspectives and political ideologies."),
            ui.p(ui.tags.strong("Key Insight:"), " Morality is not purely rational but rooted in intuitive emotional responses. Different people and cultures \"build\" different moral systems by emphasizing different foundations, like cooks using the same ingredients in different proportions."),
            
            ui.tags.hr(style="margin: 30px 0;"),
            
            ui.h4("The Six Moral Foundations"),
            
            ui.h5("1. Care/Harm", style="margin-top: 20px;"),
            ui.p(ui.tags.strong("Core Concern:"), " Protection from cruelty and suffering; nurturing and compassion"),
            ui.p(ui.tags.strong("Triggered by:"), " Violence, suffering, or distress; vulnerable individuals (children, animals, victims); stories of pain or protection"),
            ui.p(ui.tags.strong("Valence:")),
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("Positive:"), " Empathy, kindness, caregiving, mercy"),
                ui.tags.li(ui.tags.strong("Negative:"), " Cruelty, neglect, harm, indifference to suffering")
            ),
            ui.p(ui.tags.em("Political Pattern: Emphasized strongly by liberals/progressives; moderately by conservatives"), style="font-style: italic; color: #555;"),
            
            ui.h5("2. Fairness/Cheating", style="margin-top: 20px;"),
            ui.p(ui.tags.strong("Core Concern:"), " Justice, reciprocity, and proportional treatment"),
            ui.p(ui.tags.strong("Triggered by:"), " Unequal treatment or outcomes; rule-breaking, exploitation, or free-riding; issues of rights, justice, and equality"),
            ui.p(ui.tags.strong("Valence:")),
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("Positive:"), " Fair dealing, reciprocity, justice, equity"),
                ui.tags.li(ui.tags.strong("Negative:"), " Cheating, exploitation, injustice, unfair advantage")
            ),
            ui.p(ui.tags.em("Political Pattern: Valued across the spectrum but interpreted differently—progressives emphasize equality; conservatives emphasize proportionality"), style="font-style: italic; color: #555;"),
            
            ui.h5("3. Loyalty/Betrayal", style="margin-top: 20px;"),
            ui.p(ui.tags.strong("Core Concern:"), " Group cohesion, patriotism, and solidarity"),
            ui.p(ui.tags.strong("Triggered by:"), " Threats to the in-group; acts of loyalty or betrayal; national, tribal, or team identity"),
            ui.p(ui.tags.strong("Valence:")),
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("Positive:"), " Patriotism, solidarity, team spirit, sacrifice for the group"),
                ui.tags.li(ui.tags.strong("Negative:"), " Betrayal, disloyalty, treason, selfish individualism")
            ),
            ui.p(ui.tags.em("Political Pattern: Emphasized more by conservatives; less central to liberal morality"), style="font-style: italic; color: #555;"),
            
            ui.h5("4. Authority/Subversion", style="margin-top: 20px;"),
            ui.p(ui.tags.strong("Core Concern:"), " Respect for hierarchy, tradition, and legitimate leadership"),
            ui.p(ui.tags.strong("Triggered by:"), " Challenges to authority or tradition; issues of obedience, respect, and social order; hierarchical relationships"),
            ui.p(ui.tags.strong("Valence:")),
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("Positive:"), " Respect, deference, obedience, duty, order"),
                ui.tags.li(ui.tags.strong("Negative:"), " Disobedience, disrespect, rebellion, chaos")
            ),
            ui.p(ui.tags.em("Political Pattern: Emphasized more by conservatives; progressives prioritize equality over hierarchy"), style="font-style: italic; color: #555;"),
            
            ui.h5("5. Sanctity/Degradation", style="margin-top: 20px;"),
            ui.p(ui.tags.strong("Core Concern:"), " Purity, sacredness, and elevation vs. contamination and disgust"),
            ui.p(ui.tags.strong("Triggered by:"), " Violations of sacred values or bodily purity; disgust reactions (physical or moral); religious or spiritual themes"),
            ui.p(ui.tags.strong("Valence:")),
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("Positive:"), " Purity, sanctity, temperance, cleanliness, divine connection"),
                ui.tags.li(ui.tags.strong("Negative:"), " Degradation, contamination, desecration, disgust")
            ),
            ui.p(ui.tags.em("Political Pattern: Strongly emphasized by religious conservatives; less salient for secular liberals (though progressives may apply it to environmental or bodily autonomy issues)"), style="font-style: italic; color: #555;"),
            
            ui.h5("6. Liberty/Oppression", style="margin-top: 20px;"),
            ui.p(ui.tags.strong("Core Concern:"), " Freedom from domination and tyranny"),
            ui.p(ui.tags.strong("Triggered by:"), " Bullying, authoritarianism, or illegitimate control; restrictions on autonomy or expression; power imbalances and coercion"),
            ui.p(ui.tags.strong("Valence:")),
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("Positive:"), " Freedom, autonomy, resistance to tyranny"),
                ui.tags.li(ui.tags.strong("Negative:"), " Oppression, domination, bullying, coercion")
            ),
            ui.p(ui.tags.em("Political Pattern: Valued across the spectrum but differently—progressives resist corporate/hierarchical power; libertarians/conservatives resist government overreach"), style="font-style: italic; color: #555;"),
            
            ui.tags.hr(style="margin: 30px 0;"),
            
            ui.h4("Understanding Moral Profiles"),
            ui.p(ui.tags.strong("Intensity:"), " How strongly a foundation is triggered (0.0 = not present, 1.0 = maximum)"),
            ui.p(ui.tags.strong("Valence:"), " Whether the content appeals to the positive virtue (care, fairness, loyalty) or condemns its violation (harm, cheating, betrayal)"),
            ui.p(ui.tags.strong("Moral Signature:"), " The overall pattern of which foundations are activated reveals the content's moral \"flavor\" and predicts which audiences will find it compelling or repellent."),
            ui.p(ui.tags.strong("Application:"), " This framework helps explain why people with different moral profiles talk past each other—they're literally speaking different moral languages, prioritizing different foundations as most important."),
            
            style="margin: 30px 0; padding: 30px; background: #f9f9f9; border-radius: 8px; border: 1px solid #ddd;"
        )
        
        result_content = ui.div(*foundation_cards) if foundation_cards else ui.p("No significant moral foundations triggered.")
        
        return ui.div(result_content, theory_description)
    
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
        
        
        theory_description = ui.div(
            ui.tags.hr(style="margin: 40px 0 30px 0; border-top: 2px solid #ccc;"),
            ui.h3("Tribal Resonance Theory", style="margin-top: 30px;"),
            
            ui.h4("Core Principles", style="margin-top: 20px;"),
            ui.p("Tribal Resonance Theory examines how content appeals to specific social identity groups—\"tribes\"—based on shared values, worldviews, and cultural narratives. This framework recognizes that humans are tribal creatures who form strong in-group identities around political ideologies, cultural values, economic philosophies, and social movements."),
            ui.p(ui.tags.strong("Key Insight:"), " Content doesn't just convey information; it signals tribal membership and values. Messages that resonate with a tribe's core beliefs, grievances, and aspirations will be amplified within that community, while messages threatening the tribe's identity will be rejected or attacked."),
            
            ui.tags.hr(style="margin: 30px 0;"),
            
            ui.h4("Understanding Tribal Resonance"),
            
            ui.h5("Resonance Score", style="margin-top: 15px;"),
            ui.p(ui.tags.strong("Definition:"), " A measure (0.0 to 1.0) of how strongly content aligns with and appeals to a specific tribe's worldview, values, and emotional triggers."),
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("High Resonance (0.7+):"), " Content deeply validates the tribe's beliefs, uses their language, addresses their concerns, and reinforces their identity. Likely to be shared enthusiastically within the tribe."),
                ui.tags.li(ui.tags.strong("Medium Resonance (0.4-0.7):"), " Content touches on tribal themes but may have mixed messages or lack emotional intensity."),
                ui.tags.li(ui.tags.strong("Low Resonance (0.0-0.4):"), " Content is neutral, irrelevant, or misaligned with tribal values.")
            ),
            
            ui.h5("Sentiment", style="margin-top: 15px;"),
            ui.p(ui.tags.strong("Definition:"), " Whether the content portrays the tribe positively, negatively, or neutrally."),
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("Positive:"), " Content celebrates, defends, or validates the tribe's values and identity. Appeals to tribal pride and solidarity."),
                ui.tags.li(ui.tags.strong("Negative:"), " Content criticizes, threatens, or mocks the tribe. May trigger defensive reactions or outrage."),
                ui.tags.li(ui.tags.strong("Neutral:"), " Content acknowledges the tribe without strong positive or negative framing.")
            ),
            
            ui.h5("Hooks", style="margin-top: 15px;"),
            ui.p(ui.tags.strong("Definition:"), " Specific narrative elements, phrases, values, or emotional triggers in the content that connect with the tribe's core concerns."),
            ui.p(ui.tags.strong("Examples:")),
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("Progressive Left:"), " Social justice, systemic inequality, climate crisis, corporate power, marginalized voices"),
                ui.tags.li(ui.tags.strong("MAGA Conservative:"), " America First, deep state, media bias, traditional values under attack, economic nationalism"),
                ui.tags.li(ui.tags.strong("Libertarian:"), " Government overreach, individual freedom, free markets, personal responsibility, civil liberties"),
                ui.tags.li(ui.tags.strong("Religious Conservative:"), " Moral decline, religious freedom, family values, sanctity of life, biblical principles")
            ),
            ui.p(ui.tags.strong("Function:"), " Hooks are the \"entry points\" that make content feel relevant and emotionally resonant to a tribe. They signal \"this content speaks to US and OUR concerns.\""),
            
            ui.tags.hr(style="margin: 30px 0;"),
            
            ui.h4("Major Political/Cultural Tribes"),
            
            ui.h5("Progressive Left", style="margin-top: 15px;"),
            ui.p(ui.tags.strong("Core Values:"), " Social justice, equity, environmental protection, systemic change, intersectionality, collective action"),
            ui.p(ui.tags.strong("Key Concerns:"), " Systemic racism, climate change, wealth inequality, corporate power, LGBTQ+ rights, reproductive rights"),
            ui.p(ui.tags.em("Narrative Style: Emphasizes structural oppression, marginalized voices, and the need for transformative policy"), style="font-style: italic; color: #555;"),
            
            ui.h5("MAGA Conservative", style="margin-top: 15px;"),
            ui.p(ui.tags.strong("Core Values:"), " American nationalism, traditional culture, anti-establishment, populist economics, strong borders"),
            ui.p(ui.tags.strong("Key Concerns:"), " Immigration, cultural change, \"deep state\" elites, media bias, economic decline in rural/working-class communities"),
            ui.p(ui.tags.em("Narrative Style: Us vs. elite establishment; nostalgia for traditional America; distrust of mainstream institutions"), style="font-style: italic; color: #555;"),
            
            ui.h5("Establishment Conservative", style="margin-top: 15px;"),
            ui.p(ui.tags.strong("Core Values:"), " Free markets, limited government, rule of law, institutional stability, personal responsibility"),
            ui.p(ui.tags.strong("Key Concerns:"), " Government overreach, fiscal responsibility, constitutional order, meritocracy, law and order"),
            ui.p(ui.tags.em("Narrative Style: Appeals to tradition, incremental change, respect for institutions, and market solutions"), style="font-style: italic; color: #555;"),
            
            ui.h5("Libertarian", style="margin-top: 15px;"),
            ui.p(ui.tags.strong("Core Values:"), " Individual liberty, minimal government, free markets, personal autonomy, non-aggression principle"),
            ui.p(ui.tags.strong("Key Concerns:"), " Government tyranny, taxation, regulation, civil liberties, privacy, self-determination"),
            ui.p(ui.tags.em("Narrative Style: Freedom vs. coercion; skepticism of all authority; emphasis on voluntary cooperation"), style="font-style: italic; color: #555;"),
            
            ui.h5("Religious Conservative", style="margin-top: 15px;"),
            ui.p(ui.tags.strong("Core Values:"), " Traditional morality, religious freedom, sanctity of life, family values, biblical authority"),
            ui.p(ui.tags.strong("Key Concerns:"), " Abortion, religious liberty, secular culture, marriage, moral education, religious expression in public life"),
            ui.p(ui.tags.em("Narrative Style: Moral absolutes, sacred vs. profane, defense of faith, spiritual warfare"), style="font-style: italic; color: #555;"),
            
            ui.h5("Centrist/Moderate", style="margin-top: 15px;"),
            ui.p(ui.tags.strong("Core Values:"), " Pragmatism, compromise, institutional stability, evidence-based policy, civility"),
            ui.p(ui.tags.strong("Key Concerns:"), " Partisan extremism, polarization, functional governance, economic stability"),
            ui.p(ui.tags.em("Narrative Style: \"Both sides\" framing, appeals to reason and moderation, critique of ideological rigidity"), style="font-style: italic; color: #555;"),
            
            ui.h5("Tech Optimist", style="margin-top: 15px;"),
            ui.p(ui.tags.strong("Core Values:"), " Innovation, progress, entrepreneurship, disruption, technological solutions to problems"),
            ui.p(ui.tags.strong("Key Concerns:"), " Regulation stifling innovation, Luddite resistance to change, global competitiveness"),
            ui.p(ui.tags.em("Narrative Style: Future-focused, optimistic about human ingenuity, dismissive of precautionary principle"), style="font-style: italic; color: #555;"),
            
            ui.h5("Democratic Socialist", style="margin-top: 15px;"),
            ui.p(ui.tags.strong("Core Values:"), " Economic democracy, worker power, universal programs, anti-capitalism, internationalism"),
            ui.p(ui.tags.strong("Key Concerns:"), " Billionaire class, healthcare access, student debt, worker exploitation, corporate dominance"),
            ui.p(ui.tags.em("Narrative Style: Class struggle, solidarity, \"us vs. oligarchy,\" critique of neoliberalism"), style="font-style: italic; color: #555;"),
            
            ui.tags.hr(style="margin: 30px 0;"),
            
            ui.h4("Polarization Risk"),
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("High Risk:"), " Content strongly resonates with one tribe while threatening or attacking another, likely to amplify division and tribal conflict."),
                ui.tags.li(ui.tags.strong("Medium Risk:"), " Content appeals to tribal values but doesn't explicitly attack out-groups."),
                ui.tags.li(ui.tags.strong("Low Risk:"), " Content has broad appeal or minimal tribal signaling.")
            ),
            
            ui.tags.hr(style="margin: 30px 0;"),
            
            ui.h4("Application"),
            ui.p("Understanding tribal resonance helps predict:"),
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("Amplification patterns:"), " Which communities will share and promote content"),
                ui.tags.li(ui.tags.strong("Backlash potential:"), " Which tribes will resist or attack the message"),
                ui.tags.li(ui.tags.strong("Echo chamber effects:"), " How content reinforces existing tribal boundaries"),
                ui.tags.li(ui.tags.strong("Persuasion limits:"), " Why factually accurate content often fails to cross tribal lines")
            ),
            ui.p(ui.tags.strong("Strategic Insight:"), " Content creators must decide whether to maximize resonance within a target tribe (risking polarization) or craft messages that bridge tribal divides (risking lower engagement)."),
            
            style="margin: 30px 0; padding: 30px; background: #f9f9f9; border-radius: 8px; border: 1px solid #ddd;"
        )
        
        return ui.div(ui.div(*tribe_cards), theory_description)
    
    # Linguistic Analysis UI
    @output
    @render.ui
    def linguistic_ui():
        results = analysis_results.get()
        if not results:
            return ui.p("No analysis available.")
        
        linguistic = results.get("linguistic")
        if not linguistic:
            return ui.div(
                ui.p("Linguistic analysis not available. Switch to Deep mode and re-run analysis.", style="color:#666;"),
                class_="foundation-card"
            )
        
        import json as _json
        sections = []
        
        # 1. Agency & Responsibility (Passive Voice)
        agency = linguistic.get("agency_analysis", {})
        passive_pct = agency.get("passive_voice_percent", 0)
        examples = agency.get("hidden_actor_examples", [])
        interp = agency.get("responsibility_interpretation", "")
        
        # Slider visualization (horizontal bar)
        slider_pos = passive_pct  # 0-100
        slider_html = f"""
        <div style="margin:20px 0;">
            <div style="display:flex; justify-content:space-between; font-size:0.85em; color:#666; margin-bottom:6px;">
                <span>High Agency (Actors Clear)</span>
                <span>High Obfuscation (Actors Hidden)</span>
            </div>
            <div style="position:relative; height:30px; background:linear-gradient(to right, #28a745 0%, #ffc107 50%, #dc3545 100%); border-radius:15px;">
                <div style="position:absolute; left:{slider_pos}%; top:50%; transform:translate(-50%,-50%); width:16px; height:16px; background:#fff; border:3px solid #333; border-radius:50%;"></div>
            </div>
            <div style="text-align:center; margin-top:8px; font-weight:600;">{passive_pct}% Passive Voice</div>
        </div>
        """
        
        agency_card = ui.div(
            ui.h4("1. Agency & Responsibility"),
            ui.HTML(slider_html),
            ui.p(interp, style="font-size:0.9em; color:#333;"),
            ui.tags.strong("Hidden Actor Examples:", style="font-size:0.9em;"),
            ui.tags.ul(*[ui.tags.li(ex, style="font-size:0.85em;") for ex in examples[:4]]) if examples else ui.p("No passive constructions detected.", style="font-size:0.85em; color:#666;"),
            class_="foundation-card"
        )
        sections.append(agency_card)
        
        # 2. Othering Index (Us vs Them)
        polar = linguistic.get("polarization_metrics", {})
        ratio = polar.get("us_vs_them_ratio", 0.0)
        ingroup = polar.get("ingroup_pronouns", {})
        outgroup = polar.get("outgroup_pronouns", {})
        outgroup_label = polar.get("most_used_outgroup_label")
        polar_interp = polar.get("polarization_interpretation", "")
        
        in_total = sum(ingroup.values()) if ingroup else 0
        out_total = sum(outgroup.values()) if outgroup else 0
        
        # Diverging bar chart data
        bar_data = {
            "type": "bar",
            "orientation": "h",
            "y": ["In-Group (We/Us/Our)", "Out-Group (They/Them/Those)"],
            "x": [in_total, out_total],
            "marker": {"color": ["#007bff", "#dc3545"]},
            "text": [f"{in_total} uses", f"{out_total} uses"],
            "textposition": "outside"
        }
        bar_layout = {
            "margin": {"l": 180, "r": 40, "t": 20, "b": 40},
            "height": 180,
            "xaxis": {"title": "Pronoun Count"},
            "paper_bgcolor": "#f8f9fa"
        }
        bar_json = _json.dumps({"data": [bar_data], "layout": bar_layout})
        bar_html = f"""
        <div id="othering_chart" style="width:100%;height:180px;"></div>
        <script>
          (function(){{
            var spec = {bar_json};
            if (window.Plotly && document.getElementById('othering_chart')) {{
              Plotly.newPlot('othering_chart', spec.data, spec.layout, {{displayModeBar:false}});
            }}
          }})();
        </script>
        """
        
        othering_card = ui.div(
            ui.h4("2. Othering Index (Us vs. Them)"),
            ui.p(f"Ratio: {ratio:.2f} | Out-Group Label: {outgroup_label or 'None detected'}", style="font-weight:600;"),
            ui.HTML(bar_html),
            ui.p(polar_interp, style="font-size:0.9em; color:#333; margin-top:12px;"),
            class_="foundation-card"
        )
        sections.append(othering_card)
        
        # 3. Dogmatism Score (Certainty Gauge)
        cert = linguistic.get("certainty_metrics", {})
        dogma_score = cert.get("dogmatism_score", 0)
        high_mod = cert.get("high_modality_count", 0)
        low_mod = cert.get("low_modality_count", 0)
        modals = cert.get("dominant_modals", [])
        cert_interp = cert.get("certainty_interpretation", "")
        
        # Gauge visualization (semi-circle indicator)
        gauge_color = "#28a745" if dogma_score < 40 else "#ffc107" if dogma_score < 70 else "#dc3545"
        gauge_html = f"""
        <div style="text-align:center; margin:20px 0;">
            <div style="font-size:3em; font-weight:700; color:{gauge_color};">{dogma_score}</div>
            <div style="font-size:1.1em; color:#666;">Certainty Score (0-100)</div>
            <div style="margin-top:12px; font-size:0.9em; color:#666;">
                High Modality: {high_mod} | Low Modality: {low_mod}
            </div>
        </div>
        """
        
        dogma_card = ui.div(
            ui.h4("3. Dogmatism Score"),
            ui.HTML(gauge_html),
            ui.p(cert_interp, style="font-size:0.9em; color:#333;"),
            ui.tags.strong("Dominant Modals:", style="font-size:0.9em;"),
            ui.p(", ".join(modals) if modals else "None detected", style="font-size:0.85em; color:#666;"),
            class_="foundation-card"
        )
        sections.append(dogma_card)
        
        # 4. Complexity & Populism
        read = linguistic.get("readability", {})
        grade = read.get("grade_level", 0.0)
        style_class = read.get("style_classification", "Unknown")
        lex_density = read.get("lexical_density", 0.0)
        read_interp = read.get("complexity_interpretation", "")
        
        # Simple bar comparing to national average (assume 8th grade = national avg)
        nat_avg = 8.0
        grade_color = "#007bff" if grade > nat_avg else "#28a745"
        complexity_html = f"""
        <div style="margin:20px 0;">
            <div style="font-size:1.8em; font-weight:600; color:{grade_color}; text-align:center;">
                Grade Level: {grade:.1f}
            </div>
            <div style="text-align:center; font-size:0.9em; color:#666; margin-top:6px;">
                National Average: {nat_avg} | Style: {style_class}
            </div>
            <div style="margin-top:12px; font-size:0.9em; color:#666; text-align:center;">
                Lexical Density: {lex_density:.1%}
            </div>
        </div>
        """
        
        complexity_card = ui.div(
            ui.h4("4. Complexity & Populism Fingerprint"),
            ui.HTML(complexity_html),
            ui.p(read_interp, style="font-size:0.9em; color:#333;"),
            class_="foundation-card"
        )
        sections.append(complexity_card)
        
        # 5. Persuasion Signature (Spider/Radar chart with 15 devices)
        persuasion = linguistic.get("persuasion_signature", {})
        density = persuasion.get("rhetorical_density_score", 0.0)
        valence = persuasion.get("net_valence_score", 0.0)
        classification = persuasion.get("classification", "Unknown")
        devices = persuasion.get("devices", [])
        pers_interp = persuasion.get("signature_interpretation", "")
        
        # Build spider/radar chart data
        # Separate into positive and negative traces based on valence_score
        device_names = []
        positive_counts = []
        negative_counts = []
        
        for device in devices:
            name = device.get("name", "Unknown")
            count = device.get("count", 0)
            val_score = device.get("valence_score", 0.0)
            
            device_names.append(name)
            # Split into positive/negative for color coding
            if val_score >= 0:
                positive_counts.append(count)
                negative_counts.append(0)
            else:
                negative_counts.append(count)
                positive_counts.append(0)
        
        # If no devices, use empty placeholders
        if not device_names:
            device_names = ["Strawman", "False Dichotomy", "Ad Hominem", "Slippery Slope", "Whataboutism",
                           "Loaded Language", "Dog Whistle", "Proof by Gallup", "Motte-and-Bailey", "Anaphora",
                           "Catastrophizing", "Appeal to Authority", "Bandwagon", "Euphemism/Dysphemism", "Epistemic Closure"]
            positive_counts = [0] * 15
            negative_counts = [0] * 15
        
        # Radar chart traces (positive in green, negative in red)
        radar_data = []
        
        if any(positive_counts):
            radar_data.append({
                "type": "scatterpolar",
                "r": positive_counts,
                "theta": device_names,
                "fill": "toself",
                "fillcolor": "rgba(40, 167, 69, 0.2)",
                "line": {"color": "#28a745", "width": 2},
                "marker": {"color": "#28a745", "size": 6},
                "name": "Positive Valence"
            })
        
        if any(negative_counts):
            radar_data.append({
                "type": "scatterpolar",
                "r": negative_counts,
                "theta": device_names,
                "fill": "toself",
                "fillcolor": "rgba(220, 53, 69, 0.2)",
                "line": {"color": "#dc3545", "width": 2},
                "marker": {"color": "#dc3545", "size": 6},
                "name": "Negative Valence"
            })
        
        radar_layout = {
            "polar": {
                "radialaxis": {"visible": True, "range": [0, max(max(positive_counts + negative_counts, default=1) * 1.2, 5)]}
            },
            "margin": {"l": 80, "r": 80, "t": 40, "b": 40},
            "height": 520,
            "paper_bgcolor": "#f8f9fa",
            "showlegend": True,
            "legend": {"x": 0.5, "y": -0.1, "xanchor": "center", "orientation": "h"}
        }
        
        radar_json = _json.dumps({"data": radar_data, "layout": radar_layout})
        radar_html = f"""
        <div id="persuasion_radar" style="width:100%;height:520px;"></div>
        <script>
          (function(){{
            var spec = {radar_json};
            if (window.Plotly && document.getElementById('persuasion_radar')) {{
              Plotly.newPlot('persuasion_radar', spec.data, spec.layout, {{displayModeBar:false}});
            }}
          }})();
        </script>
        """
        
        # Build device list with examples
        device_list_items = []
        for device in devices:
            name = device.get("name", "Unknown")
            count = device.get("count", 0)
            val_score = device.get("valence_score", 0.0)
            examples = device.get("examples", [])
            
            if count > 0:
                color = "#28a745" if val_score >= 0 else "#dc3545"
                device_list_items.append(
                    ui.tags.li(
                        ui.tags.span(f"{name}: {count} uses (valence: {val_score:+.2f})", style=f"font-weight:600; color:{color};"),
                        ui.tags.ul(*[ui.tags.li(f'"{ex}"', style="font-size:0.8em; color:#666; font-style:italic;") for ex in examples[:2]]) if examples else "",
                        style="margin-bottom:8px;"
                    )
                )
        
        # 5. Quantifier Vagueness
        quant = linguistic.get("quantifier_vagueness", {})
        vague_count = quant.get("vague_quantifier_count", 0)
        precise_count = quant.get("precise_quantifier_count", 0)
        vague_ratio = quant.get("vagueness_ratio", 0.0)
        vague_examples = quant.get("vague_examples", [])
        precise_examples = quant.get("precise_examples", [])
        quant_interp = quant.get("vagueness_interpretation", "")
        
        # Horizontal bar chart comparing vague vs precise
        total_quant = vague_count + precise_count
        vague_pct = (vague_count / total_quant * 100) if total_quant > 0 else 0
        precise_pct = (precise_count / total_quant * 100) if total_quant > 0 else 0
        
        quant_bar_data = {
            "type": "bar",
            "orientation": "h",
            "y": ["Precise (with data)", "Vague (rhetorical)"],
            "x": [precise_count, vague_count],
            "marker": {"color": ["#28a745", "#dc3545"]},
            "text": [f"{precise_count} ({precise_pct:.0f}%)", f"{vague_count} ({vague_pct:.0f}%)"],
            "textposition": "outside"
        }
        quant_bar_layout = {
            "margin": {"l": 180, "r": 40, "t": 20, "b": 40},
            "height": 180,
            "xaxis": {"title": "Quantifier Count"},
            "paper_bgcolor": "#f8f9fa"
        }
        quant_bar_json = _json.dumps({"data": [quant_bar_data], "layout": quant_bar_layout})
        quant_bar_html = f"""
        <div id="quantifier_chart" style="width:100%;height:180px;"></div>
        <script>
          (function(){{
            var spec = {quant_bar_json};
            if (window.Plotly && document.getElementById('quantifier_chart')) {{
              Plotly.newPlot('quantifier_chart', spec.data, spec.layout, {{displayModeBar:false}});
            }}
          }})();
        </script>
        """
        
        vagueness_score = int(vague_ratio * 100)
        score_color = "#28a745" if vagueness_score < 30 else "#ffc107" if vagueness_score < 70 else "#dc3545"
        
        quantifier_card = ui.div(
            ui.h4("5. Quantifier Vagueness"),
            ui.p(f"Vagueness Ratio: {vague_ratio:.2f} | Score: ", 
                 ui.tags.span(f"{vagueness_score}", style=f"font-weight:700; color:{score_color};"), 
                 "/100", style="font-weight:600;"),
            ui.HTML(quant_bar_html),
            ui.p(quant_interp, style="font-size:0.9em; color:#333; margin-top:12px;"),
            ui.tags.strong("Vague Examples:", style="font-size:0.9em; margin-top:8px; display:block;"),
            ui.tags.ul(*[ui.tags.li(f'"{ex}"', style="font-size:0.85em; color:#dc3545; font-style:italic;") for ex in vague_examples[:3]]) if vague_examples else ui.p("None detected.", style="font-size:0.85em; color:#666;"),
            ui.tags.strong("Precise Examples:", style="font-size:0.9em; margin-top:8px; display:block;"),
            ui.tags.ul(*[ui.tags.li(f'"{ex}"', style="font-size:0.85em; color:#28a745; font-style:italic;") for ex in precise_examples[:3]]) if precise_examples else ui.p("None detected.", style="font-size:0.85em; color:#666;"),
            class_="foundation-card"
        )
        sections.append(quantifier_card)
        
        persuasion_card = ui.div(
            ui.h4("6. Persuasion Signature (15 Rhetorical Devices)"),
            ui.p(f"Density: {density:.1f} devices/1k words | Net Valence: {valence:+.2f} | Classification: {classification}", style="font-weight:600;"),
            ui.HTML(radar_html),
            ui.p(pers_interp, style="font-size:0.9em; color:#333; margin-top:12px;"),
            ui.tags.strong("Detected Rhetorical Devices:", style="font-size:0.9em; margin-top:12px; display:block;"),
            ui.tags.ul(*device_list_items, style="font-size:0.85em;") if device_list_items else ui.p("No devices detected.", style="font-size:0.85em; color:#666;"),
            class_="foundation-card"
        )
        sections.append(persuasion_card)
        
        framework_description = ui.div(
            ui.tags.hr(style="margin: 40px 0 30px 0; border-top: 2px solid #ccc;"),
            ui.h3("Linguistic Forensics Framework", style="margin-top: 30px;"),
            
            ui.h4("Core Principles", style="margin-top: 20px;"),
            ui.p("Linguistic Forensics analyzes how language choices shape perception, assign responsibility, and reveal hidden ideological frames. This framework examines the subtle ways that grammatical structures, word choices, and rhetorical patterns influence how we understand events, assign blame, and construct narratives about reality."),
            ui.p(ui.tags.strong("Key Insight:"), " Language is never neutral. Every linguistic choice—passive vs. active voice, certainty vs. hedging, us vs. them framing—carries ideological weight and shapes how audiences perceive causality, agency, and moral responsibility."),
            
            ui.tags.hr(style="margin: 30px 0;"),
            
            ui.h4("Analytical Dimensions"),
            
            ui.h5("1. Agency & Responsibility (Passive Voice Analysis)", style="margin-top: 20px;"),
            ui.p(ui.tags.strong("What It Measures:"), " Percentage of passive voice constructions that obscure who is responsible for actions"),
            ui.p(ui.tags.strong("Examples of Hidden Agency:")),
            ui.tags.ul(
                ui.tags.li(ui.tags.em("\"Mistakes were made\""), " → Who made them?"),
                ui.tags.li(ui.tags.em("\"Protesters were shot\""), " → Who shot them?"),
                ui.tags.li(ui.tags.em("\"The decision was reached\""), " → Who decided?")
            ),
            ui.p(ui.tags.strong("Interpretation:")),
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("Low passive voice (0-20%):"), " Clear assignment of agency and responsibility"),
                ui.tags.li(ui.tags.strong("Moderate (20-40%):"), " Mixed; some accountability evasion"),
                ui.tags.li(ui.tags.strong("High (40%+):"), " Systematic obscuring of actors, likely institutional defensiveness or spin")
            ),
            
            ui.h5("2. Othering Index (Us vs. Them Language)", style="margin-top: 20px;"),
            ui.p(ui.tags.strong("What It Measures:"), " Ratio of out-group pronouns (they/them/those) to in-group pronouns (we/us/our), plus identification of specific out-group labels"),
            ui.p(ui.tags.strong("Pronoun Patterns:")),
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("In-group:"), " \"We believe...\", \"Our values...\", \"Us patriots...\""),
                ui.tags.li(ui.tags.strong("Out-group:"), " \"They want to...\", \"Those people...\", \"Them liberals/conservatives...\"")
            ),
            ui.p(ui.tags.strong("Interpretation:")),
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("Ratio < 1.0:"), " More inclusive, emphasizes shared identity"),
                ui.tags.li(ui.tags.strong("Ratio 1.0-2.0:"), " Balanced or moderate tribal signaling"),
                ui.tags.li(ui.tags.strong("Ratio > 2.0:"), " High polarization, strong us-vs-them framing, tribal warfare rhetoric")
            ),
            
            ui.h5("3. Dogmatism Score (Certainty vs. Hedging)", style="margin-top: 20px;"),
            ui.p(ui.tags.strong("What It Measures:"), " Balance between high-certainty language (\"clearly\", \"obviously\", \"must\") and hedging language (\"might\", \"possibly\", \"could\"), scored 0-100"),
            ui.p(ui.tags.strong("Language Patterns:")),
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("High Modality:"), " \"Undeniably\", \"certainly\", \"without doubt\", \"proven fact\""),
                ui.tags.li(ui.tags.strong("Low Modality:"), " \"Perhaps\", \"appears to\", \"suggests\", \"may indicate\"")
            ),
            ui.p(ui.tags.strong("Interpretation:")),
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("Score 0-40:"), " Cautious, academic, or appropriately tentative given evidence"),
                ui.tags.li(ui.tags.strong("Score 40-70:"), " Moderate confidence, balanced epistemic commitment"),
                ui.tags.li(ui.tags.strong("Score 70-100:"), " Highly dogmatic, ideologically rigid, propaganda-like certainty")
            ),
            
            ui.h5("4. Complexity & Populism Fingerprint", style="margin-top: 20px;"),
            ui.p(ui.tags.strong("What It Measures:"), " Reading grade level (Flesch-Kincaid), lexical density, and stylistic classification"),
            ui.p(ui.tags.strong("Style Classifications:")),
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("Academic/Elite (Grade 13+):"), " Complex sentences, technical vocabulary, high lexical density"),
                ui.tags.li(ui.tags.strong("Mainstream (Grade 8-12):"), " Standard journalism or educated discourse"),
                ui.tags.li(ui.tags.strong("Populist (Grade < 8):"), " Simple, direct, accessible to broad audience")
            ),
            ui.p(ui.tags.strong("Interpretation:"), " Lower grade level doesn't mean \"dumb\"—it can indicate effective communication or deliberate populist appeal. High grade level can signal expertise or elitist gatekeeping."),
            
            ui.h5("5. Quantifier Vagueness", style="margin-top: 20px;"),
            ui.p(ui.tags.strong("What It Measures:"), " Ratio of vague quantifiers (\"many\", \"most\", \"some\") to precise quantifiers (specific numbers, percentages, data with sources)"),
            ui.p(ui.tags.strong("Vague Quantifiers:")),
            ui.tags.ul(
                ui.tags.li("\"Many experts agree...\""),
                ui.tags.li("\"A growing number of studies...\""),
                ui.tags.li("\"Most people believe...\""),
                ui.tags.li("\"Increasingly widespread concern...\""),
                ui.tags.li("\"Significant evidence suggests...\"")
            ),
            ui.p(ui.tags.strong("Precise Quantifiers:")),
            ui.tags.ul(
                ui.tags.li("\"73% of climate scientists in a 2023 survey...\""),
                ui.tags.li("\"14 peer-reviewed studies published between 2020-2024...\""),
                ui.tags.li("\"2,400 participants across 8 countries...\""),
                ui.tags.li("\"Crime rates decreased 23% from 2015 to 2023 (FBI data)...\"")
            ),
            ui.p(ui.tags.strong("Interpretation:")),
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("Low Vagueness (< 0.3):"), " Evidence-based, empirically grounded, verifiable claims"),
                ui.tags.li(ui.tags.strong("Moderate (0.3-0.7):"), " Mixed; some data, some impressionistic claims"),
                ui.tags.li(ui.tags.strong("High Vagueness (> 0.7):"), " Rhetorical inflation, weak evidential basis, opinion presented as fact")
            ),
            ui.p(ui.tags.strong("Why It Matters:"), " Vague quantifiers allow speakers to make sweeping claims while evading accountability. Precise quantifiers enable verification and critical evaluation. High vagueness often indicates that the evidence doesn't actually support the claim being made."),
            
            ui.h5("6. Persuasion Signature (15 Rhetorical Devices)", style="margin-top: 20px;"),
            ui.p(ui.tags.strong("What It Measures:"), " Density (devices per 1000 words) and valence (positive/negative) of 15 specific rhetorical techniques"),
            
            ui.p(ui.tags.strong("The 15 Rhetorical Devices:"), style="margin-top: 15px;"),
            
            ui.tags.strong("Negative-Valence Devices (Fallacies & Manipulation):"),
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("Strawman Argument:"), " Misrepresenting opponent's position to make it easier to attack"),
                ui.tags.li(ui.tags.strong("False Dichotomy:"), " Presenting complex issues as having only two options"),
                ui.tags.li(ui.tags.strong("Ad Hominem:"), " Attacking the person rather than their argument"),
                ui.tags.li(ui.tags.strong("Slippery Slope:"), " Claiming a small step inevitably leads to extreme consequences"),
                ui.tags.li(ui.tags.strong("Whataboutism:"), " Deflecting criticism by pointing to hypocrisy elsewhere"),
                ui.tags.li(ui.tags.strong("Dog Whistle:"), " Coded language that appears normal but signals specific meaning to in-group"),
                ui.tags.li(ui.tags.strong("Proof by Gallup:"), " Overwhelming reader with so many arguments that refutation becomes impossible"),
                ui.tags.li(ui.tags.strong("Motte-and-Bailey:"), " Conflating a modest claim (the Motte) with a controversial one (the Bailey)"),
                ui.tags.li(ui.tags.strong("Catastrophizing:"), " Emphasizing worst-case scenarios to induce fear or panic"),
                ui.tags.li(ui.tags.strong("Epistemic Closure:"), " Rhetoric that prevents consideration of outside sources (\"fake news\", \"propaganda\")")
            ),
            
            ui.tags.strong("Mixed-Valence Devices (Can Be Positive or Negative):"),
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("Loaded Language:"), " Words with strong emotional connotations (positive or negative)"),
                ui.tags.li(ui.tags.strong("Appeal to Authority:"), " Citing expert opinion (positive if legitimate, negative if false authority)"),
                ui.tags.li(ui.tags.strong("Euphemism/Dysphemism:"), " Softening harsh realities OR degrading neutral concepts")
            ),
            
            ui.tags.strong("Positive-Valence Devices (Persuasive but Not Necessarily Manipulative):"),
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("Anaphora:"), " Repetition of words/phrases at beginning of clauses for rhythmic or rallying effect"),
                ui.tags.li(ui.tags.strong("Bandwagon:"), " Arguing something is right because many people believe it (appeal to consensus)")
            ),
            
            ui.p(ui.tags.strong("Density Interpretation:"), style="margin-top: 15px;"),
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("Low Density (< 10 per 1k words):"), " Straightforward, minimal rhetorical manipulation"),
                ui.tags.li(ui.tags.strong("Moderate (10-25):"), " Normal persuasive writing"),
                ui.tags.li(ui.tags.strong("High (25-50):"), " Heavily rhetorical, activist or opinion writing"),
                ui.tags.li(ui.tags.strong("Extreme (> 50):"), " Propaganda-level manipulation, every sentence loaded with devices")
            ),
            
            ui.p(ui.tags.strong("Net Valence Interpretation:")),
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("Positive:"), " Persuasive but relatively honest; uses rallying rhetoric more than fallacies"),
                ui.tags.li(ui.tags.strong("Neutral:"), " Mixed bag of positive and negative techniques"),
                ui.tags.li(ui.tags.strong("Negative:"), " Heavy use of logical fallacies, bad-faith argumentation, manipulative tactics")
            ),
            
            ui.tags.hr(style="margin: 30px 0;"),
            
            ui.h4("Application"),
            ui.p("Linguistic Forensics helps you:"),
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("Detect spin:"), " Identify when language obscures rather than clarifies"),
                ui.tags.li(ui.tags.strong("Assess credibility:"), " Distinguish evidence-based claims from rhetorical manipulation"),
                ui.tags.li(ui.tags.strong("Understand persuasion:"), " See how linguistic choices prime audiences for specific conclusions"),
                ui.tags.li(ui.tags.strong("Recognize tribal signaling:"), " Spot us-vs-them framing and in-group/out-group dynamics"),
                ui.tags.li(ui.tags.strong("Evaluate confidence:"), " Distinguish appropriate certainty from ideological dogmatism")
            ),
            ui.p(ui.tags.strong("Strategic Insight:"), " Changing the language changes the debate. Those who control the terms of discussion often control the outcome. Recognizing these patterns helps you resist manipulation and think more critically about persuasive content."),
            
            style="margin: 30px 0; padding: 30px; background: #f9f9f9; border-radius: 8px; border: 1px solid #ddd;"
        )
        
        sections.append(framework_description)
        
        return ui.div(*sections)
    
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
