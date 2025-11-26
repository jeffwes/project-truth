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
            ui.nav_panel("Reality Taxonomy", ui.div(ui.h3("Harari's Reality Classification"), ui.p("Classification by reality type."), ui.output_ui("taxonomy_ui"))),
            ui.nav_panel("Moral Foundations", ui.div(ui.h3("Moral Foundations (Haidt)"), ui.output_ui("foundations_ui"))),
            ui.nav_panel("Tribal Resonance", ui.div(ui.h3("Predicted Social Tribe Resonance"), ui.output_ui("tribes_ui"))),
            ui.nav_panel("Linguistic Analysis", ui.div(ui.h3("Linguistic Forensics"), ui.output_ui("linguistic_ui"))),
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

        # Build Sunburst chart UI component (explicit hierarchy root -> classification -> subcategories)
        sunburst_html = ""
        if values:
            sb_labels = ["Reality"]
            sb_parents = [""]
            sb_values = [sum(values)]
            sb_customdata = ["\n".join(node_examples.get("Reality", [])) or "Overall taxonomy"]

            def push(label, parent, value, examples_key=None):
                sb_labels.append(label)
                sb_parents.append(parent)
                sb_values.append(value)
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

            sunburst_data = {
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
            layout = {
                "margin": {"l": 0, "r": 0, "t": 30, "b": 0},
                "paper_bgcolor": "#ffffff",
                "height": 540
            }
                        fig_json = _json.dumps({"data": [sunburst_data], "layout": layout})
                        # Build second (direct) sunburst: myth/empathy directly under classification
                        sb2_labels = ["Reality (Direct)"]
                        sb2_parents = [""]
                        sb2_values = [sum(values)]
                        sb2_custom = ["Direct counts by myth & empathy bias"]

                        def push2(label, parent, value, examples_key=None):
                                sb2_labels.append(label)
                                sb2_parents.append(parent)
                                sb2_values.append(value)
                                ex = node_examples.get(examples_key or label, [])
                                sb2_custom.append("\n".join(ex) or label)

                        for label, cnt in class_counts.items():
                                if cnt > 0:
                                        push2(label, "Reality (Direct)", cnt, label)

                        for k, v in fact_counts.items():
                                if v > 0:
                                        push2(f"Fact: {k.replace('_',' ').title()}", "Objective", v)

                        # Myth directly under Intersubjective
                        myth_direct_counts = {m: 0 for m in myth_keys}
                        for a in assertions:
                                if a.get("classification") == "intersubjective":
                                        mt = (a.get("myth_taxonomy") or {}).get("category")
                                        if mt in myth_direct_counts:
                                                myth_direct_counts[mt] += 1
                        for m, c in myth_direct_counts.items():
                                if c > 0:
                                        push2(f"Myth: {m.replace('_',' ').title()}", "Intersubjective", c, f"Myth: {m.replace('_',' ').title()}")

                        # Empathy bias directly under Subjective
                        empathy_direct_counts = {e: 0 for e in empathy_keys}
                        for a in assertions:
                                if a.get("classification") == "subjective":
                                        es = (a.get("empathy_span") or {}).get("focus_bias")
                                        if es in empathy_direct_counts:
                                                empathy_direct_counts[es] += 1
                        for e, c in empathy_direct_counts.items():
                                if c > 0:
                                        push2(f"Empathy: {e.replace('_',' ').title()}", "Subjective", c, f"Empathy: {e.replace('_',' ').title()}")

                        sunburst2_data = {
                                "type": "sunburst",
                                "labels": sb2_labels,
                                "parents": sb2_parents,
                                "values": sb2_values,
                                "customdata": sb2_custom,
                                "branchvalues": "total",
                                "marker": {"line": {"width": 1}},
                                "hovertemplate": "<b>%{label}</b><br>%{customdata}<extra></extra>",
                                "maxdepth": 3
                        }
                        layout2 = {"margin": {"l": 0, "r": 0, "t": 30, "b": 0}, "paper_bgcolor": "#ffffff", "height": 540}
                        fig2_json = _json.dumps({"data": [sunburst2_data], "layout": layout2})

                        sunburst_html = f"""
                        <div style=\"display:flex; gap:20px; flex-wrap:wrap;\">
                            <div style=\"flex:1 1 480px; min-width:360px;\">
                                <div id=\"sunburst_rt\" style=\"width:100%;height:540px;margin-bottom:12px;border:1px solid #e0e0e0;border-radius:4px;\"></div>
                            </div>
                            <div style=\"flex:1 1 480px; min-width:360px;\">
                                <div id=\"sunburst_rt2\" style=\"width:100%;height:540px;margin-bottom:12px;border:1px solid #e0e0e0;border-radius:4px;\"></div>
                            </div>
                        </div>
                        <script>
                            (function(){{
                                var spec1 = {fig_json};
                                if (window.Plotly && document.getElementById('sunburst_rt')) {{
                                    Plotly.newPlot('sunburst_rt', spec1.data, spec1.layout, {{displayModeBar:false}});
                                }}
                                var spec2 = {fig2_json};
                                if (window.Plotly && document.getElementById('sunburst_rt2')) {{
                                    Plotly.newPlot('sunburst_rt2', spec2.data, spec2.layout, {{displayModeBar:false}});
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
        
        # Prepend Sunburst chart if available
        if sunburst_html:
            sections.insert(0, ui.HTML(sunburst_html))
        
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
        
        persuasion_card = ui.div(
            ui.h4("5. Persuasion Signature (15 Rhetorical Devices)"),
            ui.p(f"Density: {density:.1f} devices/1k words | Net Valence: {valence:+.2f} | Classification: {classification}", style="font-weight:600;"),
            ui.HTML(radar_html),
            ui.p(pers_interp, style="font-size:0.9em; color:#333; margin-top:12px;"),
            ui.tags.strong("Detected Rhetorical Devices:", style="font-size:0.9em; margin-top:12px; display:block;"),
            ui.tags.ul(*device_list_items, style="font-size:0.85em;") if device_list_items else ui.p("No devices detected.", style="font-size:0.85em; color:#666;"),
            class_="foundation-card"
        )
        sections.append(persuasion_card)
        
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
