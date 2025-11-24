# Project Truth — Harari Fact Analyzer

This repository contains a small prototype: a Shiny for Python app that
extracts assertions from text, runs grounded fact-checks using Google's
Generative AI (Gemini) with optional web grounding, and classifies
assertions according to Harari's Objective/Subjective/Intersubjective tiers.

Quick start (using the project venv):

```bash
# activate the venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# download spaCy model
python -m spacy download en_core_web_sm

# run tests
pytest -q

# run the app (dev server)
shiny run --reload app.py
```

Files of interest:
- `app.py` — Shiny UI and glue to ingestion/analysis.
- `src/data_ingestion.py` — URL scraping, YouTube transcript helper.
- `src/analysis_engine.py` — `extract_facts`, `check_facts`, `classify_harari`.
- `src/gemini/client.py` — Gemini helper that prefers the `google.generativeai` SDK and falls back to REST.

Notes:
- The Gemini helper attempts to use the SDK's `GenerativeModel.generate_content`
  and can enable Google Search retrieval via `enable_search_tool=True` (requires
  the feature to be enabled for your API key/project).
- The code attempts to enforce JSON outputs from the model via `response_schema`
  when the SDK supports `GenerationConfig.response_schema` (best-effort).

Optional: use a `.env` file for local development
------------------------------------------------
You can create a `.env` file in the project root with the following content:

```
GEMINI_API_KEY=your_api_key_here
```

If `python-dotenv` is installed, the app will load `.env` on startup so you
don't need to export the key in every shell. The dev server still needs to be
started after creating/updating `.env` so the server process picks up the key.

