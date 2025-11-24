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

# Project Truth — Harari Fact Analyzer

A small prototype Shiny for Python app that:
- extracts factual assertions from input text (spaCy)
- runs grounded fact-checks using Google Generative AI (Gemini) with optional web grounding
- classifies assertions into Harari's tiers (Objective / Intersubjective / Subjective)

Quick start (recommended: use the provided virtualenv)

1. Activate the venv and install dependencies

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

2. Download the spaCy model (only needed once)

```bash
python -m spacy download en_core_web_sm
```

3. Create a `.env` (recommended) with your Gemini key

```bash
printf 'GEMINI_API_KEY="%s"\n' "YOUR_KEY_HERE" > .env
```

4. Run tests

```bash
pytest -q
```

5. Run the Shiny dev server (single-process recommended for local API key loading)

```bash
# from project root
PYTHONPATH=src .venv/bin/shiny run app.py --host 127.0.0.1 --port 8787
```

Files of interest
- `app.py` — Shiny UI and glue to ingestion/analysis
- `src/data_ingestion.py` — URL scraping and YouTube transcript helpers
- `src/analysis_engine.py` — `extract_facts`, `check_facts`, `classify_harari`
- `src/gemini/client.py` — Gemini helper (SDK-first, REST fallback)

Notes & troubleshooting
- `.env` contains secrets; ensure `.env` is listed in `.gitignore` (it is by default)
- The app lazy-loads heavy libraries (spaCy) to reduce import-time overhead, but the first request that triggers spaCy may take longer.
- If the Shiny reloader spawns worker processes that don't see `.env`, start the server without `--reload` so the server process imports `.env` directly.
- Gemini SDK vs REST: the helper prefers the official SDK when available and will try to enable grounding via the SDK's tool config; REST fallback is included for environments without the SDK.

Security
- Never commit API keys. If a secret is pushed accidentally, remove it and rotate the key.

Contributing
- Open an issue or PR on the GitHub repo: https://github.com/jeffwes/project-truth

License
- Add a license file if you want this project published with a permissive license.

Enjoy! If you'd like, I can add a simple GitHub Actions workflow to run tests on each push.

