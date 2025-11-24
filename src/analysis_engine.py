"""Analysis engine: fact extraction, LLM-based fact-checking, and Harari classification.

This module uses spaCy for simple extraction heuristics and calls Gemini
via the `gemini.client.call_gemini` helper created earlier. If the
`google.genai` SDK is present, it can be incorporated later.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple

from gemini.client import call_gemini


def _get_spacy_module():
    try:
        import spacy as _spacy

        return _spacy
    except Exception:
        return None


SUBJECTIVE_MARKERS = [
    "i think",
    "i believe",
    "in my opinion",
    "seems",
    "appears",
    "i feel",
]


def extract_facts(text: str, min_tokens: int = 5) -> List[str]:
    """Extract candidate factual assertions from `text`.

    Heuristic: sentence segmentation via spaCy if available, else simple split.
    Keep sentences with a high density of nouns/proper nouns/verbs and
    without subjective markers.
    """
    if not text:
        return []

    spacy = _get_spacy_module()
    if spacy is None:
        # fallback naive splitting
        sents = re.split(r"(?<=[.!?])\s+", text)
    else:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        sents = [sent.text.strip() for sent in doc.sents]

    out: List[str] = []
    for sent in sents:
        s = sent.strip()
        if not s:
            continue
        # length filter
        if len(s.split()) < min_tokens:
            continue
        low = s.lower()
        # subjective filter
        if any(marker in low for marker in SUBJECTIVE_MARKERS):
            continue

        # crude POS density: count words with capitalized or containing digits
        noun_like = len(re.findall(r"\b[A-Z][a-z]+\b", s))
        verb_like = len(re.findall(r"\b(is|are|was|were|has|have|do|does|did|be|been|become|becomes)\b", low))
        nouns = len(re.findall(r"\b\w+\b", s))
        score = noun_like + verb_like
        if score >= 1:
            out.append(s)

    return out


def _extract_json_from_text(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return ""
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except Exception:
                    return ""
    return ""


def check_facts(assertion_list: List[str]) -> List[Dict[str, Any]]:
    """Fact-check each assertion via Gemini.

    The function constructs a JSON-only prompt instructing the model to
    output rating/summary/source_url. It uses `call_gemini` as the transport.
    """
    results: List[Dict[str, Any]] = []
    system = (
        "You are an expert fact-checker. Your task is to verify or refute the provided claim"
        " using current web search results. Output only a structured JSON object containing"
        " keys: rating (VERIFIED|DISPUTED|UNVERIFIED), summary (string), source_url (string|null)."
    )

    for assertion in assertion_list:
        prompt = (
            f"System instruction: {system}\n\nUser: Fact-check this claim: {assertion}\n"
            "Return a single JSON object exactly as described."
        )

        # Request grounding/search tool when fact-checking (best-effort)
        resp = call_gemini(prompt, return_raw=True, enable_search_tool=True, timeout=60)
        entry = {"assertion": assertion, "rating": "UNVERIFIED", "summary": "", "source_url": None, "raw": None}
        if not resp.get("ok"):
            entry.update({"summary": resp.get("error"), "raw": resp.get("raw")})
            results.append(entry)
            continue

        raw = resp.get("raw") or resp.get("text") or ""
        entry["raw"] = raw

        # the assistant is instructed to return JSON; try to extract
        extracted = ""
        js = None
        if isinstance(raw, dict):
            # try to pull the assistant candidate text then extract JSON from it
            try:
                cand = (raw.get("candidates") or [None])[0]
                content = cand.get("content") if isinstance(cand, dict) else None
                parts = content.get("parts") if isinstance(content, dict) else None
                text_block = parts[0].get("text") if parts and isinstance(parts, list) and isinstance(parts[0], dict) else str(raw)
            except Exception:
                text_block = str(raw)
            extracted = _extract_json_from_text(str(text_block))
            if extracted:
                try:
                    js = json.loads(extracted)
                except Exception:
                    js = None
        else:
            extracted = _extract_json_from_text(str(raw))
            if extracted:
                try:
                    js = json.loads(extracted)
                except Exception:
                    js = None

        if js:
            rating = js.get("rating") or js.get("verdict") or "UNVERIFIED"
            summary = js.get("summary") or js.get("explanation") or ""
            source = js.get("source_url") or None
            entry.update({"rating": rating, "summary": summary, "source_url": source})
        else:
            # fallback: place raw text into summary
            entry.update({"summary": str(raw)})

        results.append(entry)

    return results


def classify_harari(fact_checked_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Classify each assertion into Harari tiers using Gemini.

    Returns items extended with `harari_tier` and `justification`.
    """
    out: List[Dict[str, Any]] = []
    system = (
        "You are an expert in Yuval Harari's philosophy. Classify the assertion into one of:"
        " Objective, Subjective, or Intersubjective. Output only a JSON object with keys: harari_tier, justification."
        " Definitions: Objective: exists independent of human belief. Subjective: exists only within a single person's mind."
        " Intersubjective: exists within shared beliefs/communication networks (money, laws, nations, corporations)."
    )

    for item in fact_checked_results:
        assertion = item.get("assertion")
        prompt = (
            f"System instruction: {system}\n\nUser: Classify the following assertion: {assertion}\n"
            "Return a single JSON object exactly as described."
        )
        resp = call_gemini(prompt, return_raw=True, timeout=60)
        entry = item.copy()
        entry.update({"harari_tier": None, "justification": None})
        if not resp.get("ok"):
            entry.update({"justification": resp.get("error")})
            out.append(entry)
            continue

        raw = resp.get("raw") or resp.get("text") or ""
        extracted = ""
        js = None
        if isinstance(raw, dict):
            try:
                cand = (raw.get("candidates") or [None])[0]
                content = cand.get("content") if isinstance(cand, dict) else None
                parts = content.get("parts") if isinstance(content, dict) else None
                text_block = parts[0].get("text") if parts and isinstance(parts, list) and isinstance(parts[0], dict) else str(raw)
            except Exception:
                text_block = str(raw)
            extracted = _extract_json_from_text(str(text_block))
            if extracted:
                try:
                    js = json.loads(extracted)
                except Exception:
                    js = None
        else:
            extracted = _extract_json_from_text(str(raw))
            if extracted:
                try:
                    js = json.loads(extracted)
                except Exception:
                    js = None

        if js:
            entry["harari_tier"] = js.get("harari_tier") or js.get("tier")
            entry["justification"] = js.get("justification") or js.get("reason")
        else:
            entry["justification"] = str(raw)

        out.append(entry)

    return out
