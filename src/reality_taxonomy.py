"""Reality Taxonomy Analyzer - Classifies assertions using Harari's framework."""
import json
from datetime import date
from typing import List, Dict, Any
from src.gemini_client import GeminiClient


class RealityTaxonomyAnalyzer:
    """
    Analyzes content using Yuval Noah Harari's Reality Taxonomy.
    
    Classifies assertions into:
    - Objective Reality: Independent of human consciousness
    - Subjective Reality: Exists only in individual consciousness
    - Intersubjective Reality: Shared beliefs/myths that shape society
    """
    
    def __init__(self, gemini_client: GeminiClient):
        """Initialize with Gemini client."""
        self.client = gemini_client
    
    def extract_assertions(self, content: str, max_assertions: int = 10) -> List[str]:
        """
        Extract key assertions from content.
        
        Args:
            content: Text content to analyze
            max_assertions: Maximum number of assertions to extract
            
        Returns:
            List of assertion strings
        """
        if not content or not content.strip():
            return []
        
        prompt = f"""Extract the top {max_assertions} most important assertions or claims from this text.

Each assertion should be:
- A standalone statement that can be classified as objective, subjective, or intersubjective
- Specific enough to analyze for reality type
- Representative of the text's key messages

Return ONLY a JSON array of strings, like: ["assertion 1", "assertion 2", ...]

TEXT:
{content[:8000]}"""
        
        result = self.client.generate_json(prompt)
        
        if not result["ok"]:
            # Fallback to simple sentence extraction
            return self._extract_sentences_fallback(content, max_assertions)
        
        try:
            assertions = result["data"]
            if isinstance(assertions, list):
                return [str(a).strip() for a in assertions if a][:max_assertions]
            return []
        except Exception:
            return self._extract_sentences_fallback(content, max_assertions)
    
    def _extract_sentences_fallback(self, content: str, max_count: int) -> List[str]:
        """Fallback: Extract sentences using simple heuristics."""
        import re
        sentences = re.split(r'[.!?]+\s+', content)
        
        # Filter and return top sentences
        filtered = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent.split()) >= 5:  # At least 5 words
                filtered.append(sent)
            if len(filtered) >= max_count:
                break
        
        return filtered
    
    def classify_assertion(self, assertion: str) -> Dict[str, Any]:
        """
        Classify a single assertion into Harari's taxonomy.
        
        Args:
            assertion: The assertion to classify
            
        Returns:
            Dict with 'assertion', 'classification', 'confidence', 'reasoning' keys
        """
        prompt = f"""Classify this assertion using Yuval Noah Harari's Reality Taxonomy.

ASSERTION: "{assertion}"

HARARI'S REALITY TAXONOMY - COMPREHENSIVE FRAMEWORK:

This framework distinguishes three fundamentally different types of reality based on their relationship to human consciousness. Understanding these categories reveals how different types of claims should be evaluated and what forms of evidence are appropriate for each.

TIER 1: OBJECTIVE REALITY
Definition: Phenomena that exist independently of human beliefs, perceptions, or consciousness. These truths existed before humans evolved and would continue to exist if humanity disappeared tomorrow.

Characteristics:
- Verifiable through empirical observation and measurement
- Consistent across cultures and time periods
- Governed by natural laws (physics, chemistry, biology)
- Can be studied through scientific method
- Examples: Gravity, radioactive decay, molecular structures, river flow rates, planetary orbits, biological viruses, chemical reactions

Political Significance: Claims presented as objective facts ("Studies show...", "The data proves...") should be held to empirical standards. Misinformation often disguises intersubjective assertions (ideological beliefs) as objective truths.

TIER 2: SUBJECTIVE REALITY
Definition: Phenomena that exist solely within individual consciousness. These experiences are real to the individual but cannot be directly shared or verified by others.

Characteristics:
- Exists only in a single person's mind
- Private, first-person experiences
- Cannot be empirically measured from outside
- Examples: Personal pain, individual preferences ("I love chocolate"), private thoughts, emotions felt by one person, aesthetic experiences

Political Significance: Subjective experiences are genuine but not universalizable. Rhetoric that conflates personal feelings with objective truth ("I feel threatened, therefore you are threatening") collapses important distinctions.

TIER 3: INTERSUBJECTIVE REALITY (Shared Myths)
Definition: Phenomena that exist only because many humans collectively believe in them. These "myths" (not meaning false, but meaning shared stories) organize human society. They are real in their consequences but would vanish if collective belief ceased.

Three Sub-Tiers:

3A. FOUNDATIONAL MYTHS (Most Deeply Embedded)
- Examples: Money, property rights, corporate personhood, national sovereignty, marriage as a legal institution
- So deeply embedded they feel like natural laws
- Require massive institutional support
- Extremely difficult to change

3B. SACRED NARRATIVES (Meaning-Making Systems)
- Examples: Religious doctrines, political ideologies, human rights frameworks, justice concepts, national origin stories
- Provide moral frameworks and group identity
- Contested between groups
- Can evolve but face strong resistance

3C. CONTINGENT CONSTRUCTIONS (Most Fluid)
- Examples: Fashion trends, social etiquette norms, professional status hierarchies, celebrity fame
- Change relatively quickly
- Less institutionally embedded
- More consciously recognized as constructed

Political Significance: Much political conflict involves competing intersubjective realities (different "myths" about justice, rights, national identity). Recognizing something as intersubjective doesn't make it less real or less important—money and laws shape our lives profoundly—but it reveals that these realities can be renegotiated through collective belief change.

KEY INSIGHT: The most powerful rhetoric obscures category boundaries—presenting intersubjective beliefs ("America is a Christian nation") as objective facts, or elevating subjective preferences ("I find this offensive") to the status of universal moral truths.

Return JSON with this structure:
{{
    "classification": "objective" | "subjective" | "intersubjective",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of why this classification fits"
}}"""
        
        result = self.client.generate_json(prompt, timeout=30)
        
        if not result["ok"]:
            return {
                "assertion": assertion,
                "classification": "unknown",
                "confidence": 0.0,
                "reasoning": f"Classification failed: {result['error']}",
                "error": result["error"]
            }
        
        try:
            data = result["data"]
            classification = data.get("classification", "unknown").lower()
            confidence = float(data.get("confidence", 0.0))
            reasoning = data.get("reasoning", "")
            
            return {
                "assertion": assertion,
                "classification": classification,
                "confidence": confidence,
                "reasoning": reasoning,
                "error": None
            }
        except Exception as e:
            return {
                "assertion": assertion,
                "classification": "unknown",
                "confidence": 0.0,
                "reasoning": "",
                "error": f"Failed to parse classification: {str(e)}"
            }
    
    def classify_batch(self, assertions: List[str]) -> List[Dict[str, Any]]:
        """Classify multiple assertions in a single Gemini call to reduce latency.

        Returns list of classification dicts (same shape as classify_assertion).
        Falls back to per-assertion classification if batch parsing fails.
        """
        if not assertions:
            return []
        # Build batch prompt
        joined = json.dumps(assertions[:50])  # safety bound
        prompt = f"""Classify each assertion in the following JSON array using Harari's Reality Taxonomy.

For EACH assertion return an object with:
{{"assertion": original, "classification": one of objective|subjective|intersubjective, "confidence": 0.0-1.0, "reasoning": brief explanation}}

Return ONLY a JSON array of objects.

ASSERTIONS:
{joined}
"""
        result = self.client.generate_json(prompt, timeout=60)
        if not result.get("ok"):
            # Fallback to individual classification
            return [self.classify_assertion(a) for a in assertions]
        try:
            data = result.get("data")
            if isinstance(data, list):
                out: List[Dict[str, Any]] = []
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    assertion = item.get("assertion") or ""
                    classification = str(item.get("classification", "unknown")).lower()
                    confidence = float(item.get("confidence", 0.0))
                    reasoning = item.get("reasoning", "")
                    out.append({
                        "assertion": assertion,
                        "classification": classification,
                        "confidence": confidence,
                        "reasoning": reasoning,
                        "error": None,
                    })
                # If counts mismatch fallback per assertion
                if len(out) != len(assertions):
                    return [self.classify_assertion(a) for a in assertions]
                return out
            # else fallback
            return [self.classify_assertion(a) for a in assertions]
        except Exception:
            return [self.classify_assertion(a) for a in assertions]

    def _enrich_objective(self, assertions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        objs = [a for a in assertions if a.get("classification") == "objective"]
        if not objs:
            return assertions
        prompt_payload = [o.get("assertion", "") for o in objs]
        prompt_json = json.dumps(prompt_payload[:40])  # safety cap
        current_date = date.today().isoformat()
        prompt = f"""Provide STRICT, up-to-date fact-check metadata for EACH objective reality assertion in this JSON array.

CURRENT_DATE: {current_date}

Temporal Resolution:
- Resolve relative time phrases ("today", "yesterday", "Tuesday", "last week") to concrete ISO dates based on CURRENT_DATE.
- If the assertion references a recent event (breaking news), prioritize sources published AFTER the referenced event date.

For each assertion return an object:
{{"assertion": original,
    "fact_check": {{
        "status": "verified"|"partially_verified"|"disputed"|"unclear",
        "verification_confidence": 0.0-1.0,
        "evidence": ["concise paraphrase of strongest supporting or refuting evidence"],
        "notes": "method + temporal resolution + any uncertainty; list search terms if unclear",
        "sources": [{{"title": "source title", "url": "https://...", "date": "YYYY-MM-DD"}}]
    }}
}}

Recency & Reliability Instructions (DO FIRST):
1. PRIORITIZE live/current information: conceptually simulate performing a web search BEFORE using internal knowledge.
2. Prefer sources < 90 days old for breaking events; if none found, use the most authoritative recent source (< 24 months) and mark older items explicitly.
3. Avoid hallucinating exact figures if not found; set status="unclear" with notes describing attempted search angle.
4. If multiple conflicting credible sources exist, use status="disputed" and include one supporting and one opposing evidence item.
5. Reject sources lacking provenance (no publisher/date) or obvious opinion pieces unless noting viewpoint.
 6. If initial internal knowledge does NOT yield a source < 90 days, ITERATIVELY simulate web search queries: start with 2-4 high-signal keywords from the assertion (include location or actor names), then refine by adding a timeframe term (e.g., "2025", "recent", last month). Continue until a recent source is found or you exhaust plausible variations. Record the final set of attempted query terms in `notes` if status becomes "unclear" or "disputed".
 7. Never backfill dates: if only older sources (>24 months) exist, explicitly state "no recent source located" in `notes`.

Return ONLY a JSON array; no commentary outside JSON.

ASSERTIONS:
{prompt_json}
"""
        result = self.client.generate_json(prompt, timeout=80)
        if not result.get("ok"):
            # Populate default fact_check structure
            for o in objs:
                o["fact_check"] = {
                    "status": "unclear",
                    "verification_confidence": 0.0,
                    "evidence": [],
                    "notes": result.get("error", "enrichment failed"),
                    "sources": []
                }
            return assertions
        try:
            data = result.get("data", [])
            mapping = {}
            for item in data:
                if isinstance(item, dict):
                    mapping[item.get("assertion", "").strip()] = item
            for o in objs:
                key = o.get("assertion", "").strip()
                enriched = mapping.get(key)
                if enriched and isinstance(enriched.get("fact_check"), dict):
                    o["fact_check"] = enriched["fact_check"]
                else:
                    o["fact_check"] = {
                        "status": "unclear",
                        "verification_confidence": 0.0,
                        "evidence": [],
                        "notes": "missing enrichment",
                        "sources": []
                    }
        except Exception as e:
            for o in objs:
                o["fact_check"] = {
                    "status": "unclear",
                    "verification_confidence": 0.0,
                    "evidence": [],
                    "notes": f"parse error: {str(e)}",
                    "sources": []
                }
        return assertions

    def _enrich_intersubjective(self, assertions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        inters = [a for a in assertions if a.get("classification") == "intersubjective"]
        if not inters:
            return assertions
        payload = [i.get("assertion", "") for i in inters]
        prompt_json = json.dumps(payload[:50])
        prompt = f"""Analyze EACH intersubjective assertion for stability and myth taxonomy.

Return ONLY a JSON array of objects like:
{{"assertion": original,
  "stability_index": {{"status": "naturalized"|"contested"|"ambiguous", "cues": ["phrase"], "reasoning": "brief"}},
  "myth_taxonomy": {{"category": "tribal_national"|"legal_bureaucratic"|"economic"|"divine_ideological"|"other", "confidence": 0.0-1.0, "reasoning": "brief"}}
}}

Guidance:
- naturalized: framed as permanent, unquestioned ("is", institutional declaratives).
- contested: language signalling challenge / fragility ("being questioned", "under attack").
- ambiguous: insufficient linguistic cues.

ASSERTIONS:
{prompt_json}
"""
        result = self.client.generate_json(prompt, timeout=70)
        if not result.get("ok"):
            for i in inters:
                i["stability_index"] = {"status": "ambiguous", "cues": [], "reasoning": result.get("error", "enrichment failed")}
                i["myth_taxonomy"] = {"category": "other", "confidence": 0.0, "reasoning": result.get("error", "enrichment failed")}
            return assertions
        try:
            data = result.get("data", [])
            mapping = {}
            for item in data:
                if isinstance(item, dict):
                    mapping[item.get("assertion", "").strip()] = item
            for i in inters:
                key = i.get("assertion", "").strip()
                enriched = mapping.get(key)
                if enriched:
                    i["stability_index"] = enriched.get("stability_index", {"status": "ambiguous", "cues": [], "reasoning": "missing"})
                    i["myth_taxonomy"] = enriched.get("myth_taxonomy", {"category": "other", "confidence": 0.0, "reasoning": "missing"})
                else:
                    i["stability_index"] = {"status": "ambiguous", "cues": [], "reasoning": "not returned"}
                    i["myth_taxonomy"] = {"category": "other", "confidence": 0.0, "reasoning": "not returned"}
        except Exception as e:
            for i in inters:
                i["stability_index"] = {"status": "ambiguous", "cues": [], "reasoning": f"parse error: {str(e)}"}
                i["myth_taxonomy"] = {"category": "other", "confidence": 0.0, "reasoning": f"parse error: {str(e)}"}
        return assertions

    def _enrich_subjective(self, assertions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        subs = [a for a in assertions if a.get("classification") == "subjective"]
        if not subs:
            return assertions
        payload = [s.get("assertion", "") for s in subs]
        prompt_json = json.dumps(payload[:50])
        prompt = f"""For EACH subjective assertion provide viral arousal and empathy span metrics.

Return ONLY a JSON array of objects:
{{"assertion": original,
  "viral_arousal": {{"category": "high"|"low"|"neutral", "emotion_tags": ["anger"], "arousal_score": 0.0-1.0, "rationale": "brief"}},
  "empathy_span": {{"sides_described": ["group"], "focus_bias": "one_sided"|"balanced"|"unclear", "entities_with_emotion": int, "notes": "brief"}}
}}

Emotion guidance: high -> anger,awe,anxiety,excitement,fear,hope; low -> sadness,contentment; neutral -> descriptive/no strong affect.

ASSERTIONS:
{prompt_json}
"""
        result = self.client.generate_json(prompt, timeout=70)
        if not result.get("ok"):
            for s in subs:
                s["viral_arousal"] = {"category": "neutral", "emotion_tags": [], "arousal_score": 0.0, "rationale": result.get("error", "enrichment failed")}
                s["empathy_span"] = {"sides_described": [], "focus_bias": "unclear", "entities_with_emotion": 0, "notes": result.get("error", "enrichment failed")}
            return assertions
        try:
            data = result.get("data", [])
            mapping = {}
            for item in data:
                if isinstance(item, dict):
                    mapping[item.get("assertion", "").strip()] = item
            for s in subs:
                key = s.get("assertion", "").strip()
                enriched = mapping.get(key)
                if enriched:
                    s["viral_arousal"] = enriched.get("viral_arousal", {"category": "neutral", "emotion_tags": [], "arousal_score": 0.0, "rationale": "missing"})
                    s["empathy_span"] = enriched.get("empathy_span", {"sides_described": [], "focus_bias": "unclear", "entities_with_emotion": 0, "notes": "missing"})
                else:
                    s["viral_arousal"] = {"category": "neutral", "emotion_tags": [], "arousal_score": 0.0, "rationale": "not returned"}
                    s["empathy_span"] = {"sides_described": [], "focus_bias": "unclear", "entities_with_emotion": 0, "notes": "not returned"}
        except Exception as e:
            for s in subs:
                s["viral_arousal"] = {"category": "neutral", "emotion_tags": [], "arousal_score": 0.0, "rationale": f"parse error: {str(e)}"}
                s["empathy_span"] = {"sides_described": [], "focus_bias": "unclear", "entities_with_emotion": 0, "notes": f"parse error: {str(e)}"}
        return assertions

    def _compute_enrichment_summaries(self, assertions: List[Dict[str, Any]]) -> Dict[str, Any]:
        fact = {"verified": 0, "partially_verified": 0, "disputed": 0, "unclear": 0}
        stab = {"naturalized": 0, "contested": 0, "ambiguous": 0}
        myth = {"tribal_national": 0, "legal_bureaucratic": 0, "economic": 0, "divine_ideological": 0, "other": 0}
        arousal = {"high": 0, "low": 0, "neutral": 0}
        empathy = {"one_sided": 0, "balanced": 0, "unclear": 0}
        for a in assertions:
            fc = a.get("fact_check") or {}
            fact_status = fc.get("status")
            if fact_status in fact:
                fact[fact_status] += 1
            si = a.get("stability_index") or {}
            st = si.get("status")
            if st in stab:
                stab[st] += 1
            mt = a.get("myth_taxonomy") or {}
            cat = mt.get("category")
            if cat in myth:
                myth[cat] += 1
            va = a.get("viral_arousal") or {}
            acat = va.get("category")
            if acat in arousal:
                arousal[acat] += 1
            es = a.get("empathy_span") or {}
            bias = es.get("focus_bias")
            if bias in empathy:
                empathy[bias] += 1
        return {
            "fact_check_summary": fact,
            "stability_summary": stab,
            "myth_summary": myth,
            "arousal_summary": arousal,
            "empathy_summary": empathy
        }

    def analyze_content(self, content: str, max_assertions: int = 10, quick_mode: bool = False) -> Dict[str, Any]:
        """Full pipeline with optional enrichment (skipped in quick mode)."""
        assertions = self.extract_assertions(content, max_assertions)
        if not assertions:
            return {
                "assertions": [],
                "summary": {"total": 0, "objective": 0, "subjective": 0, "intersubjective": 0, "unknown": 0},
                "error": "No assertions extracted from content"
            }
        if quick_mode:
            assertions = assertions[: min(len(assertions), max_assertions, 8)]
        classified = self.classify_batch(assertions)
        # Basic summary
        summary = {"objective": 0, "subjective": 0, "intersubjective": 0, "unknown": 0}
        for result in classified:
            summary[result.get("classification", "unknown")] = summary.get(result.get("classification", "unknown"), 0) + 1
        summary["total"] = len(classified)
        # Enrichment only in deep mode
        if not quick_mode:
            classified = self._enrich_objective(classified)
            classified = self._enrich_intersubjective(classified)
            classified = self._enrich_subjective(classified)
            enrich_summary = self._compute_enrichment_summaries(classified)
        else:
            enrich_summary = {}
        return {
            "assertions": classified,
            "summary": summary,
            **enrich_summary,
            "error": None
        }


# Convenience function
def analyze_reality_taxonomy(content: str, gemini_client: GeminiClient = None) -> Dict[str, Any]:
    """Analyze content using Harari's Reality Taxonomy."""
    if gemini_client is None:
        gemini_client = GeminiClient()
    
    analyzer = RealityTaxonomyAnalyzer(gemini_client)
    return analyzer.analyze_content(content)
