"""Reality Taxonomy Analyzer - Classifies assertions using Harari's framework."""
import json
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

CLASSIFICATION FRAMEWORK:

1. OBJECTIVE REALITY: Phenomena that exist independently of human consciousness
   - Examples: Radioactivity, river flow rates, biological viruses, gravity
   - These existed before humans and would continue without us
   
2. SUBJECTIVE REALITY: Phenomena that exist only in individual consciousness
   - Examples: Personal pain, individual feelings, private thoughts, personal preferences
   - Unique to each person, cannot be directly shared
   
3. INTERSUBJECTIVE REALITY: Phenomena sustained by shared beliefs
   - Examples: Money, laws, nations, corporations, human rights, religions
   - Exist because many humans believe in them; would vanish if belief ceased
   - The "shared myths" that organize human society

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

    def analyze_content(self, content: str, max_assertions: int = 10, quick_mode: bool = False) -> Dict[str, Any]:
        """
        Full pipeline: Extract assertions and classify each one.
        
        Args:
            content: Text content to analyze
            max_assertions: Maximum assertions to extract
            
        Returns:
            Dict with 'assertions' list and 'summary' statistics
        """
        # Extract assertions
        assertions = self.extract_assertions(content, max_assertions)
        
        if not assertions:
            return {
                "assertions": [],
                "summary": {
                    "total": 0,
                    "objective": 0,
                    "subjective": 0,
                    "intersubjective": 0,
                    "unknown": 0
                },
                "error": "No assertions extracted from content"
            }
        
        # Decide classification strategy
        if quick_mode:
            # In quick mode: reduce assertions for speed and batch classify
            assertions = assertions[: min(len(assertions), max_assertions, 8)]
            classified = self.classify_batch(assertions)
        else:
            # Use batch classification first; if fallback occurred, still acceptable
            classified = self.classify_batch(assertions)

        summary = {"objective": 0, "subjective": 0, "intersubjective": 0, "unknown": 0}
        for result in classified:
            classification = result.get("classification", "unknown")
            summary[classification] = summary.get(classification, 0) + 1
        
        summary["total"] = len(classified)
        
        return {
            "assertions": classified,
            "summary": summary,
            "error": None
        }


# Convenience function
def analyze_reality_taxonomy(content: str, gemini_client: GeminiClient = None) -> Dict[str, Any]:
    """Analyze content using Harari's Reality Taxonomy."""
    if gemini_client is None:
        gemini_client = GeminiClient()
    
    analyzer = RealityTaxonomyAnalyzer(gemini_client)
    return analyzer.analyze_content(content)
