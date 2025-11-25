"""Linguistic Analysis Module - Analyzes language patterns for manipulation detection."""
import json
from typing import Dict, Any, List
from src.gemini_client import GeminiClient


class LinguisticAnalyzer:
    """
    Analyzes linguistic patterns to detect persuasion tactics and manipulation.
    
    Five core modules:
    1. Agency & Responsibility (passive voice detection)
    2. Othering Index (us vs. them pronoun analysis)
    3. Dogmatism Score (modal verb certainty)
    4. Complexity & Populism (reading level)
    5. Persuasion Signature (rhetorical device density & valence)
    """
    
    def __init__(self, gemini_client: GeminiClient):
        """Initialize with Gemini client."""
        self.client = gemini_client
    
    def analyze_content(self, content: str) -> Dict[str, Any]:
        """
        Full linguistic forensics analysis.
        
        Args:
            content: Text content to analyze
            
        Returns:
            Dict with agency_analysis, polarization_metrics, certainty_metrics, 
            readability, and persuasion_signature keys
        """
        if not content or not content.strip():
            return self._empty_result()
        
        prompt = f"""Perform a COMPREHENSIVE linguistic forensics analysis on this text.

Return ONLY valid JSON with this EXACT structure:
{{
  "agency_analysis": {{
    "passive_voice_percent": 0-100,
    "hidden_actor_examples": ["example 1", "example 2"],
    "responsibility_interpretation": "brief analysis of how passive voice obscures accountability"
  }},
  "polarization_metrics": {{
    "us_vs_them_ratio": 0.0-10.0,
    "ingroup_pronouns": {{"we": count, "us": count, "our": count}},
    "outgroup_pronouns": {{"they": count, "them": count, "those": count}},
    "most_used_outgroup_label": "specific phrase or null",
    "polarization_interpretation": "brief analysis"
  }},
  "certainty_metrics": {{
    "dogmatism_score": 0-100,
    "high_modality_count": int,
    "low_modality_count": int,
    "dominant_modals": ["must", "will"],
    "certainty_interpretation": "brief analysis of epistemic stance"
  }},
  "readability": {{
    "grade_level": 0.0-20.0,
    "avg_sentence_length": float,
    "lexical_density": 0.0-1.0,
    "style_classification": "Academic|Populist|Conversational|Technical",
    "complexity_interpretation": "brief analysis"
  }},
  "persuasion_signature": {{
    "rhetorical_density_score": 0.0-100.0,
    "net_valence_score": -1.0 to +1.0,
    "classification": "Hit Piece|Manifesto|Dry Report|Balanced Analysis|Advocacy",
    "devices": [
      {{"name": "Strawman Argument", "count": int, "valence_score": -1.0 to +1.0, "examples": ["quote"]}},
      {{"name": "False Dichotomy", "count": int, "valence_score": -1.0 to +1.0, "examples": ["quote"]}},
      {{"name": "Ad Hominem", "count": int, "valence_score": -1.0 to +1.0, "examples": ["quote"]}},
      {{"name": "Slippery Slope", "count": int, "valence_score": -1.0 to +1.0, "examples": ["quote"]}},
      {{"name": "Whataboutism", "count": int, "valence_score": -1.0 to +1.0, "examples": ["quote"]}},
      {{"name": "Loaded Language", "count": int, "valence_score": -1.0 to +1.0, "examples": ["quote"]}},
      {{"name": "Dog Whistle", "count": int, "valence_score": -1.0 to +1.0, "examples": ["quote"]}},
      {{"name": "Proof by Gallup", "count": int, "valence_score": -1.0 to +1.0, "examples": ["quote"]}},
      {{"name": "Motte-and-Bailey", "count": int, "valence_score": -1.0 to +1.0, "examples": ["quote"]}},
      {{"name": "Anaphora", "count": int, "valence_score": -1.0 to +1.0, "examples": ["quote"]}},
      {{"name": "Catastrophizing", "count": int, "valence_score": -1.0 to +1.0, "examples": ["quote"]}},
      {{"name": "Appeal to Authority", "count": int, "valence_score": -1.0 to +1.0, "examples": ["quote"]}},
      {{"name": "Bandwagon", "count": int, "valence_score": -1.0 to +1.0, "examples": ["quote"]}},
      {{"name": "Euphemism/Dysphemism", "count": int, "valence_score": -1.0 to +1.0, "examples": ["quote"]}},
      {{"name": "Epistemic Closure", "count": int, "valence_score": -1.0 to +1.0, "examples": ["quote"]}}
    ],
    "signature_interpretation": "brief analysis"
  }}
}}

ANALYSIS INSTRUCTIONS:

1. AGENCY & RESPONSIBILITY:
   - Count passive voice constructions (was/were + past participle).
   - Identify where actors are hidden or responsibility is obscured.
   - Calculate passive_voice_percent = (passive sentences / total sentences) * 100.

2. POLARIZATION (US VS THEM):
   - Count: we, us, our, ours (ingroup).
   - Count: they, them, their, those people, the other side (outgroup).
   - Calculate ratio = outgroup_count / max(ingroup_count, 1).
   - Identify specific outgroup labels ("The Radical Left", "Big Tech", etc.).

3. CERTAINTY (DOGMATISM):
   - High modality: must, will, always, never, definitely, undoubtedly, is, are (declarative certainty).
   - Low modality: might, could, may, possibly, perhaps, seems, suggests.
   - dogmatism_score = (high_modality / (high_modality + low_modality)) * 100.

4. READABILITY & COMPLEXITY:
   - Estimate Flesch-Kincaid grade level (0-20 scale).
   - Lexical density = (content words / total words).
   - Style: Academic (>12), Conversational (6-10), Populist (<8 with simple vocab).

5. PERSUASION SIGNATURE:
   - Identify these SPECIFIC 15 rhetorical devices with counts and valence:
     
     1. STRAWMAN ARGUMENT: Misrepresenting opponent's argument to make it easier to attack (Valence: negative)
     2. FALSE DICHOTOMY (Binary Thinking): Presenting complex issues as having only two sides (Valence: negative)
     3. AD HOMINEM (Attacking the Person): Attacking character rather than argument (Valence: negative)
     4. SLIPPERY SLOPE: Asserting small first step inevitably leads to chain of negative events (Valence: negative)
     5. WHATABOUTISM: Deflection tactic that attempts to discredit opponent's position by charging hypocrisy (Valence: negative)
     6. LOADED LANGUAGE (Emotive Diction): Words with strong positive or negative connotations (Valence: can be positive or negative - mark each instance)
     7. DOG WHISTLE: Coded language that appears normal but communicates specific message to target subgroup (Valence: negative)
     8. PROOF BY GALLUP (Overwhelming the Reader): Excessive number of arguments to make refutation impossible (Valence: negative)
     9. MOTTE-AND-BAILEY: Conflating two positions - one modest (the Motte) one controversial (the Bailey) (Valence: negative)
     10. ANAPHORA: Repetition of word/phrase at beginning of successive clauses for hypnotic/rallying effect (Valence: positive)
     11. CATASTROPHIZING (Appeal to Fear): Emphasizing negative consequences to induce anxiety/panic (Valence: negative)
     12. APPEAL TO AUTHORITY (False Authority): Using "expert" opinion when expert not qualified (Valence: can be positive or negative)
     13. BANDWAGON: Arguing something is right simply because many people believe it (Valence: positive)
     14. EUPHEMISM / DYSPHEMISM: Mild words masking harsh realities OR harsh words degrading neutral concepts (Valence: positive for euphemism, negative for dysphemism)
     15. EPISTEMIC CLOSURE: Rhetoric designed to prevent reader from considering outside sources, framing external critique as "fake news" or "enemy propaganda" (Valence: negative)
   
   - For EACH device, return: {{"name": "device_name", "count": int, "valence_score": -1.0 to +1.0, "examples": ["quote1", "quote2"]}}
   - Count ALL 15 devices (use count=0 if not found).
   - Density = total_device_count / (word_count / 1000).
   - Net valence = average of all valence_scores weighted by count.
   - Classification based on density + valence quadrant.

TEXT (first 10k chars):
{content[:10000]}
"""
        
        result = self.client.generate_json(prompt, timeout=90)
        
        if not result.get("ok"):
            return {
                "agency_analysis": {"passive_voice_percent": 0, "hidden_actor_examples": [], "responsibility_interpretation": f"Analysis failed: {result.get('error', 'unknown error')}"},
                "polarization_metrics": {"us_vs_them_ratio": 0.0, "ingroup_pronouns": {}, "outgroup_pronouns": {}, "most_used_outgroup_label": None, "polarization_interpretation": "Analysis failed"},
                "certainty_metrics": {"dogmatism_score": 0, "high_modality_count": 0, "low_modality_count": 0, "dominant_modals": [], "certainty_interpretation": "Analysis failed"},
                "readability": {"grade_level": 0.0, "avg_sentence_length": 0.0, "lexical_density": 0.0, "style_classification": "Unknown", "complexity_interpretation": "Analysis failed"},
                "persuasion_signature": {"rhetorical_density_score": 0.0, "net_valence_score": 0.0, "classification": "Unknown", "devices": [], "signature_interpretation": "Analysis failed"},
                "error": result.get("error")
            }
        
        try:
            data = result.get("data", {})
            
            # Normalize and validate each module
            agency = data.get("agency_analysis", {})
            if not isinstance(agency, dict):
                agency = {}
            
            polarization = data.get("polarization_metrics", {})
            if not isinstance(polarization, dict):
                polarization = {}
            
            certainty = data.get("certainty_metrics", {})
            if not isinstance(certainty, dict):
                certainty = {}
            
            readability = data.get("readability", {})
            if not isinstance(readability, dict):
                readability = {}
            
            persuasion = data.get("persuasion_signature", {})
            if not isinstance(persuasion, dict):
                persuasion = {}
            
            return {
                "agency_analysis": {
                    "passive_voice_percent": int(agency.get("passive_voice_percent", 0)),
                    "hidden_actor_examples": agency.get("hidden_actor_examples", [])[:5],
                    "responsibility_interpretation": agency.get("responsibility_interpretation", "")
                },
                "polarization_metrics": {
                    "us_vs_them_ratio": float(polarization.get("us_vs_them_ratio", 0.0)),
                    "ingroup_pronouns": polarization.get("ingroup_pronouns", {}),
                    "outgroup_pronouns": polarization.get("outgroup_pronouns", {}),
                    "most_used_outgroup_label": polarization.get("most_used_outgroup_label"),
                    "polarization_interpretation": polarization.get("polarization_interpretation", "")
                },
                "certainty_metrics": {
                    "dogmatism_score": int(certainty.get("dogmatism_score", 0)),
                    "high_modality_count": int(certainty.get("high_modality_count", 0)),
                    "low_modality_count": int(certainty.get("low_modality_count", 0)),
                    "dominant_modals": certainty.get("dominant_modals", [])[:6],
                    "certainty_interpretation": certainty.get("certainty_interpretation", "")
                },
                "readability": {
                    "grade_level": float(readability.get("grade_level", 0.0)),
                    "avg_sentence_length": float(readability.get("avg_sentence_length", 0.0)),
                    "lexical_density": float(readability.get("lexical_density", 0.0)),
                    "style_classification": readability.get("style_classification", "Unknown"),
                    "complexity_interpretation": readability.get("complexity_interpretation", "")
                },
                "persuasion_signature": {
                    "rhetorical_density_score": float(persuasion.get("rhetorical_density_score", 0.0)),
                    "net_valence_score": float(persuasion.get("net_valence_score", 0.0)),
                    "classification": persuasion.get("classification", "Unknown"),
                    "devices": persuasion.get("devices", []),
                    "signature_interpretation": persuasion.get("signature_interpretation", "")
                },
                "error": None
            }
            
        except Exception as e:
            return {
                **self._empty_result(),
                "error": f"Failed to parse linguistic analysis: {str(e)}"
            }
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty structure when no content available."""
        return {
            "agency_analysis": {
                "passive_voice_percent": 0,
                "hidden_actor_examples": [],
                "responsibility_interpretation": "No content to analyze"
            },
            "polarization_metrics": {
                "us_vs_them_ratio": 0.0,
                "ingroup_pronouns": {},
                "outgroup_pronouns": {},
                "most_used_outgroup_label": None,
                "polarization_interpretation": "No content to analyze"
            },
            "certainty_metrics": {
                "dogmatism_score": 0,
                "high_modality_count": 0,
                "low_modality_count": 0,
                "dominant_modals": [],
                "certainty_interpretation": "No content to analyze"
            },
            "readability": {
                "grade_level": 0.0,
                "avg_sentence_length": 0.0,
                "lexical_density": 0.0,
                "style_classification": "Unknown",
                "complexity_interpretation": "No content to analyze"
            },
            "persuasion_signature": {
                "rhetorical_density_score": 0.0,
                "net_valence_score": 0.0,
                "classification": "Unknown",
                "devices": [],
                "signature_interpretation": "No content to analyze"
            },
            "error": "No content provided"
        }


# Convenience function
def analyze_linguistics(content: str, gemini_client: GeminiClient = None) -> Dict[str, Any]:
    """Analyze content for linguistic manipulation patterns."""
    if gemini_client is None:
        gemini_client = GeminiClient()
    
    analyzer = LinguisticAnalyzer(gemini_client)
    return analyzer.analyze_content(content)
