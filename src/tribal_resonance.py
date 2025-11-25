"""Tribal Resonance Predictor - Predicts which social tribes will resonate with content."""
import json
from typing import Dict, Any, List
from src.gemini_client import GeminiClient


class TribalResonancePredictor:
    """
    Predicts which social/political tribes will resonate with content.
    
    Based on moral foundations profile and content analysis, predicts:
    - Which tribes will find this content compelling
    - Why specific tribes will or won't resonate
    - Potential polarization effects
    """
    
    # Common tribal categories (can be expanded)
    TRIBE_CATEGORIES = {
        "progressive_left": "Progressive Left - High care, fairness, liberty. Lower authority, sanctity, loyalty.",
        "conservative_right": "Conservative Right - High loyalty, authority, sanctity. Moderate care, fairness, liberty.",
        "libertarian": "Libertarian - Very high liberty. Low authority. Moderate fairness.",
        "centrist": "Centrist/Moderate - Balanced across foundations. Pragmatic orientation.",
        "populist": "Populist - High loyalty, anti-authority (elite), fairness concerns. Variable on other foundations.",
        "religious_conservative": "Religious Conservative - Very high sanctity, authority. High loyalty, care. Moderate liberty.",
        "social_justice": "Social Justice - Very high care, fairness, liberty. Lower loyalty, authority, sanctity."
    }
    
    def __init__(self, gemini_client: GeminiClient):
        """Initialize with Gemini client."""
        self.client = gemini_client
    
    def predict_resonance(
        self,
        content: str,
        moral_foundations: Dict[str, Any],
        reality_taxonomy: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Predict which tribes will resonate with content.
        
        Args:
            content: Original content text
            moral_foundations: Result from MoralFoundationsAnalyzer
            reality_taxonomy: Optional result from RealityTaxonomyAnalyzer
            
        Returns:
            Dict with tribe predictions and resonance scores
        """
        if not content or not content.strip():
            return {
                "predictions": [],
                "polarization_risk": "unknown",
                "error": "Empty content provided"
            }
        
        # Build context from moral foundations
        foundations_summary = self._format_foundations_for_prompt(moral_foundations)
        
        # Build context from reality taxonomy if available
        taxonomy_summary = ""
        if reality_taxonomy:
            taxonomy_summary = self._format_taxonomy_for_prompt(reality_taxonomy)
        
        # Build tribe descriptions
        tribes_desc = "\n".join([f"- {name}: {desc}" for name, desc in self.TRIBE_CATEGORIES.items()])
        
        prompt = f"""Predict which social/political tribes will resonate with this content.

CONTENT EXCERPT:
{content[:4000]}

MORAL FOUNDATIONS PROFILE:
{foundations_summary}

{taxonomy_summary}

TRIBAL CATEGORIES:
{tribes_desc}

For each tribe, predict:
1. Resonance score (0.0-1.0): How strongly will this tribe resonate?
2. Reasoning: Why will they find it compelling or reject it?
3. Specific hooks: What elements will attract or repel them?

Also assess:
- Polarization risk: Will this content divide or unite tribes?
- Cross-tribal appeal: Does it bridge or reinforce divides?

Return JSON:
{{
    "tribes": [
        {{
            "name": "tribe_key",
            "resonance_score": 0.0-1.0,
            "sentiment": "positive" | "negative" | "neutral",
            "reasoning": "why they'll resonate or reject",
            "hooks": ["specific appealing elements"]
        }},
        ...
    ],
    "polarization_risk": "low" | "medium" | "high",
    "polarization_explanation": "explanation",
    "cross_tribal_potential": "Summary of bridges or divides",
    "overall_tribal_signature": "One sentence summary of who this resonates with most"
}}"""
        
        result = self.client.generate_json(prompt, timeout=60)
        
        if not result["ok"]:
            return {
                "predictions": [],
                "polarization_risk": "unknown",
                "error": result["error"]
            }
        
        try:
            data = result["data"]
            
            tribes = data.get("tribes", [])
            polarization_risk = data.get("polarization_risk", "unknown")
            polarization_explanation = data.get("polarization_explanation", "")
            cross_tribal = data.get("cross_tribal_potential", "")
            signature = data.get("overall_tribal_signature", "")
            
            # Sort tribes by resonance score
            tribes_sorted = sorted(
                tribes,
                key=lambda x: x.get("resonance_score", 0.0),
                reverse=True
            )
            
            return {
                "predictions": tribes_sorted,
                "polarization_risk": polarization_risk,
                "polarization_explanation": polarization_explanation,
                "cross_tribal_potential": cross_tribal,
                "tribal_signature": signature,
                "error": None
            }
            
        except Exception as e:
            return {
                "predictions": [],
                "polarization_risk": "unknown",
                "error": f"Failed to parse tribal predictions: {str(e)}"
            }
    
    def _format_foundations_for_prompt(self, moral_foundations: Dict[str, Any]) -> str:
        """Format moral foundations data for prompt."""
        foundations = moral_foundations.get("foundations", {})
        
        if not foundations:
            return "No moral foundations data available."
        
        lines = []
        for name, data in foundations.items():
            if isinstance(data, dict):
                triggered = data.get("triggered", False)
                intensity = data.get("intensity", 0.0)
                valence = data.get("valence", "neutral")
                
                if triggered:
                    lines.append(f"- {name.title()}: {intensity:.1f} intensity, {valence} valence")
        
        return "\n".join(lines) if lines else "No significant moral foundations triggered."
    
    def _format_taxonomy_for_prompt(self, reality_taxonomy: Dict[str, Any]) -> str:
        """Format reality taxonomy data for prompt."""
        summary = reality_taxonomy.get("summary", {})
        
        if not summary:
            return ""
        
        total = summary.get("total", 0)
        if total == 0:
            return ""
        
        obj = summary.get("objective", 0)
        subj = summary.get("subjective", 0)
        inter = summary.get("intersubjective", 0)
        
        return f"""REALITY TAXONOMY PROFILE:
- Objective assertions: {obj}/{total} ({obj/total*100:.0f}%)
- Subjective assertions: {subj}/{total} ({subj/total*100:.0f}%)
- Intersubjective assertions: {inter}/{total} ({inter/total*100:.0f}%)
"""
    
    def get_top_tribes(self, predictions: Dict[str, Any], top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Get top N tribes most likely to resonate.
        
        Returns:
            List of tribe prediction dicts, sorted by resonance score
        """
        all_predictions = predictions.get("predictions", [])
        return all_predictions[:top_n]


# Convenience function
def predict_tribal_resonance(
    content: str,
    moral_foundations: Dict[str, Any],
    reality_taxonomy: Dict[str, Any] = None,
    gemini_client: GeminiClient = None
) -> Dict[str, Any]:
    """Predict tribal resonance for content."""
    if gemini_client is None:
        gemini_client = GeminiClient()
    
    predictor = TribalResonancePredictor(gemini_client)
    return predictor.predict_resonance(content, moral_foundations, reality_taxonomy)
