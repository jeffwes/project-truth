"""Moral Foundations Theory analyzer."""
import json
from typing import Dict, Any, List
from src.gemini_client import GeminiClient


class MoralFoundationsAnalyzer:
    """
    Analyzes content using Moral Foundations Theory (Jonathan Haidt).
    
    Six foundations:
    1. Care/Harm - Sensitivity to suffering
    2. Fairness/Cheating - Justice and reciprocity  
    3. Loyalty/Betrayal - Group cohesion and tribalism
    4. Authority/Subversion - Respect for hierarchy and tradition
    5. Sanctity/Degradation - Purity and disgust
    6. Liberty/Oppression - Freedom from domination
    """
    
    # Foundation definitions for prompts
    FOUNDATIONS = {
        "care": "Care/Harm - Sensitivity to suffering, compassion, nurturing. Triggered by suffering, distress, vulnerability, protection of weak.",
        "fairness": "Fairness/Cheating - Justice, reciprocity, equality. Triggered by cooperation, fairness, proportionality, rights violations.",
        "loyalty": "Loyalty/Betrayal - Group cohesion, patriotism, tribal identity. Triggered by in-group solidarity, betrayal, team spirit, national identity.",
        "authority": "Authority/Subversion - Respect for hierarchy, tradition, leadership. Triggered by deference to authority, tradition, order, rebellion against legitimate power.",
        "sanctity": "Sanctity/Degradation - Purity, sacredness, disgust. Triggered by purity/contamination, sacred values, moral disgust, elevated ideals.",
        "liberty": "Liberty/Oppression - Freedom from domination, individual autonomy. Triggered by tyranny, oppression, freedom, resistance to control."
    }
    
    def __init__(self, gemini_client: GeminiClient):
        """Initialize with Gemini client."""
        self.client = gemini_client
    
    def analyze_foundations(self, content: str) -> Dict[str, Any]:
        """
        Analyze which moral foundations are triggered by content.
        
        Args:
            content: Text content to analyze
            
        Returns:
            Dict with foundation scores, triggers, and analysis
        """
        if not content or not content.strip():
            return {
                "foundations": {},
                "overall_moral_profile": "",
                "error": "Empty content provided"
            }
        
        # Build comprehensive prompt
        foundations_desc = "\n".join([f"{i+1}. {desc}" for i, desc in enumerate(self.FOUNDATIONS.values())])
        
        prompt = f"""Analyze this content using Moral Foundations Theory (Jonathan Haidt).

CONTENT:
{content[:8000]}

MORAL FOUNDATIONS THEORY - COMPREHENSIVE FRAMEWORK:

Moral Foundations Theory (Jonathan Haidt) identifies six innate psychological systems that shape moral judgment across cultures. These "foundations" are like taste receptors—everyone has them, but political tribes differ dramatically in which foundations they weight most heavily. Understanding which foundations a piece of content activates reveals its tribal targeting and persuasive strategy.

THE SIX FOUNDATIONS:

1. CARE/HARM
Core Concern: Suffering, compassion, nurturing, protection of vulnerable
Triggered By: Children in distress, victims of cruelty, medical suffering, vulnerable populations, protective instincts
Political Pattern:
- Progressives: VERY HIGH (primary moral foundation)
- Conservatives: MODERATE (balanced with other foundations)
Examples: Healthcare access, child welfare, refugee protection, pandemic response, animal welfare

2. FAIRNESS/CHEATING  
Core Concern: Justice, reciprocity, proportionality, equality, rights violations
Triggered By: Unfair treatment, exploitation, unequal distribution, cheating, rule violations
Political Pattern:
- Progressives: VERY HIGH (equality/equity emphasis)
- Libertarians: HIGH (procedural fairness, meritocracy)
- Conservatives: MODERATE (proportional fairness, "you get what you deserve")
Examples: Taxation debates, affirmative action, economic inequality, legal rights, meritocracy vs. equity

3. LOYALTY/BETRAYAL
Core Concern: Group cohesion, patriotism, tribal solidarity, team identity
Triggered By: National symbols, in-group pride, betrayal of the group, team sports, military service
Political Pattern:
- Conservatives: VERY HIGH ("America First", military reverence)
- Progressives: LOW (universal humanity emphasis, skepticism of nationalism)
Examples: Flag rituals, immigration restrictionism, "anti-American" accusations, "with us or against us" rhetoric

4. AUTHORITY/SUBVERSION
Core Concern: Respect for hierarchy, tradition, leadership, social order
Triggered By: Deference to legitimate authority, respect for tradition, order vs. chaos, rebellion against rightful power
Political Pattern:
- Conservatives: VERY HIGH (respect for police, military, religious authority, tradition)
- Progressives: LOW (questioning authority, dismantling hierarchies)
Examples: "Back the Blue", religious authority, traditional gender roles, "law and order", institutional deference

5. SANCTITY/DEGRADATION
Core Concern: Purity, sacredness, disgust, elevated ideals vs. contamination
Triggered By: Purity/contamination, sacred values, moral disgust, spiritual elevation, bodily integrity
Political Pattern:
- Religious Conservatives: VERY HIGH (sexual purity, religious sanctity)
- Progressives: MODERATE-LOW (but can be HIGH around environmental purity, body autonomy as sacred)
Examples: Sexual morality, religious reverence, "sanctity of life", environmental purity, bodily autonomy, GMO fears

6. LIBERTY/OPPRESSION
Core Concern: Freedom from domination, resistance to tyranny, individual autonomy
Triggered By: Tyranny, oppression, bullying, freedom, resistance to control, individual rights
Political Pattern:
- Libertarians: VERY HIGH (primary foundation)
- Populists: HIGH (anti-elite, anti-establishment)
- Progressives: HIGH (anti-oppression, liberation movements)
- Conservatives: MODERATE (freedom within order)
Examples: Free speech, gun rights, vaccine mandates, anti-authoritarianism, liberation movements, "don't tread on me"

KEY INSIGHTS:

- Progressives rely heavily on Care and Fairness, with lower emphasis on Loyalty, Authority, and Sanctity. This makes progressive messaging focused on suffering reduction and equality.

- Conservatives weight all six foundations more evenly, with especially high Loyalty, Authority, and Sanctity. This makes conservative messaging emphasize tradition, order, and group cohesion.

- Libertarians spike on Liberty/Oppression and downplay Authority, making them suspicious of both government power and social conformity pressure.

- Most political conflict stems from tribes talking past each other because they're activating different moral foundations. What progressives see as compassion (high Care), conservatives may see as naive bleeding-heart sentimentality that ignores group boundaries (low Loyalty) and proper order (low Authority).

MORAL FOUNDATIONS TO ANALYZE:
{foundations_desc}

For each foundation, determine:
1. Is it triggered? (true/false)
2. Intensity (0.0-1.0): How strongly is it invoked?
3. Specific triggers: What language/concepts activate this foundation?
4. Valence: Is it appealed to positively or violated/attacked?

Return JSON in this structure:
{{
    "care": {{
        "triggered": true/false,
        "intensity": 0.0-1.0,
        "triggers": ["specific phrase or concept", ...],
        "valence": "positive" | "negative" | "neutral",
        "explanation": "brief explanation"
    }},
    "fairness": {{ ... }},
    "loyalty": {{ ... }},
    "authority": {{ ... }},
    "sanctity": {{ ... }},
    "liberty": {{ ... }},
    "overall_profile": "Summary of the content's moral signature in 1-2 sentences"
}}"""
        
        result = self.client.generate_json(prompt, timeout=60)
        
        if not result["ok"]:
            return {
                "foundations": {},
                "overall_profile": "",
                "error": result["error"]
            }
        
        try:
            data = result["data"]
            
            # Extract foundation scores
            foundations = {}
            for foundation_key in self.FOUNDATIONS.keys():
                if foundation_key in data:
                    foundations[foundation_key] = data[foundation_key]
            
            overall_profile = data.get("overall_profile", "")
            
            return {
                "foundations": foundations,
                "overall_profile": overall_profile,
                "error": None
            }
            
        except Exception as e:
            return {
                "foundations": {},
                "overall_profile": "",
                "error": f"Failed to parse moral foundations analysis: {str(e)}"
            }
    
    def get_foundation_summary(self, analysis: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract simple intensity scores for each foundation.
        
        Args:
            analysis: Result from analyze_foundations()
            
        Returns:
            Dict mapping foundation name to intensity (0.0-1.0)
        """
        foundations = analysis.get("foundations", {})
        summary = {}
        
        for name, data in foundations.items():
            if isinstance(data, dict):
                intensity = data.get("intensity", 0.0)
                summary[name] = float(intensity)
            else:
                summary[name] = 0.0
        
        return summary
    
    def get_top_foundations(self, analysis: Dict[str, Any], top_n: int = 3) -> List[tuple]:
        """
        Get the top N most strongly triggered foundations.
        
        Returns:
            List of (foundation_name, intensity) tuples, sorted by intensity
        """
        summary = self.get_foundation_summary(analysis)
        sorted_foundations = sorted(summary.items(), key=lambda x: x[1], reverse=True)
        return sorted_foundations[:top_n]


# Convenience function
def analyze_moral_foundations(content: str, gemini_client: GeminiClient = None) -> Dict[str, Any]:
    """Analyze content using Moral Foundations Theory."""
    if gemini_client is None:
        gemini_client = GeminiClient()
    
    analyzer = MoralFoundationsAnalyzer(gemini_client)
    return analyzer.analyze_foundations(content)
