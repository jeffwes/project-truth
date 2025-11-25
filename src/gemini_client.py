"""Gemini API client for The Resonance Engine."""
import os
import json
import requests
from typing import Dict, Any, Optional, List


class GeminiClient:
    """Client for Google Gemini 3.0 Pro API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize client with API key from environment or parameter."""
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY must be set in environment or passed to constructor")
        
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self.default_model = "gemini-3-pro-preview"
    
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
        timeout: int = 60
    ) -> Dict[str, Any]:
        """
        Generate content using Gemini API.
        
        Args:
            prompt: The input text prompt
            model: Model name (defaults to gemini-3-pro-preview)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            response_format: 'json' for JSON mode, None for text
            timeout: Request timeout in seconds
            
        Returns:
            Dict with 'ok', 'text', 'raw', 'error' keys
        """
        model = model or self.default_model
        url = f"{self.base_url}/{model}:generateContent"
        
        # Build request payload
        generation_config = {"temperature": temperature}
        if max_tokens:
            generation_config["maxOutputTokens"] = max_tokens
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": generation_config
        }
        
        # Add JSON response format if requested
        if response_format == "json":
            payload["generationConfig"]["response_mime_type"] = "application/json"
        
        try:
            response = requests.post(
                url,
                params={"key": self.api_key},
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=timeout
            )
            
            if response.status_code != 200:
                return {
                    "ok": False,
                    "text": None,
                    "raw": None,
                    "error": f"HTTP {response.status_code}: {response.text[:500]}"
                }
            
            data = response.json()
            
            # Extract text from response structure
            text = self._extract_text(data)
            
            return {
                "ok": True,
                "text": text,
                "raw": data,
                "error": None
            }
            
        except requests.exceptions.Timeout:
            return {
                "ok": False,
                "text": None,
                "raw": None,
                "error": f"Request timed out after {timeout} seconds"
            }
        except Exception as e:
            return {
                "ok": False,
                "text": None,
                "raw": None,
                "error": f"Request failed: {str(e)}"
            }
    
    def _extract_text(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract text content from API response."""
        try:
            candidates = data.get("candidates", [])
            if not candidates:
                return None
            
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            
            if not parts:
                return None
            
            # Combine all text parts
            text_parts = [part.get("text", "") for part in parts]
            return "".join(text_parts)
            
        except Exception:
            return None
    
    def generate_json(
        self,
        prompt: str,
        model: Optional[str] = None,
        timeout: int = 60
    ) -> Dict[str, Any]:
        """
        Generate JSON response using Gemini.
        
        Returns:
            Dict with 'ok', 'data' (parsed JSON), 'raw', 'error' keys
        """
        result = self.generate(
            prompt=prompt,
            model=model,
            response_format="json",
            temperature=0.3,  # Lower temp for structured output
            timeout=timeout
        )
        
        if not result["ok"]:
            return {
                "ok": False,
                "data": None,
                "raw": result["raw"],
                "error": result["error"]
            }
        
        # Parse JSON from text
        try:
            text = result["text"]
            if not text:
                return {
                    "ok": False,
                    "data": None,
                    "raw": result["raw"],
                    "error": "Empty response from API"
                }
            
            # Try direct JSON parse
            data = json.loads(text)
            
            return {
                "ok": True,
                "data": data,
                "raw": result["raw"],
                "error": None
            }
            
        except json.JSONDecodeError as e:
            return {
                "ok": False,
                "data": None,
                "raw": result["raw"],
                "error": f"Failed to parse JSON: {str(e)}"
            }


# Convenience function for simple use cases
def call_gemini(prompt: str, **kwargs) -> Dict[str, Any]:
    """Simple function interface to Gemini API."""
    client = GeminiClient()
    return client.generate(prompt, **kwargs)
