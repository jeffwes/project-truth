"""Content ingestion from browser tabs, YouTube, and text input."""
import re
from typing import Dict, Any, Optional
from urllib.parse import urlparse, parse_qs


class ContentIngester:
    """Handles ingestion of content from various sources."""
    
    def ingest_text(self, text: str) -> Dict[str, Any]:
        """
        Ingest raw text content.
        
        Returns:
            Dict with 'content', 'source_type', 'metadata' keys
        """
        if not text or not text.strip():
            return {
                "content": "",
                "source_type": "text",
                "metadata": {},
                "error": "Empty text provided"
            }
        
        return {
            "content": text.strip(),
            "source_type": "text",
            "metadata": {
                "char_count": len(text),
                "word_count": len(text.split())
            },
            "error": None
        }
    
    def ingest_url(self, url: str) -> Dict[str, Any]:
        """
        Ingest content from a URL (browser tab).
        
        For MVP, extracts URL and returns placeholder for browser integration.
        Future: Could use requests + BeautifulSoup for scraping.
        
        Returns:
            Dict with 'content', 'source_type', 'url', 'metadata' keys
        """
        if not url or not url.strip():
            return {
                "content": "",
                "source_type": "url",
                "url": None,
                "metadata": {},
                "error": "Empty URL provided"
            }
        
        # Validate URL format
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return {
                    "content": "",
                    "source_type": "url",
                    "url": url,
                    "metadata": {},
                    "error": "Invalid URL format"
                }
        except Exception as e:
            return {
                "content": "",
                "source_type": "url",
                "url": url,
                "metadata": {},
                "error": f"URL parsing error: {str(e)}"
            }
        
        # Check if YouTube URL
        if self._is_youtube_url(url):
            return self.ingest_youtube(url)
        
        return {
            "content": f"[Content from {url}]\n\nNote: For MVP, paste article text directly. Browser integration coming in future version.",
            "source_type": "url",
            "url": url,
            "metadata": {
                "domain": parsed.netloc,
                "path": parsed.path
            },
            "error": None
        }
    
    def ingest_youtube(self, url: str) -> Dict[str, Any]:
        """
        Ingest content from YouTube URL.
        
        For MVP, extracts video ID and returns placeholder.
        Future: Could integrate youtube-transcript-api for captions.
        
        Returns:
            Dict with 'content', 'source_type', 'video_id', 'metadata' keys
        """
        video_id = self._extract_youtube_id(url)
        
        if not video_id:
            return {
                "content": "",
                "source_type": "youtube",
                "video_id": None,
                "url": url,
                "metadata": {},
                "error": "Could not extract YouTube video ID"
            }
        
        return {
            "content": f"[YouTube Video: {video_id}]\n\nNote: For MVP, paste video transcript directly. Auto-transcript extraction coming in future version.",
            "source_type": "youtube",
            "video_id": video_id,
            "url": url,
            "metadata": {
                "video_id": video_id,
                "youtube_url": f"https://www.youtube.com/watch?v={video_id}"
            },
            "error": None
        }
    
    def _is_youtube_url(self, url: str) -> bool:
        """Check if URL is a YouTube link."""
        youtube_domains = ["youtube.com", "youtu.be", "www.youtube.com"]
        try:
            parsed = urlparse(url)
            return any(domain in parsed.netloc for domain in youtube_domains)
        except Exception:
            return False
    
    def _extract_youtube_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL."""
        try:
            parsed = urlparse(url)
            
            # Handle youtu.be short links
            if "youtu.be" in parsed.netloc:
                return parsed.path.lstrip("/").split("?")[0]
            
            # Handle youtube.com watch links
            if "youtube.com" in parsed.netloc:
                if "/watch" in parsed.path:
                    query_params = parse_qs(parsed.query)
                    video_ids = query_params.get("v", [])
                    if video_ids:
                        return video_ids[0]
                
                # Handle /embed/ links
                if "/embed/" in parsed.path:
                    return parsed.path.split("/embed/")[1].split("?")[0]
            
            return None
            
        except Exception:
            return None


# Convenience function
def ingest_content(source: str, source_type: str = "auto") -> Dict[str, Any]:
    """
    Ingest content from various sources.
    
    Args:
        source: URL, text content, or YouTube link
        source_type: 'auto', 'text', 'url', or 'youtube'
    
    Returns:
        Ingestion result dict
    """
    ingester = ContentIngester()
    
    if source_type == "auto":
        # Auto-detect source type
        if source.strip().startswith(("http://", "https://")):
            return ingester.ingest_url(source)
        else:
            return ingester.ingest_text(source)
    elif source_type == "text":
        return ingester.ingest_text(source)
    elif source_type == "url":
        return ingester.ingest_url(source)
    elif source_type == "youtube":
        return ingester.ingest_youtube(source)
    else:
        return {
            "content": "",
            "source_type": "unknown",
            "metadata": {},
            "error": f"Unknown source_type: {source_type}"
        }
