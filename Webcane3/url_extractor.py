"""
URL Extractor for WebCane3.
Uses Groq LLM to extract website URLs from natural language goals.
"""

import re
from typing import Optional

from .config import Config

# Groq imports
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


class URLExtractor:
    """
    Extracts website URLs from natural language goal descriptions.
    Uses Groq LLM for intelligent extraction.
    """
    
    # Common website mappings
    KNOWN_SITES = {
        "youtube": "https://www.youtube.com",
        "google": "https://www.google.com",
        "flipkart": "https://www.flipkart.com",
        "amazon": "https://www.amazon.in",
        "wikipedia": "https://www.wikipedia.org",
        "github": "https://github.com",
        "twitter": "https://twitter.com",
        "x": "https://x.com",
        "facebook": "https://www.facebook.com",
        "instagram": "https://www.instagram.com",
        "linkedin": "https://www.linkedin.com",
        "reddit": "https://www.reddit.com",
        "stackoverflow": "https://stackoverflow.com",
        "netflix": "https://www.netflix.com",
        "spotify": "https://www.spotify.com",
    }
    
    def __init__(self, api_key: str = None):
        """Initialize the URL extractor."""
        self.groq_client = None
        
        if GROQ_AVAILABLE:
            try:
                api_key = api_key or Config.GROQ_API_KEY
                if api_key:
                    self.groq_client = Groq(api_key=api_key)
                    print("[URLExtractor] Ready")
            except Exception as e:
                print(f"[URLExtractor] Setup failed: {e}")
    
    def extract_url(self, goal: str) -> str:
        """
        Extract the target website URL from a goal description.
        
        Args:
            goal: Natural language goal (e.g., "Go to flipkart and search watches")
            
        Returns:
            URL string (defaults to google.com if not found)
        """
        goal_lower = goal.lower()
        
        # Check for known sites first (fast path)
        for site_name, site_url in self.KNOWN_SITES.items():
            if site_name in goal_lower:
                print(f"[URLExtractor] Found known site: {site_name} -> {site_url}")
                return site_url
        
        # Check for explicit URLs in the goal
        url_match = re.search(r'https?://[^\s]+', goal)
        if url_match:
            url = url_match.group(0).rstrip('.,!?')
            print(f"[URLExtractor] Found explicit URL: {url}")
            return url
        
        # Check for domain patterns (e.g., "go to example.com")
        domain_match = re.search(r'\b([a-zA-Z0-9-]+\.(com|org|net|io|co|in|edu))\b', goal)
        if domain_match:
            domain = domain_match.group(0)
            url = f"https://www.{domain}"
            print(f"[URLExtractor] Found domain: {url}")
            return url
        
        # Use LLM for intelligent extraction
        if self.groq_client:
            return self._extract_with_llm(goal)
        
        # Default fallback
        print("[URLExtractor] No URL found, defaulting to Google")
        return "https://www.google.com"
    
    def _extract_with_llm(self, goal: str) -> str:
        """Use Groq LLM to extract URL from goal."""
        try:
            prompt = f"""Extract the target website URL from this task. Output ONLY the URL, nothing else.

Task: "{goal}"

If a specific website is mentioned (e.g., YouTube, Flipkart, Amazon), return its URL.
If no website is mentioned, return: https://www.google.com

URL:"""
            
            response = self.groq_client.chat.completions.create(
                model=Config.GROQ_DOM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.0
            )
            
            result = response.choices[0].message.content.strip()
            
            # Validate it looks like a URL
            if result.startswith('http'):
                print(f"[URLExtractor] LLM extracted: {result}")
                return result
            
            # Try to make it a URL
            if '.' in result:
                url = f"https://www.{result}"
                print(f"[URLExtractor] LLM extracted (fixed): {url}")
                return url
            
        except Exception as e:
            print(f"[URLExtractor] LLM error: {e}")
        
        return "https://www.google.com"
