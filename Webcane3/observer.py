"""
Observer agent for WebCane3.
Provides visual page description using Groq Vision model.
Saves screenshots to folder for debugging.
"""

import os
import time
import base64
from typing import Optional

from .config import Config

# Groq imports
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("[Observer] groq not installed. Run: pip install groq")


class Observer:
    """
    Visual page observer using Groq Vision model.
    Describes what's visible on the page to provide context for planning.
    """
    
    # Screenshot save path
    SCREENSHOT_PATH = os.path.join(os.path.dirname(__file__), "current_screenshot.png")
    
    def __init__(self, api_key: str = None):
        """
        Initialize the observer.
        
        Args:
            api_key: Groq API key (uses GROQ_API_KEY3 from config)
        """
        self.client = None
        self.model_name = Config.GROQ_VISION_MODEL
        self.available = False
        self.last_description = None
        
        if not GROQ_AVAILABLE:
            print("[Observer] Groq SDK not available")
            return
        
        try:
            api_key = api_key or Config.GROQ_API_KEY3
            if not api_key:
                print("[Observer] No Groq API key (GROQ_API_KEY3) provided")
                return
            
            self.client = Groq(api_key=api_key)
            self.available = True
            print("[Observer] Ready (Groq)")
            
        except Exception as e:
            print(f"[Observer] Setup failed: {e}")
    
    def _save_screenshot(self, screenshot_bytes: bytes):
        """Save screenshot to file for debugging."""
        try:
            with open(self.SCREENSHOT_PATH, 'wb') as f:
                f.write(screenshot_bytes)
            print(f"[Observer] Screenshot saved: {self.SCREENSHOT_PATH}")
        except Exception as e:
            print(f"[Observer] Failed to save screenshot: {e}")
    
    def describe_page(self, screenshot_bytes: bytes, save_screenshot: bool = True) -> Optional[str]:
        """
        Analyze a screenshot and return a description of what's visible.
        
        Args:
            screenshot_bytes: PNG screenshot as bytes
            save_screenshot: Whether to save screenshot to file
            
        Returns:
            Description of the page, or None on failure
        """
        # Save screenshot for debugging
        if save_screenshot and screenshot_bytes:
            self._save_screenshot(screenshot_bytes)
        
        if not self.available:
            print("[Observer] Not available, returning None")
            return None
        
        try:
            # Rate limiting
            time.sleep(Config.API_DELAY)
            
            # Encode screenshot
            b64_image = base64.b64encode(screenshot_bytes).decode('utf-8')
            
            prompt = """Describe this webpage screenshot concisely.
Focus on:
1. What website/page is this?
2. What is the current state (home page, search results, video playing, etc.)?
3. What interactive elements are visible (videos, search results, buttons)?

Keep it brief (2-4 sentences). Example:
"This is YouTube search results for '4K videos'. Multiple video thumbnails are displayed in a grid. Each video shows title, channel name, and view count. A search bar is visible at the top."

Your description:"""
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64_image}"}
                            }
                        ]
                    }
                ],
                max_tokens=300,
                temperature=0.2
            )
            
            description = response.choices[0].message.content.strip()
            self.last_description = description
            
            # Print full context for debugging
            print("\n" + "=" * 50)
            print("OBSERVER CONTEXT (passed to planner):")
            print("=" * 50)
            print(description)
            print("=" * 50 + "\n")
            
            return description
            
        except Exception as e:
            print(f"[Observer] Failed: {e}")
            return None
    
    def describe_failure_context(
        self, 
        screenshot_bytes: bytes, 
        failed_action: str,
        failure_reason: str
    ) -> Optional[str]:
        """
        Analyze the page after a failure to provide context for replanning.
        """
        if not self.available:
            return None
        
        # Save screenshot
        if screenshot_bytes:
            self._save_screenshot(screenshot_bytes)
        
        try:
            time.sleep(Config.API_DELAY)
            
            b64_image = base64.b64encode(screenshot_bytes).decode('utf-8')
            
            prompt = f"""A web automation action just failed. Analyze this screenshot.

FAILED ACTION: {failed_action}
FAILURE REASON: {failure_reason}

Answer:
1. What page is currently displayed?
2. Why might the action have failed?
3. What should be tried instead?

Keep response under 4 sentences."""
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64_image}"}
                            }
                        ]
                    }
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            analysis = response.choices[0].message.content.strip()
            print(f"[Observer] Failure analysis: {analysis}")
            return analysis
            
        except Exception as e:
            print(f"[Observer] Failure analysis failed: {e}")
            return None
