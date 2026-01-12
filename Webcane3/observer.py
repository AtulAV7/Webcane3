"""
Observer agent for WebCane3 ReAct workflow.
Provides action-oriented page analysis using Groq Vision model.
"""

import os
import time
import base64
import json
from typing import Optional, Dict, List

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
    Action-oriented page observer for ReAct workflow.
    Analyzes pages to provide context for the Supervisor's next action decision.
    """
    
    # Debug file paths
    DEBUG_DIR = os.path.dirname(__file__)
    SCREENSHOT_PATH = os.path.join(DEBUG_DIR, "current_screenshot.png")
    OBSERVATION_PATH = os.path.join(DEBUG_DIR, "last_observation.json")
    
    def __init__(self, api_key: str = None):
        """
        Initialize the observer.
        
        Args:
            api_key: Groq API key (uses GROQ_API_KEY3 from config)
        """
        self.client = None
        self.model_name = Config.GROQ_OBSERVER_MODEL
        self.available = False
        self.last_observation = None
        
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
            print(f"[Observer] Ready ({self.model_name})")
            
        except Exception as e:
            print(f"[Observer] Setup failed: {e}")
    
    def _save_screenshot(self, screenshot_bytes: bytes):
        """Save screenshot to file for debugging."""
        try:
            with open(self.SCREENSHOT_PATH, 'wb') as f:
                f.write(screenshot_bytes)
            print(f"[Observer] Screenshot saved to current_screenshot.png")
        except Exception as e:
            print(f"[Observer] Failed to save screenshot: {e}")
    
    def _save_observation(self, observation: Dict):
        """Save observation JSON to file for debugging."""
        try:
            with open(self.OBSERVATION_PATH, 'w', encoding='utf-8') as f:
                json.dump(observation, f, indent=2, ensure_ascii=False)
            print(f"[Observer] Observation saved to last_observation.json")
        except Exception as e:
            print(f"[Observer] Failed to save observation: {e}")
    
    def analyze_for_action(
        self,
        screenshot_bytes: bytes,
        goal: str,
        last_action: Optional[Dict] = None,
        last_action_success: Optional[bool] = None
    ) -> Dict:
        """
        Analyze a screenshot with action-oriented focus.
        
        This is the primary method for the ReAct workflow. It provides
        structured analysis that the Supervisor can use to decide the next action.
        
        Args:
            screenshot_bytes: PNG screenshot as bytes
            goal: The user's goal
            last_action: The previous action taken (if any)
            last_action_success: Whether the last action succeeded
            
        Returns:
            Dict with:
                - page_state: Current page description
                - blockers: List of detected popups/modals
                - previous_action_result: Analysis of last action outcome
                - key_elements: Notable interactive elements
        """
        # Save screenshot for debugging
        if screenshot_bytes:
            self._save_screenshot(screenshot_bytes)
        
        if not self.available:
            print("[Observer] Not available, returning minimal observation")
            return {
                "page_state": "Unknown page state",
                "blockers": [],
                "previous_action_result": "UNKNOWN",
                "key_elements": []
            }
        
        try:
            # Rate limiting
            time.sleep(Config.API_DELAY)
            
            # Encode screenshot
            b64_image = base64.b64encode(screenshot_bytes).decode('utf-8')
            
            # Build context about last action
            last_action_context = ""
            if last_action:
                action_type = last_action.get('action', 'unknown')
                target = last_action.get('target', 'unknown')
                query = last_action.get('query', '')
                success_text = "SUCCESS" if last_action_success else "FAILED" if last_action_success is False else "UNKNOWN"
                last_action_context = f"""
PREVIOUS ACTION: {action_type} on "{target}" {f'with query "{query}"' if query else ''}
RESULT: {success_text}
"""
            
            prompt = f"""Analyze this webpage screenshot for a web automation task.

GOAL: {goal}
{last_action_context}

Provide a DETAILED JSON response with these fields:

1. "page_state": A detailed description of the current page including:
   - What website this is
   - What page/section we're on (homepage, search results, product page, etc.)
   - DETAILED CONTENT STATE:
     * Checkbox states: Mention exactly which checkboxes are CHECKED or UNCHECKED
     * Radio buttons: Mention which option is selected
     * Dropdowns: Mention the current selected value
     * Input fields: Mention if they have text in them or are empty
     * Buttons: Mention if any are disabled or loading
     * Errors: Describe any visible error messages or validation alerts

2. "blockers": An array of blocking elements that must be handled first:
   - Popups, modals, overlays
   - Cookie consent banners
   - Login/signup prompts
   - Age verification
   - Any element covering the main content
   Empty array if none visible.

3. "previous_action_result": If there was a previous action, analyze if it worked:
   - "SUCCESS - only do if expected action is there for cases like typing something into a textbox[detailed evidence]" (e.g., "SUCCESS - search results now visible for 'phones', URL changed to /search")
   - "FAILED - [detailed reason]" (e.g., "FAILED - still on login page, error message 'Invalid password' visible")
   - "PARTIAL - [explanation]"
   - "N/A" if no previous action

4. "key_elements": Array of 5-8 notable interactive elements visible:
   - Be specific with descriptions and STATES (e.g., "Submit button [Disabled]", "Type checkbox [Checked]")
   - Include position hints
   - For lists/grids: mention items by name and position
   - Don't be hallucinated and output an empty textbox with 'text' as the value just seeing the goal demands.
   - Example: ["Search bar (empty) at top", "Submit button (enabled) bottom right", "'Terms' checkbox (unchecked)", "Category dropdown (selected: 'All')"]

5. "goal_blockers": Any visible text/messages that would PREVENT achieving the goal:
   - "Product currently unavailable" or "Out of stock" (blocks add-to-cart goals)
   - "Login required" or "Please sign in" (blocks actions requiring authentication)
   - "Item no longer exists" or "Page not found"
   - "Maximum quantity reached" or "Already in cart"
   - ANY status message, warning, or notice that indicates the goal CANNOT be achieved
   - This is DIFFERENT from UI blockers (popups) - these are informational messages
   - Return empty array if no such messages visible

6. "goal_progress": Brief assessment of how close we are to completing the goal:
   - "NOT_STARTED"
   - "IN_PROGRESS"
   - "ALMOST_DONE"
   - "COMPLETE"
   - "BLOCKED" (use this if goal_blockers has items)
   - "IMPOSSIBLE" (use if goal cannot be achieved based on visible information)

Example response:
{{
    "page_state": "Amazon product page for Samsung Galaxy. Shows product title, price Rs 74,999, and 'Currently unavailable' notice. No Add to Cart button visible.",
    "blockers": [],
    "goal_blockers": ["Currently unavailable - We don't know when or if this item will be back in stock"],
    "previous_action_result": "SUCCESS - Navigated to product page",
    "key_elements": ["Product title", "Price display", "Unavailable notice", "Similar products section"],
    "goal_progress": "IMPOSSIBLE"
}}

Respond with ONLY the JSON object, no other text."""

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
                max_tokens=800,
                temperature=0.2
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Try to parse JSON
            try:
                # Handle potential markdown code blocks
                if result_text.startswith("```"):
                    result_text = result_text.split("```")[1]
                    if result_text.startswith("json"):
                        result_text = result_text[4:]
                    result_text = result_text.strip()
                
                observation = json.loads(result_text)
                
                # Validate required fields
                if "page_state" not in observation:
                    observation["page_state"] = "Unknown"
                if "blockers" not in observation:
                    observation["blockers"] = []
                if "previous_action_result" not in observation:
                    observation["previous_action_result"] = "N/A"
                if "key_elements" not in observation:
                    observation["key_elements"] = []
                if "goal_progress" not in observation:
                    observation["goal_progress"] = "IN_PROGRESS"
                
            except json.JSONDecodeError:
                # Fallback: create structured response from text
                observation = {
                    "page_state": result_text[:300],
                    "blockers": [],
                    "previous_action_result": "N/A",
                    "key_elements": [],
                    "goal_progress": "UNKNOWN"
                }
            
            self.last_observation = observation
            
            # Save observation to file
            self._save_observation(observation)
            
            # Print detailed observation for debugging
            print("\n" + "=" * 60)
            print("OBSERVER ANALYSIS:")
            print("=" * 60)
            print(f"Page State: {observation.get('page_state', 'Unknown')[:200]}...")
            if observation.get('blockers'):
                print(f"BLOCKERS DETECTED: {observation['blockers']}")
            print(f"Previous Action: {observation.get('previous_action_result', 'N/A')}")
            print(f"Goal Progress: {observation.get('goal_progress', 'UNKNOWN')}")
            print(f"Key Elements ({len(observation.get('key_elements', []))}):")
            for i, elem in enumerate(observation.get('key_elements', [])[:5], 1):
                print(f"  {i}. {elem}")
            print("=" * 60 + "\n")
            
            return observation
            
        except Exception as e:
            print(f"[Observer] Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "page_state": "Analysis failed",
                "blockers": [],
                "previous_action_result": "UNKNOWN",
                "key_elements": [],
                "goal_progress": "UNKNOWN"
            }
    
    def describe_page(self, screenshot_bytes: bytes, save_screenshot: bool = True) -> Optional[str]:
        """
        Legacy method: Simple page description.
        Kept for backward compatibility.
        
        Args:
            screenshot_bytes: PNG screenshot as bytes
            save_screenshot: Whether to save screenshot to file
            
        Returns:
            Description of the page, or None on failure
        """
        if save_screenshot and screenshot_bytes:
            self._save_screenshot(screenshot_bytes)
        
        if not self.available:
            return None
        
        try:
            time.sleep(Config.API_DELAY)
            b64_image = base64.b64encode(screenshot_bytes).decode('utf-8')
            
            prompt = """Describe this webpage screenshot in detail.
Focus on:
1. What website/page is this?
2. What is the current state (home page, search results, video playing, etc.)?
3. What interactive elements are visible?
4. Any popups, modals, or blockers?

Be comprehensive (3-5 sentences)."""
            
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
                max_tokens=400,
                temperature=0.2
            )
            
            description = response.choices[0].message.content.strip()
            return description
            
        except Exception as e:
            print(f"[Observer] describe_page failed: {e}")
            return None
