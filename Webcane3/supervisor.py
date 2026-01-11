"""
Supervisor agent for WebCane3 ReAct workflow.
Single-step decision maker using NVIDIA DeepSeek-V3.2 with thinking.
Uses LangChain NVIDIA AI Endpoints for better streaming support.
"""

import json
import re
import time
from typing import Dict, List, Optional

from .config import Config

# LangChain NVIDIA imports
try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    LANGCHAIN_NVIDIA_AVAILABLE = True
except ImportError:
    LANGCHAIN_NVIDIA_AVAILABLE = False
    print("[Supervisor] langchain-nvidia-ai-endpoints not installed. Run: pip install langchain-nvidia-ai-endpoints")


class Supervisor:
    """
    Step-by-step decision maker for ReAct workflow.
    
    Uses NVIDIA DeepSeek-V3.1-Terminus with thinking to decide ONE action at a time based on:
    - Current page observation
    - Detected blockers
    - Execution history (for loop detection)
    - Original goal
    
    Outputs either an action to execute or a termination signal (COMPLETE/FAILED).
    """
    
    # Supported actions
    SUPPORTED_ACTIONS = [
        "navigate",     # Go to a URL
        "search",       # Macro: click search + type + enter
        "click",        # Click an element
        "type",         # Type text into focused field
        "scroll_find",  # Scroll to find a visual element
        "scroll",       # Simple scroll
        "press_key",    # Press keyboard key
        "dismiss",      # Dismiss popup/modal
        "COMPLETE",     # Goal achieved
        "FAILED"        # Cannot complete goal
    ]
    
    def __init__(self):
        """Initialize the Supervisor with LangChain NVIDIA client."""
        self.model_name = Config.NVIDIA_SUPERVISOR_MODEL
        self.api_key = Config.NVIDIA_API_KEY2
        self.client = None
        self.available = False
        
        if not LANGCHAIN_NVIDIA_AVAILABLE:
            print("[Supervisor] LangChain NVIDIA SDK not available")
            return
        
        if not self.api_key:
            print("[Supervisor] NVIDIA_API_KEY2 not configured")
            return
        
        try:
            self.client = ChatNVIDIA(
                model=self.model_name,
                api_key=self.api_key,
                temperature=0.2,
                top_p=0.7,
                max_tokens=4096,  # Increased for thinking mode - needs room for both reasoning + JSON output
                model_kwargs={"chat_template_kwargs": {"thinking": True}}
            )
            self.available = True
            print(f"[Supervisor] Ready ({self.model_name}) with thinking enabled")
        except Exception as e:
            print(f"[Supervisor] Setup failed: {e}")
    
    def decide_next_action(
        self,
        goal: str,
        observation: str,
        blockers: List[str],
        execution_history: List[Dict],
        current_url: str,
        key_elements: List[str] = None
    ) -> Dict:
        """
        Decide the next single action to take.
        
        Args:
            goal: The user's goal
            observation: Current page state description
            blockers: List of detected blocking elements
            execution_history: Last N actions taken
            current_url: Current browser URL
            key_elements: Notable interactive elements on page
            
        Returns:
            Dict with action specification:
            - Action: {"action": "search", "target": "search bar", "query": "samsung phones"}
            - Complete: {"action": "COMPLETE", "reason": "Goal achieved"}
            - Failed: {"action": "FAILED", "reason": "Unable to proceed"}
        """
        if not self.available:
            return self._fallback_decision(goal, observation, blockers)
        
        try:
            # Build history context
            history_text = self._format_history(execution_history[-5:])
            
            # Build blockers context
            blockers_text = "None" if not blockers else "\n".join(f"- {b}" for b in blockers)
            
            # Build key elements context
            elements_text = "Unknown" if not key_elements else "\n".join(f"- {e}" for e in key_elements[:5])
            
            # Check for repeated failures
            loop_warning = self._detect_loops(execution_history)
            
            prompt = f"""You are a web automation supervisor. Decide the NEXT SINGLE ACTION to take.

GOAL: {goal}

CURRENT STATE:
- URL: {current_url}
- Page: {observation}

BLOCKING ELEMENTS (handle these FIRST if any):
{blockers_text}

KEY ELEMENTS VISIBLE:
{elements_text}

RECENT ACTIONS (last 5):
{history_text}

{loop_warning}

AVAILABLE ACTIONS:
1. navigate - Go to a URL. Example: {{"action": "navigate", "target": "https://amazon.in"}}
2. search - Find search input, type query, press Enter (MACRO). Example: {{"action": "search", "target": "search bar", "query": "samsung phones"}}
3. click - Click an element by description. Example: {{"action": "click", "target": "first product result"}}
4. type - Type text into already-focused field. Example: {{"action": "type", "target": "samsung phones"}}
5. scroll_find - Scroll to find a visual element. Example: {{"action": "scroll_find", "target": "video with nature thumbnail"}}
6. scroll - Simple scroll. Example: {{"action": "scroll", "target": "down"}}
7. press_key - Press a key. Example: {{"action": "press_key", "target": "Enter"}}
8. dismiss - Dismiss popup/modal. Example: {{"action": "dismiss", "target": "close button"}}
9. COMPLETE - Goal is FULLY achieved. Use this ONLY when the goal is done.
10. FAILED - Cannot proceed. Example: {{"action": "FAILED", "reason": "Login required"}}

CRITICAL RULES FOR GOAL COMPLETION:
- ONLY do what the user asked for, nothing more!
- If goal is "search X" -> COMPLETE once search results are showing
- If goal is "go to site and search X" -> COMPLETE once search results are showing
- If goal is "click X" -> COMPLETE once X is clicked and page responds
- If goal is "search X and click Y" -> COMPLETE once Y is clicked
- Do NOT automatically click on results unless the goal explicitly asks to!
- Do NOT do extra actions beyond what the goal specifies!

EXAMPLES:
- Goal: "search 4k videos" + Page shows search results -> {{"action": "COMPLETE", "reason": "Search results for 4k videos are now displayed"}}
- Goal: "search animals" + Page shows animal search results -> {{"action": "COMPLETE", "reason": "Search results for animals are showing"}}
- Goal: "find laptop deals and click first one" + Clicked first deal -> {{"action": "COMPLETE", "reason": "Clicked on first laptop deal"}}

DECISION RULES:
1. If there are BLOCKERS, handle them FIRST
2. Use "search" macro for any search task
3. CHECK THE GOAL CAREFULLY - if it's achieved, output COMPLETE immediately!
4. If stuck after 3+ failed attempts, try alternative or FAILED
5. Be specific with target descriptions

CRITICAL - NAVIGATION GUARDS:
- NEVER navigate to a random/different URL unless the goal explicitly says to
- If you're already on a page with the needed elements, STAY on that page
- If actions are succeeding, CONTINUE on the current page
- Only use "navigate" if you're on the wrong site OR goal says "go to..."

CONTINUE MODE:
- If goal starts with "Now..." or implies continuing, you're on the right page
- Focus on completing remaining steps, don't navigate away
- Check RECENT ACTIONS to see what's already done - don't repeat!

Output ONLY a valid JSON object. No explanation."""

            time.sleep(Config.API_DELAY)
            
            print("[Supervisor] Thinking and deciding next action...")
            
            # Use LangChain streaming
            reasoning_content = ""
            response_content = ""
            
            for chunk in self.client.stream([{"role": "user", "content": prompt}]):
                # Get reasoning content if available
                if chunk.additional_kwargs and "reasoning_content" in chunk.additional_kwargs:
                    reasoning_content += chunk.additional_kwargs["reasoning_content"]
                
                # Get main content
                if chunk.content:
                    response_content += chunk.content
            
            # Print reasoning for debugging
            if reasoning_content:
                print("\n[Supervisor] Thinking:")
                print("-" * 40)
                print(reasoning_content[:500] + ("..." if len(reasoning_content) > 500 else ""))
                print("-" * 40)
            
            # Handle empty response - try to extract from reasoning
            if not response_content.strip():
                print("[Supervisor] WARNING: Empty response content!")
                if reasoning_content:
                    # Try to find JSON in reasoning
                    decision = self._parse_decision(reasoning_content)
                    if decision.get('action') != 'FAILED':
                        print("[Supervisor] Extracted decision from reasoning content")
                    else:
                        # Use fallback decision - DO NOT assume COMPLETE
                        # The reasoning might just be analyzing the goal, not saying it's done
                        print("[Supervisor] Using fallback decision logic...")
                        decision = self._fallback_decision(goal, observation, blockers)
                else:
                    decision = self._fallback_decision(goal, observation, blockers)
            else:
                # Parse JSON response
                decision = self._parse_decision(response_content)
            
            # Validate action
            if decision.get('action') not in self.SUPPORTED_ACTIONS:
                print(f"[Supervisor] Invalid action: {decision.get('action')}")
                decision = self._normalize_action(decision)
            
            # Print decision
            print("\n" + "-" * 50)
            print("SUPERVISOR DECISION:")
            print(f"  Action: {decision.get('action')}")
            if decision.get('target'):
                print(f"  Target: {decision.get('target')}")
            if decision.get('query'):
                print(f"  Query: {decision.get('query')}")
            if decision.get('reason'):
                print(f"  Reason: {decision.get('reason')}")
            print("-" * 50)
            
            return decision
            
        except Exception as e:
            print(f"[Supervisor] Error: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_decision(goal, observation, blockers)
    
    def _format_history(self, history: List[Dict]) -> str:
        """Format execution history for prompt."""
        if not history:
            return "No actions taken yet"
        
        lines = []
        for entry in history:
            action = entry.get('action', 'unknown')
            target = entry.get('target', '')[:30]
            success = "SUCCESS" if entry.get('success') else "FAILED"
            
            # Include Vision confirmation if available (proves what was clicked)
            vision_info = ""
            if entry.get('vision_confirmed'):
                vision_info = f" [Vision confirmed: {entry['vision_confirmed'][:50]}...]"
            
            lines.append(f"- {action} '{target}' -> {success}{vision_info}")
        
        return "\n".join(lines)
    
    def _detect_loops(self, history: List[Dict]) -> str:
        """Detect if we're stuck in a loop."""
        if len(history) < 2:
            return ""
        
        recent = history[-3:]
        failed_targets = [
            h.get('target', '')
            for h in recent
            if not h.get('success')
        ]
        
        if len(failed_targets) >= 2:
            from collections import Counter
            counts = Counter(failed_targets)
            for target, count in counts.items():
                if count >= 2:
                    return f"""
WARNING: Action on "{target}" has FAILED {count} times recently!
You MUST try a DIFFERENT approach:
- Try a different element selector/description
- Scroll first to find the element
- Check if there's a blocker preventing interaction
- Consider if goal is achievable
"""
        
        return ""
    
    def _parse_decision(self, content: str) -> Dict:
        """Parse JSON decision from LLM response."""
        content = content.strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        if "```" in content:
            match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except:
                    pass
        
        match = re.search(r'\{[^{}]*\}', content)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
        
        print(f"[Supervisor] Could not parse: {content[:100]}")
        return {"action": "FAILED", "reason": "Could not parse supervisor response"}
    
    def _normalize_action(self, decision: Dict) -> Dict:
        """Normalize unknown actions to supported ones."""
        action = decision.get('action', '').lower()
        
        mappings = {
            'find_and_click': 'click',
            'find_click': 'click',
            'tap': 'click',
            'enter': 'press_key',
            'input': 'type',
            'write': 'type',
            'go': 'navigate',
            'goto': 'navigate',
            'open': 'navigate',
            'close': 'dismiss',
            'done': 'COMPLETE',
            'complete': 'COMPLETE',
            'success': 'COMPLETE',
            'fail': 'FAILED',
            'error': 'FAILED',
            'wait': 'scroll',
        }
        
        if action in mappings:
            decision['action'] = mappings[action]
            
            if decision['action'] == 'press_key' and not decision.get('target'):
                decision['target'] = 'Enter'
        
        return decision
    
    def _fallback_decision(self, goal: str, observation: str, blockers: List[str]) -> Dict:
        """Simple rule-based fallback when API fails."""
        print("[Supervisor] Using fallback decision logic")
        
        if blockers:
            return {
                "action": "dismiss",
                "target": "popup close button"
            }
        
        goal_lower = goal.lower()
        
        sites = {
            'youtube': 'https://www.youtube.com',
            'amazon': 'https://www.amazon.in',
            'flipkart': 'https://www.flipkart.com',
            'google': 'https://www.google.com'
        }
        
        for site, url in sites.items():
            if site in goal_lower and url not in observation.lower():
                return {
                    "action": "navigate",
                    "target": url
                }
        
        if 'search' in goal_lower:
            match = re.search(r'search\s+(?:for\s+)?(.+?)(?:\s+and|\s+on|\s*$)', goal_lower)
            if match:
                query = match.group(1).strip()
                return {
                    "action": "search",
                    "target": "search bar",
                    "query": query
                }
        
        if 'first' in goal_lower or 'click' in goal_lower:
            return {
                "action": "click",
                "target": "first result or product"
            }
        
        return {
            "action": "FAILED",
            "reason": "Fallback logic could not determine next action"
        }
