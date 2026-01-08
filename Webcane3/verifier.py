"""
Verifier agent for WebCane3.
Cascading verification: URL_CHANGE -> DOM_VALUE -> VISION_OUTCOME
"""

import time
from typing import Dict, Optional
import ollama
import base64

from .config import Config

# Groq imports
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


class Verifier:
    """
    Verification agent with cascading strategy.
    
    Priority based on planner hint:
    1. URL_CHANGE: Check URL changed
    2. DOM_VALUE: Check DOM state via LLM
    3. VISION_OUTCOME: Compare screenshots visually
    
    On failure, cascades to next method.
    """
    
    def __init__(self, groq_api_key: str = None):
        """
        Initialize the verifier.
        
        Args:
            groq_api_key: Groq API key (uses GROQ_API_KEY2 from config)
        """
        self.groq_client = None
        self.local_model = Config.OLLAMA_MODEL
        
        if GROQ_AVAILABLE:
            try:
                # Use GROQ_API_KEY2 for verification
                api_key = groq_api_key or Config.GROQ_API_KEY2
                if api_key:
                    self.groq_client = Groq(api_key=api_key)
                    print("[Verifier] Groq ready (KEY2)")
            except Exception as e:
                print(f"[Verifier] Groq setup failed: {e}")
    
    def verify_action(
        self,
        action: Dict,
        verify_method: str,
        before_state: Dict,
        after_state: Dict,
        before_screenshot: bytes = None,
        after_screenshot: bytes = None,
        goal: str = None,
        current_step: int = 0,
        total_steps: int = 0
    ) -> Dict:
        """
        Verify if an action succeeded using cascading verification.
        
        Args:
            action: The action that was executed
            verify_method: Hint from planner (URL_CHANGE, NONE, DOM_VALUE, VISION_OUTCOME)
            before_state: Page state before action
            after_state: Page state after action
            before_screenshot: Screenshot before action
            after_screenshot: Screenshot after action
            goal: Original goal for goal satisfaction check
            current_step: Current step index (0-based)
            total_steps: Total number of steps in plan
            
        Returns:
            VerificationResult dictionary
        """
        action_type = action.get('action', '').lower()
        target = action.get('target', '')
        
        print(f"[Verifier] Checking {action_type}: {target} (method: {verify_method})")
        
        # NONE: Trust the action
        if verify_method == 'NONE':
            return {
                'success': True,
                'method_used': 'NONE',
                'confidence': 0.9,
                'reason': 'Action trusted (no verification)',
                'goal_satisfied': False,
                'suggested_action': None
            }
        
        # Start with the specified method, cascade on failure
        methods = self._get_cascade_order(verify_method)
        
        for method in methods:
            result = self._verify_with_method(
                method, action, before_state, after_state,
                before_screenshot, after_screenshot
            )
            
            if result['success']:
                # Check goal satisfaction with step info
                goal_satisfied = self._check_goal_satisfaction(
                    goal, action, after_state, after_screenshot,
                    current_step, total_steps
                )
                result['goal_satisfied'] = goal_satisfied
                return result
            
            print(f"[Verifier] {method} failed, trying next...")
        
        # All methods failed
        return {
            'success': False,
            'method_used': 'ALL_FAILED',
            'confidence': 0.0,
            'reason': 'All verification methods failed',
            'goal_satisfied': False,
            'suggested_action': 'Retry or replan'
        }
    
    def _get_cascade_order(self, primary: str) -> list:
        """Get verification cascade order based on primary method."""
        all_methods = ['URL_CHANGE', 'DOM_VALUE', 'VISION_OUTCOME']
        
        if primary in all_methods:
            # Start with primary, then others
            order = [primary] + [m for m in all_methods if m != primary]
        else:
            order = all_methods
        
        return order
    
    def _verify_with_method(
        self,
        method: str,
        action: Dict,
        before_state: Dict,
        after_state: Dict,
        before_screenshot: bytes,
        after_screenshot: bytes
    ) -> Dict:
        """Run verification with specific method."""
        
        if method == 'URL_CHANGE':
            return self._verify_url_change(action, before_state, after_state)
        elif method == 'DOM_VALUE':
            return self._verify_dom_value(action, before_state, after_state)
        elif method == 'VISION_OUTCOME':
            return self._verify_vision(action, before_screenshot, after_screenshot)
        else:
            return {'success': False, 'reason': f'Unknown method: {method}'}
    
    def _verify_url_change(
        self,
        action: Dict,
        before_state: Dict,
        after_state: Dict
    ) -> Dict:
        """Verify URL changed as expected."""
        before_url = before_state.get('url', '')
        after_url = after_state.get('url', '')
        
        action_type = action.get('action', '').lower()
        target = action.get('target', '').lower()
        
        # Check for URL change
        url_changed = before_url != after_url
        
        if action_type == 'navigate':
            # Check if target is in new URL
            target_clean = target.replace('https://', '').replace('http://', '').strip('/')
            success = target_clean in after_url.lower()
        else:
            # For clicks, any URL change is considered success
            success = url_changed
        
        return {
            'success': success,
            'method_used': 'URL_CHANGE',
            'confidence': 0.95 if success else 0.1,
            'reason': f"URL {'changed to ' + after_url if url_changed else 'unchanged'}",
            'goal_satisfied': False,
            'suggested_action': None if success else 'Retry action'
        }
    
    def _verify_dom_value(
        self,
        action: Dict,
        before_state: Dict,
        after_state: Dict
    ) -> Dict:
        """Verify DOM state changed appropriately using Groq LLM."""
        action_type = action.get('action', '').lower()
        target = action.get('target', '')
        
        # Quick heuristics first
        if action_type == 'type':
            # For typing, we assume success if no error
            return {
                'success': True,
                'method_used': 'DOM_VALUE',
                'confidence': 0.85,
                'reason': f'Text typed: {target}',
                'goal_satisfied': False,
                'suggested_action': None
            }
        
        # Check element count change
        before_count = before_state.get('element_count', 0)
        after_count = after_state.get('element_count', 0)
        change = after_count - before_count
        
        if abs(change) > 5:
            # Significant DOM change
            return {
                'success': True,
                'method_used': 'DOM_VALUE',
                'confidence': 0.7,
                'reason': f'DOM changed (elements: {before_count} -> {after_count})',
                'goal_satisfied': False,
                'suggested_action': None
            }
        
        # Use LLM for ambiguous cases
        if self.groq_client:
            return self._verify_dom_with_llm(action, before_state, after_state)
        
        # Fallback: assume success for non-critical actions
        return {
            'success': True,
            'method_used': 'DOM_VALUE',
            'confidence': 0.5,
            'reason': 'Unable to verify DOM change',
            'goal_satisfied': False,
            'suggested_action': None
        }
    
    def _verify_dom_with_llm(
        self,
        action: Dict,
        before_state: Dict,
        after_state: Dict
    ) -> Dict:
        """Use Groq LLM for DOM verification."""
        try:
            prompt = f"""Verify if this web action succeeded based on state changes.

ACTION: {action.get('action')}: {action.get('target')}
DESCRIPTION: {action.get('description')}

BEFORE:
- URL: {before_state.get('url')}
- Title: {before_state.get('title')}
- Elements: {before_state.get('element_count')}

AFTER:
- URL: {after_state.get('url')}
- Title: {after_state.get('title')}
- Elements: {after_state.get('element_count')}

Did the action succeed? Answer with JSON:
{{"success": true/false, "confidence": 0.0-1.0, "reason": "explanation"}}
JSON:"""
            
            response = self.groq_client.chat.completions.create(
                model=Config.GROQ_DOM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1
            )
            
            import json
            import re
            
            result_text = response.choices[0].message.content.strip()
            # Extract JSON
            match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if match:
                result = json.loads(match.group(0))
                return {
                    'success': result.get('success', False),
                    'method_used': 'DOM_VALUE',
                    'confidence': result.get('confidence', 0.5),
                    'reason': result.get('reason', 'LLM verification'),
                    'goal_satisfied': False,
                    'suggested_action': None if result.get('success') else 'Retry'
                }
            
        except Exception as e:
            print(f"[Verifier] DOM LLM error: {e}")
        
        return {
            'success': False,
            'method_used': 'DOM_VALUE',
            'confidence': 0.0,
            'reason': 'DOM verification failed',
            'goal_satisfied': False,
            'suggested_action': 'Try visual verification'
        }
    
    def _verify_vision(
        self,
        action: Dict,
        before_screenshot: bytes,
        after_screenshot: bytes
    ) -> Dict:
        """Verify action visually by comparing screenshots using Groq vision model."""
        if not before_screenshot or not after_screenshot:
            return {
                'success': False,
                'method_used': 'VISION_OUTCOME',
                'confidence': 0.0,
                'reason': 'Screenshots not available',
                'goal_satisfied': False,
                'suggested_action': 'Retry with screenshots'
            }
        
        if not self.groq_client:
            return self._verify_vision_local(action, before_screenshot, after_screenshot)
        
        try:
            # Encode screenshots as base64
            before_b64 = base64.b64encode(before_screenshot).decode('utf-8')
            after_b64 = base64.b64encode(after_screenshot).decode('utf-8')
            
            prompt = f"""Compare these two webpage screenshots (before and after an action).

ACTION PERFORMED: {action.get('action')}: {action.get('target')}
EXPECTED OUTCOME: {action.get('description')}

Did the action succeed? Look for:
1. Did the expected change occur?
2. Is the page different in a meaningful way?
3. Did we navigate to a new page or open new content?

Answer with JSON:
{{"success": true/false, "confidence": 0.0-1.0, "reason": "what changed or why it failed"}}
JSON:"""
            
            response = self.groq_client.chat.completions.create(
                model=Config.GROQ_VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "BEFORE screenshot:"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{before_b64}"}
                            },
                            {"type": "text", "text": "AFTER screenshot:"},
                            {
                                "type": "image_url", 
                                "image_url": {"url": f"data:image/png;base64,{after_b64}"}
                            },
                            {"type": "text", "text": prompt}
                        ]
                    }
                ],
                max_tokens=150,
                temperature=0.1
            )
            
            import json
            import re
            
            result_text = response.choices[0].message.content.strip()
            match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if match:
                result = json.loads(match.group(0))
                return {
                    'success': result.get('success', False),
                    'method_used': 'VISION_OUTCOME',
                    'confidence': result.get('confidence', 0.5),
                    'reason': result.get('reason', 'Visual verification'),
                    'goal_satisfied': False,
                    'suggested_action': None if result.get('success') else 'Replan'
                }
            
        except Exception as e:
            print(f"[Verifier] Vision error: {e}")
        
        return self._verify_vision_local(action, before_screenshot, after_screenshot)
    
    def _verify_vision_local(
        self,
        action: Dict,
        before_screenshot: bytes,
        after_screenshot: bytes
    ) -> Dict:
        """Local fallback for vision verification - compare image sizes."""
        try:
            # Simple heuristic: different sizes = something changed
            if len(before_screenshot) != len(after_screenshot):
                return {
                    'success': True,
                    'method_used': 'VISION_OUTCOME',
                    'confidence': 0.6,
                    'reason': 'Screenshots differ in size',
                    'goal_satisfied': False,
                    'suggested_action': None
                }
            
            return {
                'success': False,
                'method_used': 'VISION_OUTCOME',
                'confidence': 0.3,
                'reason': 'No visual change detected',
                'goal_satisfied': False,
                'suggested_action': 'Retry action'
            }
            
        except:
            return {
                'success': False,
                'method_used': 'VISION_OUTCOME',
                'confidence': 0.0,
                'reason': 'Visual verification failed',
                'goal_satisfied': False,
                'suggested_action': 'Retry'
            }
    
    def _check_goal_satisfaction(
        self,
        goal: str,
        action: Dict,
        after_state: Dict,
        after_screenshot: bytes,
        current_step: int = 0,
        total_steps: int = 0
    ) -> bool:
        """
        Check if the overall goal is satisfied after this action.
        
        IMPORTANT: Only return True if this is genuinely the final action
        that completes the goal. Check step progress.
        """
        if not goal:
            return False
        
        goal_lower = goal.lower()
        action_type = action.get('action', '').lower()
        after_url = after_state.get('url', '').lower()
        
        # If we're not on the last step, don't mark as complete
        # This prevents early termination
        if total_steps > 0 and current_step < total_steps - 1:
            return False
        
        # Video/play goals: check if on watch page AND this was a click action
        if any(w in goal_lower for w in ['play', 'watch']):
            if ('watch' in after_url or '/shorts/' in after_url):
                if action_type == 'find_and_click':
                    print("[Verifier] Goal satisfied: Video page reached via click")
                    return True
        
        # Search goals: ONLY satisfied after pressing Enter on search results
        if 'search' in goal_lower:
            if 'results' in after_url or 'q=' in after_url:
                if action_type == 'press_key' and action.get('target', '').lower() == 'enter':
                    print("[Verifier] Goal satisfied: Search results page reached")
                    return True
        
        # Navigate-ONLY goals: goal is JUST navigation (no "and" with other actions)
        # e.g., "go to youtube" but NOT "go to youtube and search"
        if ('navigate' in goal_lower or 'go to' in goal_lower):
            # Check if goal has additional actions (contains "and" + action words)
            has_additional_actions = (
                ' and ' in goal_lower and 
                any(w in goal_lower for w in ['search', 'find', 'click', 'type', 'play', 'watch', 'select'])
            )
            if not has_additional_actions and action_type == 'navigate':
                print("[Verifier] Goal satisfied: Simple navigation complete")
                return True
        
        # Click goals on specific items
        if 'click' in goal_lower and action_type == 'find_and_click':
            # Check if we clicked the intended target
            target = action.get('target', '').lower()
            if any(word in target for word in goal_lower.split() if len(word) > 3):
                print("[Verifier] Goal satisfied: Click action completed")
                return True
        
        return False
    
    def verify_goal_completion(
        self,
        goal: str,
        screenshot: bytes,
        current_url: str
    ) -> Dict:
        """
        Perform final visual verification that the goal was achieved.
        Called after all steps complete successfully.
        
        Args:
            goal: The original goal
            screenshot: Current page screenshot
            current_url: Current URL
            
        Returns:
            Dict with success, reason, needs_replan
        """
        print("\n" + "=" * 50)
        print("FINAL GOAL VERIFICATION")
        print("=" * 50)
        print(f"Goal: {goal}")
        print(f"Current URL: {current_url}")
        
        if not screenshot or not self.groq_client:
            # Can't do visual verification, assume success
            print("[Verifier] No screenshot or API available, assuming goal complete")
            return {
                'success': True,
                'reason': 'Unable to perform visual verification',
                'needs_replan': False
            }
        
        try:
            # Generate expected outcome from goal
            expected_outcome = self._generate_expected_outcome(goal)
            print(f"[Verifier] Expected outcome: {expected_outcome}")
            
            # Compare screenshot with expected outcome
            b64_screenshot = base64.b64encode(screenshot).decode('utf-8')
            
            prompt = f"""Analyze this screenshot and determine if the goal has been achieved.

GOAL: "{goal}"
EXPECTED OUTCOME: "{expected_outcome}"
CURRENT URL: {current_url}

Look at the screenshot and answer:
1. Does the page show the expected result for this goal?
2. Is there evidence the goal was completed successfully?

Respond with ONLY one word: SUCCESS or FAILURE

If the page clearly shows the goal was achieved (e.g., search results visible, video playing, product page loaded), respond SUCCESS.
If the page does NOT show the expected outcome, respond FAILURE.

Answer:"""
            
            response = self.groq_client.chat.completions.create(
                model=Config.GROQ_VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_screenshot}"}}
                        ]
                    }
                ],
                max_tokens=50,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip().upper()
            print(f"[Verifier] Final verification result: {result}")
            
            if "SUCCESS" in result:
                print("[Verifier] GOAL ACHIEVED!")
                return {
                    'success': True,
                    'reason': 'Visual verification confirms goal completion',
                    'needs_replan': False
                }
            else:
                print("[Verifier] GOAL NOT ACHIEVED - Needs replanning")
                return {
                    'success': False,
                    'reason': 'Visual verification shows goal not achieved',
                    'needs_replan': True
                }
                
        except Exception as e:
            print(f"[Verifier] Final verification error: {e}")
            # On error, don't replan - assume steps completed
            return {
                'success': True,
                'reason': f'Verification error: {e}',
                'needs_replan': False
            }
    
    def _generate_expected_outcome(self, goal: str) -> str:
        """Generate expected outcome description from goal using simple heuristics."""
        goal_lower = goal.lower()
        
        # Search goals
        if 'search' in goal_lower:
            query = goal_lower.split('search')[-1].strip()
            return f"Search results page showing results for '{query}'"
        
        # Navigate/go to goals
        if 'go to' in goal_lower or 'navigate' in goal_lower:
            site = goal_lower.replace('go to', '').replace('navigate to', '').strip()
            return f"The {site} website homepage or main page"
        
        # Click goals
        if 'click' in goal_lower:
            return f"The page that appears after clicking the specified element"
        
        # Play/watch goals
        if 'play' in goal_lower or 'watch' in goal_lower:
            return f"A video player showing the video or video playing on the page"
        
        # Buy/cart goals
        if 'buy' in goal_lower or 'cart' in goal_lower or 'add' in goal_lower:
            return f"Item added to cart or purchase confirmation"
        
        # Default
        return f"Visual evidence that '{goal}' was completed successfully"
