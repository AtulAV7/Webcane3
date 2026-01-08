"""
Planning agent for WebCane3.
Task decomposition with verification hints per step.
Supports Groq (gpt-oss-120b) or Gemini Flash as backend.
"""

import json
import re
from typing import List, Dict, Optional

from .config import Config

# Groq imports
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Gemini imports
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class Planner:
    """
    Planning agent that decomposes goals into action steps.
    Each step includes a verification method hint.
    
    Supports two backends:
    - Groq: openai/gpt-oss-120b (uses GROQ_API_KEY3)
    - Gemini: gemini-2.5-flash
    
    Verification types:
    - URL_CHANGE: Check URL changed as expected
    - NONE: Trust the action (no verification)
    - DOM_VALUE: Check DOM element value/state
    - VISION_OUTCOME: Visual screenshot comparison
    """
    
    VALID_ACTIONS = ['navigate', 'find_and_click', 'type', 'wait', 'scroll', 'press_key']
    VALID_VERIFY = ['URL_CHANGE', 'NONE', 'DOM_VALUE', 'VISION_OUTCOME']
    
    # Groq model for planning
    GROQ_PLANNING_MODEL = "openai/gpt-oss-120b"
    
    def __init__(self, use_groq: bool = False):
        """
        Initialize the planner.
        
        Args:
            use_groq: If True, use Groq gpt-oss-120b. If False, use Gemini Flash.
        """
        self.use_groq = use_groq
        self.gemini_client = None
        self.groq_client = None
        self.gemini_available = False
        self.groq_available = False
        
        # Setup Groq if selected
        if use_groq and GROQ_AVAILABLE:
            try:
                api_key = Config.GROQ_API_KEY3
                if api_key:
                    self.groq_client = Groq(api_key=api_key)
                    self.groq_available = True
                    print(f"[Planner] Using Groq ({self.GROQ_PLANNING_MODEL})")
            except Exception as e:
                print(f"[Planner] Groq setup failed: {e}")
        
        # Setup Gemini if selected or as fallback
        if GENAI_AVAILABLE:
            try:
                api_key = Config.GEMINI_API_KEY
                if api_key:
                    self.gemini_client = genai.Client(api_key=api_key)
                    self.gemini_available = True
                    if not use_groq:
                        print(f"[Planner] Using Gemini ({Config.GEMINI_PLANNING_MODEL})")
                    else:
                        print("[Planner] Gemini fallback ready")
            except Exception as e:
                print(f"[Planner] Gemini setup failed: {e}")
    
    def decompose_task(
        self, 
        goal: str, 
        current_url: str = "about:blank",
        page_description: str = None,
        failure_context: Dict = None
    ) -> List[Dict]:
        """
        Decompose a goal into atomic action steps with verification hints.
        
        Args:
            goal: High-level task description
            current_url: Current page URL
            page_description: Visual description of current page
            failure_context: Context from previous failure (for replanning)
            
        Returns:
            List of action dictionaries with verify field
        """
        if not goal or not goal.strip():
            print("[Planner] Empty goal")
            return []
        
        goal = self._simplify_goal(goal)
        
        print(f"[Planner] Planning: {goal}")
        if failure_context:
            print(f"[Planner] Replanning after failure: {failure_context.get('reason', 'Unknown')}")
        
        prompt = self._create_prompt(goal, current_url, page_description, failure_context)
        
        # Try primary model based on selection
        if self.use_groq and self.groq_available:
            plan = self._try_groq(prompt)
            if plan:
                return self._finalize_plan(plan)
            print("[Planner] Groq failed, trying Gemini fallback...")
        
        # Try Gemini (primary if not using Groq, or fallback)
        if self.gemini_available:
            plan = self._try_gemini(prompt)
            if plan:
                return self._finalize_plan(plan)
            print("[Planner] Gemini failed")
        
        print("[Planner] All planners failed")
        return []
    
    def _try_groq(self, prompt: str) -> Optional[List[Dict]]:
        """Try Groq API for planning with reasoning."""
        try:
            response = self.groq_client.chat.completions.create(
                model=self.GROQ_PLANNING_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                reasoning_effort="high",
                temperature=1
            )
            
            content = response.choices[0].message.content
            
            # Print reasoning output for debugging
            print("\n[Planner] Groq Reasoning Output:")
            print("-" * 40)
            print(content[:1500] if len(content) > 1500 else content)
            print("-" * 40)
            
            return self._parse_plan(content)
        except Exception as e:
            print(f"[Planner] Groq error: {e}")
            return None
    
    def _simplify_goal(self, goal: str) -> str:
        """Clean and simplify goal text."""
        goal = ' '.join(goal.split())
        goal = goal.lower()
        for phrase in [' please ', ' could you ', ' can you ', ' i want to ', ' i need to ']:
            goal = goal.replace(phrase, ' ')
        return goal.strip()
    
    def _create_prompt(
        self, 
        goal: str, 
        current_url: str,
        page_description: str = None,
        failure_context: Dict = None
    ) -> str:
        """Create planning prompt with verification hints."""
        
        # Detect if this is a continuation task
        is_continuation = Config.is_context_continuation(goal)
        
        context = f"CURRENT URL: {current_url}"
        if page_description:
            context += f"\nPAGE STATE: {page_description}"
        
        failure_section = ""
        if failure_context:
            failure_section = f"""
REPLANNING AFTER FAILURE:
- Failed action: {failure_context.get('failed_action', 'Unknown')}
- Failure reason: {failure_context.get('reason', 'Unknown')}
- Current page state: {failure_context.get('current_state', 'Unknown')}

RECOVERY INSTRUCTIONS:
1. Analyze WHY the action failed based on the failure reason
2. Look at the current page state to understand what's visible
3. Generate an ALTERNATIVE approach that avoids the failed action
4. Do NOT repeat the exact same action that failed
5. Consider: Is the element visible? Is scrolling needed? Is a different selector needed?
"""
        
        continuation_notice = ""
        if is_continuation:
            continuation_notice = """
CONTINUATION TASK: Act on the CURRENT PAGE.
- Do NOT add navigate steps
- Work with visible elements
"""
        
        return f"""You are a web automation planner. Break down goals into atomic actions.

{context}
{continuation_notice}
{failure_section}
GOAL: {goal}

ACTIONS:
- navigate: Go to URL (target = URL) - SKIP if already on target site
- find_and_click: Click element (target = element description)
- type: Type text (target = text) - REQUIRES find_and_click BEFORE to focus input
- press_key: Press key (target = Enter, Tab, Escape)
- scroll: Scroll (target = down, up)
- wait: Wait (target = seconds)

VERIFICATION:
- URL_CHANGE: After navigation or page-changing clicks
- NONE: For clicking input fields
- DOM_VALUE: After typing
- VISION_OUTCOME: For visual element clicks (thumbnails, images)

STRICT SEARCH PROTOCOL - ALWAYS follow these 3 steps for ANY search:
1. [find_and_click] the search input field FIRST
2. [type] the search query
3. [press_key] Enter to submit

EXAMPLES:

Goal: "Search for 8k videos"
[
  {{"step": 1, "action": "find_and_click", "target": "search bar", "description": "Focus search box", "verify": "NONE"}},
  {{"step": 2, "action": "type", "target": "8k videos", "description": "Type query", "verify": "DOM_VALUE"}},
  {{"step": 3, "action": "press_key", "target": "Enter", "description": "Submit search", "verify": "URL_CHANGE"}}
]

Goal: "Go to flipkart and search watches"
[
  {{"step": 1, "action": "navigate", "target": "https://www.flipkart.com", "description": "Open Flipkart", "verify": "URL_CHANGE"}},
  {{"step": 2, "action": "find_and_click", "target": "search box", "description": "Focus search", "verify": "NONE"}},
  {{"step": 3, "action": "type", "target": "watches", "description": "Type query", "verify": "DOM_VALUE"}},
  {{"step": 4, "action": "press_key", "target": "Enter", "description": "Submit", "verify": "URL_CHANGE"}}
]

Goal: "Click the first video thumbnail"
[
  {{"step": 1, "action": "find_and_click", "target": "first video thumbnail", "description": "Click video", "verify": "VISION_OUTCOME"}}
]

RULES:
1. NEVER use [type] without a preceding [find_and_click] to focus the input
2. For visual tasks (thumbnails, images): use descriptive target
3. Check CURRENT URL before adding navigate

Plan for: {goal}
JSON:"""
    
    def _try_gemini(self, prompt: str) -> Optional[List[Dict]]:
        """Try Gemini API for planning with thinking enabled."""
        try:
            response = self.gemini_client.models.generate_content(
                model=Config.GEMINI_PLANNING_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=1,
                    max_output_tokens=2000,
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=1024
                    )
                )
            )
            
            # Print thinking if available
            if hasattr(response, 'candidates') and response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'thought') and part.thought:
                        print("\n[Planner] Gemini Thinking:")
                        print("-" * 40)
                        print(part.text[:1000] if len(part.text) > 1000 else part.text)
                        print("-" * 40)
            
            return self._parse_plan(response.text)
        except Exception as e:
            print(f"[Planner] Gemini error: {e}")
            return None
    
    def _try_local(self, prompt: str, goal: str) -> Optional[List[Dict]]:
        """Try local Ollama model with simplified prompt."""
        try:
            simple_prompt = f"""Convert this task to JSON steps. Output ONLY valid JSON array.

TASK: {goal}

Valid actions: navigate, find_and_click, type, scroll, press_key, wait
Valid verify: URL_CHANGE, NONE, DOM_VALUE, VISION_OUTCOME

Example for "search for cats":
[{{"step":1,"action":"find_and_click","target":"search box","description":"Click search","verify":"NONE"}},{{"step":2,"action":"type","target":"cats","description":"Type query","verify":"DOM_VALUE"}},{{"step":3,"action":"press_key","target":"Enter","description":"Submit","verify":"URL_CHANGE"}}]

JSON for "{goal}":"""
            
            response = ollama.generate(
                model=self.local_model,
                prompt=simple_prompt,
                stream=False,
                options={'temperature': 0.1, 'num_predict': 400}
            )
            return self._parse_plan(response['response'])
        except Exception as e:
            print(f"[Planner] Local error: {e}")
            return None
    
    def _parse_plan(self, response: str) -> Optional[List[Dict]]:
        """Parse LLM response to extract action plan."""
        try:
            # Try direct JSON parse
            try:
                plan = json.loads(response.strip())
                if isinstance(plan, list):
                    return plan
            except:
                pass
            
            # Try extracting from markdown code blocks
            match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                try:
                    plan = json.loads(match.group(1))
                    if isinstance(plan, list):
                        return plan
                except:
                    pass
            
            # Try extracting [...] pattern
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                try:
                    plan = json.loads(match.group(0))
                    if isinstance(plan, list):
                        return plan
                except:
                    pass
            
            print(f"[Planner] Could not parse: {response[:200]}...")
            return None
            
        except Exception as e:
            print(f"[Planner] Parse error: {e}")
            return None
    
    def _finalize_plan(self, plan: Optional[List[Dict]]) -> List[Dict]:
        """Validate and clean the plan."""
        if not plan:
            return []
        
        cleaned = []
        for idx, step in enumerate(plan, 1):
            if not isinstance(step, dict):
                continue
            
            action = step.get('action', '').lower()
            if action not in self.VALID_ACTIONS:
                print(f"[Planner] Invalid action '{action}' at step {idx}, skipping")
                continue
            
            verify = step.get('verify', 'NONE').upper()
            if verify not in self.VALID_VERIFY:
                verify = 'NONE'
            
            cleaned.append({
                'step': idx,
                'action': action,
                'target': str(step.get('target', '')),
                'description': step.get('description', f"{action} action"),
                'verify': verify
            })
        
        if cleaned:
            print(f"[Planner] Generated {len(cleaned)} steps")
            for s in cleaned:
                print(f"  {s['step']}. [{s['action']}] {s['target'][:40]} (verify: {s['verify']})")
        
        return cleaned
    
    def replan_after_failure(
        self,
        original_goal: str,
        failed_step: Dict,
        failure_reason: str,
        current_url: str,
        page_description: str = None
    ) -> List[Dict]:
        """
        Generate a recovery plan after step failure.
        Plans from current state, not from beginning.
        """
        print("[Planner] Generating recovery plan...")
        
        failure_context = {
            'failed_action': f"{failed_step.get('action', 'unknown')}: {failed_step.get('target', 'unknown')}",
            'reason': failure_reason,
            'current_state': page_description or 'Unknown'
        }
        
        return self.decompose_task(
            goal=original_goal,
            current_url=current_url,
            page_description=page_description,
            failure_context=failure_context
        )
