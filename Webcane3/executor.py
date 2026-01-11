"""
Executor agent for WebCane3 ReAct workflow.
Hybrid DOM/Vision action execution with Macro-Actions for atomic task bundles.
"""

import time
import os
import base64
import re
import requests
from typing import Dict, List, Optional

from .config import Config
from .browser_controller import BrowserController
from .som_annotator import SoMAnnotator

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


class Executor:
    """
    Hybrid action executor with Macro-Actions.
    
    Uses DOM text matching (System 1) with Vision fallback (System 2).
    Includes Macro-Actions that bundle multiple steps for speed.
    """
    
    def __init__(
        self, 
        browser: BrowserController,
        groq_api_key: str = None,
        gemini_api_key: str = None
    ):
        """
        Initialize the executor.
        
        Args:
            browser: BrowserController instance
            groq_api_key: Groq API key for DOM text matching
            gemini_api_key: Gemini API key for Vision fallback
        """
        self.browser = browser
        self.annotator = SoMAnnotator()
        
        # Groq client for DOM text matching
        self.groq_client = None
        if GROQ_AVAILABLE:
            try:
                api_key = groq_api_key or Config.GROQ_API_KEY
                if api_key:
                    self.groq_client = Groq(api_key=api_key)
                    print("[Executor] Groq ready for DOM matching")
            except Exception as e:
                print(f"[Executor] Groq setup failed: {e}")
        
        # Gemini client for Vision (fallback)
        self.gemini_client = None
        if GENAI_AVAILABLE:
            try:
                api_key = gemini_api_key or Config.GEMINI_API_KEY
                if api_key:
                    self.gemini_client = genai.Client(api_key=api_key)
                    print("[Executor] Gemini ready for Vision")
            except Exception as e:
                print(f"[Executor] Gemini setup failed: {e}")
        
        # Stats
        self.stats = {
            'dom_success': 0,
            'vision_success': 0,
            'macro_success': 0,
            'failures': 0
        }
        
        # Track last action for typing safety
        self.last_action_was_click = False
        
        # Store Vision agent's reasoning for verification (passed to Supervisor)
        self.last_vision_reasoning = None
    
    def execute_action(self, action: Dict) -> Dict:
        """
        Execute a single action step.
        
        Args:
            action: Action dictionary with action, target, query fields
            
        Returns:
            Result dictionary with success, method, error fields
        """
        action_type = action.get('action', '').lower()
        target = action.get('target', '')
        query = action.get('query', '')
        
        print(f"[Executor] {action_type}: {target}" + (f" (query: {query})" if query else ""))
        
        # Handle macro actions
        if action_type == 'search':
            return self.execute_search(target, query)
        elif action_type == 'scroll_find':
            return self.execute_scroll_find(target)
        elif action_type == 'dismiss':
            return self._execute_dismiss(target)
        
        # Handle standard actions
        if action_type == 'navigate':
            self.last_action_was_click = False
            return self._execute_navigate(target)
        elif action_type == 'click' or action_type == 'find_and_click':
            result = self._execute_click(target)
            self.last_action_was_click = result.get('success', False)
            return result
        elif action_type == 'type':
            if not self.last_action_was_click:
                print("[Executor] WARNING: Typing without prior click - attempting to focus first...")
                self._try_focus_input()
            return self._execute_type(target)
        elif action_type == 'press_key':
            self.last_action_was_click = False
            return self._execute_press_key(target)
        elif action_type == 'scroll':
            return self._execute_scroll(target)
        elif action_type == 'strong_scroll':
            return self._execute_strong_scroll(target)
        elif action_type == 'wait':
            return self._execute_wait(target)
        elif action_type == 'go_back':
            return self._execute_go_back()
        else:
            return {'success': False, 'error': f'Unknown action: {action_type}'}
    
    # ==================== MACRO ACTIONS ====================
    
    def execute_search(self, target: str, query: str) -> Dict:
        """
        Macro: Find search input -> Click -> Type query -> Press Enter.
        Bundles 3 steps into 1 to avoid slow loops.
        
        Args:
            target: Search input description (e.g., "search bar", "search box")
            query: The text to search for
            
        Returns:
            Result dictionary
        """
        print(f"[Executor] MACRO: Search for '{query}'")
        
        # Get current elements
        elements = self.browser.extract_elements()
        
        # Step 1: Try to find search input directly by looking for input/textarea elements
        search_input_id = self._find_search_input(elements)
        
        click_success = False
        
        if search_input_id >= 0:
            print(f"[Executor] Found search input at element {search_input_id}")
            click_success = self.browser.click_element(search_input_id, elements)
            if click_success:
                print(f"[Executor] Clicked search input successfully")
        
        if not click_success:
            # Fallback: Try DOM text matching
            print(f"[Executor] Direct search failed, trying DOM matching for: {target}")
            click_result = self._execute_click(target)
            click_success = click_result.get('success', False)
            
            if not click_success:
                # Try alternative search targets
                alternatives = ["search", "Search", "search input", "search field", "search box"]
                for alt in alternatives:
                    if alt.lower() != target.lower():
                        print(f"[Executor] Trying alternative: {alt}")
                        click_result = self._execute_click(alt)
                        if click_result.get('success'):
                            click_success = True
                            break
        
        if not click_success:
            self.stats['failures'] += 1
            return {
                'success': False,
                'method': 'search_macro',
                'error': f'Could not find search input: {target}'
            }
        
        time.sleep(0.5)
        
        # Step 2: Clear any existing text and type the query
        try:
            self.browser.page.keyboard.press("Control+a")
            time.sleep(0.1)
        except:
            pass
        
        type_result = self._execute_type(query)
        if not type_result.get('success'):
            self.stats['failures'] += 1
            return {
                'success': False,
                'method': 'search_macro',
                'error': f'Could not type query: {query}'
            }
        
        time.sleep(0.3)
        
        # Step 3: Press Enter
        enter_result = self._execute_press_key("Enter")
        if not enter_result.get('success'):
            self.stats['failures'] += 1
            return {
                'success': False,
                'method': 'search_macro',
                'error': 'Could not press Enter'
            }
        
        self.stats['macro_success'] += 1
        time.sleep(Config.STEP_DELAY)
        
        return {
            'success': True,
            'method': 'search_macro',
            'query': query
        }
    
    def _find_search_input(self, elements: List[Dict]) -> int:
        """
        Find a search input element by analyzing element properties.
        
        Args:
            elements: List of extracted DOM elements
            
        Returns:
            Element ID or -1 if not found
        """
        search_keywords = ['search', 'query', 'find', 'q', 'keywords']
        
        for el in elements:
            tag = el.get('tag', '').lower()
            text = (el.get('text', '') or '').lower()
            el_type = (el.get('type', '') or '').lower()
            html_class = (el.get('html_classes', '') or '').lower()
            html_id = (el.get('html_id', '') or '').lower()
            
            # Check if it's an input or textarea
            if tag in ['input', 'textarea']:
                # Check if type is text/search
                if el_type in ['text', 'search', 'button']:
                    # Check for search-related keywords in text, class, or id
                    combined = f"{text} {html_class} {html_id}"
                    if any(kw in combined for kw in search_keywords):
                        print(f"[Executor] Found search input: id={el['id']}, text='{el.get('text', '')}'")
                        return el['id']
        
        # Second pass: look for any input with search-like text
        for el in elements:
            tag = el.get('tag', '').lower()
            text = (el.get('text', '') or '').lower()
            
            if tag in ['input', 'textarea'] and 'search' in text:
                print(f"[Executor] Found search input (text match): id={el['id']}")
                return el['id']
        
        print("[Executor] No search input found in elements")
        return -1
    
    def execute_scroll_find(self, target: str, max_scrolls: int = None) -> Dict:
        """
        Macro for visual elements: Take screenshot -> Vision check ->
        Not found? Scroll -> Repeat up to max_scrolls times.
        
        Args:
            target: Visual element description (e.g., "video with cat thumbnail")
            max_scrolls: Maximum scroll attempts (uses config default if None)
            
        Returns:
            Result dictionary
        """
        max_scrolls = max_scrolls or Config.MAX_SCROLL_ATTEMPTS
        print(f"[Executor] MACRO: Scroll-find '{target}' (max {max_scrolls} scrolls)")
        
        for attempt in range(max_scrolls):
            print(f"[Executor] Scroll-find attempt {attempt + 1}/{max_scrolls}")
            
            # Extract elements and take screenshot
            elements = self.browser.extract_elements()
            if not elements:
                self.browser.scroll('down', 600)
                time.sleep(1.5)
                continue
            
            # Try to find element by vision
            element_id = self._find_element_by_vision(elements, target)
            
            if element_id >= 0:
                # Found! Try to click it
                if self.browser.click_element(element_id, elements):
                    self.stats['macro_success'] += 1
                    time.sleep(Config.STEP_DELAY)
                    return {
                        'success': True,
                        'method': 'scroll_find_macro',
                        'attempts': attempt + 1,
                        'element_id': element_id
                    }
            
            # Not found, scroll and retry
            if attempt < max_scrolls - 1:
                self.browser.scroll('down', 600)
                time.sleep(1.5)
        
        self.stats['failures'] += 1
        return {
            'success': False,
            'method': 'scroll_find_macro',
            'error': f'Element not found after {max_scrolls} scroll attempts: {target}'
        }
    
    def _execute_dismiss(self, target: str) -> Dict:
        """
        Dismiss a popup, modal, or banner.
        
        Args:
            target: What to dismiss (e.g., "close button", "popup")
            
        Returns:
            Result dictionary
        """
        print(f"[Executor] Dismissing: {target}")
        
        # Common dismiss targets to try
        dismiss_targets = [
            target,
            "close button",
            "close",
            "dismiss",
            "x button",
            "cancel",
            "no thanks",
            "skip",
            "not now"
        ]
        
        for dismiss_target in dismiss_targets:
            result = self._execute_click(dismiss_target)
            if result.get('success'):
                return {
                    'success': True,
                    'method': 'dismiss',
                    'target_used': dismiss_target
                }
        
        # Try pressing Escape as fallback
        try:
            self.browser.press_key("Escape")
            time.sleep(0.5)
            return {
                'success': True,
                'method': 'dismiss_escape'
            }
        except:
            pass
        
        return {
            'success': False,
            'method': 'dismiss',
            'error': f'Could not dismiss: {target}'
        }
    
    # ==================== STANDARD ACTIONS ====================
    
    def _try_focus_input(self) -> bool:
        """Try to focus an input element before typing."""
        try:
            elements = self.browser.extract_elements()
            input_keywords = ['search', 'input', 'query', 'text']
            
            for el in elements:
                tag = el.get('tag', '').lower()
                text = (el.get('text', '') or '').lower()
                
                if tag in ['input', 'textarea']:
                    if any(kw in text for kw in input_keywords) or el.get('type') in ['text', 'search']:
                        if self.browser.click_element(el['id'], elements):
                            print(f"[Executor] Auto-focused input element {el['id']}")
                            time.sleep(0.5)
                            return True
            return False
        except:
            return False
    
    def _soft_validate_element(self, element: Dict, target: str) -> bool:
        """
        Soft validation: check if element roughly matches target description.
        Not strict - just catches obvious mismatches like clicking wrong element type.
        
        Returns True if element seems to match, False if definitely wrong.
        """
        if not element:
            return False
        
        target_lower = target.lower()
        el_text = (element.get('text', '') or '').lower()
        el_tag = element.get('tag', '').lower()
        el_type = element.get('type', '').lower()
        
        # Extract key words from target
        keywords = []
        for word in ['cart', 'add', 'buy', 'button', 'link', 'search', 'submit', 
                     'login', 'sign', 'play', 'video', 'image', 'thumbnail']:
            if word in target_lower:
                keywords.append(word)
        
        # If no specific keywords, accept anything (visual task like "lion thumbnail")
        if not keywords:
            return True
        
        # Check if element matches at least one keyword
        combined = f"{el_text} {el_tag} {el_type}"
        for kw in keywords:
            if kw in combined:
                return True
        
        # Special cases
        if 'cart' in target_lower and ('cart' in combined or 'add' in combined):
            return True
        if 'button' in target_lower and el_tag in ['button', 'a', 'input']:
            return True
        if 'link' in target_lower and el_tag == 'a':
            return True
        
        # If element text is empty, be lenient (might be icon button)
        if not el_text.strip():
            return True
        
        # Default: accept (soft validation, not strict)
        return True
    
    def _execute_navigate(self, url: str) -> Dict:
        """Navigate to URL."""
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        success = self.browser.navigate(url)
        time.sleep(Config.STEP_DELAY)
        
        return {
            'success': success,
            'method': 'direct',
            'error': None if success else 'Navigation failed'
        }
    
    def _execute_click(self, target: str) -> Dict:
        """
        Find and click an element using DOM text matching with Vision fallback.
        For visual tasks (thumbnails, images), prioritizes Vision over DOM.
        """
        visual_keywords = ['thumbnail', 'image', 'picture', 'photo', 'icon', 
                          'video with', 'look', 'appear', 'color', 'green', 
                          'blue', 'red', 'first', 'second', 'third']
        is_visual_task = any(kw in target.lower() for kw in visual_keywords)
        
        if is_visual_task:
            print(f"[Executor] Visual task detected - prioritizing Vision agent")
        
        for scroll_attempt in range(Config.MAX_SCROLL_ATTEMPTS + 1):
            if scroll_attempt > 0:
                print(f"[Executor] Scroll attempt {scroll_attempt}/{Config.MAX_SCROLL_ATTEMPTS}")
                self.browser.scroll('down', 600)
                time.sleep(1.5)
            
            elements = self.browser.extract_elements()
            if not elements:
                continue
            
            page_info = self.browser.get_page_info()
            
            # For visual tasks, try Vision FIRST
            if is_visual_task:
                print("[Executor] Trying Vision first for visual task...")
                element_id = self._find_element_by_vision(elements, target)
                
                if element_id >= 0 and element_id < len(elements):
                    # Get the element for validation
                    clicked_element = elements[element_id] if element_id < len(elements) else None
                    
                    # Soft validation: check if element roughly matches target
                    if clicked_element and self._soft_validate_element(clicked_element, target):
                        if self.browser.click_element(element_id, elements):
                            self.stats['vision_success'] += 1
                            time.sleep(Config.STEP_DELAY)
                            return {
                                'success': True,
                                'method': 'vision',
                                'element_id': element_id,
                                'scroll_attempts': scroll_attempt,
                                'vision_reasoning': self.last_vision_reasoning
                            }
                    else:
                        # Validation failed - try nearby elements (off-by-one fix)
                        print(f"[Executor] Vision ID {element_id} doesn't match '{target}', checking nearby...")
                        for offset in [-1, 1, -2, 2]:
                            nearby_id = element_id + offset
                            if 0 <= nearby_id < len(elements):
                                nearby_el = elements[nearby_id]
                                if self._soft_validate_element(nearby_el, target):
                                    print(f"[Executor] Found better match at ID {nearby_id}")
                                    if self.browser.click_element(nearby_id, elements):
                                        self.stats['vision_success'] += 1
                                        time.sleep(Config.STEP_DELAY)
                                        return {
                                            'success': True,
                                            'method': 'vision_corrected',
                                            'element_id': nearby_id,
                                            'scroll_attempts': scroll_attempt,
                                            'vision_reasoning': f"Corrected from {element_id} to {nearby_id}"
                                        }
                        # No valid nearby element found - proceed with original anyway (lenient)
                        print(f"[Executor] No better match found, proceeding with ID {element_id}")
                        if self.browser.click_element(element_id, elements):
                            self.stats['vision_success'] += 1
                            time.sleep(Config.STEP_DELAY)
                            return {
                                'success': True,
                                'method': 'vision',
                                'element_id': element_id,
                                'scroll_attempts': scroll_attempt,
                                'vision_reasoning': self.last_vision_reasoning
                            }
                print("[Executor] Vision failed, trying DOM fallback...")
            
            # Try DOM text matching (System 1)
            element_id = self._find_element_by_text(elements, target, page_info)
            
            if element_id >= 0:
                if self.browser.click_element(element_id, elements):
                    self.stats['dom_success'] += 1
                    time.sleep(Config.STEP_DELAY)
                    return {
                        'success': True,
                        'method': 'dom',
                        'element_id': element_id,
                        'scroll_attempts': scroll_attempt
                    }
            
            # For non-visual tasks, try Vision as fallback
            if not is_visual_task:
                print("[Executor] DOM failed, trying Vision fallback...")
                element_id = self._find_element_by_vision(elements, target)
                
                if element_id >= 0 and element_id < len(elements):
                    if self.browser.click_element(element_id, elements):
                        self.stats['vision_success'] += 1
                        time.sleep(Config.STEP_DELAY)
                        return {
                            'success': True,
                            'method': 'vision',
                            'element_id': element_id,
                            'scroll_attempts': scroll_attempt,
                            'vision_reasoning': self.last_vision_reasoning  # What Vision saw
                        }
        
        self.stats['failures'] += 1
        return {
            'success': False,
            'method': 'failed',
            'error': f'Element not found: {target}',
            'scroll_attempts': Config.MAX_SCROLL_ATTEMPTS
        }
    
    def _find_element_by_text(
        self, 
        elements: List[Dict], 
        target: str,
        page_info: Dict
    ) -> int:
        """
        Find element using Groq LLM text matching.
        
        Returns:
            Element ID or -1 if not found
        """
        if not self.groq_client:
            print("[DOM Agent] Groq client not available, using local fallback")
            return self._find_element_local(elements, target)
        
        try:
            time.sleep(Config.API_DELAY)
            
            # Format elements for prompt - include type for better matching
            elem_list = []
            valid_ids = set()  # Track valid element IDs
            for el in elements[:80]:
                text = el['text'][:50] if el['text'] else ""
                el_type = el.get('type', el['tag'])
                tag = el['tag']
                elem_list.append(f"[{el['id']}] <{tag}> type={el_type}: \"{text}\"")
                valid_ids.add(el['id'])
            
            prompt = f"""Find the element that best matches this task: "{target}"

IMPORTANT HINTS:
- For "search bar/input/field": look for <input> or <textarea> elements with "search" text
- For "button": look for <button> or <a> elements
- For "link": look for <a> elements
- Match by the text content, not just the tag

Elements on page:
{chr(10).join(elem_list)}

Return ONLY the element ID number that best matches. If none match, return -1.
ID:"""
            
            print("\n" + "=" * 50)
            print("[DOM Agent] Sending query to Groq...")
            print(f"[DOM Agent] Target: \"{target}\"")
            print(f"[DOM Agent] Elements count: {len(elements)}")
            
            # Note: gpt-oss-120b requires max_completion_tokens instead of max_tokens
            response = self.groq_client.chat.completions.create(
                model=Config.GROQ_DOM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=100,  # Use max_completion_tokens for reasoning models
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip()
            
            # Print DOM agent output
            print(f"[DOM Agent] Raw response: \"{result}\"")
            
            # Extract number from response
            match = next((int(n) for n in result.split() if n.lstrip('-').isdigit()), -1)
            
            print(f"[DOM Agent] Parsed ID: {match}")
            
            # Validate: check if this ID exists in our elements (not just < len)
            if match >= 0 and match in valid_ids:
                # Find the element to print its details
                for el in elements:
                    if el['id'] == match:
                        print(f"[DOM Agent] Matched element: [{match}] {el['tag']}: \"{el.get('text', '')[:30]}\"")
                        break
                print("=" * 50 + "\n")
                return match
            else:
                print(f"[DOM Agent] Invalid ID {match} (not in valid_ids: {sorted(list(valid_ids)[:10])}...)")
                print("=" * 50 + "\n")
                return -1
            
        except Exception as e:
            print(f"[DOM Agent] Groq error: {e}")
            import traceback
            traceback.print_exc()
            return self._find_element_local(elements, target)
    
    def _find_element_local(self, elements: List[Dict], target: str) -> int:
        """Local fallback for element finding."""
        try:
            target_lower = target.lower()
            keywords = target_lower.split()
            
            for el in elements:
                text = (el.get('text', '') or '').lower()
                if any(kw in text for kw in keywords):
                    return el['id']
            
            return -1
        except:
            return -1
    
    def _try_nvidia_vision(self, annotated_bytes: bytes, target: str) -> int:
        """
        Try NVIDIA API (Mistral Large) for vision analysis.
        
        Args:
            annotated_bytes: Annotated screenshot bytes
            target: The target element description to find
        
        Returns:
            Element ID/index or -1 if failed
        """
        if not Config.NVIDIA_API_KEY:
            print("[Executor] NVIDIA Vision: No API key configured")
            return -1
        
        try:
            b64_image = base64.b64encode(annotated_bytes).decode('utf-8')
            
            headers = {
                "Authorization": f"Bearer {Config.NVIDIA_API_KEY}",
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
            
            nvidia_prompt = f"""Look at this screenshot with numbered red boxes around interactive elements.

TASK: Find the element that best matches: "{target}"

INSTRUCTIONS:
1. Look at the VISUAL content inside each numbered red box
2. Find the box that visually matches what is described
3. For thumbnails/images - look at what the image shows
4. For buttons/links - read the text inside the box
5. Prefer elements in the main content area (center/below header)

Provide brief reasoning (1-2 sentences), then write: ANSWER: [number]
If no element matches, write: ANSWER: -1"""
            
            payload = {
                "model": Config.NVIDIA_VISION_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": nvidia_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
                        ]
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.15,
                "stream": False
            }
            
            print("[Executor] NVIDIA Vision: Calling API...")
            response = requests.post(Config.NVIDIA_VISION_URL, headers=headers, json=payload, timeout=30)
            
            if response.status_code != 200:
                print(f"[Executor] NVIDIA Vision error: {response.status_code}")
                return -1
            
            result_json = response.json()
            result = result_json['choices'][0]['message']['content'].strip()
            
            print("\n[Executor] NVIDIA Vision Output:")
            print("-" * 50)
            print(result[:300] if len(result) > 300 else result)
            print("-" * 50)
            
            # Store reasoning for verification (passed to Supervisor)
            self.last_vision_reasoning = result[:200]  # Store first 200 chars
            
            # Extract answer
            answer_match = re.search(r'ANSWER:\s*(-?\d+)', result, re.IGNORECASE)
            if answer_match:
                return int(answer_match.group(1))
            
            # Fallback: extract any number at end
            numbers = re.findall(r'(-?\d+)', result)
            if numbers:
                return int(numbers[-1])
            
            return -1
            
        except Exception as e:
            print(f"[Executor] NVIDIA Vision error: {e}")
            return -1
    
    def _find_element_by_vision(self, elements: List[Dict], target: str) -> int:
        """
        Find element using Vision analysis with SoM annotations.
        
        The Vision agent returns DISPLAY INDICES (0, 1, 2...) from the annotated image.
        We use the SoM annotator's index_to_id_map to convert to actual element IDs.
        
        Returns:
            Element ID or -1 if not found
        """
        time.sleep(Config.API_DELAY)
        
        som_image_path = os.path.join(os.path.dirname(__file__), "som_annotated.png")
        
        try:
            screenshot = self.browser.take_screenshot()
            if not screenshot:
                print("[Vision Agent] No screenshot available")
                return -1
            
            annotated_bytes, filtered = self.annotator.annotate(screenshot, elements)
            
            if not filtered:
                print("[Vision Agent] No elements to annotate")
                return -1
            
            # Save SoM annotated image
            try:
                with open(som_image_path, 'wb') as f:
                    f.write(annotated_bytes)
                print(f"[Vision Agent] SoM image saved ({len(filtered)} elements annotated)")
            except Exception as e:
                print(f"[Vision Agent] Failed to save SoM image: {e}")
            
            # Try NVIDIA API first
            display_index = self._try_nvidia_vision(annotated_bytes, target)
            if display_index >= 0:
                # Convert display index to actual element ID using SoM mapping
                element_id = self.annotator.get_element_id(display_index)
                if element_id >= 0:
                    print(f"[Vision Agent] NVIDIA: display index {display_index} -> element ID {element_id}")
                    return element_id
                else:
                    print(f"[Vision Agent] NVIDIA returned index {display_index} but mapping failed")
            
            # Fallback to Gemini
            if self.gemini_client:
                try:
                    gemini_prompt = f"""Look at this screenshot with numbered red boxes around interactive elements.

TASK: Find the element that best matches: "{target}"

INSTRUCTIONS:
1. Look at the VISUAL content inside each numbered red box
2. Find the box that visually matches what is described
3. For thumbnails/images - look at what the image shows
4. For buttons/links - read the text inside the box
5. Prefer elements in the main content area (center/below header)

Provide brief reasoning (1-2 sentences), then write: ANSWER: [number]
If no element matches, write: ANSWER: -1"""
                    
                    response = self.gemini_client.models.generate_content(
                        model=Config.GEMINI_PLANNING_MODEL,
                        contents=[
                            types.Part.from_bytes(data=annotated_bytes, mime_type="image/png"),
                            gemini_prompt
                        ],
                        config=types.GenerateContentConfig(
                            temperature=0.2,
                            max_output_tokens=700,
                        )
                    )
                    
                    result = response.text.strip()
                    
                    print("\n[Vision Agent] Gemini Output:")
                    print("-" * 40)
                    print(result[:300] if len(result) > 300 else result)
                    print("-" * 40)
                    
                    answer_match = re.search(r'ANSWER:\s*(-?\d+)', result, re.IGNORECASE)
                    if answer_match:
                        display_index = int(answer_match.group(1))
                    else:
                        display_index = next((int(n) for n in result.split() if n.lstrip('-').isdigit()), -1)
                    
                    if display_index >= 0:
                        # Convert display index to actual element ID
                        element_id = self.annotator.get_element_id(display_index)
                        if element_id >= 0:
                            print(f"[Vision Agent] Gemini: display index {display_index} -> element ID {element_id}")
                            return element_id
                    
                except Exception as e:
                    print(f"[Vision Agent] Gemini error: {e}")
            
            print("[Vision Agent] All vision agents failed to find element")
            return -1
            
        except Exception as e:
            print(f"[Vision Agent] Error: {e}")
            import traceback
            traceback.print_exc()
            return -1
    
    def _execute_type(self, text: str) -> Dict:
        """Type text into focused element."""
        success = self.browser.type_text(text)
        return {
            'success': success,
            'method': 'direct',
            'error': None if success else 'Type failed'
        }
    
    def _execute_press_key(self, key: str) -> Dict:
        """Press a keyboard key."""
        success = self.browser.press_key(key)
        time.sleep(Config.STEP_DELAY)
        return {
            'success': success,
            'method': 'direct',
            'error': None if success else 'Key press failed'
        }
    
    def _execute_scroll(self, target: str) -> Dict:
        """
        Scroll the page using mouse wheel.
        Target can be: "down", "up", "down 800", "up 400", etc.
        """
        parts = target.lower().split()
        direction = parts[0] if parts and parts[0] in ['up', 'down'] else 'down'
        
        pixels = 600  # default
        if len(parts) > 1 and parts[1].isdigit():
            pixels = int(parts[1])
        
        success = self.browser.scroll(direction, pixels)
        time.sleep(1)
        return {
            'success': success,
            'method': 'mouse_wheel',
            'pixels': pixels,
            'error': None if success else 'Scroll failed'
        }
    
    def _execute_strong_scroll(self, target: str) -> Dict:
        """
        Strong scroll for YouTube Shorts, Instagram Reels, etc.
        Uses 1200px to move to next short/reel.
        """
        direction = target.lower() if target.lower() in ['up', 'down'] else 'down'
        success = self.browser.strong_scroll(direction)
        time.sleep(1.5)
        return {
            'success': success,
            'method': 'strong_scroll',
            'pixels': 1200,
            'error': None if success else 'Strong scroll failed'
        }
    
    def _execute_wait(self, seconds: str) -> Dict:
        """Wait for specified seconds."""
        try:
            wait_time = int(seconds) if seconds.isdigit() else 2
            time.sleep(wait_time)
            return {'success': True, 'method': 'direct'}
        except:
            return {'success': False, 'method': 'direct', 'error': 'Invalid wait time'}
    
    def _execute_go_back(self) -> Dict:
        """Navigate back to previous page."""
        success = self.browser.go_back()
        time.sleep(Config.STEP_DELAY)
        return {
            'success': success,
            'method': 'direct',
            'error': None if success else 'Go back failed'
        }
    
    def get_stats(self) -> Dict:
        """Get execution statistics."""
        return self.stats.copy()
