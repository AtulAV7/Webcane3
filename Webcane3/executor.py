"""
Executor agent for WebCane3.
Hybrid DOM/Vision action execution with scroll-retry logic.
"""

import time
from typing import Dict, List, Optional
import ollama

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
    Hybrid action executor.
    Uses DOM text matching (System 1) with Vision fallback (System 2).
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
        
        self.local_model = Config.OLLAMA_MODEL
        
        # Stats
        self.stats = {
            'dom_success': 0,
            'vision_success': 0,
            'failures': 0
        }
        
        # Track last action for typing safety
        self.last_action_was_click = False
    
    def execute_action(self, action: Dict) -> Dict:
        """
        Execute a single action step.
        
        Args:
            action: Action dictionary with action, target, description
            
        Returns:
            Result dictionary with success, method, error fields
        """
        action_type = action.get('action', '').lower()
        target = action.get('target', '')
        
        print(f"[Executor] {action_type}: {target}")
        
        if action_type == 'navigate':
            self.last_action_was_click = False
            return self._execute_navigate(target)
        elif action_type == 'find_and_click':
            result = self._execute_click(target)
            # Track successful click for typing safety
            self.last_action_was_click = result.get('success', False)
            return result
        elif action_type == 'type':
            # Check if a click happened before typing
            if not self.last_action_was_click:
                print("[Executor] WARNING: Typing without prior click - attempting to focus first...")
                # Try to click on a common search/input element first
                focus_result = self._try_focus_input()
                if not focus_result:
                    print("[Executor] WARNING: Could not focus input, typing may fail")
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
    
    def _try_focus_input(self) -> bool:
        """Try to focus an input element before typing."""
        try:
            # Look for common search input elements
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
    
    def _filter_elements_by_context(self, elements: List[Dict], target: str) -> tuple:
        """
        Filter elements to separate content from navigation.
        Returns (content_elements, nav_elements).
        
        Deprioritizes:
        - Elements in top 100px (nav area)
        - Elements with nav-related text
        
        Prioritizes:
        - Product cards (prices, buy buttons)
        - Video thumbnails
        - Main content area elements
        """
        nav_keywords = ['login', 'sign', 'menu', 'cart', 'account', 'profile', 
                       'home', 'categories', 'help', 'contact']
        content_keywords = ['add to', 'buy', 'price', '₹', '$', 'views', 'ago',
                           'product', 'item', 'result']
        
        content_elements = []
        nav_elements = []
        
        for el in elements:
            bbox = el.get('bbox', {})
            y_pos = bbox.get('y', 0)
            text = (el.get('text', '') or '').lower()
            
            # Check if nav-related
            is_nav = False
            if y_pos < 100:  # Top 100px is usually nav
                is_nav = True
            elif any(kw in text for kw in nav_keywords):
                is_nav = True
            
            # Check if content-related
            is_content = False
            if y_pos > 150:  # Below nav area
                is_content = True
            if any(kw in text for kw in content_keywords):
                is_content = True
            
            # Prioritize based on target context
            target_lower = target.lower()
            if 'product' in target_lower or 'result' in target_lower or 'first' in target_lower:
                # For product/result targets, strongly prefer content area
                if is_content and not is_nav:
                    content_elements.append(el)
                else:
                    nav_elements.append(el)
            elif 'search' in target_lower or 'input' in target_lower:
                # For search/input, nav area is fine
                if 'search' in text or 'input' in el.get('tag', '').lower():
                    content_elements.append(el)
                else:
                    nav_elements.append(el)
            else:
                # Default: prefer content, but keep nav available
                if is_content:
                    content_elements.append(el)
                else:
                    nav_elements.append(el)
        
        return content_elements, nav_elements
    
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
        # Check if this is a visual task that should prioritize Vision
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
            
            # Extract elements
            elements = self.browser.extract_elements()
            if not elements:
                continue
            
            page_info = self.browser.get_page_info()
            
            # For visual tasks, try Vision FIRST
            if is_visual_task:
                print("[Executor] Trying Vision first for visual task...")
                element_id = self._find_element_by_vision(elements, target)
                
                if element_id >= 0 and element_id < len(elements):
                    if self.browser.click_element(element_id, elements):
                        self.stats['vision_success'] += 1
                        time.sleep(Config.STEP_DELAY)
                        return {
                            'success': True,
                            'method': 'vision',
                            'element_id': element_id,
                            'scroll_attempts': scroll_attempt
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
                            'scroll_attempts': scroll_attempt
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
            return self._find_element_local(elements, target)
        
        try:
            # Rate limiting
            time.sleep(Config.API_DELAY)
            
            # Format elements for prompt
            elem_list = []
            for el in elements[:80]:  # Limit to 60 elements
                text = el['text'][:50] if el['text'] else f"[{el['tag']}]"
                elem_list.append(f"[{el['id']}] {el['tag']}: \"{text}\"")
            
            prompt = f"""Find the element that best matches this task: "{target}"

Elements:
{chr(10).join(elem_list)}

Return ONLY the element ID number that best matches. If none match, return -1.
ID:"""
            
            response = self.groq_client.chat.completions.create(
                model=Config.GROQ_DOM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip()
            # Extract number
            match = next((int(n) for n in result.split() if n.lstrip('-').isdigit()), -1)
            
            if match >= 0 and match < len(elements):
                print(f"[Executor] DOM match: element {match}")
                return match
            return -1
            
        except Exception as e:
            print(f"[Executor] Groq error: {e}")
            return self._find_element_local(elements, target)
    
    def _find_element_local(self, elements: List[Dict], target: str) -> int:
        """Local fallback for element finding."""
        try:
            # Simple keyword matching
            target_lower = target.lower()
            keywords = target_lower.split()
            
            for el in elements:
                text = (el.get('text', '') or '').lower()
                if any(kw in text for kw in keywords):
                    return el['id']
            
            return -1
        except:
            return -1
    
    def _try_nvidia_vision(self, annotated_bytes: bytes, target: str, elem_list: list) -> int:
        """
        Try NVIDIA API (Mistral Large) for vision analysis.
        
        Args:
            annotated_bytes: Annotated screenshot bytes
            target: The target element description to find
            elem_list: List of element descriptions
        
        Returns:
            Element ID/index or -1 if failed
        """
        import requests
        import base64
        import re
        
        if not Config.NVIDIA_API_KEY:
            print("[Executor] NVIDIA Vision: No API key configured")
            return -1
        
        try:
            # Encode image
            b64_image = base64.b64encode(annotated_bytes).decode('utf-8')
            
            # Build message content with image
            headers = {
                "Authorization": f"Bearer {Config.NVIDIA_API_KEY}",
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
            
            # Build NVIDIA prompt with actual target
            nvidia_prompt = f"""Look at this annotated screenshot. Each element has a red box with a number.

TASK: Find the element that best matches: "{target}"

Available elements:
{chr(10).join(elem_list[:50])}

Analyze the image and identify which numbered box matches best.
First provide brief reasoning (1-2 sentences), then on a new line write: ANSWER: [number]

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
            response = requests.post(Config.NVIDIA_API_URL, headers=headers, json=payload, timeout=30)
            
            if response.status_code != 200:
                print(f"[Executor] NVIDIA Vision error: {response.status_code} - {response.text[:200]}")
                return -1
            
            result_json = response.json()
            result = result_json['choices'][0]['message']['content'].strip()
            
            # Print full output
            print("\n[Executor] NVIDIA Vision Output:")
            print("-" * 50)
            print(result)
            print("-" * 50)
            
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
        Saves annotated screenshot and reasons over the image.
        
        Returns:
            Element ID or -1 if not found
        """
        # Rate limiting
        time.sleep(Config.API_DELAY)
        
        # Path to save SoM annotated image
        import os
        som_image_path = os.path.join(os.path.dirname(__file__), "som_annotated.png")
        
        try:
            # Take screenshot and annotate
            screenshot = self.browser.take_screenshot()
            if not screenshot:
                print("[Executor] Vision: No screenshot available")
                return -1
            
            annotated_bytes, filtered = self.annotator.annotate(screenshot, elements)
            
            if not filtered:
                print("[Executor] Vision: No elements to annotate")
                return -1
            
            # Save SoM annotated image
            try:
                with open(som_image_path, 'wb') as f:
                    f.write(annotated_bytes)
                print(f"[Executor] Vision: SoM image saved to {som_image_path}")
            except Exception as e:
                print(f"[Executor] Vision: Failed to save SoM image: {e}")
            
            # Filter elements by context - deprioritize nav, prioritize content
            content_elements, nav_elements = self._filter_elements_by_context(filtered, target)
            
            # Build element list for context with priority markers
            # Include ALL elements so box numbers match the annotated image
            elem_list = []
            # Add content elements first (preferred)
            for el in content_elements:
                text = el.get('text', '')[:40] if el.get('text') else f"[{el.get('tag', 'elem')}]"
                elem_list.append(f"Box {el['id']}: {text} [CONTENT]")
            # Add nav elements after
            for el in nav_elements:
                text = el.get('text', '')[:30] if el.get('text') else f"[{el.get('tag', 'elem')}]"
                elem_list.append(f"Box {el['id']}: {text} [NAV]")
            
            # Chain-of-thought reasoning prompt for vision
            prompt = f"""VISION ANALYSIS TASK

TARGET: "{target}"

ANALYSIS PROCESS (Follow these steps):
Step 1: IDENTIFY ELEMENT TYPE - What are you looking for? (product, button, thumbnail, link?)
Step 2: SCAN IMAGE - Look INSIDE each red numbered box
Step 3: MATCH DESCRIPTION - Which box contains content matching "{target}"?
Step 4: VALIDATE - Is this box in the main content area (not navigation)?

AVAILABLE ELEMENTS:
{chr(10).join(elem_list)}

IMPORTANT RULES:
- Boxes marked [NAV] are navigation elements - avoid unless target is a nav item
- Boxes marked [MAIN CONTENT] are preferred for product/video/content clicks
- For "first product" or "first result" - look for product cards with prices/images
- Elements in top 100px are usually navigation - prefer boxes lower on page

REASONING: First explain your step-by-step analysis (2-3 sentences).
Then on a new line write: ANSWER: [box number]

If no element matches, write: ANSWER: -1"""
            
            # Try NVIDIA API first (Mistral Large with vision)
            nvidia_result = self._try_nvidia_vision(annotated_bytes, target, elem_list)
            if nvidia_result >= 0:
                # NVIDIA returns the box ID (element ID), not an index
                # Look up the element by ID directly
                for el in filtered:
                    if el['id'] == nvidia_result:
                        print(f"[Executor] NVIDIA Vision matched: element {nvidia_result}")
                        return nvidia_result
                # If not found in filtered, might be element ID that wasn't filtered
                print(f"[Executor] NVIDIA Vision returned {nvidia_result} but not found in filtered list")
            
            # Fallback to Gemini for vision
            print("[Executor] Vision: Trying Gemini fallback...")
            if self.gemini_client:
                try:
                    response = self.gemini_client.models.generate_content(
                        model=Config.GEMINI_PLANNING_MODEL,
                        contents=[
                            types.Part.from_bytes(data=annotated_bytes, mime_type="image/png"),
                            prompt
                        ],
                        config=types.GenerateContentConfig(
                            temperature=0.2,
                            max_output_tokens=700,
                        )
                    )
                    
                    result = response.text.strip()
                    
                    # Print full vision agent output
                    print("\n[Executor] Gemini Vision Output:")
                    print("-" * 40)
                    print(result)
                    print("-" * 40)
                    
                    # Extract answer from reasoning
                    import re
                    answer_match = re.search(r'ANSWER:\s*(-?\d+)', result, re.IGNORECASE)
                    if answer_match:
                        match = int(answer_match.group(1))
                    else:
                        # Fallback: extract any number
                        match = next((int(n) for n in result.split() if n.lstrip('-').isdigit()), -1)
                    
                    if match >= 0 and match < len(filtered):
                        print(f"[Executor] Gemini Vision matched: element {filtered[match]['id']}")
                        return filtered[match]['id']
                    elif match >= 0:
                        # match might be the actual element ID, not index
                        for el in filtered:
                            if el['id'] == match:
                                print(f"[Executor] Gemini Vision matched: element {match}")
                                return match
                    
                except Exception as e:
                    print(f"[Executor] Gemini vision error: {e}")
            
            # No more fallbacks - return failure
            print("[Executor] Vision: All vision agents failed")
            return -1
            
        except Exception as e:
            print(f"[Executor] Vision error: {e}")
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
        direction = parts[0] if parts[0] in ['up', 'down'] else 'down'
        
        # Check if pixel value specified
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
        time.sleep(1.5)  # Extra wait for content to load
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
