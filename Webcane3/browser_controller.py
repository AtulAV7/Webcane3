"""
Browser Controller for WebCane3.
Playwright-based browser automation with DOM extraction.
"""

from playwright.sync_api import sync_playwright, Page, Browser
from typing import List, Dict, Optional
import json

from .config import Config


class BrowserController:
    """
    Playwright-based browser controller.
    Handles browser lifecycle, navigation, and DOM interactions.
    """
    
    def __init__(self):
        """Initialize browser controller - no browser started yet."""
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
    
    def start_browser(self, headless: bool = False) -> bool:
        """
        Launch Chromium browser with Playwright.
        Reuses existing browser if already running.
        
        Args:
            headless: Run browser in headless mode
            
        Returns:
            True on success, False on failure
        """
        # Check if browser already active
        if self.playwright and self.browser:
            try:
                if self.browser.is_connected():
                    if self.page and not self.page.is_closed():
                        print("[Browser] Session already active, reusing")
                        return True
                    else:
                        print("[Browser] Opening new page in existing session")
                        self.page = self.browser.new_page(no_viewport=True)
                        self.page.set_viewport_size({
                            'width': Config.BROWSER_VIEWPORT_WIDTH,
                            'height': Config.BROWSER_VIEWPORT_HEIGHT
                        })
                        return True
            except Exception as e:
                print(f"[Browser] Existing session invalid: {e}, restarting...")
                self.close()
        
        # Start new session
        try:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(
                headless=headless,
                args=['--start-maximized']
            )
            self.page = self.browser.new_page(no_viewport=True)
            self.page.set_viewport_size({
                'width': Config.BROWSER_VIEWPORT_WIDTH,
                'height': Config.BROWSER_VIEWPORT_HEIGHT
            })
            print("[Browser] Started successfully")
            return True
            
        except Exception as e:
            print(f"[Browser] Failed to start: {e}")
            return False
    
    def navigate(self, url: str) -> bool:
        """
        Navigate to URL and wait for page load.
        
        Args:
            url: Target URL
            
        Returns:
            True on success, False on failure
        """
        if not self.page:
            print("[Browser] Not started. Call start_browser() first.")
            return False
        
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            
        try:
            self.page.goto(url, wait_until='domcontentloaded', timeout=30000)
            
            # Wait for network idle with fallback
            try:
                self.page.wait_for_load_state('networkidle', timeout=5000)
            except:
                pass  # Ignore timeout
            
            print(f"[Browser] Navigated to: {url}")
            return True
            
        except Exception as e:
            print(f"[Browser] Navigation failed: {e}")
            return False
    
    def extract_elements(self) -> List[Dict]:
        """
        Extract interactive elements visible in the current viewport only.
        
        Returns:
            List of element dictionaries with id, tag, text, bbox, etc.
        """
        if not self.page:
            return []
        
        try:
            js_code = """
            () => {
                const elements = [];
                let id = 0;
                const selectors = [
                    'button', 'a', 'input', 'textarea', 'select',
                    '[role="button"]', '[role="link"]', '[onclick]', '[tabindex]'
                ];
                
                const allElements = new Set();
                selectors.forEach(selector => {
                    document.querySelectorAll(selector).forEach(el => allElements.add(el));
                });
                
                const vw = Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0);
                const vh = Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0);

                allElements.forEach(el => {
                    try {
                        const rect = el.getBoundingClientRect();
                        const style = window.getComputedStyle(el);
                        
                        // Visibility check
                        const isVisible = (
                            style.display !== 'none' &&
                            style.visibility !== 'hidden' &&
                            style.opacity !== '0' &&
                            el.offsetWidth > 0 &&
                            el.offsetHeight > 0
                        );
                        if (!isVisible) return;
                        
                        // Viewport check
                        const inViewport = (
                            rect.top < vh && 
                            rect.bottom > 0 &&
                            rect.left < vw && 
                            rect.right > 0
                        );
                        if (!inViewport) return;

                        // Extract text
                        let text = "";
                        const ariaLabel = el.getAttribute('aria-label');
                        const title = el.getAttribute('title');
                        const placeholder = el.getAttribute('placeholder');
                        const value = el.value;
                        const innerText = el.innerText || el.textContent;
                        
                        if (ariaLabel) text = ariaLabel;
                        else if (title) text = title;
                        else if (placeholder) text = placeholder;
                        else if (value && el.tagName === 'INPUT') text = value;
                        else if (innerText) text = innerText;
                        
                        text = (text || "").replace(/\\s+/g, ' ').trim().substring(0, 100);

                        elements.push({
                            id: id++,
                            tag: el.tagName.toLowerCase(),
                            text: text,
                            type: el.tagName === 'INPUT' ? (el.type || 'text') : 'button',
                            bbox: {
                                x: Math.round(rect.x),
                                y: Math.round(rect.y),
                                w: Math.round(rect.width),
                                h: Math.round(rect.height)
                            },
                            html_id: el.id || '',
                            html_classes: el.className || ''
                        });
                        
                    } catch (err) {
                        console.error(err);
                    }
                });
                
                return elements.sort((a, b) => a.id - b.id);
            }
            """
            
            elements = self.page.evaluate(js_code)
            print(f"[Browser] Extracted {len(elements)} elements")
            return elements
            
        except Exception as e:
            print(f"[Browser] Failed to extract elements: {e}")
            return []
    
    def take_screenshot(self) -> Optional[bytes]:
        """
        Take screenshot of the current viewport only.
        
        Returns:
            PNG screenshot bytes, or None on failure
        """
        if not self.page:
            return None
        
        try:
            screenshot_bytes = self.page.screenshot(full_page=False)
            print("[Browser] Screenshot captured")
            return screenshot_bytes
            
        except Exception as e:
            print(f"[Browser] Screenshot failed: {e}")
            return None
    
    def click_element(self, element_id: int, elements: List[Dict]) -> bool:
        """
        Click an element by its ID using mouse coordinates.
        Automatically switches to new tab if one opens.
        
        Args:
            element_id: Element ID from extract_elements()
            elements: List of elements from extract_elements()
            
        Returns:
            True on success, False on failure
        """
        if not self.page:
            return False
        
        try:
            element = next((el for el in elements if el['id'] == element_id), None)
            
            if not element:
                print(f"[Browser] Element {element_id} not found")
                return False
            
            bbox = element['bbox']
            center_x = bbox['x'] + bbox['w'] / 2
            center_y = bbox['y'] + bbox['h'] / 2
            
            # Get page count before click
            pages_before = len(self.browser.contexts[0].pages) if self.browser else 0
            
            self.page.mouse.click(center_x, center_y)
            self.page.wait_for_timeout(500)
            
            # Check if new tab opened
            if self.browser:
                pages_after = self.browser.contexts[0].pages
                if len(pages_after) > pages_before:
                    # Switch to newest tab
                    self._switch_to_newest_tab()
            
            print(f"[Browser] Clicked element {element_id} at ({center_x:.0f}, {center_y:.0f})")
            return True
            
        except Exception as e:
            print(f"[Browser] Click failed: {e}")
            return False
    
    def _switch_to_newest_tab(self):
        """Switch to the most recently opened tab."""
        try:
            pages = self.browser.contexts[0].pages
            if len(pages) > 1:
                newest_page = pages[-1]
                self.page = newest_page
                
                # Wait for new page to load
                try:
                    self.page.wait_for_load_state('domcontentloaded', timeout=5000)
                except:
                    pass
                
                print(f"[Browser] Switched to new tab: {self.page.url}")
        except Exception as e:
            print(f"[Browser] Tab switch failed: {e}")
    
    def check_for_new_tabs(self) -> bool:
        """
        Check for new tabs and switch to the newest one if found.
        Call this after actions that might open new tabs.
        
        Returns:
            True if switched to a new tab, False otherwise
        """
        if not self.browser:
            return False
        
        try:
            pages = self.browser.contexts[0].pages
            
            # If there are multiple tabs and we're not on the last one
            if len(pages) > 1:
                newest = pages[-1]
                if newest != self.page:
                    self.page = newest
                    try:
                        self.page.wait_for_load_state('domcontentloaded', timeout=5000)
                    except:
                        pass
                    print(f"[Browser] Detected and switched to new tab: {self.page.url}")
                    return True
            
            return False
        except Exception as e:
            print(f"[Browser] Tab check failed: {e}")
            return False
    
    def type_text(self, text: str) -> bool:
        """
        Type text into the currently focused element.
        
        Args:
            text: Text to type
            
        Returns:
            True on success, False on failure
        """
        if not self.page:
            return False
        try:
            self.page.keyboard.type(text, delay=50)
            print(f"[Browser] Typed: {text}")
            return True
        except Exception as e:
            print(f"[Browser] Type failed: {e}")
            return False
    
    def press_key(self, key: str) -> bool:
        """
        Press a keyboard key.
        
        Args:
            key: Key to press (Enter, Tab, Escape, etc.)
            
        Returns:
            True on success, False on failure
        """
        if not self.page:
            return False
        try:
            self.page.keyboard.press(key)
            print(f"[Browser] Pressed: {key}")
            return True
        except Exception as e:
            print(f"[Browser] Key press failed: {e}")
            return False
    
    def scroll(self, direction: str = "down", pixels: int = 600) -> bool:
        """
        Scroll the page using mouse wheel.
        
        Args:
            direction: "up" or "down"
            pixels: Number of pixels to scroll (use larger values for shorts/reels)
            
        Returns:
            True on success, False on failure
        """
        if not self.page:
            return False
        try:
            # Move mouse to center so wheel event is captured correctly
            viewport = self.page.viewport_size
            center_x = viewport['width'] / 2
            center_y = viewport['height'] / 2
            
            self.page.mouse.move(center_x, center_y)
            
            # Calculate scroll delta
            delta_y = pixels if direction.lower() == "down" else -pixels
            
            # Perform mouse wheel scroll
            self.page.mouse.wheel(0, delta_y)
            
            print(f"[Browser] Mouse wheel scroll {direction} ({pixels}px)")
            return True
        except Exception as e:
            print(f"[Browser] Scroll failed: {e}")
            return False
    
    def strong_scroll(self, direction: str = "down") -> bool:
        """
        Strong scroll for YouTube Shorts, Instagram Reels, etc.
        Uses 1200px scroll to move to next short/reel.
        
        Args:
            direction: "up" or "down" (next/previous)
            
        Returns:
            True on success, False on failure
        """
        return self.scroll(direction=direction, pixels=1200)
    
    def get_page_info(self) -> Dict:
        """
        Get current page information.
        
        Returns:
            Dictionary with url, title, and viewport dimensions
        """
        if not self.page:
            return {'url': '', 'title': '', 'viewport': {'width': 0, 'height': 0}}
        
        try:
            return {
                'url': self.page.url,
                'title': self.page.title(),
                'viewport': {
                    'width': self.page.viewport_size['width'],
                    'height': self.page.viewport_size['height']
                }
            }
        except Exception as e:
            print(f"[Browser] Failed to get page info: {e}")
            return {'url': '', 'title': '', 'viewport': {'width': 0, 'height': 0}}
    
    def get_current_state(self) -> Dict:
        """
        Get current page state for verification.
        
        Returns:
            State dictionary with url, title, element_count
        """
        page_info = self.get_page_info()
        elements = self.extract_elements()
        
        return {
            'url': page_info['url'],
            'title': page_info['title'],
            'element_count': len(elements)
        }
    
    def close(self):
        """Close browser and cleanup resources."""
        try:
            if self.page:
                self.page.close()
            if self.browser:
                self.browser.close()
            if self.playwright:
                self.playwright.stop()
            print("[Browser] Closed")
        except Exception as e:
            print(f"[Browser] Error during cleanup: {e}")
        finally:
            self.page = None
            self.browser = None
            self.playwright = None
