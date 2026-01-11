"""
Main entry point for WebCane3 ReAct workflow.
Implements the Step-by-Step Supervisor Loop architecture.

Workflow:
- START -> Router (URL Extraction) -> Start Browser -> 
- LOOP: [Observe -> Supervise -> Execute] until COMPLETE/FAILED
"""

import time
from typing import Dict, Optional

from .config import Config
from .state import WebCaneState, create_initial_state
from .browser_controller import BrowserController
from .observer import Observer
from .supervisor import Supervisor
from .executor import Executor
from .url_extractor import URLExtractor

# LangGraph imports
from langgraph.graph import StateGraph, START, END


class WebCane:
    """
    WebCane3 - ReAct-based Web Automation System.
    
    Uses a circular Supervisor loop instead of pre-planned steps:
    - Observer: Analyzes current page state
    - Supervisor: Decides ONE action based on observation
    - Executor: Executes the action
    - Loop continues until goal is COMPLETE or FAILED
    """
    
    def __init__(self):
        """Initialize WebCane3 ReAct system."""
        print("=" * 60)
        print("WEBCANE3 - ReAct Architecture")
        print("=" * 60)
        
        # Initialize components
        self.browser = BrowserController()
        self.observer = Observer()
        self.supervisor = Supervisor()
        self.executor = Executor(browser=self.browser)
        self.url_extractor = URLExtractor()
        
        # Track state
        self.is_first_task = True
        
        # Voice interface for accessibility (set during execute_goal)
        self.voice = None
        
        # Build graph
        self.graph = self._build_graph()
        
        Config.print_status()
        print("[WebCane3] Ready")
        print("=" * 60)
    
    def _is_browser_active(self) -> bool:
        """Check if browser is active with a page loaded."""
        if not self.browser.page:
            return False
        try:
            url = self.browser.page.url
            return url and url != "about:blank"
        except:
            return False
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph ReAct workflow."""
        graph = StateGraph(WebCaneState)
        
        # ==================== NODES ====================
        
        def router(state: WebCaneState) -> dict:
            """Extract URL from goal and prepare for navigation."""
            goal = state["goal"]
            starting_url = state.get("starting_url", "")
            
            # If starting_url already provided and valid, use it
            if starting_url and starting_url.startswith("http"):
                print(f"[Router] Using provided URL: {starting_url}")
                return {"starting_url": starting_url}
            
            # Extract URL from goal
            print("\n" + "-" * 50)
            print("ROUTER PHASE (URL Extraction)")
            print("-" * 50)
            extracted = self.url_extractor.extract_url(goal)
            print(f"[Router] Extracted URL: {extracted}")
            
            return {"starting_url": extracted}
        
        def start_browser(state: WebCaneState) -> dict:
            """Start browser and navigate to initial URL."""
            print("\n" + "-" * 50)
            print("BROWSER START PHASE")
            print("-" * 50)
            
            url = state.get("starting_url", "https://www.google.com")
            
            if not self.browser.start_browser(headless=False):
                return {"error": "Browser start failed"}
            
            # Voice feedback for navigation
            if self.voice:
                site_name = url.replace("https://", "").replace("http://", "").split("/")[0]
                self.voice.speak(f"Navigating to {site_name}")
            
            print(f"[Browser] Navigating to: {url}")
            if not self.browser.navigate(url):
                return {"error": f"Navigation failed to {url}"}
            
            time.sleep(Config.STEP_DELAY)
            info = self.browser.get_page_info()
            print(f"[Browser] Loaded: {info['title']}")
            
            return {"current_url": info["url"], "error": None}
        
        def observe(state: WebCaneState) -> dict:
            """Take screenshot and analyze page for action-oriented context."""
            print("\n" + "-" * 50)
            print(f"OBSERVE PHASE (Loop {state.get('loop_count', 0) + 1})")
            print("-" * 50)
            
            screenshot = None
            elements = []
            current_url = state.get("current_url", "about:blank")
            observation = {"page_state": "Unknown", "blockers": [], "key_elements": []}
            
            try:
                if self.browser.page:
                    # Check for new tabs and switch if needed
                    self.browser.check_for_new_tabs()
                    
                    page_info = self.browser.get_page_info()
                    current_url = page_info.get("url", current_url)
                    print(f"[Observe] Current URL: {current_url}")
                    
                    screenshot = self.browser.take_screenshot()
                    elements = self.browser.extract_elements()
                    print(f"[Observe] Found {len(elements)} elements")
                    
                    if screenshot and self.observer.available:
                        observation = self.observer.analyze_for_action(
                            screenshot_bytes=screenshot,
                            goal=state["goal"],
                            last_action=state.get("last_action"),
                            last_action_success=state.get("last_action_success")
                        )
                    
            except Exception as e:
                print(f"[Observe] Error: {e}")
            
            return {
                "screenshot": screenshot,
                "elements": elements,
                "current_url": current_url,
                "observation": observation.get("page_state", "Unknown"),
                "blockers": observation.get("blockers", [])
            }
        
        def supervise(state: WebCaneState) -> dict:
            """Decide the next single action based on observation."""
            print("\n" + "-" * 50)
            print("SUPERVISOR PHASE")
            print("-" * 50)
            
            # Get key elements from last observation
            key_elements = []
            if self.observer.last_observation:
                key_elements = self.observer.last_observation.get("key_elements", [])
            
            decision = self.supervisor.decide_next_action(
                goal=state["goal"],
                observation=state.get("observation", "Unknown page"),
                blockers=state.get("blockers", []),
                execution_history=state.get("execution_history", [])[-Config.EXECUTION_HISTORY_SIZE:],
                current_url=state.get("current_url", ""),
                key_elements=key_elements
            )
            
            action_type = decision.get("action", "").upper()
            
            # Handle termination signals
            if action_type == "COMPLETE":
                print("[Supervisor] Goal COMPLETE!")
                return {
                    "is_complete": True,
                    "error": None
                }
            
            if action_type == "FAILED":
                reason = decision.get("reason", "Supervisor determined goal cannot be completed")
                print(f"[Supervisor] Goal FAILED: {reason}")
                return {
                    "is_complete": False,
                    "error": reason
                }
            
            # Return action to execute
            return {
                "last_action": decision,
                "loop_count": state.get("loop_count", 0) + 1
            }
        
        def execute(state: WebCaneState) -> dict:
            """Execute the supervisor's decision."""
            print("\n" + "-" * 50)
            print("EXECUTOR PHASE")
            print("-" * 50)
            
            action = state.get("last_action")
            if not action:
                print("[Execute] No action to execute")
                return {"last_action_success": False}
            
            # Voice feedback for action (non-blocking)
            action_type = action.get("action", "")
            target = action.get("target", "")[:30]
            if self.voice and action_type:
                if action_type == "search":
                    self.voice.speak(f"Searching for {action.get('query', target)}")
                elif action_type == "click":
                    self.voice.speak(f"Clicking {target}")
                elif action_type == "type":
                    self.voice.speak(f"Typing")
                elif action_type == "navigate":
                    self.voice.speak(f"Going to {target}")
            
            result = self.executor.execute_action(action)
            success = result.get("success", False)
            
            print(f"[Execute] Result: {'SUCCESS' if success else 'FAILED'}")
            if result.get('method'):
                print(f"[Execute] Method: {result['method']}")
            if result.get('error'):
                print(f"[Execute] Error: {result['error']}")
            
            # Build history entry
            history_entry = {
                "action": action.get("action"),
                "target": action.get("target", "")[:50],
                "success": success,
                "timestamp": time.time() - state.get("start_time", time.time())
            }
            
            # Add Vision reasoning if available (helps Supervisor verify visual tasks)
            if result.get('vision_reasoning'):
                history_entry["vision_confirmed"] = result['vision_reasoning'][:100]
            
            # Wait for page to settle
            time.sleep(Config.STEP_DELAY)
            
            return {
                "last_action_success": success,
                "execution_history": [history_entry]  # Uses operator.add
            }
        
        def finalize_success(state: WebCaneState) -> dict:
            """Handle successful goal completion."""
            elapsed = time.time() - state.get("start_time", time.time())
            print("\n" + "=" * 60)
            print("SUCCESS!")
            print(f"Goal: {state['goal']}")
            print(f"Time: {elapsed:.2f}s")
            print(f"Actions taken: {len(state.get('execution_history', []))}")
            print("=" * 60)
            return {"is_complete": True}
        
        def finalize_failure(state: WebCaneState) -> dict:
            """Handle goal failure."""
            elapsed = time.time() - state.get("start_time", time.time())
            print("\n" + "=" * 60)
            print("FAILED")
            print(f"Goal: {state['goal']}")
            print(f"Error: {state.get('error', 'Unknown')}")
            print(f"Time: {elapsed:.2f}s")
            print(f"Actions taken: {len(state.get('execution_history', []))}")
            print("=" * 60)
            return {"is_complete": False}
        
        # ==================== ADD NODES ====================
        
        graph.add_node("router", router)
        graph.add_node("start_browser", start_browser)
        graph.add_node("observe", observe)
        graph.add_node("supervise", supervise)
        graph.add_node("execute", execute)
        graph.add_node("finalize_success", finalize_success)
        graph.add_node("finalize_failure", finalize_failure)
        
        # ==================== ROUTING FUNCTIONS ====================
        
        def after_start(state):
            """Route after START based on whether browser is active."""
            if state.get("_is_first_task"):
                return "router"
            return "observe"
        
        def after_browser(state):
            """Route after browser start."""
            if state.get("error"):
                return "finalize_failure"
            return "observe"
        
        def after_supervise(state):
            """Route after supervisor decision."""
            # Check for completion
            if state.get("is_complete"):
                return "finalize_success"
            
            # Check for failure
            if state.get("error"):
                return "finalize_failure"
            
            # Check loop limit
            loop_count = state.get("loop_count", 0)
            if loop_count >= Config.MAX_LOOP_ITERATIONS:
                print(f"[Router] Loop limit reached ({loop_count})")
                return "finalize_failure"
            
            # Continue to execute
            return "execute"
        
        # ==================== ADD EDGES ====================
        
        # Initial flow
        graph.add_conditional_edges(START, after_start)
        graph.add_edge("router", "start_browser")
        graph.add_conditional_edges("start_browser", after_browser)
        
        # ReAct loop
        graph.add_edge("observe", "supervise")
        graph.add_conditional_edges("supervise", after_supervise)
        graph.add_edge("execute", "observe")  # Loop back!
        
        # Terminal nodes
        graph.add_edge("finalize_success", END)
        graph.add_edge("finalize_failure", END)
        
        return graph.compile()
    
    def execute_goal(
        self,
        goal: str,
        starting_url: str = None,
        voice = None
    ) -> Dict:
        """
        Execute a web automation goal using ReAct loop.
        
        Args:
            goal: The goal to achieve
            starting_url: Optional URL (extracted from goal if not provided)
            voice: VoiceInterface instance for accessibility (optional)
            
        Returns:
            Result dictionary with success, actions_taken, elapsed_time, error
        """
        print("\n" + "=" * 60)
        print("WEBCANE3 - EXECUTING GOAL (ReAct)")
        print("=" * 60)
        print(f"Goal: {goal}")
        if starting_url:
            print(f"Starting URL: {starting_url}")
        
        # Set voice interface for accessibility feedback
        self.voice = voice
        
        # Determine if this is first task or follow-up
        is_first_task = not self._is_browser_active()
        
        if is_first_task:
            print("[Mode] NEW TASK - will extract URL and start browser")
        else:
            current_url = self.browser.page.url if self.browser.page else "unknown"
            print(f"[Mode] FOLLOW-UP - browser active at {current_url}")
        
        print("=" * 60)
        
        # Create initial state
        initial_state = create_initial_state(goal, starting_url or "")
        initial_state["_is_first_task"] = is_first_task
        
        # If follow-up, set current URL from browser
        if not is_first_task and self.browser.page:
            initial_state["current_url"] = self.browser.page.url
        
        # Run the graph
        try:
            final_state = self.graph.invoke(
                initial_state,
                config={"recursion_limit": Config.MAX_LOOP_ITERATIONS + 10}
            )
            
            history = final_state.get("execution_history", [])
            successful = len([h for h in history if h.get("success")])
            
            # Mark that we've run at least one task
            self.is_first_task = False
            
            return {
                "success": final_state.get("is_complete", False),
                "actions_taken": len(history),
                "successful_actions": successful,
                "final_url": final_state.get("current_url", ""),
                "elapsed_time": time.time() - final_state.get("start_time", time.time()),
                "error": final_state.get("error")
            }
            
        except Exception as e:
            print(f"[WebCane3] Execution error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }
    
    def close(self):
        """Close browser and cleanup."""
        print("\n[WebCane3] Closing...")
        self.browser.close()
        self.is_first_task = True
        print("[WebCane3] Done")
