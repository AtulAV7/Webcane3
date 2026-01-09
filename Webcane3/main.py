"""
Main entry point for WebCane3.
Provides the WebCane class for executing web automation goals.

Workflow:
- NEW TASK (no browser): Extract URL -> Start Browser -> Navigate -> Observe -> Plan -> Execute
- FOLLOW-UP (browser active): Observe -> Plan (may include navigation) -> Execute
"""

import time
from typing import Dict, Optional

from .config import Config
from .state import WebCaneState, create_initial_state
from .browser_controller import BrowserController
from .observer import Observer
from .planner import Planner
from .executor import Executor
from .verifier import Verifier
from .url_extractor import URLExtractor

# LangGraph imports
from langgraph.graph import StateGraph, START, END


class WebCane:
    """
    WebCane3 - LangGraph-based Web Automation System.
    
    Handles both new tasks and follow-up tasks intelligently:
    - New task: Extracts URL from goal, navigates, then plans
    - Follow-up: Observes current page, plans from there
    """
    
    def __init__(self, use_groq_planner: bool = False):
        """
        Initialize WebCane3.
        
        Args:
            use_groq_planner: If True, use Groq gpt-oss-120b for planning.
                              If False, use Gemini Flash.
        """
        print("=" * 60)
        print("WEBCANE3 - Initializing")
        print("=" * 60)
        
        # Initialize components
        self.browser = BrowserController()
        self.observer = Observer()        # Uses GROQ_API_KEY3
        self.planner = Planner(use_groq=use_groq_planner)  # Configurable
        self.executor = Executor(browser=self.browser)  # Uses GROQ_API_KEY
        self.verifier = Verifier()        # Uses GROQ_API_KEY2
        self.url_extractor = URLExtractor()  # Uses GROQ_API_KEY
        
        # Track state
        self.is_first_task = True
        self.replan_count = 0
        
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
        """Build the LangGraph workflow."""
        graph = StateGraph(WebCaneState)
        
        # Node: Extract URL from goal (for first task)
        def extract_url(state: WebCaneState) -> dict:
            goal = state["goal"]
            starting_url = state.get("starting_url", "")
            
            # If starting_url already provided and valid, use it
            if starting_url and starting_url.startswith("http"):
                print(f"[Extract] Using provided URL: {starting_url}")
                return {"starting_url": starting_url}
            
            # Extract URL from goal
            print("\n" + "-" * 50)
            print("URL EXTRACTION PHASE")
            print("-" * 50)
            extracted = self.url_extractor.extract_url(goal)
            print(f"[Extract] Extracted URL: {extracted}")
            
            return {"starting_url": extracted}
        
        # Node: Start browser and navigate
        def start_browser(state: WebCaneState) -> dict:
            print("\n" + "-" * 50)
            print("BROWSER START PHASE")
            print("-" * 50)
            
            url = state.get("starting_url", "https://www.google.com")
            
            if not self.browser.start_browser(headless=False):
                return {"error": "Browser start failed"}
            
            print(f"[Browser] Navigating to: {url}")
            if not self.browser.navigate(url):
                return {"error": f"Navigation failed to {url}"}
            
            info = self.browser.get_page_info()
            print(f"[Browser] Loaded: {info['title']}")
            
            return {"current_url": info["url"], "error": None}
        
        # Node: Observe current page
        def observe(state: WebCaneState) -> dict:
            print("\n" + "-" * 50)
            print("OBSERVATION PHASE")
            print("-" * 50)
            
            page_description = None
            screenshot = None
            elements = []
            current_url = state.get("current_url", "about:blank")
            
            try:
                if self.browser.page:
                    page_info = self.browser.get_page_info()
                    current_url = page_info.get("url", current_url)
                    print(f"[Observe] Current URL: {current_url}")
                    
                    screenshot = self.browser.take_screenshot()
                    
                    if screenshot and self.observer.available:
                        page_description = self.observer.describe_page(screenshot)
                    
                    elements = self.browser.extract_elements()
                    print(f"[Observe] Found {len(elements)} elements")
                    
            except Exception as e:
                print(f"[Observe] Error: {e}")
            
            return {
                "page_description": page_description,
                "screenshot": screenshot,
                "elements": elements,
                "current_url": current_url
            }
        
        # Node: Plan task
        def plan(state: WebCaneState) -> dict:
            print("\n" + "-" * 50)
            print("PLANNING PHASE")
            print("-" * 50)
            
            current_url = state.get("current_url", state.get("starting_url", "about:blank"))
            page_desc = state.get("page_description")
            
            # Check if this is a replan after failure
            if state.get("needs_replan"):
                idx = state.get("current_step_index", 0)
                plan_list = state.get("current_plan", [])
                last_verified_url = state.get("last_verified_url", state.get("starting_url", "about:blank"))
                last_verified_step = state.get("last_verified_step", 0)
                
                if idx < len(plan_list):
                    failed = plan_list[idx]
                    hist = state.get("execution_history", [])
                    reason = hist[-1].get("reason", "Unknown") if hist else "Unknown"
                    
                    # Build failed action info
                    failed_action = {
                        "action": failed.get('action'),
                        "target": failed.get('target'),
                        "error": reason
                    }
                    
                    print(f"\n[Smart Replan] Last verified URL: {last_verified_url}")
                    print(f"[Smart Replan] Last verified step: {last_verified_step}")
                    print(f"[Smart Replan] Failed action: {failed_action}")
                    
                    # Use smart replanning with NVIDIA Mistral
                    replan_result = self.planner.smart_replan(
                        goal=state["goal"],
                        current_url=current_url,
                        page_context=page_desc,
                        failed_action=failed_action,
                        last_verified_url=last_verified_url,
                        last_verified_step=last_verified_step
                    )
                    
                    if replan_result and replan_result.get("plan"):
                        print(f"[Smart Replan] Strategy: {replan_result.get('strategy', 'UNKNOWN')}")
                        return {
                            "current_plan": replan_result["plan"],
                            "current_step_index": 0,
                            "retry_count": 0,
                            "needs_replan": False,
                            "replan_strategy": replan_result.get("strategy"),
                            "error": None
                        }
            
            # Normal planning (not replan)
            print(f"[Planner] Goal: {state['goal']}")
            print(f"[Planner] Current URL: {current_url}")
            if page_desc:
                print(f"[Planner] Page context available: {len(page_desc)} chars")
            
            plan_result = self.planner.decompose_task(
                goal=state["goal"],
                current_url=current_url,
                page_description=page_desc,
                failure_context=None
            )
            
            if plan_result:
                return {
                    "current_plan": plan_result,
                    "current_step_index": 0,
                    "retry_count": 0,
                    "needs_replan": False,
                    "last_verified_url": current_url,  # Initialize last verified URL
                    "last_verified_step": 0,
                    "error": None
                }
            return {"current_plan": [], "error": "Planning failed"}
        
        # Node: Execute action
        def execute(state: WebCaneState) -> dict:
            idx = state.get("current_step_index", 0)
            plan = state.get("current_plan", [])
            
            if idx >= len(plan):
                return {"error": "No more steps"}
            
            action = plan[idx]
            
            print("\n" + "-" * 50)
            print(f"EXECUTION PHASE (Step {idx + 1}/{len(plan)})")
            print(f"  Action: {action.get('action')}")
            print(f"  Target: {action.get('target')}")
            print(f"  Verify: {action.get('verify')}")
            print("-" * 50)
            
            before_state = self.browser.get_current_state()
            before_screenshot = self.browser.take_screenshot()
            
            result = self.executor.execute_action(action)
            
            print(f"[Execute] Result: {'SUCCESS' if result.get('success') else 'FAILED'}")
            if result.get('method'):
                print(f"[Execute] Method: {result['method']}")
            
            time.sleep(Config.STEP_DELAY)
            after_state = self.browser.get_current_state()
            after_screenshot = self.browser.take_screenshot()
            
            return {
                "before_state": before_state,
                "after_state": after_state,
                "before_screenshot": before_screenshot,
                "after_screenshot": after_screenshot,
                "current_url": after_state.get("url", state.get("current_url", "")),
                "execution_history": [{
                    "step": idx + 1,
                    "action": action,
                    "result": result,
                    "timestamp": time.time() - state.get("start_time", time.time())
                }]
            }
        
        # Node: Verify action
        def verify(state: WebCaneState) -> dict:
            idx = state.get("current_step_index", 0)
            plan = state.get("current_plan", [])
            
            if idx >= len(plan):
                return {"error": "No step to verify"}
            
            action = plan[idx]
            verify_method = action.get("verify", "NONE")
            
            print("\n" + "-" * 50)
            print(f"VERIFICATION PHASE (Method: {verify_method})")
            print("-" * 50)
            
            hist = state.get("execution_history", [])
            last = hist[-1] if hist else {}
            exec_result = last.get("result", {})
            
            if not exec_result.get("success", False):
                print(f"[Verify] Execution failed: {exec_result.get('error', 'Unknown')}")
                return {
                    "execution_history": [{
                        "step": idx + 1,
                        "success": False,
                        "reason": exec_result.get("error", "Execution failed"),
                        "method": "EXECUTION"
                    }]
                }
            
            verification = self.verifier.verify_action(
                action=action,
                verify_method=verify_method,
                before_state=state.get("before_state", {}),
                after_state=state.get("after_state", {}),
                before_screenshot=state.get("before_screenshot"),
                after_screenshot=state.get("after_screenshot"),
                goal=state["goal"],
                current_step=idx,
                total_steps=len(plan)
            )
            
            success = verification.get("success", False)
            goal_satisfied = verification.get("goal_satisfied", False)
            
            print(f"[Verify] Result: {'SUCCESS' if success else 'FAILED'}")
            print(f"[Verify] Reason: {verification.get('reason', 'Unknown')}")
            if goal_satisfied:
                print("[Verify] GOAL SATISFIED!")
            
            # Track last verified state on success (for smart replanning recovery)
            result = {
                "is_complete": goal_satisfied,
                "execution_history": [{
                    "step": idx + 1,
                    "success": success,
                    "reason": verification.get("reason", "Unknown"),
                    "method": verification.get("method_used", "Unknown"),
                    "goal_satisfied": goal_satisfied
                }]
            }
            
            # Update last verified state on successful verification
            if success:
                result["last_verified_url"] = state.get("current_url", "")
                result["last_verified_step"] = idx
            
            return result
        
        # Node: Advance to next step
        def advance(state: WebCaneState) -> dict:
            return {
                "current_step_index": state.get("current_step_index", 0) + 1,
                "retry_count": 0
            }
        
        # Node: Handle failure
        def handle_failure(state: WebCaneState) -> dict:
            retry = state.get("retry_count", 0)
            max_retry = Config.MAX_RETRIES
            
            if retry < max_retry:
                print(f"\n[Retry] Attempt {retry + 1}/{max_retry}")
                time.sleep(1)
                return {"retry_count": retry + 1, "needs_replan": False}
            
            if self.replan_count >= Config.MAX_REPLAN_ATTEMPTS:
                print("\n[Failure] Max replans reached, giving up")
                return {"error": "Max replan attempts reached", "is_complete": False}
            
            self.replan_count += 1
            print(f"\n[Replan] Attempt {self.replan_count}/{Config.MAX_REPLAN_ATTEMPTS}")
            return {"needs_replan": True}
        
        # Node: Final visual verification of goal completion
        def final_verify(state: WebCaneState) -> dict:
            """Perform final visual verification that goal was achieved."""
            print("\n" + "-" * 50)
            print("FINAL VERIFICATION PHASE")
            print("-" * 50)
            
            # Take screenshot of current state
            screenshot = self.browser.take_screenshot()
            current_url = state.get("current_url", "")
            
            result = self.verifier.verify_goal_completion(
                goal=state["goal"],
                screenshot=screenshot,
                current_url=current_url
            )
            
            if result.get("success"):
                return {"is_complete": True, "error": None}
            else:
                if result.get("needs_replan") and self.replan_count < Config.MAX_REPLAN_ATTEMPTS:
                    self.replan_count += 1
                    print(f"[Final Verify] Goal not achieved, replanning ({self.replan_count}/{Config.MAX_REPLAN_ATTEMPTS})")
                    return {"needs_replan": True, "is_complete": False}
                else:
                    return {"is_complete": False, "error": result.get("reason", "Goal verification failed")}
        
        # Node: Finalize success
        def finalize_success(state: WebCaneState) -> dict:
            elapsed = time.time() - state.get("start_time", time.time())
            print("\n" + "=" * 60)
            print("SUCCESS!")
            print(f"Time: {elapsed:.2f}s")
            print("=" * 60)
            self.replan_count = 0
            return {"is_complete": True}
        
        # Node: Finalize failure
        def finalize_failure(state: WebCaneState) -> dict:
            print("\n" + "=" * 60)
            print(f"FAILED: {state.get('error', 'Unknown')}")
            print("=" * 60)
            self.replan_count = 0
            return {"is_complete": False}
        
        # Add nodes
        graph.add_node("extract_url", extract_url)
        graph.add_node("start_browser", start_browser)
        graph.add_node("observe", observe)
        graph.add_node("plan", plan)
        graph.add_node("execute", execute)
        graph.add_node("verify", verify)
        graph.add_node("advance", advance)
        graph.add_node("handle_failure", handle_failure)
        graph.add_node("final_verify", final_verify)
        graph.add_node("finalize_success", finalize_success)
        graph.add_node("finalize_failure", finalize_failure)
        
        # Routing functions
        def after_start(state):
            # Check if this is a new task or follow-up
            if state.get("_is_first_task"):
                return "extract_url"
            return "observe"
        
        def after_extract(state):
            return "start_browser"
        
        def after_browser(state):
            if state.get("error"):
                return "finalize_failure"
            return "observe"
        
        def after_observe(state):
            if state.get("needs_replan"):
                return "plan"
            return "plan"
        
        def after_plan(state):
            if state.get("current_plan"):
                return "execute"
            return "finalize_failure"
        
        def after_verify(state):
            if state.get("is_complete"):
                return "finalize_success"
            if state.get("error"):
                return "finalize_failure"
            
            hist = state.get("execution_history", [])
            last = hist[-1] if hist else {}
            
            if last.get("success"):
                idx = state.get("current_step_index", 0)
                plan = state.get("current_plan", [])
                if idx + 1 < len(plan):
                    return "advance"
                # Last step completed - go to final verification
                return "final_verify"
            return "handle_failure"
        
        def after_final_verify(state):
            if state.get("is_complete"):
                return "finalize_success"
            if state.get("needs_replan"):
                return "observe"  # Replan with current observation
            return "finalize_failure"
        
        def after_failure(state):
            if state.get("error"):
                return "finalize_failure"
            if state.get("needs_replan"):
                return "observe"
            return "execute"
        
        # Add edges - Start routes based on first task or not
        graph.add_conditional_edges(START, after_start)
        graph.add_edge("extract_url", "start_browser")
        graph.add_conditional_edges("start_browser", after_browser)
        graph.add_edge("observe", "plan")
        graph.add_conditional_edges("plan", after_plan)
        graph.add_edge("execute", "verify")
        graph.add_conditional_edges("verify", after_verify)
        graph.add_edge("advance", "execute")
        graph.add_conditional_edges("handle_failure", after_failure)
        graph.add_conditional_edges("final_verify", after_final_verify)
        graph.add_edge("finalize_success", END)
        graph.add_edge("finalize_failure", END)
        
        return graph.compile()
    
    def execute_goal(
        self,
        goal: str,
        starting_url: str = None,
        max_retries: int = 2
    ) -> Dict:
        """
        Execute a web automation goal.
        
        Args:
            goal: The goal to achieve
            starting_url: Optional URL (extracted from goal if not provided)
            max_retries: Maximum retry attempts per step
        """
        print("\n" + "=" * 60)
        print("WEBCANE3 - EXECUTING GOAL")
        print("=" * 60)
        print(f"Goal: {goal}")
        if starting_url:
            print(f"Starting URL: {starting_url}")
        
        # Reset replan counter
        self.replan_count = 0
        
        # Determine if this is first task or follow-up
        is_first_task = not self._is_browser_active()
        
        if is_first_task:
            print("[Mode] NEW TASK - will extract URL and start browser")
        else:
            current_url = self.browser.page.url if self.browser.page else "unknown"
            print(f"[Mode] FOLLOW-UP - browser active at {current_url}")
        
        print("=" * 60)
        
        # Create initial state
        initial_state = create_initial_state(goal, starting_url or "", max_retries)
        initial_state["_is_first_task"] = is_first_task
        
        # If follow-up, set current URL from browser
        if not is_first_task and self.browser.page:
            initial_state["current_url"] = self.browser.page.url
        
        # Run the graph with increased recursion limit
        try:
            final_state = self.graph.invoke(
                initial_state,
                config={"recursion_limit": 100}
            )
            
            history = final_state.get("execution_history", [])
            successful = len([h for h in history if h.get("success")])
            
            # Mark that we've run at least one task
            self.is_first_task = False
            
            return {
                "success": final_state.get("is_complete", False),
                "steps_completed": successful,
                "total_steps": len(final_state.get("current_plan", [])),
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
