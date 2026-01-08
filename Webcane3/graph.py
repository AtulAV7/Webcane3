"""
LangGraph StateGraph for WebCane3.
Defines the workflow nodes and edges for web automation.
"""

import time
from typing import Dict

from langgraph.graph import StateGraph, START, END

from .state import WebCaneState, create_initial_state
from .browser_controller import BrowserController
from .observer import Observer
from .planner import Planner
from .executor import Executor
from .verifier import Verifier
from .config import Config


# Node Functions

def observe_node(state: WebCaneState) -> dict:
    """
    Observe the current page state.
    Updates page_description, screenshot, elements, current_url.
    """
    browser = state.get("_browser")
    observer = state.get("_observer")
    
    page_description = None
    screenshot = None
    elements = []
    current_url = state.get("current_url", "about:blank")
    
    try:
        if browser and browser.page:
            print("\n" + "=" * 60)
            print("PHASE: OBSERVATION")
            print("=" * 60)
            
            # Get page info
            page_info = browser.get_page_info()
            current_url = page_info.get("url", current_url)
            
            # Take screenshot
            screenshot = browser.take_screenshot()
            
            # Get page description from observer
            if screenshot and observer and observer.available:
                try:
                    page_description = observer.describe_page(screenshot)
                except Exception as e:
                    print(f"[Observe] Description failed: {e}")
            
            # Extract elements
            elements = browser.extract_elements()
            
    except Exception as e:
        print(f"[Observe] Error: {e}")
    
    return {
        "page_description": page_description,
        "screenshot": screenshot,
        "elements": elements,
        "current_url": current_url
    }


def plan_node(state: WebCaneState) -> dict:
    """
    Generate or update the action plan.
    """
    planner = state.get("_planner")
    
    print("\n" + "=" * 60)
    print("PHASE: PLANNING")
    print("=" * 60)
    
    try:
        # Check if replanning after failure
        failure_context = None
        if state.get("needs_replan"):
            current_step = state["current_step_index"]
            if current_step < len(state.get("current_plan", [])):
                failed_step = state["current_plan"][current_step]
                history = state.get("execution_history", [])
                failure_reason = history[-1].get("reason", "Unknown") if history else "Unknown"
                
                failure_context = {
                    "failed_action": f"{failed_step.get('action')}: {failed_step.get('target')}",
                    "reason": failure_reason,
                    "current_state": state.get("page_description", "Unknown")
                }
        
        plan = planner.decompose_task(
            goal=state["goal"],
            current_url=state.get("current_url", state["starting_url"]),
            page_description=state.get("page_description"),
            failure_context=failure_context
        )
        
        if plan:
            return {
                "current_plan": plan,
                "current_step_index": 0,
                "retry_count": 0,
                "needs_replan": False,
                "error": None
            }
        else:
            return {
                "current_plan": [],
                "error": "Failed to generate plan"
            }
            
    except Exception as e:
        print(f"[Plan] Error: {e}")
        return {
            "current_plan": [],
            "error": f"Planning error: {str(e)}"
        }


def start_browser_node(state: WebCaneState) -> dict:
    """
    Start browser and navigate to starting URL.
    """
    browser = state.get("_browser")
    
    print("\n[Browser] Starting session...")
    
    try:
        if not browser.start_browser(headless=False):
            return {"error": "Failed to start browser"}
        
        if not browser.navigate(state["starting_url"]):
            return {"error": "Failed to navigate to starting URL"}
        
        page_info = browser.get_page_info()
        print(f"[Browser] Ready: {page_info['title']}")
        
        return {
            "current_url": page_info["url"],
            "error": None
        }
        
    except Exception as e:
        return {"error": f"Browser error: {str(e)}"}


def execute_node(state: WebCaneState) -> dict:
    """
    Execute the current action step.
    """
    executor = state.get("_executor")
    browser = state.get("_browser")
    
    step_idx = state["current_step_index"]
    plan = state.get("current_plan", [])
    
    if step_idx >= len(plan):
        return {"error": "Step index out of range"}
    
    action = plan[step_idx]
    
    print("\n" + "=" * 60)
    print(f"PHASE: EXECUTION (Step {step_idx + 1}/{len(plan)})")
    print(f"Action: {action.get('description', 'Unknown')}")
    print("=" * 60)
    
    # Capture before state
    before_state = browser.get_current_state()
    before_screenshot = browser.take_screenshot()
    
    # Execute action
    result = executor.execute_action(action)
    
    # Capture after state
    time.sleep(Config.STEP_DELAY)
    after_state = browser.get_current_state()
    after_screenshot = browser.take_screenshot()
    
    return {
        "before_state": before_state,
        "after_state": after_state,
        "before_screenshot": before_screenshot,
        "after_screenshot": after_screenshot,
        "current_url": after_state.get("url", state["current_url"]),
        "execution_history": [{
            "step": step_idx + 1,
            "action": action,
            "result": result,
            "timestamp": time.time() - state.get("start_time", time.time())
        }]
    }


def verify_node(state: WebCaneState) -> dict:
    """
    Verify if the action succeeded and check goal satisfaction.
    """
    verifier = state.get("_verifier")
    
    step_idx = state["current_step_index"]
    action = state["current_plan"][step_idx]
    verify_method = action.get("verify", "NONE")
    
    print("\n" + "=" * 60)
    print(f"PHASE: VERIFICATION (Method: {verify_method})")
    print("=" * 60)
    
    # Check if execution succeeded
    history = state.get("execution_history", [])
    last_entry = history[-1] if history else {}
    exec_result = last_entry.get("result", {})
    
    if not exec_result.get("success", False):
        print(f"[Verify] Execution failed: {exec_result.get('error', 'Unknown')}")
        return {
            "execution_history": [{
                "step": step_idx + 1,
                "success": False,
                "reason": exec_result.get("error", "Execution failed"),
                "method": "EXECUTION_CHECK"
            }]
        }
    
    # Run verification
    verification = verifier.verify_action(
        action=action,
        verify_method=verify_method,
        before_state=state.get("before_state", {}),
        after_state=state.get("after_state", {}),
        before_screenshot=state.get("before_screenshot"),
        after_screenshot=state.get("after_screenshot"),
        goal=state["goal"]
    )
    
    success = verification.get("success", False)
    goal_satisfied = verification.get("goal_satisfied", False)
    
    print(f"[Verify] Result: {'SUCCESS' if success else 'FAILED'}")
    print(f"[Verify] Reason: {verification.get('reason', 'Unknown')}")
    
    if goal_satisfied:
        print("[Verify] GOAL SATISFIED!")
    
    return {
        "is_complete": goal_satisfied,
        "execution_history": [{
            "step": step_idx + 1,
            "success": success,
            "reason": verification.get("reason", "Unknown"),
            "method": verification.get("method_used", "Unknown"),
            "goal_satisfied": goal_satisfied
        }]
    }


def advance_node(state: WebCaneState) -> dict:
    """Move to the next step."""
    return {
        "current_step_index": state["current_step_index"] + 1,
        "retry_count": 0
    }


def handle_failure_node(state: WebCaneState) -> dict:
    """Handle step failure - retry or replan."""
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", Config.MAX_RETRIES)
    
    if retry_count < max_retries:
        print(f"\n[Failure] Retry {retry_count + 1}/{max_retries}")
        time.sleep(2)
        return {
            "retry_count": retry_count + 1,
            "needs_replan": False
        }
    else:
        print("\n[Failure] Max retries reached, will replan")
        return {
            "needs_replan": True
        }


def finalize_success_node(state: WebCaneState) -> dict:
    """Finalize successful execution."""
    elapsed = time.time() - state.get("start_time", time.time())
    
    print("\n" + "=" * 60)
    print("GOAL COMPLETED SUCCESSFULLY!")
    print(f"Total time: {elapsed:.2f}s")
    print("=" * 60)
    
    return {
        "is_complete": True,
        "error": None
    }


def finalize_failure_node(state: WebCaneState) -> dict:
    """Finalize failed execution."""
    print("\n" + "=" * 60)
    print(f"EXECUTION FAILED: {state.get('error', 'Unknown error')}")
    print("=" * 60)
    
    return {
        "is_complete": False
    }


# Conditional Edges

def should_continue_after_plan(state: WebCaneState) -> str:
    """Decide next step after planning."""
    if state.get("current_plan") and len(state["current_plan"]) > 0:
        return "start_browser"
    return "finalize_failure"


def should_continue_after_browser(state: WebCaneState) -> str:
    """Decide next step after browser start."""
    if state.get("error"):
        return "finalize_failure"
    return "observe"


def should_continue_after_verify(state: WebCaneState) -> str:
    """Decide next step after verification."""
    # Check goal satisfaction
    if state.get("is_complete"):
        return "finalize_success"
    
    # Check verification result
    history = state.get("execution_history", [])
    last = history[-1] if history else {}
    
    if last.get("success"):
        # Success - check if more steps
        step_idx = state["current_step_index"]
        plan = state.get("current_plan", [])
        if step_idx + 1 < len(plan):
            return "advance"
        return "finalize_success"
    else:
        return "handle_failure"


def should_retry_or_replan(state: WebCaneState) -> str:
    """Decide whether to retry or replan."""
    if state.get("needs_replan"):
        return "observe_for_replan"
    return "execute"


def should_continue_after_observe_replan(state: WebCaneState) -> str:
    """Continue to plan after observation for replanning."""
    return "plan"


# Graph Builder

def create_webcane_graph() -> StateGraph:
    """
    Create the LangGraph StateGraph for WebCane3.
    
    Flow:
    START -> observe -> plan -> start_browser -> observe -> execute -> verify
    
    verify:
      - goal_satisfied -> finalize_success -> END
      - success + more_steps -> advance -> execute
      - failure -> handle_failure
        - retry -> execute
        - replan -> observe -> plan -> execute
    """
    graph = StateGraph(WebCaneState)
    
    # Add nodes
    graph.add_node("observe", observe_node)
    graph.add_node("plan", plan_node)
    graph.add_node("start_browser", start_browser_node)
    graph.add_node("execute", execute_node)
    graph.add_node("verify", verify_node)
    graph.add_node("advance", advance_node)
    graph.add_node("handle_failure", handle_failure_node)
    graph.add_node("observe_for_replan", observe_node)
    graph.add_node("finalize_success", finalize_success_node)
    graph.add_node("finalize_failure", finalize_failure_node)
    
    # Add edges
    graph.add_edge(START, "observe")
    graph.add_edge("observe", "plan")
    graph.add_conditional_edges("plan", should_continue_after_plan)
    graph.add_conditional_edges("start_browser", should_continue_after_browser, 
                                 {"finalize_failure": "finalize_failure", "observe": "observe"})
    graph.add_edge("observe", "execute", condition=lambda s: s.get("current_plan"))
    
    # Fix: After observe from start_browser, go to execute
    graph.add_conditional_edges("start_browser", should_continue_after_browser)
    
    graph.add_edge("execute", "verify")
    graph.add_conditional_edges("verify", should_continue_after_verify)
    graph.add_edge("advance", "execute")
    graph.add_conditional_edges("handle_failure", should_retry_or_replan)
    graph.add_edge("observe_for_replan", "plan")
    graph.add_edge("finalize_success", END)
    graph.add_edge("finalize_failure", END)
    
    return graph.compile()
