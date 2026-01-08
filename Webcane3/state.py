"""
State definition for WebCane3 LangGraph workflow.
Defines the TypedDict schema for state management across nodes.
"""

from typing import TypedDict, List, Dict, Optional, Annotated
import operator


class WebCaneState(TypedDict):
    """
    State schema for the WebCane3 LangGraph workflow.
    
    This state is passed between all nodes and maintains the complete
    context of the automation session.
    """
    
    # Core Goal
    goal: str                           # Original user goal
    starting_url: str                   # Initial URL to navigate to
    
    # Current Page Context
    current_url: str                    # Current browser URL
    page_description: Optional[str]     # Visual description from observer
    screenshot: Optional[bytes]         # Current page screenshot
    elements: List[Dict]                # Extracted DOM elements
    
    # Plan & Execution
    current_plan: List[Dict]            # Action steps with verify hints
    current_step_index: int             # Which step we're on
    execution_history: Annotated[List[Dict], operator.add]  # Log of actions
    
    # Verification
    before_screenshot: Optional[bytes]  # Screenshot before action
    after_screenshot: Optional[bytes]   # Screenshot after action
    before_state: Optional[Dict]        # Page state before action
    after_state: Optional[Dict]         # Page state after action
    
    # Control Flow
    is_complete: bool                   # Goal achieved?
    error: Optional[str]                # Error message if failed
    retry_count: int                    # Current retry attempts
    max_retries: int                    # Max retries per step
    needs_replan: bool                  # Should we replan?
    
    # Timing
    start_time: float                   # When execution started


class ActionStep(TypedDict):
    """Schema for a single action step in the plan."""
    step: int                   # Step number (1-indexed)
    action: str                 # Action type: navigate, find_and_click, type, press_key, scroll, wait
    target: str                 # Action target (URL, element description, text, key)
    description: str            # Human-readable description
    verify: str                 # Verification method: URL_CHANGE, NONE, DOM_VALUE, VISION_OUTCOME


class VerificationResult(TypedDict):
    """Schema for verification result."""
    success: bool               # Did verification pass?
    method_used: str            # Which method: URL_CHANGE, DOM_VALUE, VISION_OUTCOME
    confidence: float           # Confidence score 0.0 - 1.0
    reason: str                 # Explanation
    goal_satisfied: bool        # Is the overall goal complete?
    suggested_action: Optional[str]  # What to do if failed


class ExecutionResult(TypedDict):
    """Schema for execution result."""
    success: bool               # Did action execute?
    action: Dict                # The action that was executed
    method: str                 # dom or vision
    element_id: int             # Element ID that was acted on
    scroll_attempts: int        # How many scrolls were needed
    error: Optional[str]        # Error message if failed


def create_initial_state(
    goal: str,
    starting_url: str,
    max_retries: int = 2
) -> WebCaneState:
    """
    Create initial state for a new WebCane3 session.
    
    Args:
        goal: The user's goal to achieve
        starting_url: URL to start from
        max_retries: Maximum retry attempts per step
        
    Returns:
        Initialized WebCaneState
    """
    import time
    
    return {
        "goal": goal,
        "starting_url": starting_url,
        "current_url": starting_url,
        "page_description": None,
        "screenshot": None,
        "elements": [],
        "current_plan": [],
        "current_step_index": 0,
        "execution_history": [],
        "before_screenshot": None,
        "after_screenshot": None,
        "before_state": None,
        "after_state": None,
        "is_complete": False,
        "error": None,
        "retry_count": 0,
        "max_retries": max_retries,
        "needs_replan": False,
        "start_time": time.time(),
    }
