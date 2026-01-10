"""
State definition for WebCane3 ReAct workflow.
Simplified state schema for step-by-step Supervisor loop.
"""

from typing import TypedDict, List, Dict, Optional, Annotated
import operator


class WebCaneState(TypedDict):
    """
    State schema for the WebCane3 ReAct workflow.
    
    This state is passed between Observer -> Supervisor -> Executor nodes
    in a circular loop until goal completion or failure.
    """
    
    # Core Goal
    goal: str                           # Original user goal
    starting_url: str                   # Initial URL to navigate to
    
    # Current Page Context
    current_url: str                    # Current browser URL
    screenshot: Optional[bytes]         # Current page screenshot
    observation: Optional[str]          # Action-oriented page analysis
    blockers: List[str]                 # Detected popups/modals/blockers
    elements: List[Dict]                # Extracted DOM elements
    
    # Execution State
    last_action: Optional[Dict]         # Most recent action from supervisor
    last_action_success: Optional[bool] # Whether the last action succeeded
    execution_history: Annotated[List[Dict], operator.add]  # Log of actions (last N)
    
    # Control Flow
    is_complete: bool                   # Goal achieved?
    error: Optional[str]                # Error message if failed
    loop_count: int                     # Current iteration count (safety limit)
    
    # Timing
    start_time: float                   # When execution started


class SupervisorAction(TypedDict):
    """Schema for a single action from the Supervisor."""
    action: str                 # Action type: navigate, search, click, type, scroll_find, done, fail
    target: str                 # Action target (element description, URL, etc.)
    query: Optional[str]        # Query text for search action
    reason: Optional[str]       # Reason for COMPLETE/FAILED actions


class ExecutionHistoryEntry(TypedDict):
    """Schema for an entry in the execution history."""
    action: str                 # Action type
    target: str                 # Target description
    success: bool               # Whether action succeeded
    timestamp: float            # Seconds since start


def create_initial_state(
    goal: str,
    starting_url: str = ""
) -> WebCaneState:
    """
    Create initial state for a new WebCane3 ReAct session.
    
    Args:
        goal: The user's goal to achieve
        starting_url: URL to start from (optional, can be extracted from goal)
        
    Returns:
        Initialized WebCaneState
    """
    import time
    
    return {
        "goal": goal,
        "starting_url": starting_url,
        "current_url": starting_url or "about:blank",
        "screenshot": None,
        "observation": None,
        "blockers": [],
        "elements": [],
        "last_action": None,
        "last_action_success": None,
        "execution_history": [],
        "is_complete": False,
        "error": None,
        "loop_count": 0,
        "start_time": time.time(),
    }
