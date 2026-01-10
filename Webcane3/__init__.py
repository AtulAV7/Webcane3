"""
WebCane3 - ReAct Web Automation System

Step-by-Step Supervisor Loop Architecture:
- Observer: Analyzes page state
- Supervisor: Decides next action (using DeepSeek-V3.2)
- Executor: Executes with macro-actions
"""

from .config import Config
from .main import WebCane
from .supervisor import Supervisor
from .observer import Observer
from .executor import Executor

__all__ = ["Config", "WebCane", "Supervisor", "Observer", "Executor"]
__version__ = "3.1.0"
