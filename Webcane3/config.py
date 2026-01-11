"""
Configuration module for WebCane3.
Handles API keys, model names, and system settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv


class Config:
    """Centralized configuration for WebCane3."""
    
    # Load environment variables from Webcane3/.env
    _env_path = Path(__file__).parent / ".env"
    load_dotenv(_env_path)
    
    # API Keys - Separate keys for different components to avoid quota issues
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")          # DOM Text Agent
    GROQ_API_KEY2: str = os.getenv("GROQ_API_KEY2", "")        # (Legacy - was Verification)
    GROQ_API_KEY3: str = os.getenv("GROQ_API_KEY3", "")        # Observer Agent
    NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "")      # Vision Agent
    NVIDIA_API_KEY2: str = os.getenv("NVIDIA_API_KEY2", "")    # Supervisor (DeepSeek)
    
    # Model Names - Gemini (used for planning fallback)
    GEMINI_PLANNING_MODEL: str = "gemini-2.5-flash"
    
    # Model Names - NVIDIA API
    NVIDIA_VISION_MODEL: str = "mistralai/mistral-large-3-675b-instruct-2512"
    NVIDIA_SUPERVISOR_MODEL: str = "deepseek-ai/deepseek-v3.1-terminus"    # ReAct Supervisor with thinking
    NVIDIA_API_URL: str = "https://integrate.api.nvidia.com/v1"  # Base URL for LangChain
    NVIDIA_VISION_URL: str = "https://integrate.api.nvidia.com/v1/chat/completions"  # Full URL for requests
    
    # Model Names - Groq
    GROQ_DOM_MODEL: str = "openai/gpt-oss-120b"                # DOM Text Agent
    GROQ_OBSERVER_MODEL: str = "meta-llama/llama-4-maverick-17b-128e-instruct"  # Observer
    
    # Voice Models (Groq) - Accessibility
    GROQ_STT_MODEL: str = "whisper-large-v3-turbo"             # Speech-to-Text
    GROQ_TTS_MODEL: str = "canopylabs/orpheus-v1-english"      # Text-to-Speech  
    GROQ_TTS_VOICE: str = "autumn"                             # TTS Voice
    
    # Model Names - Ollama (Local Fallback)
    OLLAMA_MODEL: str = "llama3.2:3b"
    
    # Local Vision Model Path (Qwen3-VL)
    QWEN_MODEL_PATH: str = r"C:\Users\devan\Desktop\Major Project\WebCane_1.0\My_Local_Models\Qwen3-VL-4B"
    
    # Timeouts (seconds)
    API_TIMEOUT: int = 30
    OBSERVATION_TIMEOUT: int = 15
    API_DELAY: float = 0.5  # Delay between API calls to avoid rate limiting
    
    # ReAct Loop Settings
    MAX_LOOP_ITERATIONS: int = 25         # Safety limit for supervisor loop
    EXECUTION_HISTORY_SIZE: int = 10      # Track last N actions for loop detection
    
    # Execution Settings
    MAX_SCROLL_ATTEMPTS: int = 3          # Max scrolls in scroll_find macro
    STEP_DELAY: float = 1.5
    
    # Browser Settings
    BROWSER_VIEWPORT_WIDTH: int = 1440
    BROWSER_VIEWPORT_HEIGHT: int = 900
    
    # Context Keywords - Indicate staying on current page
    CONTEXT_KEYWORDS: list = [
        "now", "current", "this page", "in results", "on this", "here",
        "from these", "among", "in the", "visible", "showing", "displayed"
    ]
    
    @classmethod
    def validate(cls) -> dict:
        """Validate configuration and return status."""
        status = {
            "gemini_available": bool(cls.GEMINI_API_KEY),
            "groq_dom_available": bool(cls.GROQ_API_KEY),
            "groq_observer_available": bool(cls.GROQ_API_KEY3),
            "nvidia_vision_available": bool(cls.NVIDIA_API_KEY),
            "nvidia_supervisor_available": bool(cls.NVIDIA_API_KEY2),
            "qwen_available": os.path.exists(cls.QWEN_MODEL_PATH),
        }
        return status
    
    @classmethod
    def print_status(cls):
        """Print configuration status."""
        status = cls.validate()
        print("=" * 60)
        print("WEBCANE3 CONFIGURATION (ReAct Architecture)")
        print("=" * 60)
        print(f"  Gemini API: {'Available' if status['gemini_available'] else 'Not configured'}")
        print(f"  Groq DOM: {'Available' if status['groq_dom_available'] else 'Not configured'}")
        print(f"  Groq Observer: {'Available' if status['groq_observer_available'] else 'Not configured'}")
        print(f"  NVIDIA Vision: {'Available' if status['nvidia_vision_available'] else 'Not configured'}")
        print(f"  NVIDIA Supervisor: {'Available' if status['nvidia_supervisor_available'] else 'Not configured'}")
        print(f"  Qwen3-VL: {'Available' if status['qwen_available'] else 'Not found'}")
        print("=" * 60)
    
    @classmethod
    def is_context_continuation(cls, goal: str) -> bool:
        """Check if goal indicates staying on current page."""
        goal_lower = goal.lower()
        return any(kw in goal_lower for kw in cls.CONTEXT_KEYWORDS)
