"""
WebCane3 - Run Script
Interactive script to run WebCane3 ReAct automation.
Supports both text and voice input modes for accessibility.
"""

from Webcane3.main import WebCane
from Webcane3.voice_interface import VoiceInterface


def select_model():
    """Let user choose supervisor model."""
    print("\n" + "-" * 60)
    print("SUPERVISOR MODEL SELECTION")
    print("-" * 60)
    print("  [1] NVIDIA DeepSeek-V3.1 (Default) - Thinking enabled")
    print("  [2] Groq OpenAI GPT-OSS-120b - Fast inference")
    print("-" * 60)
    
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice == "1" or choice == "":
            return "deepseek"
        elif choice == "2":
            return "gpt-oss"
        else:
            print("Please enter 1 or 2")


def select_mode():
    """Let user choose input mode."""
    print("\n" + "-" * 60)
    print("ACCESSIBILITY MODE SELECTION")
    print("-" * 60)
    print("  [1] Voice Mode - Speak your goals (for visually impaired)")
    print("  [2] Text Mode  - Type your goals")
    print("-" * 60)
    
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice == "1":
            return "voice"
        elif choice == "2":
            return "text"
        else:
            print("Please enter 1 or 2")


def main():
    print("=" * 60)
    print("WEBCANE3 - ReAct Interactive Mode")
    print("=" * 60)
    print("\nInitializing...")
    
    # Model selection
    sup_model_key = select_model()
    
    # Initialize WebCane with selected model
    webcane = WebCane(supervisor_model=sup_model_key)
    
    # Initialize Voice Interface
    voice = VoiceInterface()
    
    # Select mode
    mode = select_mode()
    voice_mode = (mode == "voice")
    
    if voice_mode:
        print("\n[Voice Mode] Speak your goals when prompted.")
        if voice.tts_available:
            voice.announce("Voice mode activated. Say your goal when you hear the beep.")
    else:
        print("\n[Text Mode] Type your goals.")
    
    print("\n" + "=" * 60)
    print("READY!")
    print("Commands:")
    if voice_mode:
        print("  - Wait for prompt, then speak your goal")
        print("  - Say 'quit' or 'exit' to close")
    else:
        print("  - Type a goal to execute (e.g., 'Go to youtube and search cats')")
        print("  - Type 'quit' or 'exit' to close")
    print("=" * 60)
    
    try:
        while True:
            print("\n" + "-" * 60)
            
            # Get goal based on mode
            if voice_mode:
                if voice.tts_available:
                    voice.speak("What would you like me to do?", blocking=True)
                    # Wait for audio to fully stop before listening (prevents STT catching TTS)
                    import time
                    time.sleep(1.5)
                print("Enter goal (or speak): ", end="", flush=True)
                
                # Try voice input first
                goal = voice.listen(duration=6)
                if not goal:
                    # Fall back to text input
                    goal = input("").strip()
            else:
                goal = input("Enter goal: ").strip()
            
            # Handle empty/silent input (STT returns '.' when user is silent)
            if not goal or goal in ['.', '...', '']:
                if voice_mode:
                    print("[Voice] No input detected (user silent)")
                    if voice.tts_available:
                        voice.announce("No input detected. Goodbye!")
                    break
                else:
                    print("Please enter a goal.")
                    continue
            
            # Check for stop commands (flexible matching)
            goal_lower = goal.lower()
            stop_words = ['quit', 'exit', 'stop', 'close', 'end', 'bye', 'goodbye']
            stop_phrases = ['stop session', 'end session', 'stop the session', 'close session', 
                           'stop running', 'stop the program', 'terminate', 'shut down','stop']
            
            should_stop = (
                goal_lower in stop_words or
                any(phrase in goal_lower for phrase in stop_phrases)
            )
            
            if should_stop:
                if voice_mode and voice.tts_available:
                    voice.announce("Goodbye!")
                break
            
            # Announce goal
            if voice_mode and voice.tts_available:
                voice.speak(f"Working on: {goal[:50]}")
            
            # Execute the goal with voice feedback
            result = webcane.execute_goal(goal, voice=voice if voice_mode else None)
            
            # Print and speak result
            print("\n" + "=" * 60)
            print("RESULT")
            print("=" * 60)
            print(f"  Success: {result.get('success')}")
            print(f"  Actions taken: {result.get('actions_taken', 0)}")
            print(f"  Successful actions: {result.get('successful_actions', 0)}")
            print(f"  Time: {result.get('elapsed_time', 0):.2f}s")
            print(f"  Final URL: {result.get('final_url', 'N/A')}")
            if result.get('error'):
                print(f"  Error: {result.get('error')}")
            print("=" * 60)
            
            # Voice feedback for result
            if voice_mode and voice.tts_available:
                if result.get('success'):
                    # Say goal completed, then ask about next action (blocking)
                    voice.speak("Goal completed successfully.", blocking=True)
                    voice.speak("Do you have another action to do?", blocking=True)
                else:
                    voice.speak(f"Goal failed. {result.get('error', '')[:50]}", blocking=True)
                    voice.speak("Do you want to try something else?", blocking=True)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        voice.cleanup()
        webcane.close()


if __name__ == "__main__":
    main()
