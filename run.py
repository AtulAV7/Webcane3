"""
WebCane3 - Run Script
Interactive script to run WebCane3 with planner model selection.
"""

from Webcane3.main import WebCane


def select_planner():
    """Ask user to select planner model."""
    print("\n" + "=" * 60)
    print("SELECT PLANNER MODEL")
    print("=" * 60)
    print("  1. Gemini Flash (gemini-2.5-flash)")
    print("  2. Groq GPT-OSS (openai/gpt-oss-120b)")
    print("=" * 60)
    
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice == "1":
            return False  # use_groq = False
        elif choice == "2":
            return True   # use_groq = True
        else:
            print("Invalid choice. Please enter 1 or 2.")


def main():
    print("=" * 60)
    print("WEBCANE3 - Interactive Mode")
    print("=" * 60)
    
    # Select planner model
    use_groq = select_planner()
    
    print("\n" + "=" * 60)
    print("INITIALIZING...")
    print("=" * 60)
    
    webcane = WebCane(use_groq_planner=use_groq)
    
    print("\n" + "=" * 60)
    print("READY!")
    print("Commands:")
    print("  - Type a goal to execute (e.g., 'Go to youtube and search cats')")
    print("  - Type 'quit' or 'exit' to close")
    print("=" * 60)
    
    try:
        while True:
            print("\n" + "-" * 60)
            goal = input("Enter goal: ").strip()
            
            if goal.lower() in ['quit', 'exit', 'q']:
                break
            
            if not goal:
                print("Please enter a goal.")
                continue
            
            # Execute the goal
            result = webcane.execute_goal(goal)
            
            print("\n" + "=" * 60)
            print("RESULT")
            print("=" * 60)
            print(f"  Success: {result.get('success')}")
            print(f"  Steps: {result.get('steps_completed')}/{result.get('total_steps')}")
            print(f"  Time: {result.get('elapsed_time', 0):.2f}s")
            print(f"  Final URL: {result.get('final_url', 'N/A')}")
            if result.get('error'):
                print(f"  Error: {result.get('error')}")
            print("=" * 60)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        webcane.close()


if __name__ == "__main__":
    main()
