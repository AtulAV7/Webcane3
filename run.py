"""
WebCane3 - Run Script
Interactive script to run WebCane3 ReAct automation.
"""

from Webcane3.main import WebCane


def main():
    print("=" * 60)
    print("WEBCANE3 - ReAct Interactive Mode")
    print("=" * 60)
    print("\nInitializing...")
    
    webcane = WebCane()
    
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
            print(f"  Actions taken: {result.get('actions_taken', 0)}")
            print(f"  Successful actions: {result.get('successful_actions', 0)}")
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
