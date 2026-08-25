"""Interactive Command-Line Interface for Aster & Row Support Agent."""
import sys
import uuid
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.agent.core import AsterRowAgent


def print_banner():
    print("\n" + "=" * 70)
    print("      ASTER & ROW SUPPORT AGENT (Reliable RAG + Tool System)       ")
    print("=" * 70)
    print("Type your questions below. Type 'exit', 'quit', or 'clear' to manage session.")
    print("Try questions like:")
    print("  - 'How long do I have to return an unused backpack?'")
    print("  - 'Where is ORD-1007 and when will it arrive?'")
    print("  - 'Can I put the entire Breeze Tumbler in the dishwasher?'")
    print("  - 'Do you ship to Canada?'")
    print("=" * 70 + "\n")


def run_cli():
    print_banner()
    agent = AsterRowAgent()
    session_id = f"cli-session-{uuid.uuid4().hex[:6]}"

    while True:
        try:
            user_input = input("\n[Customer] > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit"):
                print("\nThank you for chatting with Aster & Row. Goodbye!\n")
                break

            if user_input.lower() == "clear":
                session_id = f"cli-session-{uuid.uuid4().hex[:6]}"
                print("\n[System] Conversation history cleared. New session started.")
                continue

            response = agent.process_message(user_input, session_id=session_id)

            print("\n[Aster & Row Assistant]:")
            print(response.answer)

            if response.sources:
                print("\nSources Cited:")
                for src in response.sources:
                    print(f"  * {src}")

            if response.handoff_recommended:
                print("\n[Notice]: Human Specialist Assistance Recommended.")

            if response.tool_called:
                print(f"\n[Tool Executed]: {response.tool_called} (Args: {response.tool_arguments})")

            print("-" * 70)

        except (KeyboardInterrupt, EOFError):
            print("\n\nSession terminated. Goodbye!\n")
            break


if __name__ == "__main__":
    run_cli()
