#!/usr/bin/env python3
"""
Simple chatbot to test prompts using evidence files as context.
This script loads evidence from evidence_storage.json and provides an interactive chat interface.
"""

import asyncio
import json
import sys
from pathlib import Path

import aiohttp

# Configuration
BACKEND_URL = "http://localhost:8000"
EVIDENCE_FILE = "evidence_storage.json"
NOTEBOOK_ID = "chatbot-test-session"


class ChatBot:
    def __init__(self, backend_url: str):
        self.backend_url = backend_url
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def check_health(self):
        """Check if the backend is running"""
        try:
            async with self.session.get(f"{self.backend_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Backend is healthy: {data}")
                    return True
                else:
                    print(f"❌ Backend health check failed: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Cannot connect to backend: {e}")
            print(f"Make sure the backend is running on {self.backend_url}")
            return False

    async def upload_evidence(self, evidence_list: list):
        """Upload evidence to the backend"""
        payload = {"notebookId": NOTEBOOK_ID, "evidence": evidence_list}

        try:
            async with self.session.post(
                f"{self.backend_url}/api/upload", json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ Evidence uploaded successfully!")
                    print(f"📊 Total evidence count: {data['totalEvidenceCount']}")
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ Failed to upload evidence: {response.status}")
                    print(f"Error: {error_text}")
                    return False
        except Exception as e:
            print(f"❌ Error uploading evidence: {e}")
            return False

    async def chat(self, user_message: str):
        """Send a chat message and get response"""
        payload = {"notebookId": NOTEBOOK_ID, "userMessage": user_message}

        try:
            async with self.session.post(
                f"{self.backend_url}/api/chat", json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["response"]
                else:
                    error_text = await response.text()
                    return f"❌ Chat error ({response.status}): {error_text}"
        except Exception as e:
            return f"❌ Chat error: {e}"


def load_evidence_from_file(file_path: str):
    """Load evidence from JSON file"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("evidence", [])
    except FileNotFoundError:
        print(f"❌ Evidence file not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in evidence file: {e}")
        return None
    except Exception as e:
        print(f"❌ Error loading evidence file: {e}")
        return None


def print_welcome():
    """Print welcome message"""
    print("=" * 80)
    print("🕵️  DETECTIVE AI CHATBOT - PROMPT TESTING INTERFACE")
    print("=" * 80)
    print("This chatbot uses the evidence files from evidence_storage.json as context.")
    print(
        "You can ask questions about the case, analyze evidence, or test different prompts."
    )
    print()
    print("Commands:")
    print("  - Type your question/prompt and press Enter")
    print("  - Type 'quit' or 'exit' to end the session")
    print("  - Type 'help' to see this message again")
    print("=" * 80)


def print_help():
    """Print help message"""
    print("\n📋 HELP - Example prompts you can try:")
    print("─" * 50)
    print("• 'What are the key suspects in this case?'")
    print("• 'Create a timeline of events on August 21st'")
    print("• 'What evidence points to murder vs suicide?'")
    print("• 'Analyze the alibis of all suspects'")
    print("• 'What are the main contradictions in the evidence?'")
    print("• 'Who had motive to kill Sam Altman?'")
    print("• 'What digital evidence do we have?'")
    print("• 'Generate a social network of all people involved'")
    print("─" * 50)


async def main():
    """Main chatbot loop"""
    print_welcome()

    # Load evidence from file
    evidence_file_path = Path(__file__).parent / EVIDENCE_FILE
    evidence_list = load_evidence_from_file(evidence_file_path)

    if evidence_list is None:
        print("Cannot continue without evidence data.")
        return

    print(f"📁 Loaded {len(evidence_list)} evidence files from {EVIDENCE_FILE}")

    # Initialize chatbot
    async with ChatBot(BACKEND_URL) as chatbot:
        # Check backend health
        if not await chatbot.check_health():
            print("\n💡 To start the backend, run:")
            print("   cd backend && python app.py")
            return

        # Upload evidence
        print("\n📤 Uploading evidence to backend...")
        if not await chatbot.upload_evidence(evidence_list):
            print("Cannot continue without uploading evidence.")
            return

        print("\n🚀 Chatbot ready! Start asking questions about the case.")
        print("─" * 50)

        # Chat loop
        while True:
            try:
                # Get user input
                user_input = input("\n🔍 Your question: ").strip()

                # Handle commands
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("\n👋 Goodbye! Happy investigating!")
                    break
                elif user_input.lower() in ["help", "h"]:
                    print_help()
                    continue
                elif not user_input:
                    print("Please enter a question or 'help' for examples.")
                    continue

                # Send to chatbot
                print("\n🤖 Detective AI is analyzing...")
                response = await chatbot.chat(user_input)

                # Display response
                print("\n" + "─" * 60)
                print("🕵️  DETECTIVE AI RESPONSE:")
                print("─" * 60)
                print(response)
                print("─" * 60)

            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Session ended. Goodbye!")
                break


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
