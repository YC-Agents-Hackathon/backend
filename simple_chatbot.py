#!/usr/bin/env python3
"""
Simple synchronous chatbot to test prompts using evidence files as context.
This is a simplified version that uses requests instead of aiohttp.
"""

import json
from pathlib import Path

import requests

# Configuration
BACKEND_URL = "http://localhost:8000"
EVIDENCE_FILE = "evidence_storage.json"
NOTEBOOK_ID = "simple-chatbot-test"


def check_backend():
    """Check if backend is running"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running and healthy")
            return True
        else:
            print(f"❌ Backend returned status: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to backend: {e}")
        print(f"Make sure the backend is running on {BACKEND_URL}")
        print("To start backend: cd backend && python app.py")
        return False


def load_evidence():
    """Load evidence from JSON file"""
    evidence_file = Path(__file__).parent / EVIDENCE_FILE
    try:
        with open(evidence_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            evidence_list = data.get("evidence", [])
            print(f"📁 Loaded {len(evidence_list)} evidence files")
            return evidence_list
    except Exception as e:
        print(f"❌ Error loading evidence: {e}")
        return None


def upload_evidence(evidence_list):
    """Upload evidence to backend"""
    payload = {"notebookId": NOTEBOOK_ID, "evidence": evidence_list}

    try:
        response = requests.post(f"{BACKEND_URL}/api/upload", json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Evidence uploaded! Total count: {data['totalEvidenceCount']}")
            return True
        else:
            print(f"❌ Upload failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return False


def chat_with_ai(message):
    """Send message to AI and get response with enhanced instructions"""

    payload = {"notebookId": NOTEBOOK_ID, "userMessage": message}

    try:
        response = requests.post(f"{BACKEND_URL}/api/chat", json=payload, timeout=60)
        if response.status_code == 200:
            data = response.json()
            return data["response"]
        else:
            return f"❌ Chat error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"❌ Chat error: {e}"


def main():
    """Main chatbot function"""
    print("🕵️  DETECTIVE AI - QUICK INSIGHTS MODE")
    print("=" * 50)
    print("Get direct, precise answers without confidence levels or fluff.")
    print("Perfect for rapid case analysis and prompt testing.")
    print("=" * 50)

    # Check backend
    if not check_backend():
        return

    # Load evidence
    evidence = load_evidence()
    if not evidence:
        return

    # Upload evidence
    print("\n📤 Uploading evidence...")
    if not upload_evidence(evidence):
        return

    print("\n🚀 Quick Insights Mode activated! Ask direct questions.")
    print("Type 'quit' to exit, 'help' for examples.\n")

    # Chat loop
    while True:
        try:
            user_input = input("🔍 Your question: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == "help":
                print("\n📋 Quick insight questions:")
                print("• Who killed Sam Altman?")
                print("• What happened at 8:58 PM on August 21st?")
                print("• Which alibis are fake?")
                print("• What was PROMETHEUS?")
                print("• Why was the security system disabled?")
                print("• Who had access to delete the AI model?")
                print("• What connects Musk, Nadella, and Zuckerberg?")
                continue
            elif not user_input:
                continue

            print("\n🤖 Thinking...")
            response = chat_with_ai(user_input)

            print("\n🕵️  AI Response:")
            print("-" * 40)
            print(response)
            print("-" * 40 + "\n")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main()
