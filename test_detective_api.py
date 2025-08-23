"""
Test client for Detective LLM Backend
Tests all endpoints according to PRD specifications
"""
import asyncio
import json
import httpx


async def test_health():
    """Test health endpoint"""
    print("=== Testing Health Endpoint ===")
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        print()


async def test_upload():
    """Test evidence upload endpoint"""
    print("=== Testing Upload Endpoint ===")
    
    # Test case from PRD
    data = {
        "notebookId": "case-42",
        "evidence": [
            "Camera footage shows a red sedan at 21:14 near Oak St.",
            "Witness says suspect wore a blue jacket."
        ]
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post("http://localhost:8000/api/upload", json=data)
        print(f"Status: {response.status_code}")
        print(f"Request: {json.dumps(data, indent=2)}")
        print(f"Response: {response.json()}")
        print()
        
        # Upload more evidence to the same notebook
        more_evidence = {
            "notebookId": "case-42",
            "evidence": [
                "Credit card transaction at gas station nearby at 21:20.",
                "License plate partial: X5B-..."
            ]
        }
        
        response2 = await client.post("http://localhost:8000/api/upload", json=more_evidence)
        print(f"Second upload - Status: {response2.status_code}")
        print(f"Second upload - Response: {response2.json()}")
        print()


async def test_chat():
    """Test non-streaming chat endpoint"""
    print("=== Testing Non-Streaming Chat ===")
    
    # Test case from PRD
    data = {
        "notebookId": "case-42",
        "userMessage": "What are the top 3 leads?"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post("http://localhost:8000/api/chat", json=data)
        print(f"Status: {response.status_code}")
        print(f"Request: {json.dumps(data, indent=2)}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {result['response']}")
        else:
            print(f"Error: {response.text}")
        print()


async def test_chat_stream():
    """Test streaming chat endpoint"""
    print("=== Testing Streaming Chat ===")
    
    # Test case from PRD
    data = {
        "notebookId": "case-42",
        "userMessage": "Cross-check car and jacket details."
    }
    
    print(f"Request: {json.dumps(data, indent=2)}")
    print("Streaming Response:")
    
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST", 
            "http://localhost:8000/api/chat/stream",
            json=data,
            headers={"Accept": "text/event-stream"}
        ) as response:
            if response.status_code != 200:
                print(f"Error: {response.status_code} - {await response.aread()}")
                return
            
            async for chunk in response.aiter_text():
                if chunk.strip():
                    lines = chunk.strip().split('\n')
                    for line in lines:
                        if line.startswith('data: '):
                            data_str = line[6:]
                            try:
                                data_obj = json.loads(data_str)
                                if data_obj['type'] == 'content':
                                    print(data_obj['content'], end='', flush=True)
                                elif data_obj['type'] == 'final':
                                    print(f"\n[Final]: {data_obj['content']}")
                                elif data_obj['type'] == 'done':
                                    print("\n[Stream Complete]")
                                elif data_obj['type'] == 'error':
                                    print(f"\n[Error]: {data_obj['error']}")
                            except json.JSONDecodeError:
                                print(f"\n[Raw]: {data_str}")
    print("\n")


async def test_report():
    """Test report generation endpoint"""
    print("=== Testing Report Generation ===")
    
    # Test basic report
    data = {
        "notebookId": "case-42"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post("http://localhost:8000/api/report", json=data)
        print(f"Status: {response.status_code}")
        print(f"Request: {json.dumps(data, indent=2)}")
        if response.status_code == 200:
            result = response.json()
            print(f"Report:\n{result['report']}")
        else:
            print(f"Error: {response.text}")
        print()
        
        # Test report with specific task
        task_data = {
            "notebookId": "case-42",
            "task": "Draft final incident report"
        }
        
        response2 = await client.post("http://localhost:8000/api/report", json=task_data)
        print(f"Task-specific report - Status: {response2.status_code}")
        if response2.status_code == 200:
            result2 = response2.json()
            print(f"Task Report:\n{result2['report']}")
        else:
            print(f"Error: {response2.text}")
        print()


async def test_new_notebook():
    """Test auto-creation of new notebook"""
    print("=== Testing New Notebook Auto-Creation ===")
    
    # Test with a new notebook ID
    data = {
        "notebookId": "case-99",
        "userMessage": "Hello, I need help with a new case."
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post("http://localhost:8000/api/chat", json=data)
        print(f"Status: {response.status_code}")
        print(f"Request: {json.dumps(data, indent=2)}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {result['response']}")
        else:
            print(f"Error: {response.text}")
        print()


async def main():
    """Run all tests in sequence"""
    print("🕵️ Detective LLM Backend Test Suite")
    print("=" * 50)
    print("Make sure the FastAPI server is running on http://localhost:8000")
    print("Run: uv run uvicorn app:app --reload\n")
    
    try:
        # Check server health first
        await test_health()
        
        # Test all endpoints in logical order
        await test_upload()
        await test_chat()
        await test_chat_stream()
        await test_report()
        await test_new_notebook()
        
        print("✅ All tests completed!")
        
    except httpx.ConnectError:
        print("❌ Could not connect to server. Make sure it's running on port 8000!")
    except Exception as e:
        print(f"❌ Error during testing: {e}")


if __name__ == "__main__":
    asyncio.run(main())