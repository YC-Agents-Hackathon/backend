"""
Test client for Detective LLM Backend
"""

import asyncio
import json
import httpx

NOTEBOOK_ID = "case-42"


async def test_upload():
    """Test evidence upload"""
    print("=== Testing Upload ===")
    
    url = "http://localhost:8000/api/upload"
    data = {
        "notebookId": NOTEBOOK_ID,
        "evidence": [
            "Camera footage shows a red sedan at 21:14 near Oak St.",
            "Witness says suspect wore a blue jacket."
        ]
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=data)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    print()


async def test_streaming_endpoint():
    """Test the streaming chat endpoint"""
    print("=== Testing Streaming Chat ===")
    
    url = "http://localhost:8000/api/chat/stream"
    data = {
        "notebookId": NOTEBOOK_ID,
        "userMessage": "What are the top 3 leads based on the evidence?",
    }
    
    print(f"Request: {json.dumps(data, indent=2)}")
    print("\n--- Streaming Response ---")
    
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST", 
            url, 
            json=data,
            headers={"Accept": "text/event-stream"}
        ) as response:
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                print(await response.aread())
                return
            
            async for chunk in response.aiter_text():
                if chunk.strip():
                    lines = chunk.strip().split('\n')
                    for line in lines:
                        if line.startswith('data: '):
                            data_str = line[6:]
                            try:
                                if data_str.strip():  # Only parse non-empty data
                                    data_obj = json.loads(data_str)
                                    if data_obj['type'] == 'content':
                                        print(data_obj['content'], end='', flush=True)
                                    elif data_obj['type'] == 'final':
                                        print(f"\n\nFinal output: {data_obj['content']}")
                                    elif data_obj['type'] == 'done':
                                        print("\n\n--- Stream Complete ---")
                                    elif data_obj['type'] == 'error':
                                        print(f"\nError: {data_obj['error']}")
                            except json.JSONDecodeError:
                                if data_str.strip():  # Only log non-empty parsing errors
                                    print(f"Could not parse: {data_str}")


async def test_simple_endpoint():
    """Test the non-streaming chat endpoint"""
    print("\n=== Testing Simple Chat ===")
    
    url = "http://localhost:8000/api/chat"
    data = {
        "notebookId": NOTEBOOK_ID,
        "userMessage": "Cross-check the car and jacket details.",
    }
    
    print(f"Request: {json.dumps(data, indent=2)}")
    print("\n--- Response ---")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=data)
        
        if response.status_code == 200:
            result = response.json()
            print(result['response'])
        else:
            print(f"Error: {response.status_code}")
            print(response.text)


async def test_report():
    """Test report generation"""
    print("\n=== Testing Report Generation ===")
    
    url = "http://localhost:8000/api/report"
    data = {
        "notebookId": NOTEBOOK_ID
    }
    
    print(f"Request: {json.dumps(data, indent=2)}")
    print("\n--- Report ---")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=data)
        
        if response.status_code == 200:
            result = response.json()
            print(result['report'])
        else:
            print(f"Error: {response.status_code}")
            print(f"Headers: {dict(response.headers)}")
            try:
                error_detail = response.json()
                print(f"Detail: {error_detail}")
            except:
                print(f"Raw response: {response.text}")
                print(f"Raw bytes: {response.content}")


async def main():
    """Run all tests"""
    print("🕵️ Detective LLM Backend Test Client")
    print("=" * 50)
    print("Make sure the FastAPI server is running on http://localhost:8000")
    print("Run: uv run uvicorn app:app --reload\n")
    
    try:
        # Test health endpoint first
        async with httpx.AsyncClient() as client:
            health_response = await client.get("http://localhost:8000/health")
            if health_response.status_code != 200:
                print("❌ Server not responding. Make sure it's running!")
                return
            print("✅ Server is healthy\n")
        
        # Test all endpoints in order
        await test_upload()
        await test_streaming_endpoint()
        await test_simple_endpoint()
        await test_report()
        
        print("\n✅ All tests completed!")
        
    except httpx.ConnectError:
        print("❌ Could not connect to server. Make sure it's running on port 8000!")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())