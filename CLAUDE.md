# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python backend project that uses the Dedalus Labs framework for AI agent interactions. The project is configured with uv for package management and uses Python 3.12+.

## Dependencies and Package Management

- **Package Manager**: Uses `uv` for dependency management (evidenced by `uv.lock` file)
- **Core Dependencies**: 
  - `dedalus-labs>=0.1.0a8` - Main framework for AI agent interactions
  - `dotenv>=0.9.9` - Environment variable management
  - `fastapi>=0.104.0` - Web framework for API endpoints
  - `uvicorn[standard]>=0.24.0` - ASGI server for running FastAPI
  - `sse-starlette>=1.6.0` - Server-Sent Events for streaming
- **Python Version**: Requires Python >=3.12

## Development Commands

```bash
# Install dependencies
uv sync

# Run the FastAPI server (main application)
uvicorn app:app --reload

# Alternative: Run directly with Python
python app.py

# Test the streaming endpoints
python test_client.py

# Run the original Dedalus example
python hello.py
```

## Code Architecture

### Main Application Structure

- **FastAPI Server**: `app.py` contains the main web API with streaming endpoints
- **Original Example**: `hello.py` contains basic Dedalus usage example
- **Test Client**: `test_client.py` demonstrates how to consume the streaming API
- **Async Architecture**: Built around async/await patterns using asyncio
- **Core Components**:
  - `AsyncDedalus` - Main client for Dedalus Labs API
  - `DedalusRunner` - Runner class that executes AI model requests
  - Environment configuration via dotenv

### Key Patterns

1. **Async Pattern**: All main functionality uses asyncio
2. **Environment Variables**: Load configuration via `load_dotenv()`
3. **AI Model Integration**: Uses OpenAI models (specifically gpt-4o-mini) through Dedalus framework
4. **Streaming Support**: Framework supports streaming responses via `stream_async` utility
5. **Server-Sent Events**: FastAPI streaming using SSE for real-time frontend communication

### API Endpoints

- **GET `/`** - Root endpoint with API status
- **GET `/health`** - Health check endpoint
- **POST `/api/chat/stream`** - Streaming chat endpoint with Server-Sent Events
- **POST `/api/chat`** - Non-streaming chat endpoint for simple responses

### Example Usage Patterns

The codebase demonstrates three main usage patterns:
- **FastAPI Streaming**: `runner.run(stream=True)` with SSE response formatting
- **Simple API Response**: `await runner.run()` with JSON response
- **Direct Usage**: Original `hello.py` example with console output

## File Structure

```
backend/
├── app.py            # FastAPI server with streaming endpoints
├── hello.py          # Original Dedalus example
├── test_client.py    # Test client for API endpoints
├── pyproject.toml    # Project configuration and dependencies
├── uv.lock          # Locked dependencies
├── dedalus_docs.md  # Documentation/examples for Dedalus usage
└── README.md        # Basic project description
```