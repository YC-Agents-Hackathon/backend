# Detective AI Chatbot - Prompt Testing Interface

This directory contains simple chatbot scripts to test your prompts using the evidence files as context.

## Files

- `simple_chatbot.py` - Simple synchronous chatbot (recommended for basic testing)
- `chatbot_test.py` - Advanced async chatbot with more features
- `evidence_storage.json` - Contains all the evidence files used as context
- `chatbot_requirements.txt` - Python dependencies for the chatbots

## Quick Start

### 1. Start the Backend Server

```bash
cd backend
python app.py
```

The backend should start on `http://localhost:8000`

### 2. Install Dependencies (if needed)

```bash
pip install requests aiohttp
# or
pip install -r chatbot_requirements.txt
```

### 3. Run the Simple Chatbot

```bash
cd backend
python simple_chatbot.py
```

## Usage

1. The chatbot will automatically load all evidence files from `evidence_storage.json`
2. Evidence gets uploaded to the backend API
3. You can then ask questions about the case
4. Type `help` to see example questions
5. Type `quit` to exit

## Example Questions

- "Who are the main suspects in this case?"
- "Create a timeline of events on August 21st"
- "What evidence points to murder vs suicide?"
- "Analyze the alibis of all suspects"
- "What are the main contradictions in the evidence?"
- "Who had motive to kill Sam Altman?"
- "What digital evidence do we have?"
- "Generate a social network of all people involved"

## Evidence Context

The chatbot uses these evidence files as context:
- Crime scene report
- Coroner's preliminary report  
- Digital forensics report
- Witness statements
- Server logs
- Shredded NDA reconstruction
- Flight records
- Social media forensics
- Audio analysis
- Final memo fragment
- Building security interviews

## Troubleshooting

**"Cannot connect to backend"**
- Make sure the backend is running: `python app.py`
- Check that it's accessible at `http://localhost:8000/health`

**"Evidence upload failed"**
- Ensure `evidence_storage.json` exists in the backend directory
- Check backend logs for error messages

**"Chat timeout"**
- The AI might be processing a complex query
- Try shorter, more specific questions
- Check your OpenAI API key is configured in the backend
