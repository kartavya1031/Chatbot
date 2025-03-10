# CDP Chatbot

This project is a Support Agent Chatbot that answers "how-to" questions related to four Customer Data Platforms (CDPs): Segment, mParticle, Lytics, and Zeotap.

## Features
- Automatic data scraping from official CDP documentation
- Query processing using spaCy
- Semantic search with FAISS and Sentence Transformers
- Cross-CDP comparison placeholder
- API with endpoints for asking questions and health check

## Setup

1. Clone the repository:
```bash
git clone <repository_url>
cd chatbot_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the chatbot:
```bash
python chatbot.py
```

4. Test the API using Postman or cURL:
```bash
curl -X POST http://localhost:5000/ask -H "Content-Type: application/json" -d '{"query": "How do I set up a new source in Segment?"}'
```

## Endpoints
- `/ask` - Accepts JSON query and returns answer
- `/compare` - Placeholder for cross-CDP comparison
- `/health` - Health check API
