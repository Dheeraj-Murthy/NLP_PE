# LegalRAG Frontend

Barebones frontend for the LegalRAG legal chatbot.

## Quick Start

```bash
# Using Python's built-in server
cd frontend
python -m http.server 8080

# Or using Node.js
npx serve .
```

Then open http://localhost:8080 in your browser.

## Backend Connection

When the backend is ready, uncomment the API calls in `app.js`:

1. **Chat**: Find `// TODO: Connect to backend when ready` and replace the simulated response with:
   ```javascript
   const response = await fetch(`${API_BASE_URL}/chat`, {
       method: 'POST',
       headers: { 'Content-Type': 'application/json' },
       body: JSON.stringify({ message, history: chatHistory })
   });
   const data = await response.json();
   addMessage(data.answer, 'bot');
   ```

2. **File Upload**: Find the file upload TODO and connect to `/document` endpoint.

## Endpoints Expected

| Endpoint | Method | Payload |
|----------|--------|---------|
| `/chat` | POST | `{ "message": "?", "history": [] }` |
| `/document` | POST | FormData with file + query |

## Features

- Chat interface with message bubbles
- Drag & drop file upload (PDF, DOCX, TXT)
- Typing indicator
- Citation highlighting
- Responsive design
