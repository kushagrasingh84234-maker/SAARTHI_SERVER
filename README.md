# Saarthi Server

FastAPI server with two channels sharing one brain:
- Robot: WebSocket `/ws/{session_id}` (unchanged)
- Web: `POST /api/chat` (Server-Sent Events) and `GET /api/health`

## Run locally
    python -m venv .venv && .venv\Scripts\activate    # macOS/Linux: source .venv/bin/activate
    pip install -r requirements.txt
    copy .env.example .env                              # then fill GROQ_API_KEY
    uvicorn server:app --host 0.0.0.0 --port 5000
    python tools/test_stream.py "Explain gravity in Hindi"

## Deploy on Render
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn server:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`
- Environment: `GROQ_API_KEY`, `ALLOWED_ORIGINS` (your Vercel URL), `ALLOWED_ORIGIN_REGEX`, optional `SEARCH_PROVIDER` + `SEARCH_API_KEY`, and `PYTHON_VERSION` (3.12.x)
- Use an always-on plan for real users (free instances sleep and the first reply is slow).

## Web contract
`POST /api/chat` body: `{message, channel:"web", session_id, conversation_id, images?}`
Events (`text/event-stream`, UTF-8): `status {text}`, `search {query}`, `sources {items[]}`, `token {text}`, `done {emotion}`, `error {text}`.
HTTP errors (429, 503) return JSON `{"error": "..."}`.

## Troubleshooting
- Browser CORS error: set `ALLOWED_ORIGINS` to the exact website URL (https, no trailing slash).
- 429: rate limit (`WEB_RATE_LIMIT_PER_MIN`).
- No search steps: `SEARCH_PROVIDER` and `SEARCH_API_KEY` must both be set.
- Empty answers: check the logs for "planner returned empty output" or "empty content".
