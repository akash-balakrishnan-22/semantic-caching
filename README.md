# Semantic Caching with Redis

A Streamlit app demonstrating three LLM query strategies — direct LLM calls, Redis-backed semantic caching, and in-memory fuzzy caching — with live cost and latency comparisons.

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) — Python package manager
- [Redis Stack](https://redis.io/docs/latest/operate/oss_and_stack/install/install-stack/) running locally (default port `6379`)
- A [Groq API key](https://console.groq.com/)

## Setup

**1. Clone the repo and install dependencies**

```bash
git clone <repo-url>
cd semantic-caching
uv sync
```

**2. Configure environment variables**

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

**3. Start Redis Stack with Docker**

```bash
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

This exposes Redis on port `6379` and the RedisInsight UI on port `8001`.

Alternatively, you can pull and run the `redis/redis-stack` image directly from the **Docker Desktop** GUI without using the terminal.

## Running the App

```bash
uv run streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Chat Modes

| Mode | Description |
|------|-------------|
| **Direct LLM** | Every query hits the Groq API — no caching |
| **Redis Semantic Cache** | Similar queries are served from Redis using vector similarity search |
| **Fuzzy Cache** | In-memory cache using fuzzy string matching |

## Project Structure

```
semantic-caching/
├── app.py           # Streamlit application
├── main.py          # Core caching logic
├── pyproject.toml   # Project dependencies
├── uv.lock          # Locked dependency versions
├── data/            # Sample data / exports
└── notebooks/       # Jupyter notebooks for experimentation
```
