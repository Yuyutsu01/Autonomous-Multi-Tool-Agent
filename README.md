# Autonomous Multi-Tool Agent

Welcome to the **Autonomous Multi-Tool Agent**, a robust, state-of-the-art AI system designed to intelligently parse user requests, generate execution plans, utilize tools, and validate outputs—all while providing full observability through a sleek, real-time UI.

## 🚀 What This Project Is

This project is an advanced AI agent that moves beyond simple chat completion. Instead of immediately returning an answer, the agent operates on a **Planner-Executor-Validator** loop:

1. **Memory Retrieval:** Checks a vector store (`FAISS`) and local JSON registry for previously solved tasks to avoid redundant processing.
2. **Planning:** Deconstructs complex user requests into a step-by-step sequential plan using an LLM.
3. **Execution:** Iterates over each step, using relevant tools (such as RAG over local documents, bash commands, etc.) to gather data.
4. **Validation:** An internal validation layer checks the output of each step to ensure accuracy. If a step fails, the agent retries with the failure context before proceeding.
5. **Real-time UI:** The entire thought trace, including planning, tool usage, and self-correction, is streamed via WebSockets to a minimal, high-performance UI.

## 🏗️ Architecture

- **Backend:** `FastAPI` (Python) serving REST endpoints and WebSockets for real-time streaming.
- **Frontend:** Vanilla JavaScript and HTML/CSS powered by `Vite`, providing a lightning-fast, zero-bloat interface.
- **Agent Core:** Built with `LangChain` / `Transformers` integrating RAG (Retrieval-Augmented Generation) and dynamic tool usage.
- **Storage:** Local vector embeddings and JSON stores for long-term memory.

---

## 📊 Metrics & Performance

The agent features a built-in telemetry system that tracks performance in real-time. Below are the current baseline metrics gathered from the latest benchmark run:

| Metric | Current Baseline | Target Benchmark | Description |
| :--- | :--- | :--- | :--- |
| **Planning Latency** | `1.25ms`* | `< 1200ms` | Time taken to deconstruct user request into a step-by-step plan. |
| **Execution Latency** | `5.82ms`* | Varies by tool | Time spent retrieving RAG context or executing system commands. |
| **Validation Rate** | `71.43%` | `> 85%` | Percentage of steps passing the Validator on the first attempt. |
| **Memory Cache Hit** | `20.0%` | `~30%` | Frequency of requests served from semantic memory (< 50ms total). |

*\*Note: Latency benchmarks were recorded in Mock Mode. Real-world latency varies based on LLM provider (Ollama/OpenAI).*

### Running the Benchmark
You can verify these metrics on your own machine by running the integrated benchmark suite:

```bash
# To run with real LLM (requires Ollama/OpenAI)
python backend/benchmark.py

# To run in Mock Mode (for infrastructure testing)
python backend/benchmark.py --mock
```

---

## 🐳 Running with Docker (Recommended)

The easiest way to run the entire stack is using Docker. We provide a `docker-compose.yml` that handles orchestrating the backend and frontend.

### Prerequisites
- Docker and Docker Compose installed.
- Ensure ports `8000` (Backend) and `3000` (Frontend) are available.

### Steps

1. Clone the repository and navigate to the root directory.
2. Build and start the containers:
   ```bash
   docker-compose up --build
   ```
3. Access the UI:
   - Open your browser and navigate to: `http://localhost:3000`
4. Stop the application:
   ```bash
   docker-compose down
   ```

*(Note: Data such as RAG document embeddings and agent memory are persisted through Docker volumes mounted directly from your local filesystem).*

## 🛠️ Local Development

If you prefer to run the services without Docker:

**1. Start the Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**2. Start the Frontend:**
```bash
cd frontend
npm install
npm run dev
```

Then visit `http://localhost:5173` (or the port Vite provides) in your browser.

---

## 📂 Project Structure

```text
.
├── backend/
│   ├── agent/          # Core logic (planner, executor, validator, memory)
│   ├── rag/            # Vector store logic and document loading
│   ├── tools/          # Extensible tool integration
│   ├── app.py          # FastAPI server and WebSocket endpoint
│   ├── main.py         # Original CLI entry point
│   └── requirements.txt
├── frontend/
│   ├── src/            # Vanilla JS/CSS assets
│   ├── index.html      # UI structure
│   └── package.json
├── docker-compose.yml
└── README.md
```
