# Autonomous AI Agent with RAG & Semantic Memory

An advanced, task-driven autonomous AI agent built from scratch using Python. It features a custom orchestration engine capable of generating execution plans, utilizing external tools, validating outputs, and leveraging Retrieval-Augmented Generation (RAG) and persistent semantic memory.

## 🚀 Key Features

- **Custom Agent Orchestration:** Built a modular planner-executor-validator architecture from the ground up (without frameworks like LangChain) ensuring a lightweight, highly customizable, and transparent execution flow.
- **Retrieval-Augmented Generation (RAG):** Contextualizes LLM prompts by retrieving relevant documents using FAISS vector search and `sentence-transformers`.
- **Semantic Caching & Memory:** Persists past task outputs using mathematical cosine similarity to dramatically reduce redundant LLM API calls and improve response latency for repeated queries.
- **Tool Execution & Self-Correction:** Dynamically delegates tasks to integrated tool sets (Search, File I/O) with automated retry logic based on programmable validation heuristics.
- **Local LLM Integration:** Integrates seamlessly with local Ollama models (LLaMA-3) via OpenAI-compliant APIs for offline, zero-cost inference.

## 🛠️ Tools & Technologies

- **Core:** Python
- **AI & Models:** Local LLaMA-3 (via Ollama), OpenAI Python SDK
- **Vector Search & Embeddings:** FAISS (Facebook AI Similarity Search), `sentence-transformers`
- **Data & Math:** NumPy, JSON
- **Architecture Concepts:** ReAct-style Agent Loops, Semantic Caching, RAG

## ⚙️ Setup Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Setup Ollama (Free and Local LLM):
   - Download and install [Ollama](https://ollama.com/)
   - Pull the model we are using by running this in your terminal:
   ```bash
   ollama pull llama3
   ```
   *Note: Make sure the Ollama application is running in the background before you start the agent.*

3. Run the main application:
   ```bash
   python main.py
   ```

## 📂 Project Structure
- `agent/`: Core orchestration logic (`planner.py`, `executor.py`, `validator.py`, `memory.py`)
- `tools/`: Extensible implementations of tools the agent can invoke
- `rag/`: Document ingestion, embeddings, and FAISS vector retrieval modules
- `main.py`: Entry point for the REPL application
