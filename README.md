# Autonomous Multi-Tool AI Agent

This project implements a task-driven autonomous AI agent with tool use, RAG (Retrieval-Augmented Generation) capabilities, memory of past interactions, and output validation.

## Features
- **Planner:** Breaks natural language requests down into executable steps.
- **Tools:** Supports mock search, basic file operations (read/write), and email sending.
- **RAG:** Context retrieval using `sentence-transformers` and `faiss`.
- **Memory:** Agent's task memory for semantic caching and rapid response reuse.
- **Validator:** Ensures the steps output are non-empty and logically complete.

## Setup

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
   *Make sure the Ollama application is running in the background before you start the agent.*

3. Run the main agent routine (Example):
   ```bash
   python main.py
   ```

## Structure
- `agent/`: Orchestration and AI interaction (planner, executor, validator, memory)
- `tools/`: Implementations of executable tools by the agent
- `rag/`: Retrievers and embeddings for contextual understanding
- `main.py`: Entry point for the application.
