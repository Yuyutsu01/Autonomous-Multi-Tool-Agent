import os
import glob
from sentence_transformers import SentenceTransformer

# Load embedding model
_embedder = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Basic chunking of text by characters."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def load_documents(docs_dir: str = "rag/documents") -> list[str]:
    """Load all .txt files from the specified folder."""
    all_chunks = []
    for filepath in glob.glob(os.path.join(docs_dir, "*.txt")):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                all_chunks.extend(chunk_text(content))
        except Exception as e:
            print(f"Failed to read {filepath}: {e}")
    return all_chunks

def get_embedding(text: str):
    """Return embedding vector for a given text."""
    return _embedder.encode([text])[0]
