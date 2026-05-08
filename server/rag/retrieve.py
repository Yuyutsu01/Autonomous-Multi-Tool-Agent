import faiss
import numpy as np
from rag.embed import load_documents, get_embedding, _embedder

_index = None
_documents = []

def init_retriever(docs_dir: str = "rag/documents"):
    """Initialize FAISS index and populate it with local documents."""
    global _index, _documents
    
    # Load and chunk docs
    _documents = load_documents(docs_dir)
    if not _documents:
        print("No documents found in rag/documents.")
        return
        
    # Generate embeddings for all documents
    print("Generating embeddings for RAG...")
    embeddings = _embedder.encode(_documents)
    embedding_dim = embeddings.shape[1]
    
    # Init FAISS
    _index = faiss.IndexFlatL2(embedding_dim)
    _index.add(np.array(embeddings).astype("float32"))
    print(f"Added {len(_documents)} document chunks to FAISS.")

def retrieve(query: str, top_k: int = 3) -> list[str]:
    """Retrieve top_k relevant text chunks for a query."""
    if not _index or not _documents:
        return []
        
    query_emb = get_embedding(query)
    # faiss expects a 2D array
    query_emb_2d = np.array([query_emb]).astype("float32")
    
    distances, indices = _index.search(query_emb_2d, top_k)
    
    results = []
    for idx in indices[0]:
        if idx < len(_documents) and idx != -1:
            results.append(_documents[idx])
            
    return results
