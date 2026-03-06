from src.data_loader import load_all_documents
from src.embedding import EmbeddingGenerator
if __name__ == "__main__":
    docs = load_all_documents("data")
    print(f"Loaded {len(docs)} documents from the data directory.")
    print("Example documents:", docs[0] if docs else None)
    emb_pipe = EmbeddingGenerator()
    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.embed_chunks(chunks)
    print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else "No embeddings generated")

