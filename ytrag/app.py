from src.data_loader import load_all_documents
from src.embedding import EmbeddingGenerator
from src.vectorstore import   FaissVectorStore
from src.search import RAGSearch
if __name__ == "__main__":
    # docs = load_all_documents("data")
    # print(f"Loaded {len(docs)} documents from the data directory.")
    # print("Example documents:", docs[0] if docs else None)
    # emb_pipe = EmbeddingGenerator()
    # chunks = emb_pipe.chunk_documents(docs)
    # embeddings = emb_pipe.embed_chunks(chunks)
    # print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else "No embeddings generated")
    # store = FaissVectorStore("faiss_store")
    # store.build_from_documents(docs)
    # print("[INFO] Vector store built successfully.")
    # store.load()
    # print("[INFO] Vector store loaded successfully.")
    # print(store.query("What are the health benefits of being physically active?", top_k=3))
    rag_search = RAGSearch()
    query = "What are the health benefits of being physically active?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("\nSummary:", summary)
    
    

