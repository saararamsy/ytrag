import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_openai import ChatOpenAI

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", model_name: str = "gpt-4o-mini"):
        """
        RAG Search using OpenAI LLM.
        """
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)

        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")

        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()

        # OpenAI LLM
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")

        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=openai_api_key,
            temperature=0.1,
            max_tokens=1024
        )

        print(f"[INFO] OpenAI LLM initialized: {model_name}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)

        if not context:
            return "No relevant documents found."

        prompt = f"""
        Summarize the following context for the query: '{query}'

        Context:
        {context}

        Summary:
        """

        response = self.llm.invoke(prompt)
        return response.content


# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)