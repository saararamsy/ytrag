# RAG Search and Retrieval Pipeline

This project implements a **Retrieval-Augmented Generation (RAG)** system with a modular design that supports both production-style pipelines and interactive notebook experiments.

It can:
- Load and chunk documents from a directory
- Embed documents into a vector store
- Retrieve relevant documents based on queries
- Use an LLM to summarize or answer queries
- Support enhanced RAG with metadata, context, and confidence scores
- Run experiments in Jupyter notebooks using ChromaDB

## Features
✔ Modular document loading and chunking  
✔ Embedding generation with SentenceTransformers  
✔ Vector search using FAISS (production)  
✔ Optional vector store using ChromaDB (notebook experiments)  
✔ Basic and enhanced RAG pipelines  
✔ OpenAI-powered LLM for query answering and summarization  
✔ Context-aware retrieval with metadata and citations  

## Tech Stack
- Python  
- FAISS (vector search for production pipelines)  
- ChromaDB (vector store for interactive experiments)  
- SentenceTransformers (embedding generation)  
- OpenAI / LangChain (LLM integration)  
- dotenv (environment variables management)  
- Jupyter Notebook (interactive experimentation)


