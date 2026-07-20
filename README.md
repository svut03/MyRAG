# MyRAG

A lightweight Retrieval-Augmented Generation framework built from scratch.

## Features

- FAISS Retriever
- CrossEncoder Reranker
- Chunking
- Prompt Builder
- Local Qwen Generator
- Save/Load index
- Interactive CLI

## Architecture

Chunker
    ↓
Retriever
    ↓
CrossEncoder
    ↓
PromptBuilder
    ↓
Generator

## Future work

- ColBERT Reranker
- ChromaDB
- BM25
- Hybrid Search
- FastAPI