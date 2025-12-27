# SemanticalCodeSearch

**SemanticalCodeSearch** is a semantic search engine and structural analysis tool for codebases. It combines tree-sitter parsing, knowledge graph extraction, and modern LLM retrieval techniques.

## Key Features

- **Semantic Search**: Find code by meaning, not just keywords, using vector embeddings.
- **Structural Analysis**: Codebase understanding through Tree-sitter parsing for Python and C.
- **Knowledge Graph**: Builds a relationship graph (calls, imports, inheritance) and store using FalkorDB.
- **LLM Summarization**: Automatically generates concise summaries for functions, classes, and files.
- **Context-Aware Answers**: Ask questions about your codebase and get natural language answers backed by retrieved snippets (via Gemini).
- **Hybrid Retrieval**: Combines vector search (ChromaDB) with graph-based relationship traversal.
- **High-Precision Reranking**: Uses Jina AI Reranker (v2) to ensure the most relevant code is at the top.
- **Incremental Indexing**: Efficiently updates only modified files, reusing existing embeddings and summaries.
- **Web Interface & CLI**: Modern web dashboard for visual search and a command-line tool.

## Architecture

1.  **Parsing Pass (Pass 1)**: Uses Tree-sitter to decompose files into structural snippets (functions, classes, etc.).
2.  **Structural Analysis (Pass 2)**: Identifies how snippets relate to each other (e.g., `Function A` calls `Function B`) and builds a relationship graph.
3.  **Summarization & Embedding**: 
    - Generates concise technical summaries using **Gemini 2.5 Flash Lite**.
    - Computes high-dimensional vector embeddings using **Jina Code Embeddings v1.5**.
4.  **Storage**: 
    - **ChromaDB**: High-performance vector storage for semantic retrieval.
    - **FalkorDB**: Graph database for structural relationships.
    - **SQLite**: Metadata, keyword search index, and file tracking.
5.  **Search & Rerank**: 
    - **Query Orchestration**: Uses HyDE (Hypothetical Document Embeddings) to expand queries.
    - **Hybrid Retrieval**: Combines vector search with keyword search (SQLite FTS5) using **Reciprocal Rank Fusion (RRF)**.
    - **Re-ranking**: Final precision ranking using **Jina Reranker v2**.
6.  **Answering**: Provides final natural language answers using **Gemini 3 Flash Preview** with streaming support.

## Incremental Indexing

SemanticalCodeSearch implements an efficient incremental indexing pipeline to avoid redundant processing:

- **Hash-based Change Detection**: Uses MD5 hashes stored in SQLite to identify modified files. Unchanged files are skipped during the parsing phase.
- **Structural Reuse**: If a file hasn't changed, its existing snippets and metadata are loaded from the database rather than being re-extracted.
- **Hierarchical Re-summarization**: The system detects when a child component (like a function) has changed and automatically triggers re-summarization for its parent (like a class or file) to maintain context accuracy.
- **Delta Updates**: Only new or modified snippets are processed for LLM summarization and vector embedding, significantly reducing API costs and local compute time.
- **Automatic Cleanup**: Stale data from deleted files is purged from SQLite, ChromaDB, and FalkorDB during each indexing run.

## Prerequisites

- **Python 3.13+**
- **[uv](https://github.com/astral-sh/uv)** (Recommended for dependency management)
- **Google Gemini API Key** (Required for summarization and answering)

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-repo/SemanticalCodeSearch.git
    cd SemanticalCodeSearch
    ```

2.  **Install dependencies**:
    Using `uv`:
    ```bash
    uv sync
    ```

3.  **Configure environment**:
    Create a `.env` file in the root directory:
    ```env
    GEMINI_API_KEY=your_gemini_api_key
    ```

## Usage

### 1. Indexing a Project
Before searching, you must index your codebase. Run this in the root of the project you want to index:
```bash
python main.py /path/to/your/project
```
*Note: This will perform parsing, summarization, and embedding. It uses 4-bit quantization for embeddings by default to save VRAM.*

### 2. Command Line Search
Ask questions directly from your terminal:
```bash
python main.py --query "How does the authentication flow work?"
```

### 3. Web Interface
Launch the interactive web dashboard:
```bash
python main.py --web
```
Then open `http://localhost:5000` in your browser.

## Model Details

- **Embedding Model**: `jinaai/jina-code-embeddings-1.5b` (quantized to 4-bit)
- **Reranker**: `jinaai/jina-reranker-v2-base-multilingual`
- **Summarizer**: `gemini-2.5-flash-lite`
- **Answerer**: `gemini-3-flash-preview`

## Project Structure

```text
src/
├── graph/       # Relationship extraction (Python/C)
├── IR/          # Information Retrieval data models
├── model/       # LLM, Embeddings, and Reranker logic
├── parsers/     # Tree-sitter integration and chunking
├── storage/     # Chroma, FalkorDB, and SQLite backends
├── search.py    # Hybrid search orchestrator
├── server.py    # Flask API and Web server
└── indexer.py   # Main indexing pipeline
```

## Supported Languages

- **Python** 
- **C** 
