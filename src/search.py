import logging
from typing import List, Dict, Any, Optional
from src.indexer import ProjectIndexer
from src.model.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

ANSWER_PROMPT = """You are an expert software engineer assistant. Answer the user's question about the codebase based on the provided code snippets and their context.

User Question: {query}

Relevant Code Snippets:
{snippets_context}

Instructions:
1. Provide a direct and technical answer to the user's question as plain text.
2. DO NOT use JSON format in your response.
3. Reference specific file paths and line numbers where appropriate.
4. If the code snippets don't contain enough information to answer the question, state that clearly.
5. Use the summary information if available to provide high-level context.

Answer:"""

class SearchManager:
    def __init__(self, indexer: ProjectIndexer, orchestrator: Optional[Orchestrator] = None):
        self.indexer = indexer
        self.llm = indexer.llm
        self.embedding_model = indexer.embedding_model
        self.sqlite = indexer.sqlite
        self.graph_db = indexer.graph_db
        self.chroma = indexer.chroma
        self.orchestrator = orchestrator

    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Performs a semantic search for snippets and hydrates them with metadata and relations.
        """
        if not self.chroma or not self.embedding_model:
            logger.error("Search components not initialized.")
            return []

        # Check if the vector database is empty
        count = self.chroma.collection.count()
        if count == 0:
            logger.warning("Vector database is empty. Please run indexing first.")
            return []

        # 0. Orchestrate the query (HyDE)
        final_query = query
        if self.orchestrator:
            final_query = self.orchestrator.process_query(query)

        # 1. Embed the query
        query_embedding = self.embedding_model.embed_text(final_query)
        
        # 2. Get similar snippets from ChromaDB
        results = self.chroma.query(query_embedding, n_results=n_results)
        
        hydrated_results = []
        for res in results:
            snippet_id = res["id"]
            
            # 3. Fetch full snippet data from SQLite
            snippet = self.sqlite.get_snippet(snippet_id) if self.sqlite else None
            if not snippet:
                continue

            # 3.1 Fetch parent snippet for context
            parent_snippet = None
            if snippet.parent_id:
                parent_snippet = self.sqlite.get_snippet(snippet.parent_id)

            # 4. Fetch relations from FalkorDB
            relations = self.graph_db.get_snippet_relationships(snippet_id) if self.graph_db else []
            
            hydrated_results.append({
                "snippet": snippet,
                "parent": parent_snippet,
                "score": res["distance"],
                "relations": relations,
                "document": res["document"]
            })
            
        return hydrated_results

    def answer_query(self, query: str, results: List[Dict[str, Any]]) -> str:
        """
        Uses DeepSeek LLM to generate an answer based on search results.
        """
        if not self.llm:
            return "LLM not available for answering."
        
        if not results:
            return "No search results to base an answer on."

        snippets_context = ""
        for i, res in enumerate(results, 1):
            s = res["snippet"]
            parent = res.get("parent")
            parent_info = f" [in {parent.name} ({parent.type.value})]" if parent else ""
            
            snippets_context += f"--- Snippet {i} [{s.name}{parent_info} at {s.file_path}:{s.start_line + 1}] ---\n"
            if s.summary:
                snippets_context += f"Summary: {s.summary}\n"
            snippets_context += f"Code:\n{s.content}\n\n"

        prompt = ANSWER_PROMPT.format(query=query, snippets_context=snippets_context)
        
        logger.info("Generating answer with DeepSeek...")
        answer = self.llm.complete(prompt)
        return answer

