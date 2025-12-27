import logging
import time
from typing import List, Dict, Any, Optional
from src.indexer import ProjectIndexer
from src.model.orchestrator import Orchestrator
from src.model.reranker import get_reranker

logger = logging.getLogger(__name__)

ANSWER_PROMPT = """You are an expert software engineer assistant. Answer the user's question about the codebase based on the provided code snippets and their context.

User Question: {query}

Relevant Code Snippets:
{snippets_context}

Instructions:
1. Provide a direct and technical answer to the user's question.
2. Use markdown formatting for better readability (e.g., **bold** for emphasis, `code` for identifiers, lists for steps).
3. Reference specific file paths and line numbers where appropriate.
4. If the code snippets don't contain enough information to answer the question, state that clearly.
5. Use the summary information if available to provide high-level context.

Answer:"""

class SearchManager:
    def __init__(self, indexer: ProjectIndexer, orchestrator: Optional[Orchestrator] = None, use_reranker: bool = True):
        self.indexer = indexer
        self.llm = indexer.llm
        self.embedding_model = indexer.embedding_model
        self.sqlite = indexer.sqlite
        self.graph_db = indexer.graph_db
        self.chroma = indexer.chroma
        self.orchestrator = orchestrator
        self.reranker = get_reranker() if use_reranker else None

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Performs a semantic search for snippets and hydrates them with metadata and relations.
        Returns a dictionary containing results and orchestration metadata.
        """
        start_time = time.time()
        if not self.chroma or not self.embedding_model:
            logger.error("Search components not initialized.")
            return {"results": [], "final_query": query, "hyde_used": False}

        # Check if the vector database is empty
        count = self.chroma.collection.count()
        if count == 0:
            logger.warning("Vector database is empty. Please run indexing first.")
            return {"results": [], "final_query": query, "hyde_used": False}

        # 0. Orchestrate the query (HyDE)
        final_query = query
        hyde_used = False
        if self.orchestrator:
            o_start = time.time()
            print(f"\n[Search Pipeline] 0. Orchestrating query: '{query}'")
            final_query = self.orchestrator.process_query(query)
            hyde_used = (final_query != query)
            if hyde_used:
                print("[Search Pipeline] HyDE augmented query generated.")
            print(f"[Search Pipeline] 0. Orchestration took {time.time() - o_start:.2f}s")

        # 1. Embed the query
        e_start = time.time()
        print("[Search Pipeline] 1. Embedding final query...")
        query_embedding = self.embedding_model.embed_text(final_query)
        print(f"[Search Pipeline] 1. Embedding took {time.time() - e_start:.2f}s")
        
        # 2. Get similar snippets from ChromaDB
        initial_n = max(n_results * 10, 50) if self.reranker else n_results
        r_start = time.time()
        print(f"[Search Pipeline] 2. Retrieving top {initial_n} candidates from ChromaDB...")
        results = self.chroma.query(query_embedding, n_results=initial_n)
        print(f"[Search Pipeline] 2. Retrieval took {time.time() - r_start:.2f}s")
        
        # 2.1 Rerank results if reranker is available
        if self.reranker and results:
            rk_start = time.time()
            print(f"[Search Pipeline] 2.1 Reranking {len(results)} results with {self.reranker.model_id}...")
            logger.info(f"Reranking {len(results)} results...")
            documents = [res["document"] for res in results]
            reranked_results_meta = self.reranker.rerank(query, documents, top_n=n_results)
            
            # Reorder original results based on reranker indices and update scores
            new_results = []
            for meta in reranked_results_meta:
                idx = meta["index"]
                res = results[idx]
                res["distance"] = meta["score"]  # Update score to reranker score
                new_results.append(res)
            results = new_results
            print(f"[Search Pipeline] 2.1 Reranking took {time.time() - rk_start:.2f}s")

        hydrated_results = []
        h_start = time.time()
        print("[Search Pipeline] 3. Hydrating results with metadata and relationships...")
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
            
        print(f"[Search Pipeline] 3. Hydration took {time.time() - h_start:.2f}s")
        print(f"[Search Pipeline] Search complete. {len(hydrated_results)} results hydrated in {time.time() - start_time:.2f}s\n")
        return {
            "results": hydrated_results,
            "final_query": final_query,
            "hyde_used": hyde_used
        }

    def answer_query(self, query: str, results: List[Dict[str, Any]]) -> str:
        """
        Uses Gemini LLM to generate an answer based on search results.
        """
        start_time = time.time()
        if not self.llm:
            return "LLM not available for answering."
        
        if not results:
            return "No search results to base an answer on."

        snippets_context = ""
        for i, res in enumerate(results, 1):
            s = res["snippet"]
            parent = res.get("parent")
            relations = res.get("relations", [])
            parent_info = f" [in {parent.name} ({parent.type.value})]" if parent else ""
            
            snippets_context += f"--- Snippet {i} [{s.name}{parent_info} at {s.file_path}:{s.start_line + 1}] ---\n"
            if s.summary:
                snippets_context += f"Summary: {s.summary}\n"
            
            if relations:
                rel_str = ", ".join([f"{rel_type} {target}" for rel_type, target in relations])
                snippets_context += f"Relationships: {rel_str}\n"
                
            snippets_context += f"Code:\n{s.content}\n\n"

        prompt = ANSWER_PROMPT.format(query=query, snippets_context=snippets_context)
        
        print("[Search Pipeline] 4. Generating final answer with Gemini LLM...")
        logger.info("Generating answer with Gemini...")
        answer = self.llm.complete(prompt)
        print(f"[Search Pipeline] 4. LLM Generation took {time.time() - start_time:.2f}s")
        return answer

    def stream_answer_query(self, query: str, results: List[Dict[str, Any]]):
        """
        Yields tokens for the answer as they are generated.
        """
        if not self.llm:
            yield "LLM not available for answering."
            return
        
        if not results:
            yield "No search results to base an answer on."
            return

        snippets_context = ""
        for i, res in enumerate(results, 1):
            s = res["snippet"]
            parent = res.get("parent")
            relations = res.get("relations", [])
            parent_info = f" [in {parent.name} ({parent.type.value})]" if parent else ""
            
            snippets_context += f"--- Snippet {i} [{s.name}{parent_info} at {s.file_path}:{s.start_line + 1}] ---\n"
            if s.summary:
                snippets_context += f"Summary: {s.summary}\n"
            
            if relations:
                rel_str = ", ".join([f"{rel_type} {target}" for rel_type, target in relations])
                snippets_context += f"Relationships: {rel_str}\n"
                
            snippets_context += f"Code:\n{s.content}\n\n"

        prompt = ANSWER_PROMPT.format(query=query, snippets_context=snippets_context)
        
        print("[Search Pipeline] 4. Starting streaming answer with Gemini LLM...")
        for token in self.llm.stream_complete(prompt):
            yield token

